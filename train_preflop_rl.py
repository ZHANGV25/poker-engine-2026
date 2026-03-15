"""
PPO training script for the RL-based preflop strategy.

Trains a small network (4 inputs -> 64 -> 64 -> 6 actions) to play preflop
poker. Post-flop decisions are handled by the existing CFR bot logic.

The opponent is a copy of the full CFR bot (submission/player.py).

Usage:
    python train_preflop_rl.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add submission to path so we can import the bot components
_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(_dir, "submission")
if sub_dir not in sys.path:
    sys.path.insert(0, sub_dir)

from gym_env import PokerEnv
from submission.preflop_rl import PreflopNetwork, NUM_ACTIONS, RAISE_LEVELS
from submission.equity import ExactEquityEngine
from submission.inference import DiscardInference
from submission.solver import SubgameSolver

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value


# ----------------------------------------------------------------
#  Opponent: Full CFR Bot
# ----------------------------------------------------------------

class CFROpponent:
    """Wraps the submission PlayerAgent for use as an opponent during training."""

    def __init__(self):
        from submission.player import PlayerAgent
        self._agent = PlayerAgent.__new__(PlayerAgent)
        # Manually init without the FastAPI server
        self._agent.engine = ExactEquityEngine()
        self._agent.inference = DiscardInference(self._agent.engine)
        self._agent.solver = SubgameSolver(self._agent.engine)
        self._agent._preflop_table = self._agent._load_preflop_table()
        self._agent._preflop_strategy = self._agent._load_preflop_strategy()
        self._agent._preflop_mode = "cfr"
        self._agent._preflop_rl = None
        self._agent._current_hand = -1
        self._agent._opp_weights = None

    def reset_hand(self, hand_number):
        self._agent._reset_hand(hand_number)

    def act(self, observation, info):
        return self._agent.act(observation, reward=0, terminated=False,
                               truncated=False, info=info)

    def observe(self, observation, reward, terminated, truncated, info):
        self._agent.observe(observation, reward, terminated, truncated, info)


# ----------------------------------------------------------------
#  RL Agent: Preflop Only, Delegates Post-flop to CFR
# ----------------------------------------------------------------

class RLPreflopAgent:
    """Agent that uses RL for preflop and the full CFR bot for post-flop."""

    def __init__(self):
        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)

        # Load preflop potential table
        data_path = os.path.join(sub_dir, "data", "preflop_potential.npz")
        data = np.load(data_path)
        self._preflop_table = dict(zip(
            data["bitmasks"].tolist(), data["potentials"].tolist()
        ))

        # Per-hand state
        self._opp_weights = None
        self._current_hand = -1

    def reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None

    def preflop_potential(self, my_cards):
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask, 0.5)

    def postflop_act(self, observation, info):
        """Handle all non-preflop decisions using the CFR bot logic."""
        my_cards = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        valid_actions = observation["valid_actions"]

        # Discard
        if valid_actions[DISCARD]:
            if len(opp_discards) == 3 and self._opp_weights is None:
                self._opp_weights = self.inference.infer_opponent_weights(
                    opp_discards, board, my_cards
                )
            results = self.engine.evaluate_all_keep_pairs(
                my_cards, board, opp_discards, self._opp_weights
            )
            best_keep = results[0][0]
            return (DISCARD, 0, best_keep[0], best_keep[1])

        # Post-flop betting via solver
        dead_cards = my_discards + opp_discards
        opp_range = self._opp_weights
        return self.solver.solve_and_act(
            hero_cards=my_cards,
            board=board,
            opp_range=opp_range,
            dead_cards=dead_cards,
            my_bet=observation["my_bet"],
            opp_bet=observation["opp_bet"],
            street=observation["street"],
            min_raise=observation["min_raise"],
            max_raise=observation["max_raise"],
            valid_actions=valid_actions,
            hero_is_first=True,
            time_remaining=400,
        )

    def observe(self, observation, reward, terminated, truncated, info):
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )


# ----------------------------------------------------------------
#  PPO Trainer
# ----------------------------------------------------------------

class PPOTrainer:
    def __init__(self, lr_policy=3e-4, lr_value=1e-3, gamma=0.99,
                 clip_eps=0.2, epochs_per_update=4, batch_size=64):
        self.device = torch.device("cpu")
        self.net = PreflopNetwork().to(self.device)
        self.optimizer = optim.Adam([
            {"params": self.net.net.parameters(), "lr": lr_policy},
            {"params": self.net.policy_head.parameters(), "lr": lr_policy},
            {"params": self.net.value_head.parameters(), "lr": lr_value},
        ])
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

    def _build_state(self, potential, position, my_bet, opp_bet):
        return torch.tensor(
            [potential, float(position), my_bet / 100.0, opp_bet / 100.0],
            dtype=torch.float32, device=self.device,
        )

    def _get_valid_mask(self, valid_actions, opp_bet, min_raise, max_raise):
        mask = torch.full((NUM_ACTIONS,), -1e9, device=self.device)
        if valid_actions[FOLD]:
            mask[0] = 0.0
        if valid_actions[CALL] or valid_actions[CHECK]:
            mask[1] = 0.0
        if valid_actions[RAISE]:
            any_valid_raise = False
            for i, level in enumerate(RAISE_LEVELS):
                ra = level - opp_bet
                if min_raise <= ra <= max_raise:
                    mask[2 + i] = 0.0
                    any_valid_raise = True
            if not any_valid_raise:
                mask[2] = 0.0  # allow smallest raise as fallback
        return mask

    def _map_action(self, action_idx, opp_bet, min_raise, max_raise, valid_actions):
        if action_idx == 0:
            return (FOLD, 0, 0, 0)
        elif action_idx == 1:
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)
        else:
            level = RAISE_LEVELS[action_idx - 2]
            raise_amount = level - opp_bet
            raise_amount = max(raise_amount, min_raise)
            raise_amount = min(raise_amount, max_raise)
            if raise_amount <= 0 or not valid_actions[RAISE]:
                if valid_actions[CALL]:
                    return (CALL, 0, 0, 0)
                if valid_actions[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            return (RAISE, raise_amount, 0, 0)

    def select_preflop_action(self, state, valid_mask):
        """Select action and return (action_idx, log_prob, value)."""
        with torch.no_grad():
            logits, value = self.net(state)
        masked_logits = logits + valid_mask
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def collect_batch(self, env, rl_agent, opponent, num_hands):
        """Play num_hands and collect preflop transitions.

        Returns lists of: states, actions, log_probs, values, rewards.
        Each entry corresponds to one preflop decision by the RL agent.
        """
        states = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []

        for hand_num in range(num_hands):
            obs, info = env.reset()
            info["hand_number"] = hand_num
            rl_agent.reset_hand(hand_num)
            opponent.reset_hand(hand_num)

            done = False
            preflop_data = []  # (state, action_idx, log_prob, value)
            rl_player = 0  # RL is always player 0

            while not done:
                acting = obs[0]["acting_agent"]

                if acting == rl_player:
                    observation = obs[rl_player]
                    is_preflop = observation["street"] == 0 and not observation["valid_actions"][DISCARD]

                    if is_preflop:
                        my_cards = [c for c in observation["my_cards"] if c != -1]
                        potential = rl_agent.preflop_potential(my_cards)
                        position = observation.get("blind_position", 0)
                        state = self._build_state(
                            potential, position,
                            observation["my_bet"], observation["opp_bet"]
                        )
                        valid_mask = self._get_valid_mask(
                            observation["valid_actions"],
                            observation["opp_bet"],
                            observation["min_raise"],
                            observation["max_raise"],
                        )
                        action_idx, log_prob, value = self.select_preflop_action(
                            state, valid_mask)
                        action = self._map_action(
                            action_idx, observation["opp_bet"],
                            observation["min_raise"], observation["max_raise"],
                            observation["valid_actions"],
                        )
                        preflop_data.append((state, action_idx, log_prob, value))
                    else:
                        action = rl_agent.postflop_act(observation, info)
                else:
                    observation = obs[1 - rl_player]
                    action = opponent.act(observation, info)

                obs, reward, done, truncated, info = env.step(action)
                info["hand_number"] = hand_num

                # Let both agents observe
                if acting != rl_player:
                    opponent.observe(obs[1 - rl_player], reward[1 - rl_player],
                                     done, truncated, info)
                else:
                    rl_agent.observe(obs[rl_player], reward[rl_player],
                                     done, truncated, info)

            # Hand over — assign terminal reward to all preflop decisions
            hand_reward = reward[rl_player]
            for state_t, action_idx_t, log_prob_t, value_t in preflop_data:
                states.append(state_t)
                actions.append(action_idx_t)
                old_log_probs.append(log_prob_t)
                values.append(value_t)
                rewards.append(hand_reward)

        return states, actions, old_log_probs, values, rewards

    def update(self, states, actions, old_log_probs, values, rewards):
        """PPO update using collected batch data."""
        if not states:
            return 0.0

        states_t = torch.stack(states).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Normalize rewards
        if rewards_t.std() > 1e-5:
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # Advantages
        advantages = rewards_t - values_t
        if advantages.std() > 1e-5:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs_per_update):
            logits, new_values = self.net(states_t)
            new_values = new_values.squeeze(-1)

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values, rewards_t)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.epochs_per_update

    def save_weights(self, path=None):
        if path is None:
            path = os.path.join(sub_dir, "data", "preflop_rl_weights.pth")
        torch.save(self.net.state_dict(), path)
        print(f"Saved weights to {path}")


# ----------------------------------------------------------------
#  Main Training Loop
# ----------------------------------------------------------------

def train(num_epochs=500, hands_per_epoch=64, save_interval=25):
    print("Initializing...")
    env = PokerEnv()
    rl_agent = RLPreflopAgent()
    opponent = CFROpponent()
    trainer = PPOTrainer()

    print(f"Training for {num_epochs} epochs, {hands_per_epoch} hands each")
    print(f"Total hands: {num_epochs * hands_per_epoch}")

    running_reward = 0.0
    for epoch in range(num_epochs):
        states, actions, old_log_probs, values, rewards = trainer.collect_batch(
            env, rl_agent, opponent, hands_per_epoch
        )

        loss = trainer.update(states, actions, old_log_probs, values, rewards)

        # Track performance
        epoch_reward = sum(rewards) / max(len(rewards), 1)
        running_reward = 0.95 * running_reward + 0.05 * epoch_reward

        if (epoch + 1) % 10 == 0:
            n_preflop = len(states)
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Avg reward: {epoch_reward:.2f} | "
                  f"Running: {running_reward:.2f} | "
                  f"Preflop decisions: {n_preflop}")

        if (epoch + 1) % save_interval == 0:
            trainer.save_weights()

    # Final save
    trainer.save_weights()
    print("Training complete.")


if __name__ == "__main__":
    train()
