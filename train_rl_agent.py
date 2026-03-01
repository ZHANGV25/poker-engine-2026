import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import our poker environment and opponent agent classes.
from gym_env import PokerEnv
from agents.agent import Agent
from agents.prob_agent import ProbabilityAgent

# --- Helper Functions for Preprocessing and Equity Calculation ---

def compute_equity(obs, num_simulations=100):
    """
    Compute an approximate equity (win probability) given the observation.
    New variant: 27-card deck, opp_discarded_cards (3 cards). Uses first 2 hole cards for equity.
    """
    my_cards_raw = [c for c in obs["my_cards"] if c != -1]
    my_cards = my_cards_raw[:2] if len(my_cards_raw) >= 2 else my_cards_raw
    if len(my_cards) != 2:
        return 0.5
    community_cards = [c for c in obs["community_cards"] if c != -1]
    opp_discarded = list(obs.get("opp_discarded_cards", [-1, -1, -1]))
    shown_cards = set(my_cards) | set(community_cards) | {c for c in opp_discarded if c != -1}
    deck = list(range(27))
    non_shown_cards = [card for card in deck if card not in shown_cards]

    wins = 0
    for _ in range(num_simulations):
        opp_needed = 2
        board_needed = 5 - len(community_cards)
        sample_size = opp_needed + board_needed
        if sample_size > len(non_shown_cards):
            continue
        sample = random.sample(non_shown_cards, sample_size)
        opp_full = sample[:opp_needed]
        community_full = community_cards + sample[opp_needed:]

        my_hand = [PokerEnv.int_to_card(card) for card in my_cards]
        opp_hand = [PokerEnv.int_to_card(card) for card in opp_full]
        board = [PokerEnv.int_to_card(card) for card in community_full]
        evaluator = PokerEnv().evaluator
        my_rank = evaluator.evaluate(my_hand, board)
        opp_rank = evaluator.evaluate(opp_hand, board)
        if my_rank < opp_rank:
            wins += 1
    return wins / num_simulations if num_simulations > 0 else 0.0

# New variant: 5 hole card slots, 27-card deck. Feature dim = 1 + 5 + 5 + 1+1+1+1+1 = 16
INPUT_DIM = 16

def preprocess_observation(obs):
    """
    Converts the observation into a feature tensor for the new variant.
    Features: street(1), my_cards(5), community_cards(5), my_bet, opp_bet, min_raise, max_raise, equity(1).
    Cards normalized (card+1)/28; -1 -> 0.
    """
    street = np.array([obs["street"] / 3.0])
    my_cards = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["my_cards"]])
    if len(my_cards) < 5:
        my_cards = np.resize(my_cards, 5)
    community_cards = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["community_cards"]])
    my_bet = np.array([obs["my_bet"] / 100.0])
    opp_bet = np.array([obs["opp_bet"] / 100.0])
    min_raise = np.array([obs["min_raise"] / 100.0])
    max_raise = np.array([obs["max_raise"] / 100.0])
    equity = np.array([compute_equity(obs)])
    features = np.concatenate([street, my_cards[:5], community_cards[:5], my_bet, opp_bet, min_raise, max_raise, equity])
    return torch.tensor(features, dtype=torch.float32)

# --- Define the Policy Network ---

# Which pair of indices (out of 5) to keep when discarding: (0,1), (0,2), ..., (3,4)
KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)  # 10


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=NUM_DISCARD_CLASSES):
        """
        New variant: shared base and three heads.
          - Action type: 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise: 100 classes -> [1, 100]
          - Discard: 10 classes -> which pair (i,j) of 5 hole cards to keep
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        return action_type_logits, raise_logits, discard_logits

# --- Define the RL Agent using REINFORCE ---

class RLAgent:
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = 0.99

    def select_action(self, state, valid_actions, min_raise, max_raise):
        """
        Sample an action. Returns (action_type, raise_amount, keep1, keep2), log_prob.
        For DISCARD, keep1 and keep2 are the two indices (0-4) to keep from 5 hole cards.
        """
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        mask = (valid_actions == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9

        action_type_dist = torch.distributions.Categorical(logits=masked_logits)
        raise_dist = torch.distributions.Categorical(logits=raise_logits)
        discard_dist = torch.distributions.Categorical(logits=discard_logits)

        action_type = action_type_dist.sample()
        raise_amount = raise_dist.sample()
        discard_idx = discard_dist.sample()

        log_prob = (action_type_dist.log_prob(action_type) +
                    raise_dist.log_prob(raise_amount) +
                    discard_dist.log_prob(discard_idx))

        action_type = action_type.item()
        raise_amount = raise_amount.item() + 1
        if action_type == PokerEnv.ActionType.RAISE.value:
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
        else:
            raise_amount = 0

        if action_type == PokerEnv.ActionType.DISCARD.value:
            keep1, keep2 = KEEP_PAIRS[discard_idx.item() % NUM_DISCARD_CLASSES]
        else:
            keep1, keep2 = 0, 0

        return (action_type, raise_amount, keep1, keep2), log_prob

    def update_policy(self, trajectory):
        R = 0
        returns = []
        for _, r in reversed(trajectory):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            returns = returns - returns.mean()
        loss = 0
        for (log_prob, _), R in zip(trajectory, returns):
            loss += -log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- Training Loop with Opponent Agent, CUDA support, and Weight Saving ---

def train_agent(num_episodes=500, save_every=50, weight_path=None):
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__), "agents", "rl_agent_weights.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = PokerEnv()
    agent = RLAgent(input_dim=INPUT_DIM)
    agent.policy_net.to(device)
    
    # Use ProbabilityAgent as the opponent.
    opponent_agent = ProbabilityAgent()
    print(f"Using opponent: {opponent_agent.__name__()}")

    for episode in range(num_episodes):
        obs, info = env.reset()
        trajectory = []  # stores (log_prob, reward) for RL agent's actions
        total_reward = 0
        done = False

        while not done:
            acting_agent = obs[0]["acting_agent"]
            if acting_agent == 0:
                # RL agent's turn (player 0)
                state = preprocess_observation(obs[0]).to(device)
                valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device=device)
                min_raise_val = obs[0]["min_raise"]
                max_raise_val = obs[0]["max_raise"]
                action, log_prob = agent.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
                our_turn = True
            else:
                # Opponent's turn (player 1) using the provided opponent agent.
                action = opponent_agent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                our_turn = False
                log_prob = None

            obs, reward, done, truncated, info = env.step(action)
            r = reward[0]  # RL agent's reward is at index 0
            total_reward += r
            if our_turn:
                trajectory.append((log_prob, r))

        agent.update_policy(trajectory)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

        # Save weights every 'save_every' episodes.
        if (episode + 1) % save_every == 0:
            torch.save(agent.policy_net.state_dict(), weight_path)
            print(f"Saved weights to {weight_path}")

    # Final weight save.
    torch.save(agent.policy_net.state_dict(), weight_path)
    print(f"Final weights saved to {weight_path}")

if __name__ == "__main__":
    train_agent(num_episodes=500)
