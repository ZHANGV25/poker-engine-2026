"""
RL-based preflop strategy using a small PPO-trained network.

Replaces the precomputed CFR preflop strategy with a learned policy.
The network maps (potential, position, my_bet, opp_bet) to a distribution
over 6 actions: fold, call/check, raise_4, raise_10, raise_30, all-in.

Weights are loaded from data/preflop_rl_weights.pth (trained offline).
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gym_env import PokerEnv

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value

# Action space: fold, call/check, raise to 4, raise to 10, raise to 30, all-in
NUM_ACTIONS = 6
RAISE_LEVELS = [4, 10, 30, 100]

INPUT_DIM = 4
HIDDEN_DIM = 64
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "preflop_rl_weights.pth")


class PreflopNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN_DIM, NUM_ACTIONS)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        h = self.net(x)
        return self.policy_head(h), self.value_head(h)

    def policy(self, x):
        h = self.net(x)
        return self.policy_head(h)


class PreflopRL:
    def __init__(self):
        self.available = False
        if not HAS_TORCH:
            return
        if not os.path.exists(WEIGHTS_PATH):
            return

        self.device = torch.device("cpu")
        self.net = PreflopNetwork().to(self.device)
        state = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        self.available = True

    def get_action(self, potential, position, my_bet, opp_bet,
                   valid_actions, min_raise, max_raise):
        if not self.available:
            return None

        state = torch.tensor(
            [potential, float(position), my_bet / 100.0, opp_bet / 100.0],
            dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            logits = self.net.policy(state)

        # Build validity mask for the 6 actions
        # [fold, call/check, raise_4, raise_10, raise_30, all-in]
        mask = torch.full((NUM_ACTIONS,), -1e9)
        if valid_actions[FOLD]:
            mask[0] = 0.0
        if valid_actions[CALL] or valid_actions[CHECK]:
            mask[1] = 0.0
        if valid_actions[RAISE]:
            for i, level in enumerate(RAISE_LEVELS):
                raise_amount = level - opp_bet
                if min_raise <= raise_amount <= max_raise:
                    mask[2 + i] = 0.0

        masked_logits = logits + mask

        # If no raise levels are valid but raise is valid, allow closest level
        if valid_actions[RAISE] and (mask[2:] == -1e9).all():
            mask[2] = 0.0  # allow smallest raise as fallback
            masked_logits = logits + mask

        probs = torch.softmax(masked_logits, dim=0)

        # Deterministic if one action dominates, else sample
        if probs.max() > 0.90:
            action_idx = int(torch.argmax(probs))
        else:
            action_idx = int(torch.multinomial(probs, 1))

        return self._map_action(action_idx, opp_bet, min_raise, max_raise, valid_actions)

    def _map_action(self, action_idx, opp_bet, min_raise, max_raise, valid_actions):
        if action_idx == 0:
            return (FOLD, 0, 0, 0)
        elif action_idx == 1:
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)
        else:
            # Raise actions (indices 2-5 map to RAISE_LEVELS [4, 10, 30, 100])
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
