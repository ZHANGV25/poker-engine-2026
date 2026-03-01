"""
RL agent for the new variant: 27-card deck, 5 hole cards, mandatory discard to 2 on the flop.
Uses PolicyNetwork from train_rl_agent (input_dim=16, discard head has 10 classes for which pair to keep).
"""
import os
import torch
from agents.agent import Agent
from gym_env import PokerEnv
from train_rl_agent import (
    PolicyNetwork,
    preprocess_observation,
    INPUT_DIM,
    KEEP_PAIRS,
    NUM_DISCARD_CLASSES,
)

action_types = PokerEnv.ActionType

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl_agent_weights.pth")


class RLAgent(Agent):
    def __name__(self):
        return "RLAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(input_dim=INPUT_DIM, num_discard_classes=NUM_DISCARD_CLASSES)
        if os.path.exists(WEIGHTS_PATH):
            state = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
            try:
                self.policy_net.load_state_dict(state, strict=True)
                self.logger.info(f"Loaded RL weights from {WEIGHTS_PATH}")
            except Exception as e:
                self.logger.warning(f"Could not load weights (wrong shape for new variant?): {e}. Using random policy.")
        else:
            self.logger.warning(f"No weights found at {WEIGHTS_PATH}, using random policy")
        self.policy_net.to(self.device)
        self.policy_net.eval()

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        state = preprocess_observation(observation).to(self.device)
        valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits = self.policy_net(state)

            mask = valid_actions_tensor == 0
            action_type_logits = action_type_logits.clone()
            action_type_logits[mask] = -1e9

            action_type = torch.distributions.Categorical(logits=action_type_logits).sample().item()

            if action_type == action_types.RAISE.value:
                raise_amount = torch.distributions.Categorical(logits=raise_logits).sample().item() + 1
                raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            else:
                raise_amount = 0

            if action_type == action_types.DISCARD.value:
                discard_idx = torch.distributions.Categorical(logits=discard_logits).sample().item()
                keep1, keep2 = KEEP_PAIRS[discard_idx % NUM_DISCARD_CLASSES]
            else:
                keep1, keep2 = 0, 0

        return (action_type, raise_amount, keep1, keep2)

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand ended with reward: {reward}")
