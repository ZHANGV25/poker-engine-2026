import random
import torch
import numpy as np
from agents.agent import Agent
from gym_env import PokerEnv

# Import the helper functions and PolicyNetwork from the training file.
# Adjust the import path as needed.
from train_rl_agent import PolicyNetwork, preprocess_observation

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class ProbabilityAgent(Agent):
    def __name__(self):
        return "ProbabilityAgent"

    def __init__(self, weight_path="rl_agent_weights.pth", stream: bool = False):
        super().__init__(stream)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the policy network with the same dimensions as during training.
        self.policy_net = PolicyNetwork(input_dim=13)
        self.policy_net.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.policy_net.to(self.device)
        self.policy_net.eval()

    def select_action(self, state, valid_actions, min_raise, max_raise):
        """
        Same as during training, but without gradient tracking.
        """
        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits = self.policy_net(state)
            mask = (valid_actions == 0)
            masked_logits = action_type_logits.clone()
            masked_logits[mask] = -1e9

            action_type_dist = torch.distributions.Categorical(logits=masked_logits)
            raise_dist = torch.distributions.Categorical(logits=raise_logits)
            discard_dist = torch.distributions.Categorical(logits=discard_logits)

            action_type = action_type_dist.sample()
            raise_amount = raise_dist.sample()
            discard_action = discard_dist.sample()

            action_type = action_type.item()
            raise_amount = raise_amount.item() + 1
            if action_type == PokerEnv.ActionType.RAISE.value:
                raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            else:
                raise_amount = 0

            discard_action = discard_action.item() - 1
            if action_type == PokerEnv.ActionType.DISCARD.value:
                if discard_action < 0:
                    discard_action = 0
            else:
                discard_action = -1

            return (action_type, raise_amount, discard_action)

    def act(self, observation, reward, terminated, truncated, info):
        # Preprocess the observation.
        state = preprocess_observation(observation).to(self.device)
        valid_actions_tensor = torch.tensor(observation["valid_actions"], dtype=torch.float32).to(self.device)
        min_raise_val = observation["min_raise"]
        max_raise_val = observation["max_raise"]
        action = self.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:
            self.logger.info(f"Significant hand completed with reward: {reward}")
