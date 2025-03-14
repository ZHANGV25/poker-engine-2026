from agents.agent import Agent
from gym_env import PokerEnv
import random

action_types = PokerEnv.ActionType


class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0

    def act(self, observation, reward, terminated, truncated, info):
        # Example of using the logger
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']}")

        # Always check the valid actions
        valid_actions = observation["valid_actions"]
        action_type = random.choice(valid_actions)
        return action_type, 0, -1
        if action_types.RAISE in valid_actions:
            action_type = action_types.RAISE
            
            # Always raise within the min and max raise
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
        else:
            action_type = random.choice(valid_actions)
            raise_amount = 0

        # If you discard, you will get to play another action immediately after (next act call)
        if action_type == action_types.DISCARD:
            card_to_discard = random.randint(0, 1)
        else:
            card_to_discard = -1

        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")