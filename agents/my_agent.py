import random
from agents.agent import Agent
from gym_env import PokerEnv
from agents.prob_agent import ProbabilityAgent

action_types = PokerEnv.ActionType


class MyAgent(Agent):
    def __name__(self):
        return "MyAgent"

    def __init__(self, logger=None):
        super().__init__(logger)
        # Initialize any instance variables here

    def act(self, observation, reward, terminated, truncated, info):
        # Example of using the logger
        self.logger.info(f"Street: {observation['street']}, Bankroll: {observation.get('my_bankroll', 0)}")

        # For now, we'll use the ProbabilityAgent's logic
        prob_agent = ProbabilityAgent(self.logger)
        return prob_agent.act(observation, reward, terminated, truncated, info)
        # valid_actions = observation["valid_actions"]
        # action_type = random.choice(valid_actions)
        # raise_amount = observation["min_raise"]
        # card_to_discard = -1
        # return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
