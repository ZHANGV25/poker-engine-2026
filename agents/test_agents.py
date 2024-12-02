import random
from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType


class FoldAgent(Agent):
    @staticmethod
    def name():
        return "FoldAgent"

    def act(self, observation, reward, terminated, truncated, info):
        action_type = action_types.FOLD.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class CallingStationAgent(Agent):
    @staticmethod
    def name():
        return "CallingStationAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        else:
            action_type = action_types.CHECK.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class AllInAgent(Agent):
    @staticmethod
    def name():
        return "AllInAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            raise_amount = observation["max_raise"]
        elif observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
            raise_amount = 0
        else:
            action_type = action_types.CHECK.value
            raise_amount = 0
        
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class RandomAgent(Agent):
    @staticmethod
    def name():
        return "RandomAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = [i for i, is_valid in enumerate(observation["valid_actions"]) if is_valid]
        action_type = random.choice(valid_actions)
        
        if action_type == action_types.RAISE.value:
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
        else:
            raise_amount = 0
            
        card_to_discard = -1
        if observation["valid_actions"][action_types.DISCARD.value]:
            card_to_discard = random.randint(0, 1)
            
        return action_type, raise_amount, card_to_discard


all_agent_classes = (FoldAgent, CallingStationAgent, AllInAgent, RandomAgent)