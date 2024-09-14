import random
from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType


class FoldAgent(Agent):
    def act(self, observation, reward, terminated, truncated, info):
        # fold instantly by betting -1
        bet, reveal_card = -1, -1
        return bet, reveal_card


# TODO: Implement the following agents
class CallingStationAgent(Agent):
    # Always calls/checks
    def act(self, observation, reward, terminated, truncated, info):
        # observation["street"] == 1 marks the flop, where we discard a card
        if observation["street"] == 1:
            # discard & reveal arbitrary card
            discard_card = 0
        else:
            discard_card = -1

        my_bet = observation["opp_bet"]
        return my_bet, discard_card


class AllInAgent(Agent):
    # Always goes all in
    def act(self, observation, reward, terminated, truncated, info):
        # observation["street"] == 1 marks the flop, where we discard a card
        if observation["street"] == 1:
            # discard & reveal arbitrary card
            discard_card = 0
        else:
            discard_card = -1

        my_bet = 100  # always put in the max pot from the start
        return my_bet, discard_card


class RandomAgent(Agent):
    # Randomly chooses an action
    def act(self, observation, reward, terminated, truncated, info):
        # randomly choose whether to fold, call, or raise
        action = random.choice(
            [action_types.CALL, action_types.FOLD, action_types.RAISE]
        )

        if action == action_types.FOLD:
            return -1, -1

        discard_card = -1
        if observation["street"] == 1:
            discard_card = random.choice([0, 1, 2])

        if action == action_types.CALL:
            ammount_to_call = observation["opp_bet"] - observation["my_bet"]
            return ammount_to_call, discard_card

        if action == action_types.RAISE:
            max_raise = 100
            min_raise = observation["min_raise"]
            my_bet = random.randint(min_raise, max_raise)
            return my_bet, discard_card

        assert False


class ProbabilityAgent(Agent):
    # Chooses an action based on the probability of winning
    def act(self, observation, reward, terminated, truncated, info):
        # treys has an implementation that could be odds
        # it's defined as percentage rank among all hands
        # not sure if it's able to take in extra info to reduce the
        # pool of all hands by the flop discards though
        # so we might have to do it ourselves
        pass
