"""
Basic Test Suite for PlayerAgent which checks that it never does an invalid action
"""

import time
from logging import getLogger
from starter.player import PlayerAgent
from agents.agent import Agent
from agents.prob_agent import ProbabilityAgent
from agents.test_agents import FoldAgent, CallingStationAgent, AllInAgent, RandomAgent
from gym_env import PokerEnv


NUM_HANDS = 100
TIME_PER_HAND = 1


class InvalidActionError(Exception):
    """AI GENERATED"""

    def __init__(self, action, observation):
        """
        Initialize the InvalidActionError with action and observation.

        Args:
            action: The invalid action attempted.
            observation: The observation in which the invalid action was attempted.
        """
        self.action = action
        self.observation = observation
        super().__init__(self._generate_message())

    def _generate_message(self):
        """
        Generate a detailed error message.

        Returns:
            str: A descriptive error message including the action and observation.
        """
        return f"Invalid action '{self.action}' attempted in observation '{self.observation}'."


def run_game(player_bot_num: int, test_agent_class: Agent, num_hands: int):
    """
    Run the agents against each other and check for errors
    """
    logger = getLogger(__name__)
    env = PokerEnv(logger=logger, num_hands=num_hands)
    my_agent = PlayerAgent(logger=logger)
    test_agent = test_agent_class(logger=logger)

    bots: list[Agent] = [None, None]
    bots[player_bot_num] = my_agent
    bots[1 - player_bot_num] = test_agent

    (obs0, obs1), info = env.reset()
    terminated = False
    reward = (0, 0)
    truncated = False
    while not terminated:
        acting_agent = obs0["acting_agent"]
        obs = [obs0, obs1][acting_agent]
        action = bots[acting_agent].act(
            obs, reward, terminated, truncated, info
        )
        (obs0, obs1), reward, terminated, truncated, info = env.step(action)
        if info["invalid_action"]:
            assert (bots[acting_agent]) != test_agent, print(
                "ProbabilityAgent produced an invalid action!!!!", action, obs
            )
            raise InvalidActionError(obs, action)
    return reward


def main():
    """
    Runs NUM_GAMES games between PlayerAgent and ProbabilityAgent, alternating who
    is small blind
    """
    
    # TODO: We have a weird bug in RandomAgent where it will sometimes not act
    # TODO: Sometimes the min_raise > max_raise???
    
    player_bankroll = 0
    test_bot_bankroll = 0
    for test_agent_class in [ProbabilityAgent, AllInAgent, FoldAgent, CallingStationAgent, RandomAgent]:
        print("--------------------------------")
        print("Testing user bot against", test_agent_class)
        start_time = time.time()
        for game_num in range(NUM_HANDS):
            try:
                player_bot_num = game_num % 2
                reward = run_game(player_bot_num, test_agent_class, NUM_HANDS)
                player_bankroll += reward[player_bot_num]
                test_bot_bankroll += reward[1 - player_bot_num]
                print("passed game number", game_num)
                print("bankroll:", player_bankroll)

            except InvalidActionError as e:
                print("failed game number", game_num)
                print(e)
                break
        end_time = time.time()
        print(f"Player bankroll: {player_bankroll}, Test bot bankroll: {test_bot_bankroll}")
        print(f"Time taken for {NUM_HANDS} hands: {end_time - start_time} seconds")
        print(f"Time per hand: {(end_time - start_time) / NUM_HANDS} seconds")
        assert (end_time - start_time) / NUM_HANDS < TIME_PER_HAND, "Time per hand is too long"

if __name__ == "__main__":
    main()
