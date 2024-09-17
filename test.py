from gym_env import PokerEnv
from agents.test_agents import AllInAgent, RandomAgent
from agents.agent import ActionRequest, ActionResponse
import requests
import json
import numpy

def test_agents():
    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    bot0, bot1 = AllInAgent(), RandomAgent()

    reward0 = reward1 = 0
    trunc = None

    terminated = False
    while not terminated:
        print("\n#####################")
        print("Turn:", obs0["turn"])
        print("Bot0 cards:", obs0["my_cards"], "Bot1 cards:", obs1["my_cards"])
        print("Community cards:", obs0["community_cards"])
        print("Bot0 bet:", obs0["my_bet"], "Bot1 bet:", obs1["my_bet"])
        print("#####################\n" )

        if obs0["turn"] == 0:
            action = bot0.act(obs0, reward0, terminated, trunc, info)
            bot1.observe(obs1, reward1, terminated, trunc, info)
        else:
            action = bot1.act(obs1, reward1, terminated, trunc, info)
            bot0.observe(obs0, reward0, terminated, trunc, info)

        print("bot", obs0["turn"], "did action", action)

        (obs0, obs1), (reward0, reward1), terminated, trunc, inf = env.step(
            action=action
        )
        print("Bot0 reward:", reward0, "Bot1 reward:", reward1)


def test_agents_with_api_calls():
    def _prepare_observation(observation):
        "Converts numpy arrays to lists so that they can be json serialized"
        prepared_obs = dict()
        for key, value in observation.items():
            print(type(value))
            if type(value) == numpy.ndarray:
                converted_value = value.tolist()
            else:
                converted_value = value
            prepared_obs[key] = converted_value
        return prepared_obs
    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    bot0_ep = "http://0.0.0.0:8000" + "/get_action"
    bot1_ep = "http://0.0.0.0:8001" + "/get_action"

    reward0 = reward1 = 0
    trunc = False

    terminated = False
    while not terminated:
        print("\n#####################")
        print("Turn:", obs0["turn"])
        print("Bot0 cards:", obs0["my_cards"], "Bot1 cards:", obs1["my_cards"])
        print("Community cards:", obs0["community_cards"])
        print("Bot0 bet:", obs0["my_bet"], "Bot1 bet:", obs1["my_bet"])
        print("#####################\n" )

        if obs0["turn"] == 0:
            bot0_action_request = {
                "observation": _prepare_observation(obs0),
                "reward": reward0,
                "terminated": terminated,
                "truncated": trunc,
                "info": info
            }
            print(bot0_action_request)
            # bot0_action_request = ActionRequest(observation=obs0, reward=reward0, terminated=terminated, truncated=trunc, info = info)
            bot0_action_response = requests.get(bot0_ep, json=bot0_action_request)
            action = bot0_action_response.json()
            print("Bot0 Action:", action)
            # action = bot0.get_action(bot0_action_request)
            # bot1.observe(obs1, reward1, terminated, trunc, info)
        else:
            bot1_action_request = {
                "observation": _prepare_observation(obs1),
                "reward": reward0,
                "terminated": terminated,
                "truncated": trunc,
                "info": info
            }
            print(bot1_action_request)
            bot1_action_response = requests.get(bot1_ep, json=bot1_action_request)
            action = bot1_action_response.json()
            print(action)
            print("Bot1 Action:", action)
            # action = bot1.get_action(bot1_action_request)
            # bot0.observe(obs0, reward0, terminated, trunc, info)

        action_value = action["action"]

        print("bot", obs0["turn"], "did action", action_value)

        (obs0, obs1), (reward0, reward1), terminated, trunc, inf = env.step(
            action=action_value
        )
        print("Bot0 reward:", reward0, "Bot1 reward:", reward1)
    # TODO: Implement the game loop with API calls
        

if __name__ == "__main__":
    test_agents_with_api_calls()
