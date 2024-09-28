from gym_env import PokerEnv
from agents.test_agents import AllInAgent, RandomAgent, CallingStationAgent
from agents.agent import ActionRequest, ActionResponse
import requests
import json
import numpy
import multiprocessing
import time


def run_bot(bot_class, port):
    bot = bot_class()
    bot.run(port=port)

class TestAgentError(ValueError):
    pass

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
        print("#####################\n")

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
    def _prepare_payload(obs, reward):
        def _prepare_observation():
            "Converts numpy arrays to lists so that they can be json serialized"
            prepared_obs = dict()
            for key, value in obs.items():
                if isinstance(value, numpy.ndarray):
                    converted_value = value.tolist()
                else:
                    converted_value = value
                prepared_obs[key] = converted_value
            return prepared_obs

        payload = {
            "observation": _prepare_observation(),
            "reward": reward,
            "terminated": terminated,
            "truncated": trunc,
            "info": info,
        }
        return payload

    def _call_agent_ep(method, base_url, ep, payload) -> dict:
        response = requests.request(method, base_url + ep, json=payload)
        if response.status_code // 200 != 1:
            raise TestAgentError(f"Failed API Request! - Status Code {response.status_code}")
        return response.json()

    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    GET_ACTION_EP = "/get_action"
    SEND_OBS_EP = "/post_observation"
    BASE_URL_0 = "http://127.0.0.1:8000"
    BASE_URL_1 = "http://127.0.0.1:8001"

    reward0 = reward1 = 0
    trunc = False

    terminated = False
    while not terminated:
        print("\n#####################")
        print("Turn:", obs0["turn"])
        print("Bot0 cards:", obs0["my_cards"], "Bot1 cards:", obs1["my_cards"])
        print("Community cards:", obs0["community_cards"])
        print("Bot0 bet:", obs0["my_bet"], "Bot1 bet:", obs1["my_bet"])
        print("#####################\n")
        bot0_payload = _prepare_payload(obs0, reward0)
        bot1_payload = _prepare_payload(obs1, reward1)
        if obs0["turn"] == 0:
            # Request Action
            action = _call_agent_ep("GET", BASE_URL_0, GET_ACTION_EP, bot0_payload)
            # Send Observation
            _call_agent_ep("POST", BASE_URL_1, SEND_OBS_EP, bot1_payload)
        else:
            # Request Action
            action = _call_agent_ep("GET", BASE_URL_1, GET_ACTION_EP, bot1_payload)
            # Send Observation
            _call_agent_ep("POST", BASE_URL_0, SEND_OBS_EP, bot0_payload)

        action_value = action["action"]

        print("bot", obs0["turn"], "did action", action_value)

        (obs0, obs1), (reward0, reward1), terminated, trunc, inf = env.step(
            action=action_value
        )
        print("Bot0 reward:", reward0, "Bot1 reward:", reward1)


if __name__ == "__main__":
    process0 = multiprocessing.Process(target=run_bot, args=(AllInAgent, 8000))
    process1 = multiprocessing.Process(target=run_bot, args=(CallingStationAgent, 8001))

    process0.start()
    process1.start()

    # Wait a bit for the bots to start
    time.sleep(1)

    print("Starting test")
    test_agents_with_api_calls()

    process0.terminate()
    process1.terminate()

