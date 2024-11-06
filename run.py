"""
Script to run matches between agents.
"""

import logging
import multiprocessing
import time
from typing import Any, Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
import requests
from agents.agent import Agent
from agents.test_agents import AllInAgent, CallingStationAgent, all_agent_classes
from gym_env import PokerEnv


def prepare_payload(obs: Dict[str, Any], reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the payload for API calls by converting numpy arrays to lists.

    Args:
        obs (Dict[str, Any]): The observation dictionary.
        reward (float): The reward value.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode has been truncated.
        info (Dict[str, Any]): Additional information.

    Returns:
        Dict[str, Any]: The prepared payload.
    """

    def _prepare_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in observation.items()}

    return {
        "observation": _prepare_observation(obs),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


def call_agent_api(method: str, base_url: str, endpoint: str, payload: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Make an API call to an agent with retry logic.

    Args:
        method (str): The HTTP method to use.
        base_url (str): The base URL of the agent's API.
        endpoint (str): The API endpoint to call.
        payload (Dict[str, Any]): The payload to send with the request.

    Returns:
        Dict[str, Any]: The JSON response from the API.

    Raises:
        requests.exceptions.RequestException: If all retry attempts fail.
    """
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.request(method, base_url + endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt == max_retries - 1:
                raise requests.exceptions.RequestException(f"Failed API Request after {max_retries} attempts: {str(e)}")
            delay = base_delay * (2**attempt)
            logger.info(f"Backing off for {delay} seconds before retry {attempt + 1}")
            time.sleep(delay)


def run_local_match(agent1: Agent, agent2: Agent, num_games: int = 5, logger: logging.Logger = None) -> float:
    """
    Run a local match between two agents.

    Args:
        agent1 (Agent): The first agent instance.
        agent2 (Agent): The second agent instance.
        num_games (int, optional): The number of games to play. Defaults to 5.
        logger (logging.Logger, optional): The logger object to use for logging.

    Returns:
        float: The net bankroll of agent1 vs agent2.
    """
    env = PokerEnv(num_games=num_games, logger=logger)
    (obs0, obs1), info = env.reset()
    bot0, bot1 = agent1, agent2

    reward0 = reward1 = 0
    truncated = None

    logger.info(f"Starting a new match with {num_games} games")

    terminated = False
    while not terminated:
        print_game_state(obs0, obs1)

        if obs0["turn"] == 0:
            action = bot0.act(obs0, reward0, terminated, truncated, info)
            bot1.observe(obs1, reward1, terminated, truncated, info)
        else:
            action = bot1.act(obs1, reward1, terminated, truncated, info)
            bot0.observe(obs0, reward0, terminated, truncated, info)

        print(f"Bot {obs0['turn']} did action {action}")

        (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action)
        print(f"Bot0 reward: {reward0}, Bot1 reward: {reward1}")

    return obs0["my_bankroll"] - obs1["my_bankroll"]


TIME_LIMIT_SECONDS = 420  # 7 minutes


def run_api_match(base_url_0: str, base_url_1: str, logger: logging.Logger, num_games: int = 1000) -> Dict[str, Any]:
    """
    Run a match between two API-based agents.

    Args:
        base_url_0 (str): The base URL for the first agent's API.
        base_url_1 (str): The base URL for the second agent's API.
        logger (logging.Logger): The logger object to use for logging.
        num_games (int): The number of games to play. Defaults to 1000.

    Returns:
        Dict[str, Any]: A dictionary containing the match results:
            - status: "completed", "timeout", "error"
            - winner: None if completed normally, 0 or 1 if someone won by timeout/error
            - bot0_reward/bot1_reward: Final rewards (only if status=="completed")
    """
    env = PokerEnv(num_games=num_games, logger=logger)
    (obs0, obs1), info = env.reset()

    get_action_endpoint = "/get_action"
    send_obs_endpoint = "/post_observation"

    reward0 = reward1 = 0
    truncated = False
    terminated = False
    game_count = 0
    total_reward0 = 0
    total_reward1 = 0

    # Track time used by each player
    time_used_0 = 0.0
    time_used_1 = 0.0

    def handle_player_action(current_player: int, current_payload: Dict) -> Dict:
        current_url = base_url_0 if current_player == 0 else base_url_1

        action_start = time.time()
        action = call_agent_api("GET", current_url, get_action_endpoint, current_payload, logger)
        action_duration = time.time() - action_start

        # Update time tracking
        if current_player == 0:
            nonlocal time_used_0
            time_used_0 += action_duration
            if time_used_0 > TIME_LIMIT_SECONDS:
                raise TimeoutError("Player 0 exceeded time limit")
        else:
            nonlocal time_used_1
            time_used_1 += action_duration
            if time_used_1 > TIME_LIMIT_SECONDS:
                raise TimeoutError("Player 1 exceeded time limit")

        return action

    def get_match_result(status: str, winner: Optional[int] = None, rewards: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        result = {"status": status, "winner": winner}
        if rewards:
            result.update({"bot0_reward": rewards[0], "bot1_reward": rewards[1]})
        return result

    try:
        logger.info(f"Starting a new match with {num_games} games")

        while not terminated:
            logger.debug(f"Game {game_count + 1}, Turn: {obs0['turn']}")
            log_game_state(logger, obs0, obs1)

            bot0_payload = prepare_payload(obs0, reward0, terminated, truncated, info)
            bot1_payload = prepare_payload(obs1, reward1, terminated, truncated, info)

            try:
                current_player = obs0["turn"]
                observer_url = base_url_1 if current_player == 0 else base_url_0
                current_payload = bot0_payload if current_player == 0 else bot1_payload
                observer_payload = bot1_payload if current_player == 0 else bot0_payload

                action = handle_player_action(current_player, current_payload)

                # Send observation to the other player
                call_agent_api("POST", observer_url, send_obs_endpoint, observer_payload, logger)

            except Exception as e:
                logger.error(f"Error during bot communication: {str(e)}")
                return get_match_result("error")

            action_value = action["action"]
            logger.debug(f"Bot {obs0['turn']} did action {action_value}")

            (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action_value)
            logger.debug(f"Bot0 reward: {reward0}, Bot1 reward: {reward1}")

            total_reward0 += reward0
            total_reward1 += reward1

            if info is not None and info.get("game_ended", False):
                game_count += 1
                logger.info(f"Game {game_count} ended. Bot0 total reward: {total_reward0}, Bot1 total reward: {total_reward1}")
                bankroll_log = format_bankroll_log(game_count, env.bankrolls)
                logger.info(bankroll_log)

                if game_count == num_games:
                    terminated = True

        logger.info("Match completed")
        logger.info(f"Final results - Bot0 total reward: {total_reward0}, Bot1 total reward: {total_reward1}")

        return get_match_result("completed", rewards=(total_reward0, total_reward1))

    except TimeoutError:
        return get_match_result("timeout", winner=1 - current_player)
    except Exception as e:
        logger.error(f"Match failed: {str(e)}")
        return get_match_result("error")


def print_game_state(obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
    """
    Print the current game state.

    Args:
        obs0 (Dict[str, Any]): Observation for the first agent.
        obs1 (Dict[str, Any]): Observation for the second agent.
    """
    print("\n#####################")
    print(f"Turn: {obs0['turn']}")
    print(f"Bot0 cards: {obs0['my_cards']}, Bot1 cards: {obs1['my_cards']}")
    print(f"Community cards: {obs0['community_cards']}")
    print(f"Bot0 bet: {obs0['my_bet']}, Bot1 bet: {obs1['my_bet']}")
    print("#####################\n")


def log_game_state(logger: logging.Logger, obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
    """
    Log the current game state.

    Args:
        logger (logging.Logger): The logger object to use for logging.
        obs0 (Dict[str, Any]): Observation for the first agent.
        obs1 (Dict[str, Any]): Observation for the second agent.
    """
    logger.debug("#####################")
    logger.debug(f"Turn: {obs0['turn']}")
    logger.debug(f"Bot0 cards: {obs0['my_cards']}, Bot1 cards: {obs1['my_cards']}")
    logger.debug(f"Community cards: {obs0['community_cards']}")
    logger.debug(f"Bot0 bet: {obs0['my_bet']}, Bot1 bet: {obs1['my_bet']}")
    logger.debug("#####################")


def run_all_local_matches(logger: logging.Logger) -> None:
    """Run matches between all combinations of local agents and print results."""
    agent_names = [x.name() for x in all_agent_classes]
    bankroll_matrix = []

    for i1, agent1 in enumerate(all_agent_classes):
        bankroll_matrix.append([])
        for i2, agent2 in enumerate(all_agent_classes):
            logger.info(f"{agent_names[i1]} vs {agent_names[i2]}")
            net_bankroll = run_local_match(agent1, agent2, logger=logger)
            bankroll_matrix[-1].append(net_bankroll)

    bankroll_df = pd.DataFrame(bankroll_matrix, columns=agent_names, index=agent_names)
    logger.info(f"\n{bankroll_df}")


def format_bankroll_log(game_number: int, bankrolls: list) -> str:
    """Format bankroll data as a JSON string for logging"""
    bankroll_data = {
        "type": "bankroll_update",
        "game_number": game_number,
        "bot0_bankroll": int(bankrolls[0]),
        "bot1_bankroll": int(bankrolls[1])
    }
    return json.dumps(bankroll_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    process0 = multiprocessing.Process(target=AllInAgent.run, args=(8000, logger))
    process1 = multiprocessing.Process(target=CallingStationAgent.run, args=(8001, logger))

    process0.start()
    process1.start()

    logger.info("Starting API-based match")
    run_api_match("http://127.0.0.1:8000", "http://127.0.0.1:8001", logger)

    process0.terminate()
    process1.terminate()
