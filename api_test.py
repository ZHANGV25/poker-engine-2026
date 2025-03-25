import logging
import time
from unittest.mock import Mock, patch

import pytest
import requests
from match import (
    TIME_LIMIT_SECONDS,
    AgentFailure,
    AgentFailureTracker,
    call_agent_api,
    get_match_result,
    run_api_match,
)


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def reset_failure_tracker():
    """Reset the failure tracker before each test"""
    failure_tracker = AgentFailureTracker()
    return failure_tracker


@pytest.fixture(autouse=True)
def reduce_delays():
    """Temporarily reduce delays and timeouts for testing"""
    original_time_limit = TIME_LIMIT_SECONDS
    original_sleep = time.sleep

    # Patch time.sleep to be instant
    with patch("time.sleep", return_value=None):
        # Reduce the time limit for faster timeout tests
        with patch("match.TIME_LIMIT_SECONDS", 0.1):
            yield


def test_call_agent_api_success(mock_logger):
    """Test successful API call"""
    mock_response = Mock()
    mock_response.json.return_value = {"action": [0, 0]}

    with patch("requests.request", return_value=mock_response):
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result == {"action": [0, 0]}
        assert mock_logger.error.call_count == 0


def test_call_agent_api_both_players_failing(mock_logger, reset_failure_tracker):
    """Test when both players fail multiple times"""
    tracker = reset_failure_tracker

    with patch("requests.request", side_effect=requests.exceptions.ConnectionError):
        # Fail player 0 three times
        for _ in range(3):
            with pytest.raises((requests.exceptions.ConnectionError, AgentFailure)):
                call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)

        # Fail player 1 three times
        for _ in range(3):
            with pytest.raises((requests.exceptions.ConnectionError, AgentFailure)):
                call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)

        # Now both players should be marked as failed
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert "Both players have failed" in str(exc_info.value)


def test_call_agent_api_single_player_failing(mock_logger, reset_failure_tracker):
    """Test when one player consistently fails"""
    tracker = reset_failure_tracker

    with patch("requests.request") as mock_request:
        # Make player 0's calls succeed
        mock_request.return_value.json.return_value = {"action": [0, 0]}
        call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)

        # Make player 1's calls fail
        mock_request.side_effect = requests.exceptions.ConnectionError
        for _ in range(3):
            with pytest.raises((requests.exceptions.ConnectionError, AgentFailure)):
                call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)

        # Now player 1 should be marked as failed
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert "Player 1 has failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_api_match_timeout(mock_logger):
    """Test match handling when a player exceeds time limit"""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = TimeoutError("Player 0 exceeded time limit")

        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "timeout"
        assert result["result"] == "loss"  # Player 0 timed out, so Player 1 wins


@pytest.mark.asyncio
async def test_run_api_match_both_failing(mock_logger):
    """Test match handling when both players fail"""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = AgentFailure("Both players have failed multiple times")
        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "error"
        assert "Both players have failed multiple times" in result["error"]


@pytest.mark.asyncio
async def test_run_api_match_single_failure(mock_logger):
    """Test match handling when one player fails"""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = AgentFailure("Player 1 has failed 3 times")

        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "timeout"
        assert result["result"] == "win"  # Player 1 failed, so Player 0 wins


def test_call_agent_api_retry_success(mock_logger):
    """Test successful retry after temporary failures"""
    mock_response = Mock()
    mock_response.json.return_value = {"action": [0, 0]}

    with patch("requests.request") as mock_request:
        with patch("time.sleep", return_value=None):  # Make retries instant
            # Fail twice, then succeed
            mock_request.side_effect = [
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectionError,
                mock_response,
            ]

            result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
            assert result == {"action": [0, 0]}
            assert mock_logger.error.call_count == 0


def test_match_result_format():
    """Test that get_match_result produces correctly formatted results"""
    test_cases = [
        {
            "name": "normal completion with win",
            "inputs": {"status": "completed", "rewards": (100, 50)},
            "expected": {
                "status": "completed",
                "result": "win",  # Player 0 won
                "bot0_reward": 100,
                "bot0_time_used": 0,
                "bot1_reward": 50,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "normal completion with loss",
            "inputs": {"status": "completed", "rewards": (50, 100)},
            "expected": {
                "status": "completed",
                "result": "loss",  # Player 1 won
                "bot0_reward": 50,
                "bot0_time_used": 0,
                "bot1_reward": 100,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "normal completion with tie",
            "inputs": {"status": "completed", "rewards": (100, 100)},
            "expected": {
                "status": "completed",
                "result": "tie",
                "bot0_reward": 100,
                "bot0_time_used": 0,
                "bot1_reward": 100,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "timeout",
            "inputs": {"status": "timeout", "winner": 0},
            "expected": {
                "status": "timeout",
                "result": "win",  # Player 0 won due to timeout
                "bot0_time_used": 0,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "error",
            "inputs": {"status": "error", "error": "Test error"},
            "expected": {
                "status": "error",
                "result": "invalid",
                "error": "Test error",
                "bot0_time_used": 0,
                "bot1_time_used": 0,
            },
        },
    ]

    for case in test_cases:
        result = get_match_result(**case["inputs"])
        assert result == case["expected"], f"Case '{case['name']}' failed: expected {case['expected']}, got {result}"
