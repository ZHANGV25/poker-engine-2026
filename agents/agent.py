import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, TypedDict
from pydantic import BaseModel


# I used a typedDict instead of a pydantic model because it
# was giving me issues.
class Observation(TypedDict):
    street: int
    acting_agent: int
    my_cards: List[int]
    community_cards: List[int]
    my_bet: int
    opp_bet: int
    opp_discarded_card: int
    opp_drawn_card: int
    min_raise: int
    max_raise: int
    valid_actions: List[int]


class ActionRequest(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Any


class ObservationRequest(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Any


class ActionResponse(BaseModel):
    action: Tuple[int, int, int]


class Agent(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.app = FastAPI()
        self.logger = logger or logging.getLogger(__name__)
        self.add_routes()

    @abstractmethod
    def act(self, observation, reward, terminated, truncated, info) -> tuple[int, int]:
        """
        Given the current state, return the action to take.

        Args:
            reward (int)  : 0 if terminated is false, or the profit / loss of the game
            #TODO: add the types of the arguments
        Returns:
            action (Tuple[int, int]) : (cumulative amount to bet, index of the card to discard)
        """
        pass

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        """
        Observe the result of your action. However, it's not your turn.
        """
        pass

    def add_routes(self):
        @self.app.get("/get_action")
        async def get_action(request: ActionRequest) -> ActionResponse:
            """
            API endpoint to get an action based on the current game state.
            """
            self.logger.debug(f"ActionRequest: {request}")
            try:
                action = self.act(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
                self.logger.debug(f"Action taken: {action}")
                return ActionResponse(action=action)
            except Exception as e:
                self.logger.error(f"Error in get_action: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/post_observation")
        async def post_observation(request: ObservationRequest) -> None:
            """
            API endpoint to send the observation to the bot
            """
            self.logger.debug(f"Observation: {request}")
            try:
                self.observe(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
            except Exception as e:
                self.logger.error(f"Error in post_observation: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    @classmethod
    def run(cls, port: int, logger: logging.Logger, host: str = "0.0.0.0"):
        """
        Run an API-based bot on a specified port.

        Args:
            port (int): The port number to run the bot on.
            logger (logging.Logger): The logger object to use for logging.
            host (str): The host to bind the server to. Defaults to "0.0.0.0".
        """
        bot = cls(logger)
        logger.info(f"Starting agent server on {host}:{port}")

        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.setLevel(logging.WARNING)

        uvicorn_logger.handlers.clear()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        uvicorn_logger.addHandler(handler)

        uvicorn.run(bot.app, host=host, port=port, log_config=None)
