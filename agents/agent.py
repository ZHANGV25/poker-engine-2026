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
        # Use the provided logger directly, or create a new one
        self.logger = logger or logging.getLogger(self.__name__())
        self.add_routes()

    @abstractmethod
    def __name__(self):
        """Return the name of the agent. Must be implemented by subclasses."""
        pass

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
        # Create a logger for this agent instance
        agent_logger = logging.getLogger(cls.__name__)
        agent_logger.setLevel(logging.INFO)

        # Add a handler with formatting if none exists
        if not agent_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            agent_logger.addHandler(handler)

        bot = cls(agent_logger)
        agent_logger.info(f"Starting agent server on {host}:{port}")

        uvicorn.run(bot.app, host=host, port=port, log_level="info", access_log=False)
