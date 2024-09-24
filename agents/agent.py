from abc import ABC, abstractmethod
from typing import Tuple, List, Any, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# I used a typedDict instead of a pydantic model because it
# was giving me issues.
class Observation(TypedDict):
    street: int
    turn: int
    my_cards: List[int]
    community_cards: List[int]
    my_bet: int
    opp_bet: int
    my_bankroll: int
    opp_shown_cards: List[int]
    game_num: int
    min_raise: int


class ActionRequest(BaseModel):
    observation: Observation
    reward: int
    terminated: bool
    truncated: bool
    info: Any


class ObservationRequest(BaseModel):
    observation: Observation
    reward: int
    terminated: bool
    truncated: bool
    info: Any


class ActionResponse(BaseModel):
    action: Tuple[int, int]


class Agent(ABC):
    def __init__(self):
        self.app = FastAPI()
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
            print(f"ActionRequest: {request}")
            try:
                action = self.act(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
                return ActionResponse(action=action)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/post_observation")
        async def post_observation(request: ObservationRequest) -> None:
            """
            API endpoint to send the observation to the bot
            """
            print(f"Observation: {request}")
            try:
                self.observe(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host="0.0.0.0", port=8000):
        """
        Method to start the FastAPI server from within the agent class.
        """
        uvicorn.run(self.app, host=host, port=port)
