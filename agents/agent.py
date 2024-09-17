from abc import ABC, abstractmethod
from typing import Tuple, Any
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import uvicorn


class ActionRequest(BaseModel):
    observation: Any  # Replace 'Any' with specific types if possible
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
    def act(
        self, observation, reward, terminated, truncated, info
    ) -> tuple[int, int]:
        """
        Given the current state, return the action to take.

        Args:
            reward (int)  : 0 if terminated is false, or the profit / loss of the game
            #TODO: add the types of the arguments
        Returns:
            action (Tuple[int, int]) : (cumulative amount to bet, index of the card to discard)
        """
        pass

    def observe(
        self, observation, reward, terminated, truncated, info
    ) -> None:
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
            print(request)
            try:
                action = self.act(
                    observation=request.observation,
                    reward=request.reward,
                    terminated=request.terminated,
                    truncated=request.truncated,
                    info=request.info
                )
                return ActionResponse(action=action)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host="0.0.0.0", port=8000):
        """
        Method to start the FastAPI server from within the agent class.
        """
        uvicorn.run(self.app, host=host, port=port)