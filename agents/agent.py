from abc import ABC, abstractmethod


class Agent(ABC):

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
