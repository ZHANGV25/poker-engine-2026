from abc import ABC, abstractmethod  

class Agent(ABC):    
    
    @abstractmethod     
    def act(self, observation, reward, terminated, truncated, info):         
        '''         
        Given the current state, return the action to take.          
        Args: #TODO: add the types of the arguments   
        Returns:              
            action (Tuple[int, int]) : (amount to bet, index of the card to discard)         
        '''         
        pass