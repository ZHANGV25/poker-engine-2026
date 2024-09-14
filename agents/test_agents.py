from agent.agent import Agent


class FoldAgent(Agent):
    def act(self, observation, reward, terminated, truncated, info):
        return 0, 0

# TODO: Implement the following agents
class CallingStationAgent(Agent):
    # Always calls/checks
    def act(self, observation, reward, terminated, truncated, info):
        pass

class AllInAgent(Agent):
    # Always goes all in
    def act(self, observation, reward, terminated, truncated, info):
        pass

class RandomAgent(Agent):
    # Randomly chooses an action
    def act(self, observation, reward, terminated, truncated, info):
        pass

class ProbabilityAgent(Agent):
    # Chooses an action based on the probability of winning
    def act(self, observation, reward, terminated, truncated, info):
        pass