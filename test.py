from gym_env import PokerEnv
from agents.test_agents import AllInAgent, RandomAgent


if __name__ == "__main__":
    env = PokerEnv(num_games=1000)

    (obs1, obs2), info = env.reset()
    bot1, bot2 = AllInAgent(), RandomAgent()

    reward1 = reward2 = 0
    trunc = None

    terminated = False
    while not terminated:
        if obs1["turn"] == 0:
            action = bot1.act(obs1, reward1, terminated, trunc, info)
        else:
            action = bot2.act(obs2, reward2, terminated, trunc, info)

        (obs1, obs2), (reward1, reward2), terminated, trunc, inf = env.step(
            action=action
        )

        print("bot", obs1["turn"], "did action", action)
