from gym_env import PokerEnv
from agents.test_agents import AllInAgent, RandomAgent


if __name__ == "__main__":
    env = PokerEnv(num_games=1)

    (obs1, obs2), info = env.reset()
    bot1, bot2 = AllInAgent(), RandomAgent()

    reward1 = reward2 = 0
    trunc = None

    terminated = False
    while not terminated:
        print("\n#####################")
        print("Bot1 cards:", obs1["my_cards"], "Bot2 cards:", obs2["my_cards"])
        print("Community cards:", obs1["community_cards"])
        print("Bot1 bet:", obs1["my_bet"], "Bot2 bet:", obs2["my_bet"])
        print("#####################\n" )

        if obs1["turn"] == 0:
            action = bot1.act(obs1, reward1, terminated, trunc, info)
            bot2.observe(obs2, reward2, terminated, trunc, info)
        else:
            action = bot2.act(obs2, reward2, terminated, trunc, info)
            bot1.observe(obs1, reward1, terminated, trunc, info)

        print("bot", obs1["turn"], "did action", action)

        (obs1, obs2), (reward1, reward2), terminated, trunc, inf = env.step(
            action=action
        )
        print("Bot1 reward:", reward1, "Bot2 reward:", reward2)

