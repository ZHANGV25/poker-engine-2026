from gym_env import PokerEnv
from agents.test_agents import AllInAgent, RandomAgent


def test_agents():
    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    bot0, bot1 = AllInAgent(), RandomAgent()

    reward0 = reward1 = 0
    trunc = None

    terminated = False
    while not terminated:
        print("\n#####################")
        print("Turn:", obs0["turn"])
        print("Bot0 cards:", obs0["my_cards"], "Bot1 cards:", obs1["my_cards"])
        print("Community cards:", obs0["community_cards"])
        print("Bot0 bet:", obs0["my_bet"], "Bot1 bet:", obs1["my_bet"])
        print("#####################\n" )

        if obs0["turn"] == 0:
            action = bot0.act(obs0, reward0, terminated, trunc, info)
            bot1.observe(obs1, reward1, terminated, trunc, info)
        else:
            action = bot1.act(obs1, reward1, terminated, trunc, info)
            bot0.observe(obs0, reward0, terminated, trunc, info)

        print("bot", obs0["turn"], "did action", action)

        (obs0, obs1), (reward0, reward1), terminated, trunc, inf = env.step(
            action=action
        )
        print("Bot0 reward:", reward0, "Bot1 reward:", reward1)


def test_agents_with_api_calls():
    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    bot0, bot1 = AllInAgent(), RandomAgent()

    reward0 = reward1 = 0
    trunc = None

    terminated = False

    # TODO: Implement the game loop with API calls
        

if __name__ == "__main__":
    test_agents()
