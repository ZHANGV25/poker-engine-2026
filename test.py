from gym_env import PokerEnv
from agents.test_agents import all_agent_classes
import pandas as pd


def test_agents(agent1, agent2):
    env = PokerEnv(num_games=5)

    (obs0, obs1), info = env.reset()
    bot0, bot1 = agent1(), agent2()

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
    return obs0['my_bankroll'] - obs1['my_bankroll']

def test_all_base_agents():
    agent_names = [x.name() for x in all_agent_classes]
    bankroll_matrix = []
    for i1, agent1 in enumerate(all_agent_classes):
        bankroll_matrix.append([])
        for i2, agent2 in enumerate(all_agent_classes):
            print(agent_names[i1], "vs", agent_names[i2])
            net_bankroll = test_agents(agent1, agent2)
            bankroll_matrix[-1].append(net_bankroll)
    
    bankroll_matrix = pd.DataFrame(bankroll_matrix, columns=agent_names, index=agent_names)
    print(bankroll_matrix)



def test_agents_with_api_calls():
    env = PokerEnv(num_games=5)
    bot0, bot1 = AllInAgent(), RandomAgent()
    # TODO: Implement the game loop with API calls
        

if __name__ == "__main__":
    test_all_base_agents()
