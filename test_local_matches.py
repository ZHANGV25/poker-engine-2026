#!/usr/bin/env python3
"""Run local matches: current bot vs test agents and previous version.

Tests:
1. vs CallingStation (should crush — they never fold)
2. vs AllIn (should crush — they always bet max)
3. vs ProbabilityAgent (should beat — heuristic equity bot)
4. Self-play (should break even — GTO vs GTO)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, '.')
from gym_env import PokerEnv
from local_match import run_match

N_HANDS = 200  # enough to see signal, fast enough to iterate

def test_vs_agent(agent_name, agent_class, n_hands=N_HANDS):
    from submission.player import PlayerAgent

    bot_current = PlayerAgent(stream=False)
    bot_opponent = agent_class(stream=False)

    print(f"\n{'='*60}")
    print(f"Current bot vs {agent_name} ({n_hands} hands)")
    print(f"{'='*60}")

    # Match 1: current as bot0
    t0 = time.time()
    r1 = run_match(bot_current, bot_opponent, n_hands=n_hands, verbose=False)
    ms1 = (time.time() - t0) * 1000

    # Match 2: current as bot1 (reversed position)
    bot_current2 = PlayerAgent(stream=False)
    bot_opponent2 = agent_class(stream=False)
    t0 = time.time()
    r2 = run_match(bot_opponent2, bot_current2, n_hands=n_hands, verbose=False)
    ms2 = (time.time() - t0) * 1000

    total_us = r1[0] + r2[1]
    total_them = r1[1] + r2[0]
    per_hand = total_us / (n_hands * 2)

    print(f"  Match 1 (us=P0): us={r1[0]:+d} them={r1[1]:+d} ({ms1/n_hands:.0f}ms/hand)")
    print(f"  Match 2 (us=P1): us={r2[1]:+d} them={r2[0]:+d} ({ms2/n_hands:.0f}ms/hand)")
    print(f"  TOTAL: us={total_us:+d} ({per_hand:+.1f}/hand)")

    return total_us, per_hand

def test_self_play(n_hands=N_HANDS):
    from submission.player import PlayerAgent

    bot0 = PlayerAgent(stream=False)
    bot1 = PlayerAgent(stream=False)

    print(f"\n{'='*60}")
    print(f"Self-play ({n_hands} hands)")
    print(f"{'='*60}")

    t0 = time.time()
    r = run_match(bot0, bot1, n_hands=n_hands, verbose=False)
    ms = (time.time() - t0) * 1000

    print(f"  bot0={r[0]:+d} bot1={r[1]:+d} ({ms/n_hands:.0f}ms/hand)")
    print(f"  Deviation from 0: {abs(r[0])}")

    return r

if __name__ == "__main__":
    from agents.test_agents import CallingStationAgent, AllInAgent, FoldAgent
    from agents.prob_agent import ProbabilityAgent

    results = {}

    # vs each test agent
    for name, cls in [
        ("FoldAgent", FoldAgent),
        ("CallingStation", CallingStationAgent),
        ("AllInAgent", AllInAgent),
        ("ProbabilityAgent", ProbabilityAgent),
    ]:
        try:
            total, per_hand = test_vs_agent(name, cls)
            results[name] = (total, per_hand)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (0, 0)

    # Self-play
    try:
        sp = test_self_play()
        results['self_play'] = sp
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, (total, ph) in results.items():
        if name == 'self_play':
            continue
        status = "WIN" if total > 0 else "LOSS" if total < 0 else "DRAW"
        print(f"  vs {name:<20}: {total:>+5d} chips ({ph:>+5.1f}/hand) {status}")

    if 'self_play' in results:
        sp = results['self_play']
        print(f"  Self-play:              bot0={sp[0]:+d} bot1={sp[1]:+d}")
