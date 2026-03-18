#!/usr/bin/env python3
"""Run a direct local match between two bot versions without FastAPI.

Bypasses the HTTP match runner and calls act()/observe() directly.
"""
import sys, os, random, time
import numpy as np

sys.path.insert(0, '.')
from gym_env import PokerEnv

def run_match(bot0, bot1, n_hands=200, verbose=False):
    """Run n_hands between bot0 and bot1, return chip results."""
    env = PokerEnv()
    bankrolls = [0, 0]

    for hand_num in range(n_hands):
        small_blind_player = hand_num % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind_player})
        info["hand_number"] = hand_num

        terminated = False
        reward = (0, 0)

        while not terminated:
            acting = obs0["acting_agent"] if isinstance(obs0, dict) else obs1["acting_agent"]

            if acting == 0:
                action = bot0.act(obs0, reward[0], False, False, info)
                # Notify bot1
                bot1.observe(obs1, reward[1], False, False, info)
            else:
                action = bot1.act(obs1, reward[1], False, False, info)
                # Notify bot0
                bot0.observe(obs0, reward[0], False, False, info)

            (obs0, obs1), reward, terminated, truncated, info = env.step(action)
            info["hand_number"] = hand_num

        bankrolls[0] += reward[0]
        bankrolls[1] += reward[1]

        if verbose and hand_num % 50 == 0:
            print(f"  Hand {hand_num}: bot0={bankrolls[0]:+d} bot1={bankrolls[1]:+d}")

    return bankrolls


if __name__ == "__main__":
    # Import current version
    from submission.player import PlayerAgent as CurrentBot

    # Import old GTO+ version by temporarily swapping player.py
    import importlib
    import submission.player
    # Save current module
    current_module = submission.player

    # Load old version
    old_code = open('/tmp/old_player.py').read()

    # We can't easily load two versions of the same module
    # Instead, just run current vs prob_agent
    from agents.prob_agent import ProbabilityAgent

    print("=" * 60)
    print("Match: Current (solver-first) vs ProbabilityAgent")
    print("=" * 60)

    bot0 = CurrentBot(stream=False)
    bot1 = ProbabilityAgent(stream=False)

    t0 = time.time()
    results = run_match(bot0, bot1, n_hands=100, verbose=True)
    elapsed = time.time() - t0

    print(f"\nResult: Current={results[0]:+d} Prob={results[1]:+d}")
    print(f"Time: {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/hand)")

    # Run reverse (bot1 = current, bot0 = prob)
    print("\n" + "=" * 60)
    print("Match: ProbabilityAgent vs Current (reversed positions)")
    print("=" * 60)

    bot0b = ProbabilityAgent(stream=False)
    bot1b = CurrentBot(stream=False)

    t0 = time.time()
    results2 = run_match(bot0b, bot1b, n_hands=100, verbose=True)
    elapsed2 = time.time() - t0

    print(f"\nResult: Prob={results2[0]:+d} Current={results2[1]:+d}")
    print(f"Time: {elapsed2:.1f}s ({elapsed2/100*1000:.0f}ms/hand)")

    total_current = results[0] + results2[1]
    print(f"\nOverall: Current {total_current:+d} over 200 hands")
