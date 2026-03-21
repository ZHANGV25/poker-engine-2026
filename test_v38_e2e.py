#!/usr/bin/env python3
"""
Comprehensive end-to-end test of poker bot v38.

Tests: river decision pipeline, edge cases, narrowing pipeline,
equity gate, match log replay, and timing under load.
"""

import sys
import os
import time
import math
import random
import ast
import csv
import traceback
import itertools
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, 'submission'))

RIVER_DATA_DIR = '/tmp/river_e2e_test/river'

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

failures = []
warnings = []

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def result(label, status, detail=""):
    sym = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}
    raw_status = "PASS" if "PASS" in status else ("FAIL" if "FAIL" in status else "WARN")
    s = sym.get(raw_status, "[????]")
    msg = f"  {s} {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if raw_status == "FAIL":
        failures.append(f"{label}: {detail}")
    elif raw_status == "WARN":
        warnings.append(f"{label}: {detail}")

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
section("SETUP: Import Components")

try:
    from river_lookup import RiverLookup
    from range_solver import RangeSolver
    from equity import ExactEquityEngine
    from game_tree import (ACT_CHECK, ACT_FOLD, ACT_CALL,
                           ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN)
    from inference import DiscardInference
    result("Import all components", PASS)
except Exception as e:
    result("Import all components", FAIL, str(e))
    sys.exit(1)

# Initialize engines
t0 = time.perf_counter()
equity_engine = ExactEquityEngine()
t_eq = time.perf_counter() - t0
result("ExactEquityEngine init", PASS, f"{t_eq:.2f}s")

t0 = time.perf_counter()
lookup = RiverLookup(RIVER_DATA_DIR)
t_rl = time.perf_counter() - t0
result("RiverLookup init", PASS if lookup.loaded else FAIL,
       f"{t_rl*1000:.1f}ms, loaded={lookup.loaded}")

solver = RangeSolver(equity_engine)
result("RangeSolver init", PASS)

# ═════════════════════════════════════════════════════════════
# TEST 1: Full River Decision Pipeline
# ═════════════════════════════════════════════════════════════
section("TEST 1: Full River Decision Pipeline (20 random boards)")

random.seed(42)
np.random.seed(42)

test1_pass = 0
test1_fail = 0
test1_times_acting = []
test1_times_facing = []

for trial in range(20):
    try:
        # Generate random 5-card board from 0-26
        board = sorted(random.sample(range(27), 5))
        remaining = [c for c in range(27) if c not in board]
        hero_cards = random.sample(remaining, 2)

        # --- Acting first: get_strategy ---
        t0 = time.perf_counter()
        strat = lookup.get_strategy(hero_cards, board, 8, 8)
        t_strat = time.perf_counter() - t0
        test1_times_acting.append(t_strat)

        if strat is not None:
            probs = list(strat.values())
            total_p = sum(probs)
            has_nan = any(math.isnan(p) for p in probs)
            has_inf = any(math.isinf(p) for p in probs)
            ok = (not has_nan and not has_inf
                  and all(0 <= p <= 1.001 for p in probs)
                  and abs(total_p - 1.0) < 0.02)
            if ok:
                test1_pass += 1
            else:
                test1_fail += 1
                result(f"  Trial {trial} strategy", FAIL,
                       f"nan={has_nan} inf={has_inf} sum={total_p:.4f} probs={probs}")
        else:
            # None is acceptable (board may not be precomputed)
            test1_pass += 1

        # --- Facing bet: P(bet) + narrowing + solver ---
        t0 = time.perf_counter()
        pbet = lookup.get_p_bet(board, my_bet=8, opp_bet=8)

        if pbet is not None:
            # Bayesian narrowing
            floor = 0.05
            uniform_w = 1.0 / len(pbet)
            narrowed = {}
            for hand, pb in pbet.items():
                pb_adj = max(pb, floor)
                narrowed[hand] = uniform_w * pb_adj

            total_w = sum(narrowed.values())
            if total_w > 0:
                for k in narrowed:
                    narrowed[k] /= total_w

            # Feed to solver
            valid_actions = [True, True, True, True]  # fold, raise, check, call
            action = solver.solve_and_act(
                hero_cards, board, narrowed, [],
                my_bet=8, opp_bet=16, street=3,
                min_raise=2, max_raise=100,
                valid_actions=valid_actions,
                time_remaining=500)

            t_facing = time.perf_counter() - t0
            test1_times_facing.append(t_facing)

            if action is not None:
                act_type, amount, _, _ = action
                if act_type in (0, 1, 2, 3):  # FOLD, RAISE, CHECK, CALL
                    test1_pass += 1
                else:
                    test1_fail += 1
                    result(f"  Trial {trial} solver", FAIL, f"bad action type: {act_type}")
            else:
                # None is acceptable for some boards
                test1_pass += 1
        else:
            test1_pass += 1

    except Exception as e:
        test1_fail += 1
        result(f"  Trial {trial} CRASH", FAIL, traceback.format_exc()[-200:])

avg_acting = (sum(test1_times_acting) / len(test1_times_acting) * 1000
              if test1_times_acting else 0)
max_acting = max(test1_times_acting) * 1000 if test1_times_acting else 0
avg_facing = (sum(test1_times_facing) / len(test1_times_facing)
              if test1_times_facing else 0)
max_facing = max(test1_times_facing) if test1_times_facing else 0

result("Test 1: Acting-first strategies",
       PASS if test1_fail == 0 else FAIL,
       f"{test1_pass} pass, {test1_fail} fail, avg={avg_acting:.1f}ms, max={max_acting:.1f}ms")
result("Test 1: Facing-bet solver",
       PASS if test1_fail == 0 else FAIL,
       f"avg={avg_facing:.2f}s, max={max_facing:.2f}s")

if max_facing > 5.0:
    result("Test 1: TIMING", FAIL, f"max facing-bet time {max_facing:.2f}s > 5s limit!")
else:
    result("Test 1: TIMING", PASS, f"all facing-bet decisions under 5s")


# ═════════════════════════════════════════════════════════════
# TEST 2: Edge Cases
# ═════════════════════════════════════════════════════════════
section("TEST 2: Edge Cases")

# 27 cards = 9 ranks x 3 suits
# Cards 0,9,18 are all rank-0 (2d, 2h, 2s)
# Cards 1,10,19 are all rank-1 (3d, 3h, 3s)
# etc.

# Edge case 2a: All same-rank cards on board (monotone-ish)
# Suit structure: card % 9 = rank, card // 9 = suit (approximately)
# Actually: 0-8 = suit0, 9-17 = suit1, 18-26 = suit2
# So same suit: [0,1,2,3,4] (all diamonds, ranks 2-6)
test_boards = [
    ("Same suit (all diamonds)", [0, 1, 2, 3, 4]),
    ("Same suit (all hearts)", [9, 10, 11, 12, 13]),
    ("Board index 0 (min comb)", [0, 1, 2, 3, 4]),
    ("Board index max (26,25,24,23,22)", [22, 23, 24, 25, 26]),
    ("Mixed suits", [0, 9, 18, 1, 10]),
]

for label, board in test_boards:
    try:
        remaining = [c for c in range(27) if c not in board]
        hero = remaining[:2]
        strat = lookup.get_strategy(hero, board, 8, 8)
        pbet = lookup.get_p_bet(board, my_bet=8, opp_bet=8)
        status = PASS
        detail = f"strat={'ok' if strat else 'None'}, pbet={'ok' if pbet else 'None'}"
        if strat:
            probs = list(strat.values())
            if any(math.isnan(p) or math.isinf(p) for p in probs):
                status = FAIL
                detail += " NaN/Inf in strategy!"
        result(f"  Board {label}", status, detail)
    except Exception as e:
        result(f"  Board {label}", FAIL, str(e)[:100])

# Edge case 2b: Hero hand shares card with board (should be impossible)
print()
try:
    board = [0, 1, 2, 3, 4]
    hero_overlap = [0, 5]  # card 0 is on board!
    strat = lookup.get_strategy(hero_overlap, board, 8, 8)
    if strat is None:
        result("  Overlapping hero/board", PASS, "returned None (correct)")
    else:
        # If it returns something, check if it's sane
        probs = list(strat.values())
        if any(math.isnan(p) for p in probs):
            result("  Overlapping hero/board", FAIL, f"returned strategy with NaN!")
        else:
            result("  Overlapping hero/board", WARN,
                   f"returned strategy (should not happen): {strat}")
except Exception as e:
    result("  Overlapping hero/board", FAIL, f"CRASH: {str(e)[:100]}")

# Edge case 2c: Very small and very large pots
print()
for my_b, opp_b, label in [(1, 1, "tiny pot"), (200, 200, "huge pot"),
                             (3, 7, "asymmetric small"), (50, 100, "asymmetric large")]:
    try:
        board = [0, 1, 2, 3, 4]
        remaining = [c for c in range(27) if c not in board]
        hero = remaining[:2]
        strat = lookup.get_strategy(hero, board, my_b, opp_b)
        if strat is not None:
            probs = list(strat.values())
            ok = (not any(math.isnan(p) or math.isinf(p) for p in probs)
                  and abs(sum(probs) - 1.0) < 0.02)
            result(f"  Pot {label} ({my_b},{opp_b})",
                   PASS if ok else FAIL,
                   f"sum={sum(probs):.4f}, n_acts={len(strat)}")
        else:
            result(f"  Pot {label} ({my_b},{opp_b})", PASS, "None (ok)")
    except Exception as e:
        result(f"  Pot {label} ({my_b},{opp_b})", FAIL, str(e)[:100])

# Edge case 2d: Board index boundary
print()
from math import comb as mcomb
max_board_idx = mcomb(27, 5) - 1  # = 80730-1 = 80729
result("  Board index range", PASS, f"0..{max_board_idx} ({max_board_idx+1} total)")

# Check file for index 0 and 80729
for idx in [0, 80729]:
    fpath = os.path.join(RIVER_DATA_DIR, f'river_{idx}.npz')
    exists = os.path.isfile(fpath)
    result(f"  river_{idx}.npz exists", PASS if exists else WARN, str(exists))


# ═════════════════════════════════════════════════════════════
# TEST 3: Narrowing Pipeline
# ═════════════════════════════════════════════════════════════
section("TEST 3: Narrowing Pipeline")

board = [0, 1, 2, 3, 4]
remaining = [c for c in range(27) if c not in board]

# Step 1: Create uniform opponent range (all C(22,2)=231 hands)
all_hands = list(itertools.combinations(remaining, 2))
n_hands = len(all_hands)
result(f"  Uniform range size", PASS, f"{n_hands} hands (expected {mcomb(22,2)}=231)")

uniform_range = {}
for h in all_hands:
    key = (min(h), max(h))
    uniform_range[key] = 1.0 / n_hands

# Step 2: Get P(bet|hand) and apply Bayesian narrowing
pbet = lookup.get_p_bet(board, my_bet=8, opp_bet=8)
if pbet is not None:
    result("  P(bet|hand) lookup", PASS, f"{len(pbet)} hands")

    floor = 0.05
    narrowed = {}
    for hand, w in uniform_range.items():
        pb = pbet.get(hand, 0.0)
        pb_adj = max(pb, floor)
        narrowed[hand] = w * pb_adj

    total_w = sum(narrowed.values())
    if total_w > 0:
        for k in narrowed:
            narrowed[k] /= total_w

    # Verify: sums to 1.0
    narrowed_sum = sum(narrowed.values())
    result("  Narrowed range sums to 1.0",
           PASS if abs(narrowed_sum - 1.0) < 1e-6 else FAIL,
           f"sum={narrowed_sum:.8f}")

    # Verify: no negative weights
    neg_weights = [v for v in narrowed.values() if v < 0]
    result("  No negative weights",
           PASS if not neg_weights else FAIL,
           f"{len(neg_weights)} negative values")

    # Verify: hands with high P(bet) have higher weight after narrowing
    pbet_vals = sorted(pbet.values())
    high_thresh = pbet_vals[int(0.8 * len(pbet_vals))] if pbet_vals else 0.5
    low_thresh = pbet_vals[int(0.2 * len(pbet_vals))] if pbet_vals else 0.5

    high_avg = np.mean([narrowed[h] for h in narrowed
                        if pbet.get(h, 0) >= high_thresh])
    low_avg = np.mean([narrowed[h] for h in narrowed
                       if pbet.get(h, 0) <= low_thresh])
    result("  High-P(bet) hands have higher weight",
           PASS if high_avg >= low_avg else WARN,
           f"high_avg={high_avg:.6f} vs low_avg={low_avg:.6f}")

    # Step 3: Feed narrowed range to solver
    hero_cards = remaining[:2]
    valid_actions = [True, True, True, True]
    t0 = time.perf_counter()
    action = solver.solve_and_act(
        hero_cards, board, narrowed, [],
        my_bet=8, opp_bet=16, street=3,
        min_raise=2, max_raise=100,
        valid_actions=valid_actions,
        time_remaining=500)
    t_solve = time.perf_counter() - t0

    if action is not None:
        act_type, amount, _, _ = action
        act_names = {0: 'FOLD', 1: 'RAISE', 2: 'CHECK', 3: 'CALL'}
        result("  Solver with narrowed range", PASS,
               f"action={act_names.get(act_type, '?')} amt={amount} time={t_solve:.2f}s")
    else:
        result("  Solver with narrowed range", WARN, "returned None")
else:
    result("  P(bet|hand) lookup", WARN, "returned None for this board")


# ═════════════════════════════════════════════════════════════
# TEST 4: Equity Gate
# ═════════════════════════════════════════════════════════════
section("TEST 4: Equity Gate")

# Set up a scenario where hero has a weak hand facing a bet
board = [0, 1, 2, 3, 4]  # 2d,3d,4d,5d,6d
remaining = [c for c in range(27) if c not in board]

# Strong hand: 7d,8d (cards 5,6) - continues the straight
# Weak hand: 2h,3h (cards 9,10) - pairs but not great against narrow range
strong_hand = [7, 8]  # 8d, Ad - high cards
weak_hand = [9, 10]   # 2h, 3h - low pair

# Build a narrowed range that's skewed towards strong hands
narrowed = {}
for h in itertools.combinations(remaining, 2):
    key = (min(h), max(h))
    # Give higher weight to high cards (simulating opponent who bets strong)
    avg_card = (h[0] + h[1]) / 2.0
    narrowed[key] = max(0.01, avg_card / 26.0)

total_w = sum(narrowed.values())
for k in narrowed:
    narrowed[k] /= total_w

# Compute equity for strong and weak hands
t0 = time.perf_counter()
eq_strong = equity_engine.compute_equity(strong_hand, board, [], narrowed)
eq_weak = equity_engine.compute_equity(weak_hand, board, [], narrowed)
t_eq_comp = time.perf_counter() - t0

result("  Equity computation time", PASS, f"{t_eq_comp*1000:.1f}ms")
result("  Strong hand equity", PASS, f"{eq_strong:.4f}")
result("  Weak hand equity", PASS, f"{eq_weak:.4f}")

# Simulate pot odds: hero faces a pot-size bet
# Pot = 16 (8+8 before bet), opponent bets 16 more -> total pot 32
# Hero needs to call 16 to win 32 -> pot odds = 16/48 = 0.333
pot_odds = 16.0 / (16 + 16 + 16)
result("  Pot odds", PASS, f"{pot_odds:.4f}")

# Equity gate check
if eq_strong >= pot_odds:
    result("  Strong hand passes equity gate", PASS,
           f"{eq_strong:.4f} >= {pot_odds:.4f}")
else:
    result("  Strong hand passes equity gate", WARN,
           f"{eq_strong:.4f} < {pot_odds:.4f} (would be overridden to FOLD)")

if eq_weak < pot_odds:
    result("  Weak hand blocked by equity gate", PASS,
           f"{eq_weak:.4f} < {pot_odds:.4f} -> override to FOLD")
else:
    result("  Weak hand blocked by equity gate", WARN,
           f"{eq_weak:.4f} >= {pot_odds:.4f} (would NOT be overridden)")

# Now test equity for clearly dominant vs clearly dominated
# The Ace-high board: use the ace
ace_board = [8, 17, 26, 0, 1]  # Ad, Ah, As, 2d, 3d (three aces on board!)
remaining_ace = [c for c in range(27) if c not in ace_board]
# With 3 aces on board, having high kickers matters
hero_high = [7, 16]  # 9d, 9h (pair of 9s as kicker)
hero_low = [2, 11]   # 4d, 4h (pair of 4s as kicker)

eq_high = equity_engine.compute_equity(hero_high, ace_board, [])
eq_low = equity_engine.compute_equity(hero_low, ace_board, [])
result("  9-kicker equity (3 aces board)", PASS, f"{eq_high:.4f}")
result("  4-kicker equity (3 aces board)", PASS, f"{eq_low:.4f}")
if eq_high > eq_low:
    result("  9-kicker > 4-kicker", PASS, "equity ordering correct")
else:
    result("  9-kicker > 4-kicker", FAIL,
           f"{eq_high:.4f} <= {eq_low:.4f}")


# ═════════════════════════════════════════════════════════════
# TEST 5: Match Log Replay
# ═════════════════════════════════════════════════════════════
section("TEST 5: Match Log Replay")

_27CARD_STR = [
    '2d','3d','4d','5d','6d','7d','8d','9d','Ad',
    '2h','3h','4h','5h','6h','7h','8h','9h','Ah',
    '2s','3s','4s','5s','6s','7s','8s','9s','As',
]
_CARD_TO_INT = {c.lower(): i for i, c in enumerate(_27CARD_STR)}

def card_str_to_int(cs):
    return _CARD_TO_INT.get(cs.lower())

import glob as globmod
match_files = sorted(globmod.glob(os.path.expanduser('~/Downloads/match_*.txt')))

if not match_files:
    result("  Match logs", WARN, "No match_*.txt in ~/Downloads")
else:
    result("  Match logs found", PASS, f"{len(match_files)} files")

    # Process the 3 most recent logs
    logs_to_test = match_files[-3:]
    total_river_decisions = 0
    total_lookups = 0
    total_crashes = 0
    total_solver_tests = 0
    total_solver_crashes = 0
    solver_times = []

    for log_path in logs_to_test:
        log_name = os.path.basename(log_path)
        with open(log_path) as f:
            first_line = f.readline().strip()
            content = f.read()

        lines = [l for l in content.split('\n') if l.strip()]
        rows = list(csv.DictReader(lines))

        river_rows = []
        for row in rows:
            if row.get('street') != 'River':
                continue
            if row.get('active_team') != '0':
                continue
            try:
                t0_cards = ast.literal_eval(row['team_0_cards'])
                board_cards = ast.literal_eval(row['board_cards'])
                t0_disc = ast.literal_eval(row.get('team_0_discarded', '[]'))
                t1_disc = ast.literal_eval(row.get('team_1_discarded', '[]'))
                if len(t0_cards) != 2 or len(board_cards) != 5:
                    continue
                river_rows.append({
                    'hand': row.get('hand_number', '?'),
                    'hero_strs': t0_cards,
                    'board_strs': board_cards,
                    'action': row.get('action_type', '?'),
                    'my_bet': int(row.get('team_0_bet', 0)),
                    'opp_bet': int(row.get('team_1_bet', 0)),
                    'hero_disc_strs': t0_disc,
                    'opp_disc_strs': t1_disc,
                })
            except Exception:
                continue

        n_river = len(river_rows)
        total_river_decisions += n_river

        n_lookup_ok = 0
        n_crash = 0
        n_solver_ok = 0
        n_solver_crash = 0

        # Test up to 20 river decisions per log
        test_rows = river_rows[:20]
        for dec in test_rows:
            hero_ints = [card_str_to_int(c) for c in dec['hero_strs']]
            board_ints = [card_str_to_int(c) for c in dec['board_strs']]

            if None in hero_ints or None in board_ints:
                continue

            # Test 1: lookup
            try:
                strat = lookup.get_strategy(hero_ints, board_ints,
                                            dec['my_bet'], dec['opp_bet'])
                if strat is not None:
                    probs = list(strat.values())
                    if (not any(math.isnan(p) or math.isinf(p) for p in probs)
                            and abs(sum(probs) - 1.0) < 0.02):
                        n_lookup_ok += 1
                    else:
                        n_crash += 1
                        result(f"    {log_name} hand {dec['hand']}", FAIL,
                               f"Bad strategy: sum={sum(probs):.4f}")
                else:
                    n_lookup_ok += 1  # None is ok
            except Exception as e:
                n_crash += 1
                result(f"    {log_name} hand {dec['hand']} lookup CRASH", FAIL,
                       str(e)[:100])

            # Test 2: solver (for facing-bet scenarios, where opp_bet > my_bet)
            if dec['opp_bet'] > dec['my_bet']:
                try:
                    pbet = lookup.get_p_bet(board_ints,
                                            my_bet=dec['my_bet'],
                                            opp_bet=dec['opp_bet'])
                    if pbet is not None:
                        # Quick narrowing
                        narrowed = {}
                        known = set(board_ints) | set(hero_ints)
                        avail = [c for c in range(27) if c not in known]
                        for h in itertools.combinations(avail, 2):
                            key = (min(h), max(h))
                            pb = pbet.get(key, 0.0)
                            narrowed[key] = max(pb, 0.05) / len(avail)

                        tw = sum(narrowed.values())
                        if tw > 0:
                            for k in narrowed:
                                narrowed[k] /= tw

                        dead = []
                        for cs in dec['hero_disc_strs'] + dec['opp_disc_strs']:
                            ci = card_str_to_int(cs)
                            if ci is not None:
                                dead.append(ci)

                        t0 = time.perf_counter()
                        action = solver.solve_and_act(
                            hero_ints, board_ints, narrowed, dead,
                            my_bet=dec['my_bet'], opp_bet=dec['opp_bet'],
                            street=3, min_raise=2, max_raise=100,
                            valid_actions=[True, True, True, True],
                            time_remaining=300)
                        t_s = time.perf_counter() - t0
                        solver_times.append(t_s)
                        total_solver_tests += 1

                        if action is not None:
                            act_type, _, _, _ = action
                            if act_type in (0, 1, 2, 3):
                                n_solver_ok += 1
                            else:
                                n_solver_crash += 1
                        else:
                            n_solver_ok += 1  # None ok
                except Exception as e:
                    n_solver_crash += 1
                    total_solver_crashes += 1
                    result(f"    {log_name} hand {dec['hand']} solver CRASH", FAIL,
                           str(e)[:150])

        total_lookups += n_lookup_ok
        total_crashes += n_crash
        total_solver_crashes += n_solver_crash

        print(f"  {log_name}: {n_river} river decisions, "
              f"tested {len(test_rows)}, "
              f"lookups ok={n_lookup_ok}, solver ok={n_solver_ok}, "
              f"crashes={n_crash + n_solver_crash}")

    result("  Total match log lookups", PASS if total_crashes == 0 else FAIL,
           f"{total_lookups} ok, {total_crashes} crashes")
    result("  Total solver tests", PASS if total_solver_crashes == 0 else FAIL,
           f"{total_solver_tests} tested, {total_solver_crashes} crashes")

    if solver_times:
        avg_t = sum(solver_times) / len(solver_times)
        max_t = max(solver_times)
        result("  Solver timing (match replay)",
               PASS if max_t < 5.0 else FAIL,
               f"avg={avg_t:.2f}s, max={max_t:.2f}s, n={len(solver_times)}")


# ═════════════════════════════════════════════════════════════
# TEST 6: Timing Under Load (50 river decisions)
# ═════════════════════════════════════════════════════════════
section("TEST 6: Timing Under Load (50 decisions)")

random.seed(123)
np.random.seed(123)

acting_times = []
facing_times = []
n_acting_ok = 0
n_facing_ok = 0
n_acting_crash = 0
n_facing_crash = 0

for trial in range(50):
    board = sorted(random.sample(range(27), 5))
    remaining = [c for c in range(27) if c not in board]
    hero = random.sample(remaining, 2)

    # Acting first timing
    try:
        t0 = time.perf_counter()
        strat = lookup.get_strategy(hero, board, 8, 8)
        dt = time.perf_counter() - t0
        acting_times.append(dt)
        n_acting_ok += 1
    except Exception as e:
        n_acting_crash += 1
        result(f"  Trial {trial} acting crash", FAIL, str(e)[:80])

    # Facing bet timing (every 3rd trial to keep total time manageable)
    if trial % 3 == 0:
        try:
            pbet = lookup.get_p_bet(board, my_bet=4, opp_bet=4)
            if pbet:
                narrowed = {}
                for h in itertools.combinations(remaining, 2):
                    key = (min(h), max(h))
                    pb = pbet.get(key, 0.0)
                    narrowed[key] = max(pb, 0.05)
                tw = sum(narrowed.values())
                if tw > 0:
                    for k in narrowed:
                        narrowed[k] /= tw

                t0 = time.perf_counter()
                action = solver.solve_and_act(
                    hero, board, narrowed, [],
                    my_bet=4, opp_bet=12, street=3,
                    min_raise=2, max_raise=100,
                    valid_actions=[True, True, True, True],
                    time_remaining=300)
                dt = time.perf_counter() - t0
                facing_times.append(dt)
                n_facing_ok += 1
        except Exception as e:
            n_facing_crash += 1
            result(f"  Trial {trial} facing crash", FAIL, str(e)[:80])

# Report acting-first times
if acting_times:
    avg_a = sum(acting_times) / len(acting_times) * 1000
    max_a = max(acting_times) * 1000
    p95_a = sorted(acting_times)[int(0.95 * len(acting_times))] * 1000
    result("  Acting-first lookup (50 trials)",
           PASS if max_a < 5000 else FAIL,
           f"avg={avg_a:.1f}ms, p95={p95_a:.1f}ms, max={max_a:.1f}ms")

# Report facing-bet times
if facing_times:
    avg_f = sum(facing_times) / len(facing_times)
    max_f = max(facing_times)
    p95_f = sorted(facing_times)[int(0.95 * len(facing_times))]
    timeout_count = sum(1 for t in facing_times if t > 5.0)
    result("  Facing-bet solver (17 trials)",
           PASS if timeout_count == 0 else FAIL,
           f"avg={avg_f:.2f}s, p95={p95_f:.2f}s, max={max_f:.2f}s, "
           f"timeouts(>5s)={timeout_count}")

result("  Crashes", PASS if (n_acting_crash + n_facing_crash) == 0 else FAIL,
       f"acting={n_acting_crash}, facing={n_facing_crash}")


# ═════════════════════════════════════════════════════════════
# BONUS: Inference Pipeline Integration
# ═════════════════════════════════════════════════════════════
section("BONUS: Inference + River Pipeline Integration")

# Test the full chain: discard inference -> narrowing -> solver
try:
    inference = DiscardInference(equity_engine)
    board = [0, 1, 2]  # flop: 2d, 3d, 4d
    opp_discards = [9, 10, 11]  # 2h, 3h, 4h
    my_cards = [18, 19]  # 2s, 3s (just 2 cards, post-discard)

    t0 = time.perf_counter()
    opp_weights = inference.infer_opponent_weights(opp_discards, board, my_cards)
    t_inf = time.perf_counter() - t0

    n_nonzero = sum(1 for v in opp_weights.values() if v > 0.001)
    total_w = sum(opp_weights.values())
    result("  Discard inference", PASS,
           f"{n_nonzero} likely hands (of {len(opp_weights)}), "
           f"sum={total_w:.4f}, time={t_inf*1000:.1f}ms")

    # Now extend to river and test lookup
    river_board = [0, 1, 2, 3, 4]  # add turn+river: 5d, 6d
    hero_cards = [18, 19]  # 2s, 3s

    pbet = lookup.get_p_bet(river_board, my_bet=8, opp_bet=8)
    if pbet:
        # Combine inference weights with P(bet) narrowing
        combined = {}
        for hand, w in opp_weights.items():
            if hand in pbet:
                pb = max(pbet[hand], 0.05)
                combined[hand] = w * pb

        tw = sum(combined.values())
        if tw > 0:
            for k in combined:
                combined[k] /= tw

        n_live = sum(1 for v in combined.values() if v > 0.001)
        result("  Combined inference + P(bet) narrowing", PASS,
               f"{n_live} live hands after double narrowing")

        # Feed to solver
        dead = list(opp_discards)
        t0 = time.perf_counter()
        action = solver.solve_and_act(
            hero_cards, river_board, combined, dead,
            my_bet=8, opp_bet=16, street=3,
            min_raise=2, max_raise=100,
            valid_actions=[True, True, True, True],
            time_remaining=300)
        t_s = time.perf_counter() - t0

        if action:
            act_names = {0: 'FOLD', 1: 'RAISE', 2: 'CHECK', 3: 'CALL'}
            result("  Full pipeline solver", PASS,
                   f"action={act_names.get(action[0], '?')}, time={t_s:.2f}s")
        else:
            result("  Full pipeline solver", WARN, "returned None")
    else:
        result("  P(bet) for integration test", WARN, "returned None")
except Exception as e:
    result("  Integration pipeline", FAIL, traceback.format_exc()[-300:])


# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"\n  Total FAILURES: {len(failures)}")
for f in failures:
    print(f"    [FAIL] {f}")

print(f"\n  Total WARNINGS: {len(warnings)}")
for w in warnings:
    print(f"    [WARN] {w}")

if not failures:
    print("\n  v38 PASSED all critical tests. Safe to ship.")
else:
    print(f"\n  v38 has {len(failures)} FAILURE(s). DO NOT SHIP without fixing.")

print()
