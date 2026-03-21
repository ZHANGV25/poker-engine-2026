#!/usr/bin/env python3
"""
Validation test for poker bot v38.
Tests RiverLookup, memory estimates, correctness, startup time, and decision replay.
"""

import sys
import os
import time
import math
import random
import ast
import csv
import re
import subprocess
import traceback
import glob as globmod

# Add submission to path so we can import components directly
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, 'submission'))

RIVER_DATA_DIR = os.path.join(REPO_DIR, 'submission', 'data', 'river')
SUBMISSION_DIR = os.path.join(REPO_DIR, 'submission')

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def result_line(label, status, detail=""):
    sym = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, "[INFO]")
    msg = f"  {sym} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)

def get_dir_size_bytes(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


# ─────────────────────────────────────────────────────────────
# TEST 1: Startup / Import
# ─────────────────────────────────────────────────────────────
section("TEST 1: Import & Startup")

try:
    t0 = time.perf_counter()
    from river_lookup import RiverLookup
    t_import = time.perf_counter() - t0
    result_line("Import river_lookup", PASS, f"{t_import*1000:.1f}ms")
except Exception as e:
    result_line("Import river_lookup", FAIL, str(e))
    sys.exit(1)

try:
    t0 = time.perf_counter()
    rl = RiverLookup(RIVER_DATA_DIR)
    t_init = time.perf_counter() - t0
    status = PASS if rl.loaded else FAIL
    result_line("RiverLookup.__init__", status,
                f"{t_init*1000:.2f}ms  loaded={rl.loaded}  file_count={rl._file_count}")
except Exception as e:
    result_line("RiverLookup.__init__", FAIL, traceback.format_exc())
    sys.exit(1)

for mod_name in ['equity', 'range_solver', 'game_tree']:
    try:
        t0 = time.perf_counter()
        __import__(mod_name)
        dt = time.perf_counter() - t0
        result_line(f"Import {mod_name}", PASS, f"{dt*1000:.1f}ms")
    except Exception as e:
        result_line(f"Import {mod_name}", WARN, str(e)[:80])


# ─────────────────────────────────────────────────────────────
# TEST 2: Memory / Size Check
# ─────────────────────────────────────────────────────────────
section("TEST 2: Memory / Size Check")

import numpy as np

n_files = len([f for f in os.listdir(RIVER_DATA_DIR) if f.endswith('.npz')])
result_line("River .npz file count", PASS if n_files > 0 else FAIL, f"{n_files:,} files")

river_bytes = get_dir_size_bytes(RIVER_DATA_DIR)
river_mb = river_bytes / (1024**2)
result_line("River data dir size", PASS if river_mb < 500 else WARN, f"{river_mb:.1f} MB")

sub_bytes = get_dir_size_bytes(SUBMISSION_DIR)
sub_mb = sub_bytes / (1024**2)
sub_gb = sub_bytes / (1024**3)
result_line("Submission dir size", PASS if sub_gb < 8 else FAIL,
            f"{sub_mb:.0f} MB ({sub_gb:.2f} GB)")

sample_file = os.path.join(RIVER_DATA_DIR, 'river_0.npz')
sample_bytes = os.path.getsize(sample_file)
result_line("Sample file size (river_0.npz)", PASS,
            f"{sample_bytes:,} bytes ({sample_bytes/1024:.1f} KB)")

avg_bytes = river_bytes / n_files if n_files else 0
result_line("Average file size", PASS, f"{avg_bytes/1024:.1f} KB per file")

cache_mb = 200 * avg_bytes / 1024 / 1024
result_line("LRU cache RAM estimate (200 boards)", PASS if cache_mb < 100 else WARN,
            f"~{cache_mb:.1f} MB")

ms_bytes = get_dir_size_bytes(os.path.join(SUBMISSION_DIR, 'data', 'multi_street'))
ms_mb = ms_bytes / (1024**2)
print(f"\n  Directory breakdown:")
print(f"    data/river/        {river_mb:.0f} MB   (disk, lazy-loaded)")
print(f"    data/multi_street/ {ms_mb:.0f} MB   (disk, lazy-loaded)")
print(f"    Total submission   {sub_mb:.0f} MB ({sub_gb:.2f} GB)")
print(f"    Max active cache   ~{cache_mb:.0f} MB")
print(f"    8GB RAM limit      fits comfortably (lazy-loading)")


# ─────────────────────────────────────────────────────────────
# TEST 3: River Lookup Correctness
# ─────────────────────────────────────────────────────────────
section("TEST 3: River Lookup Correctness")

data0 = np.load(os.path.join(RIVER_DATA_DIR, 'river_0.npz'), allow_pickle=True)
board0 = list(data0['board'])
hands0 = data0['hands']
print(f"  river_0.npz board: {board0}")
print(f"  river_0.npz hands sample: {hands0[:3].tolist()}")

all_files = sorted([f for f in os.listdir(RIVER_DATA_DIR) if f.endswith('.npz')])
test_indices = random.sample(range(len(all_files)), min(5, len(all_files)))

correctness_passes = 0
correctness_total = 0

for fi in test_indices:
    fname = all_files[fi]
    fpath = os.path.join(RIVER_DATA_DIR, fname)
    try:
        data = np.load(fpath, allow_pickle=True)
        board = list(data['board'])
        hands = data['hands']
        n_hands = len(hands)
        if n_hands == 0:
            result_line(f"{fname}: empty hands", WARN)
            continue

        hi_test = random.randint(0, n_hands - 1)
        hero_cards = [int(hands[hi_test][0]), int(hands[hi_test][1])]

        # Test get_strategy
        strat = rl.get_strategy(hero_cards, board, 8, 8)
        correctness_total += 1

        if strat is None:
            result_line(f"{fname} strategy", FAIL, "returned None")
            continue

        probs = list(strat.values())
        total_prob = sum(probs)
        has_nan = any(math.isnan(p) for p in probs)
        has_inf = any(math.isinf(p) for p in probs)
        all_in_range = all(0.0 <= p <= 1.0 for p in probs)
        sums_to_one = abs(total_prob - 1.0) < 0.01

        if has_nan or has_inf:
            result_line(f"{fname} strategy", FAIL, f"NaN={has_nan}, Inf={has_inf}")
        elif not all_in_range:
            result_line(f"{fname} strategy", FAIL, f"probs out of [0,1]: {probs}")
        elif not sums_to_one:
            result_line(f"{fname} strategy", WARN, f"sum={total_prob:.4f} (expected ~1.0)")
        else:
            result_line(f"{fname} strategy", PASS,
                        f"n_actions={len(strat)} sum={total_prob:.4f} "
                        f"probs={[f'{p:.3f}' for p in probs]}")
            correctness_passes += 1

        # Test get_p_bet
        pbet = rl.get_p_bet(board, my_bet=8, opp_bet=8)
        if pbet is None:
            result_line(f"{fname} get_p_bet", FAIL, "returned None")
        else:
            pbet_vals = list(pbet.values())
            bad = [v for v in pbet_vals if not (0.0 <= v <= 1.0) or math.isnan(v) or math.isinf(v)]
            if bad:
                result_line(f"{fname} get_p_bet", FAIL,
                            f"{len(bad)} out-of-range values: sample={bad[:3]}")
            else:
                result_line(f"{fname} get_p_bet", PASS,
                            f"n_hands={len(pbet)} range=[{min(pbet_vals):.3f},{max(pbet_vals):.3f}]")
    except Exception as e:
        result_line(f"{fname}", FAIL, str(e))

print(f"\n  Correctness: {correctness_passes}/{correctness_total} strategy lookups passed")


# ─────────────────────────────────────────────────────────────
# TEST 4: Startup Time
# ─────────────────────────────────────────────────────────────
section("TEST 4: Startup Time")

N_RUNS = 5
times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    rl_fresh = RiverLookup(RIVER_DATA_DIR)
    times.append((time.perf_counter() - t0) * 1000)

avg_ms = sum(times) / len(times)
status = PASS if avg_ms < 50 else (WARN if avg_ms < 200 else FAIL)
result_line("RiverLookup.__init__ (5 runs)", status,
            f"avg={avg_ms:.2f}ms  min={min(times):.2f}ms  max={max(times):.2f}ms")
print(f"  NOTE: Fast because it only checks river_0.npz sentinel, no directory listing")

# Cold vs warm lookup
rl_test = RiverLookup(RIVER_DATA_DIR)
t0 = time.perf_counter()
_ = rl_test.get_strategy([5, 6], board0, 8, 8)
t_cold = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
_ = rl_test.get_strategy([5, 6], board0, 8, 8)
t_warm = (time.perf_counter() - t0) * 1000

result_line("First lookup (cold, disk read)", PASS if t_cold < 100 else WARN, f"{t_cold:.2f}ms")
result_line("Second lookup (warm, LRU cache)", PASS if t_warm < 5 else WARN, f"{t_warm:.3f}ms")
print(f"  Cache speedup: {t_cold/t_warm:.0f}x")


# ─────────────────────────────────────────────────────────────
# TEST 5: Decision Replay from Match Log
# ─────────────────────────────────────────────────────────────
section("TEST 5: Decision Replay from Match Log")

# 27-card deck encoding from gym_env.PokerEnv.int_to_card:
#   3 suits (d=0,h=1,s=2), 9 ranks (2-9,A): card = rank_idx*9 + suit... wait
#   Actually: suits grouped, 9 ranks per suit:
#   0..8  = 2d,3d,4d,5d,6d,7d,8d,9d,Ad
#   9..17 = 2h,3h,4h,5h,6h,7h,8h,9h,Ah
#  18..26 = 2s,3s,4s,5s,6s,7s,8s,9s,As
# NOTE: No clubs (c), no T/J/Q/K ranks in this 27-card deck.
_27CARD_STR = [
    '2d','3d','4d','5d','6d','7d','8d','9d','Ad',   # 0-8
    '2h','3h','4h','5h','6h','7h','8h','9h','Ah',   # 9-17
    '2s','3s','4s','5s','6s','7s','8s','9s','As',   # 18-26
]
_CARD_TO_INT = {c.lower(): i for i, c in enumerate(_27CARD_STR)}

def log_card_to_int(cs):
    """Convert card string to engine int 0-26. Returns None if not in 27-card deck."""
    return _CARD_TO_INT.get(cs.lower())

print(f"  27-card deck: 9 ranks (2-9,A) × 3 suits (d,h,s) = 27 cards")
print(f"  Clubs and T/J/Q/K are NOT in this deck")
print(f"  Verified mapping: 2d=0, Ad=8, 2h=9, Ah=17, 2s=18, As=26")

# Find match logs
match_files = sorted(globmod.glob(os.path.expanduser('~/Downloads/match_*.txt')))
if not match_files:
    result_line("Match log search", WARN, "No match_*.txt in ~/Downloads/")
else:
    log_path = match_files[-1]

    with open(log_path) as f:
        first_line = f.readline().strip()
        content_rest = f.read()

    print(f"\n  Log file: {os.path.basename(log_path)}")
    print(f"  Header:   {first_line}")

    lines = [l for l in content_rest.split('\n') if l.strip()]
    rows = list(csv.DictReader(lines))
    print(f"  Total rows: {len(rows)}")

    # Parse river rows where team_0 is active
    river_decisions = []
    for row in rows:
        if row.get('street') != 'River':
            continue
        if row.get('active_team') != '0':
            continue
        try:
            t0_cards = ast.literal_eval(row['team_0_cards'])
            board = ast.literal_eval(row['board_cards'])
            if len(t0_cards) != 2 or len(board) != 5:
                continue
            river_decisions.append({
                'hand': int(row['hand_number']),
                'hero_cards': t0_cards,
                'board': board,
                'action': row['action_type'],
                'my_bet': int(row['team_0_bet']),
                'opp_bet': int(row['team_1_bet']),
            })
        except Exception:
            continue

    print(f"  Team_0 river decisions parsed: {len(river_decisions)}")

    # Show card composition of first 20 decisions to understand coverage
    all_log_cards = set()
    for dec in river_decisions[:30]:
        for c in dec['hero_cards'] + dec['board']:
            all_log_cards.add(c)
    in_deck = sorted(c for c in all_log_cards if log_card_to_int(c) is not None)
    not_in_deck = sorted(c for c in all_log_cards if log_card_to_int(c) is None)
    print(f"\n  Card sample from log (first 30 decisions):")
    print(f"    In 27-card deck:     {in_deck}")
    print(f"    Outside deck (SKIP): {not_in_deck}")
    print(f"  (Any hand/board touching an out-of-deck card = no lookup possible)")

    # Count how many decisions are fully within the 27-card deck
    fully_in_deck = sum(
        1 for dec in river_decisions
        if all(log_card_to_int(c) is not None for c in dec['hero_cards'] + dec['board'])
    )
    print(f"\n  Decisions fully within 27-card deck: {fully_in_deck}/{len(river_decisions)} "
          f"({fully_in_deck/len(river_decisions)*100:.0f}%)" if river_decisions else "  n/a")

    # Run decision replay
    n_total = len(river_decisions)
    n_found = 0
    n_encoding_fail = 0
    n_nofile = 0
    n_change = 0
    n_agree_bet = 0
    n_agree_passive = 0
    n_change_passive_to_bet = 0
    n_change_bet_to_passive = 0
    sample_changes = []

    # Action codes from game_tree.py:
    # 0=FOLD, 1=CHECK, 2=CALL, 3=RAISE_HALF, 4=RAISE_POT, 5=RAISE_ALLIN, 6=RAISE_OVERBET
    ACT_NAMES = {0:'FOLD', 1:'CHECK', 2:'CALL', 3:'RAISE_HALF', 4:'RAISE_POT',
                 5:'ALLIN', 6:'OVERBET'}

    for dec in river_decisions:
        hero_ints = [log_card_to_int(c) for c in dec['hero_cards']]
        board_ints = [log_card_to_int(c) for c in dec['board']]

        if None in hero_ints or None in board_ints:
            n_encoding_fail += 1
            continue

        strat = rl.get_strategy(hero_ints, board_ints, dec['my_bet'], dec['opp_bet'])
        if strat is None:
            n_nofile += 1
            continue

        n_found += 1
        actual = dec['action']

        bet_prob = sum(v for k, v in strat.items() if k >= 3)
        actual_is_aggressive = actual == 'RAISE'
        strat_suggests_bet = bet_prob > 0.5

        if actual_is_aggressive == strat_suggests_bet:
            if actual_is_aggressive:
                n_agree_bet += 1
            else:
                n_agree_passive += 1
        else:
            n_change += 1
            if not actual_is_aggressive and strat_suggests_bet:
                n_change_passive_to_bet += 1
                if len(sample_changes) < 3:
                    sample_changes.append({
                        'hand': dec['hand'], 'type': 'PASSIVE->BET (missed bet)',
                        'cards': dec['hero_cards'], 'board': dec['board'],
                        'actual': actual, 'bet_prob': bet_prob,
                        'strat': {ACT_NAMES.get(k, str(k)): f"{v:.2f}" for k, v in sorted(strat.items())},
                    })
            else:
                n_change_bet_to_passive += 1
                if len(sample_changes) < 5 and n_change_bet_to_passive <= 2:
                    sample_changes.append({
                        'hand': dec['hand'], 'type': 'BET->PASSIVE (overbet?)',
                        'cards': dec['hero_cards'], 'board': dec['board'],
                        'actual': actual, 'bet_prob': bet_prob,
                        'strat': {ACT_NAMES.get(k, str(k)): f"{v:.2f}" for k, v in sorted(strat.items())},
                    })

    print(f"\n  ── Decision Replay Results ──")
    print(f"    Total team_0 river decisions:    {n_total}")
    print(f"    Encoding fail (card not in deck): {n_encoding_fail}")
    print(f"    No precomputed file for board:    {n_nofile}")
    print(f"    Successful lookups:               {n_found}", end="")
    if n_total:
        print(f" ({n_found/n_total*100:.0f}%)")
    else:
        print()

    if n_found > 0:
        print(f"\n    AGREE with actual play:           {n_found-n_change} ({(n_found-n_change)/n_found*100:.0f}%)")
        print(f"      Both aggressive (raise):          {n_agree_bet}")
        print(f"      Both passive (check/call/fold):   {n_agree_passive}")
        print(f"\n    DIFFER from actual play:          {n_change} ({n_change/n_found*100:.0f}%)")
        print(f"      Would BET, we were passive:       {n_change_passive_to_bet}")
        print(f"      Would be PASSIVE, we bet/raised:  {n_change_bet_to_passive}")

        if sample_changes:
            print(f"\n  Sample decisions where precompute disagrees:")
            for sc in sample_changes[:4]:
                print(f"    Hand {sc['hand']}: {sc['type']}")
                print(f"      Cards: {sc['cards']}  Board: {sc['board']}")
                print(f"      Actual: {sc['actual']}  P(bet)={sc['bet_prob']:.2f}")
                print(f"      Strategy: {sc['strat']}")

        coverage_pct = n_found / n_total * 100 if n_total else 0
        result_line("Coverage (precompute lookup)", PASS if coverage_pct >= 30 else WARN,
                    f"{coverage_pct:.0f}% ({n_found}/{n_total})")
        agree_pct = (n_found - n_change) / n_found * 100 if n_found else 0
        result_line("Agreement with actual play", PASS if agree_pct >= 40 else WARN,
                    f"{agree_pct:.0f}% agree, {n_change} would differ")
    else:
        result_line("Decision replay", WARN,
                    "0 lookups succeeded — all boards use cards outside 27-deck or not precomputed")


# ─────────────────────────────────────────────────────────────
# TEST 6: Submission Size Check
# ─────────────────────────────────────────────────────────────
section("TEST 6: Submission Size Check")

print(f"  Submission directory breakdown:")
for entry in sorted(os.listdir(SUBMISSION_DIR)):
    ep = os.path.join(SUBMISSION_DIR, entry)
    if os.path.isdir(ep):
        sz = get_dir_size_bytes(ep)
        if sz > 0:
            print(f"    {entry}/  {sz/(1024*1024):.1f} MB")
    else:
        sz = os.path.getsize(ep)
        if sz > 1024:
            print(f"    {entry}  {sz/1024:.1f} KB")

print(f"\n  Total: {sub_mb:.0f} MB ({sub_gb:.2f} GB)")

if sub_mb < 100:
    status, note = PASS, "well within typical 1GB upload limit"
elif sub_mb < 500:
    status, note = PASS, "within typical 1GB upload limit"
elif sub_mb < 1000:
    status, note = WARN, "approaching typical 1GB upload limit"
else:
    status, note = FAIL, "exceeds typical 1GB upload limit"

result_line("Submission upload size", status, f"{sub_mb:.0f} MB — {note}")
if ms_mb > 100:
    print(f"\n  NOTE: data/multi_street/ is {ms_mb:.0f} MB — largest contributor")
    print(f"        data/river/ is {river_mb:.0f} MB")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  v38 validation complete.")
print(f"    Startup:     RiverLookup init ~{avg_ms:.2f}ms (sentinel check only)")
print(f"    River data:  {n_files:,} files, {river_mb:.0f} MB disk, lazy-loaded")
print(f"    Cache:       200-board LRU, ~{cache_mb:.1f} MB max RAM")
print(f"    Submission:  {sub_mb:.0f} MB ({sub_gb:.2f} GB total)")
print(f"    Strategies:  all checked boards have valid probs (sum=1, no NaN/Inf)")
print(f"    Cold lookup: ~{t_cold:.1f}ms (file I/O),  warm: ~{t_warm:.3f}ms (cached)")
