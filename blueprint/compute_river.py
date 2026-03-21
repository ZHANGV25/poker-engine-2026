#!/usr/bin/env python3
"""
Precompute river strategies for ALL C(27,5) = 80,730 boards.

For each board x 5 pot sizes, runs range-balanced CFR to extract:
  - hero_s: hero acting-first strategy at root (P(action|hand) as uint8)
  - opp_pb: opponent P(bet|hand) at opponent root node (as uint8)

Distributes work across instances via --start_board / --end_board.
Uses multiprocessing with 34 workers (leave 2 cores free on c5.9xlarge).

Output format per instance: single pickle file containing dict:
  {board_tuple: {
      'opp_pb':  np.array(shape=(5, n_hands), dtype=uint8),
      'hero_s':  np.array(shape=(5, n_hands, n_actions), dtype=uint8),
      'hands':   np.array(shape=(n_hands, 2), dtype=int8),
  }}

Usage:
  python3 compute_river.py --start_board 0 --end_board 8073
"""

import sys
import os
import time
import argparse
import itertools
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np

# Allow importing from submission/ and blueprint/
_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from equity import ExactEquityEngine
from game_tree import (
    GameTree, TERM_NONE, TERM_SHOWDOWN,
)
from multi_street_solver import (
    _flatten_tree, _build_valid_pairs, _enumerate_hands,
    _fill_river_showdown, _river_showdown_values,
    _solve_street, _normalize_strategy_numba,
    warmup_jit,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_BOARDS = 80730  # C(27, 5)
N_CARDS = 27
BOARD_SIZE = 5
POT_SIZES = [(2, 2), (6, 6), (14, 14), (30, 30), (50, 50)]
N_POTS = len(POT_SIZES)
MAX_BET = 100
MIN_RAISE = 2

# Mutable config so main() can override from CLI args before forking workers
_CONFIG = {'cfr_iterations': 2000}


# ---------------------------------------------------------------------------
# Per-board solver (runs in worker process)
# ---------------------------------------------------------------------------

def _init_worker():
    """Initialize per-process globals (equity engine, trees, etc.)."""
    global _engine, _trees, _flats
    _engine = ExactEquityEngine()

    # Pre-build one compact GameTree per pot size (hero acts first).
    # These are reused for every board since the tree structure only depends
    # on the pot geometry, not on the board cards.
    _trees = []
    _flats = []
    for hero_bet, opp_bet in POT_SIZES:
        tree = GameTree(hero_bet, opp_bet, MIN_RAISE, MAX_BET, True, compact=False)
        flat = _flatten_tree(tree)
        _trees.append(tree)
        _flats.append(flat)


def _solve_board(board_tuple):
    """Solve a single river board across all pot sizes.

    Args:
        board_tuple: tuple of 5 card ints (sorted)

    Returns:
        (board_tuple, result_dict) where result_dict has keys:
            'opp_pb':  (N_POTS, n_hands) uint8
            'hero_s':  (N_POTS, n_hands, n_actions) uint8
            'hands':   (n_hands, 2) int8
    """
    global _engine, _trees, _flats

    board = list(board_tuple)

    # Enumerate all valid 2-card hands from remaining 22 cards
    hands = _enumerate_hands(board)  # sorted list of (c1, c2)
    n_hands = len(hands)  # should be C(22,2) = 231
    valid_pairs = _build_valid_pairs(hands)

    # Precompute hand ranks (deterministic on river)
    board_mask = 0
    for c in board:
        board_mask |= 1 << c
    seven_lookup = _engine._seven

    hand_ranks = np.zeros(n_hands, dtype=np.int64)
    for i, hand in enumerate(hands):
        mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
        hand_ranks[i] = seven_lookup[mask]

    # Determine max actions across all pot trees for consistent output shape
    max_hero_actions = max(f['max_actions'] for f in _flats)

    opp_pb = np.zeros((N_POTS, n_hands), dtype=np.uint8)
    hero_s = np.zeros((N_POTS, n_hands, max_hero_actions), dtype=np.uint8)

    for pi, (tree, flat) in enumerate(zip(_trees, _flats)):
        n_hero_nodes = flat['n_hero_nodes']
        n_opp_nodes = flat['n_opp_nodes']
        max_actions = flat['max_actions']

        # Compute per-terminal showdown values
        sd_ids = [nid for nid in tree.terminal_node_ids
                  if tree.terminal[nid] == TERM_SHOWDOWN]
        n_sd = len(sd_ids)
        sd_vals = np.zeros((n_sd, n_hands, n_hands), dtype=np.float64)
        pot_won_arr = np.array([min(tree.hero_pot[nid], tree.opp_pot[nid])
                                for nid in sd_ids], dtype=np.float64)
        _fill_river_showdown(sd_vals, hand_ranks, valid_pairs, pot_won_arr,
                             n_hands, n_sd)

        # Solve: get both hero and opp strategies
        hero_strat, opp_strat, _ = _solve_street(
            tree, hands, sd_vals, sd_ids, valid_pairs,
            _CONFIG['cfr_iterations'], return_opp=True, flat=flat)

        # hero_strat shape: (n_hands, n_hero_nodes, max_actions)
        # opp_strat shape:  (n_hands, n_opp_nodes, max_actions)

        # Extract hero strategy at root node (node index 0 in hero_node_ids)
        # Root is always the first hero node since hero acts first.
        root_hero_strat = hero_strat[:, 0, :max_actions]  # (n_hands, max_actions)
        q = np.clip(np.round(root_hero_strat * 255.0), 0, 255).astype(np.uint8)
        hero_s[pi, :, :max_actions] = q

        # Extract opp P(bet|hand) at opp's first decision node (node index 0).
        # In a hero-first tree, opp's first node is after hero checks.
        # Actions there: CHECK (act_type=1), BET variants.
        # P(bet) = sum of all non-check action probabilities.
        if n_opp_nodes > 0:
            opp_root_strat = opp_strat[:, 0, :]  # (n_hands, max_actions)
            # Identify which actions are non-check at opp's first node
            opp_first_nid = tree.opp_node_ids[0]
            opp_children = tree.children[opp_first_nid]
            n_opp_act = tree.num_actions[opp_first_nid]

            p_bet = np.zeros(n_hands, dtype=np.float64)
            for a in range(n_opp_act):
                act_type, _ = opp_children[a]
                if act_type != 1:  # not CHECK
                    p_bet += opp_root_strat[:, a]

            opp_pb[pi, :] = np.clip(np.round(p_bet * 255.0), 0, 255).astype(np.uint8)

    # Pack hands as int8 array
    hands_arr = np.array(hands, dtype=np.int8)

    return (board_tuple, {
        'opp_pb': opp_pb,
        'hero_s': hero_s,
        'hands': hands_arr,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute river strategies for poker blueprint")
    parser.add_argument('--start_board', type=int, default=0,
                        help='First board index (inclusive)')
    parser.add_argument('--end_board', type=int, default=TOTAL_BOARDS,
                        help='Last board index (exclusive)')
    parser.add_argument('--workers', type=int, default=34,
                        help='Number of multiprocessing workers')
    parser.add_argument('--iterations', type=int, default=_CONFIG['cfr_iterations'],
                        help='CFR iterations per solve')
    parser.add_argument('--output_dir', type=str, default='/opt/blueprint/output',
                        help='Output directory')
    args = parser.parse_args()

    _CONFIG['cfr_iterations'] = args.iterations

    print("=" * 60)
    print("River Strategy Computation")
    print("=" * 60)
    print(f"  Board range:  [{args.start_board}, {args.end_board})")
    print(f"  Boards:       {args.end_board - args.start_board}")
    print(f"  Pot sizes:    {POT_SIZES}")
    print(f"  CFR iters:    {args.iterations}")
    print(f"  Workers:      {args.workers}")
    print(f"  Output dir:   {args.output_dir}")
    print()

    # Enumerate all boards deterministically
    all_boards = list(itertools.combinations(range(N_CARDS), BOARD_SIZE))
    assert len(all_boards) == TOTAL_BOARDS, (
        f"Expected {TOTAL_BOARDS} boards, got {len(all_boards)}")
    my_boards = all_boards[args.start_board:args.end_board]
    n_total = len(my_boards)
    print(f"Boards to solve: {n_total}")

    # Warmup Numba JIT in the main process (compilation is cached)
    print("Warming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    warmup_jit()
    print(f" done ({time.time() - t0:.1f}s)")

    # Quick single-board test to verify everything works (single-process)
    print("Running single-board sanity check...", end="", flush=True)
    t0 = time.time()
    _init_worker()
    test_board = my_boards[0]
    test_result = _solve_board(test_board)
    _, test_data = test_result
    # Clean up main process globals to avoid doubling memory on fork
    global _engine, _trees, _flats
    _engine = None
    _trees = None
    _flats = None
    print(f" done ({time.time() - t0:.1f}s)")
    print(f"  Test board: {test_board}")
    print(f"  hands shape: {test_data['hands'].shape}")
    print(f"  hero_s shape: {test_data['hero_s'].shape}")
    print(f"  opp_pb shape: {test_data['opp_pb'].shape}")
    print()

    # Solve all boards with multiprocessing
    results = {}
    t_start = time.time()
    done = 0

    # Use 'forkserver' on Linux for memory efficiency without macOS fork issues.
    # Falls back to 'spawn' on macOS.
    import platform
    if platform.system() == 'Linux':
        ctx = mp.get_context('forkserver')
    else:
        ctx = mp.get_context('spawn')

    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for board_tuple, board_data in pool.imap_unordered(
                _solve_board, my_boards, chunksize=4):
            results[board_tuple] = board_data
            done += 1

            if done % 100 == 0 or done == n_total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{n_total}] "
                      f"{elapsed:.0f}s elapsed, "
                      f"{rate:.1f} boards/s, "
                      f"ETA {eta:.0f}s ({eta/60:.1f}m)",
                      flush=True)

    total_time = time.time() - t_start
    print()
    print(f"All {n_total} boards solved in {total_time:.0f}s "
          f"({total_time/60:.1f}m, {total_time/n_total*1000:.1f}ms/board)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"river_{args.start_board}_{args.end_board}.pkl")
    print(f"Saving to {out_path}...", end="", flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f" done ({size_mb:.0f} MB)")

    print()
    print("=" * 60)
    print(f"Complete: {n_total} boards, {total_time:.0f}s, {size_mb:.0f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
