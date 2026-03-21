#!/usr/bin/env python3
"""
Precompute river strategies + P(bet|hand) for all 80,730 possible 5-card boards.

For each board:
- Runs DCFR range solver at 5000 iterations (vs 300 at runtime)
- Extracts per-hand strategy at root (CHECK vs BET sizes)
- Extracts P(bet|hand) = sum of all non-check action probabilities
- Stores as compact uint8 arrays

Output: per-batch .npz files with strategies and P(bet|hand).

Usage:
    # Quick test (10 boards):
    python compute_river_strategies.py --start 0 --end 10 --n_workers 1

    # Full run on one machine (split across 10 machines):
    python compute_river_strategies.py --start 0 --end 8073 --n_workers 8
    python compute_river_strategies.py --start 8073 --end 16146 --n_workers 8
    # ... etc (10 machines, 8073 boards each)

    # Upload results to S3:
    aws s3 sync output_river/ s3://poker-blueprint-2026/river_v2/
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
import itertools

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from equity import ExactEquityEngine
from range_solver import RangeSolver
from game_tree import GameTree, ACT_CHECK

logger = logging.getLogger(__name__)

# Pot sizes to solve
POT_SIZES = [(4, 4), (16, 16), (40, 40)]

# DCFR iterations — high quality for precompute
ITERATIONS = 5000


def enumerate_all_river_boards():
    """Generate all C(27,5) = 80,730 possible 5-card boards."""
    return list(itertools.combinations(range(27), 5))


def solve_one_board(args):
    """Solve one river board across all pot sizes."""
    board_idx, board_cards, output_dir, iters_count = args

    engine = ExactEquityEngine()
    solver = RangeSolver(engine)

    board = list(board_cards)
    known = set(board)
    remaining = [c for c in range(27) if c not in known]
    hero_hands = list(itertools.combinations(remaining, 2))
    n_hands = len(hero_hands)

    if n_hands < 3:
        return board_idx, 0, -1

    opp_w = np.ones(n_hands, dtype=np.float64) / n_hands

    result = {
        'board': np.array(board, dtype=np.int8),
        'hands': np.array(hero_hands, dtype=np.int8),
    }

    t0 = time.time()

    for pot_idx, (hero_bet, opp_bet) in enumerate(POT_SIZES):
        # Build tree (full, not compact)
        tree = solver._get_tree(hero_bet, opp_bet, 2, 100, compact=False)

        # Compute equity + not_blocked
        eq, nb = solver._compute_equity_and_mask(
            hero_hands, hero_hands, board, [], 3)

        # Terminal values
        tv = solver._compute_terminal_values(tree, eq, nb)

        # Solve with DCFR at high iteration count
        strat = solver._run_dcfr(tree, opp_w, tv, n_hands, n_hands, iters_count)

        # Extract root strategy
        n_act = tree.num_actions[0]
        root_children = tree.children[0]

        # Per-hand strategy as uint8 (0-255)
        strat_q = np.zeros((n_hands, n_act), dtype=np.uint8)
        for hi in range(n_hands):
            s = strat[hi, :n_act]
            total = s.sum()
            if total > 0:
                s = s / total
            else:
                s = np.ones(n_act) / n_act
            strat_q[hi] = np.clip(s * 255, 0, 255).astype(np.uint8)

        # P(bet|hand) = 1 - P(check|hand)
        check_idx = None
        for ai, (act_type, _) in enumerate(root_children):
            if act_type == ACT_CHECK:
                check_idx = ai
                break

        p_bet = np.zeros(n_hands, dtype=np.uint8)
        for hi in range(n_hands):
            if check_idx is not None:
                check_prob = strat_q[hi, check_idx] / 255.0
                p_bet[hi] = int(np.clip((1.0 - check_prob) * 255, 0, 255))
            else:
                p_bet[hi] = 128  # no check action = ambiguous

        # Action types at root
        act_types = np.array([root_children[a][0] for a in range(n_act)],
                             dtype=np.int8)

        result[f's_p{pot_idx}'] = strat_q      # (n_hands, n_act)
        result[f'pb_p{pot_idx}'] = p_bet        # (n_hands,)
        result[f'a_p{pot_idx}'] = act_types     # (n_act,)

    elapsed = time.time() - t0

    # Save
    out_path = os.path.join(output_dir, f'river_{board_idx}.npz')
    np.savez_compressed(out_path, **result)

    return board_idx, n_hands, elapsed


def solve_batch(args_list, n_workers):
    """Solve a batch of boards with multiprocessing."""
    if n_workers <= 1:
        results = [solve_one_board(a) for a in args_list]
    else:
        with mp.Pool(n_workers) as pool:
            results = pool.map(solve_one_board, args_list)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0,
                        help='Start board index')
    parser.add_argument('--end', type=int, default=100,
                        help='End board index (exclusive)')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', default='output_river',
                        help='Output directory')
    parser.add_argument('--iters', type=int, default=ITERATIONS,
                        help='DCFR iterations per solve')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    all_boards = enumerate_all_river_boards()
    total_boards = len(all_boards)
    logger.info(f'Total river boards: {total_boards}')

    start = min(args.start, total_boards)
    end = min(args.end, total_boards)
    boards = all_boards[start:end]
    n_boards = len(boards)

    logger.info(f'Computing boards {start}-{end} ({n_boards} boards, '
                f'{args.n_workers} workers, {ITERATIONS} iters)')

    tasks = [(start + i, board, args.output_dir, args.iters)
             for i, board in enumerate(boards)]

    # Process in batches of 100 for progress reporting
    batch_size = 100
    total_elapsed = 0
    total_done = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        t0 = time.time()
        results = solve_batch(batch, args.n_workers)
        batch_elapsed = time.time() - t0
        total_elapsed += batch_elapsed

        successes = sum(1 for _, _, t in results if t > 0)
        total_done += len(batch)

        avg_per_board = total_elapsed / total_done
        remaining = (n_boards - total_done) * avg_per_board

        logger.info(f'Batch {batch_start//batch_size + 1}: '
                    f'{successes}/{len(batch)} ok, '
                    f'{total_done}/{n_boards} total, '
                    f'{avg_per_board:.1f}s/board, '
                    f'ETA {remaining/60:.0f}min')

    logger.info(f'Done: {total_done} boards in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
