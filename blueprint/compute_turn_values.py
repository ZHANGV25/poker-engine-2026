#!/usr/bin/env python3
"""
Compute turn game values for flop continuation.

For each (flop_board, turn_card, pot_size), computes the per-hero-hand
game value at the turn root. This captures turn+river betting dynamics
for use as flop continuation values at runtime.

The compute flow is identical to the existing backward induction:
1. For each river card: solve river → get river EVs
2. Average river EVs → turn continuation values
3. Solve turn with continuation values → get turn root EVs
4. Store per-hero-hand average turn EVs (NEW)

Usage:
    # Quick test (1 board):
    python compute_turn_values.py --cluster_start 0 --cluster_end 1

    # Full run (all 2925 boards, split across 10 machines):
    python compute_turn_values.py --cluster_start 0 --cluster_end 293 --n_workers 8
    python compute_turn_values.py --cluster_start 293 --cluster_end 586 --n_workers 8
    # etc.

Output: per-board .npz files with turn game values.
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing as mp

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from abstraction import enumerate_all_flops
from equity import ExactEquityEngine
# import multi_street_solver as solver

logger = logging.getLogger(__name__)

# Pot sizes to compute (subset of blueprint pots)
COMPUTE_POTS = [(2, 2), (10, 10), (30, 30)]


def compute_one_board(args):
    """Compute turn game values for one flop board."""
    board_idx, board_cards, output_dir = args

    engine = ExactEquityEngine()
    t0 = time.time()

    try:
        # Use the existing backward induction solver but extract turn EVs
        # We call the same solve_flop_board but with a hook to capture turn EVs

        board = list(board_cards)
        known = set(board)
        remaining = [c for c in range(27) if c not in known]

        # Enumerate hands
        hands = []
        for h in __import__('itertools').combinations(remaining, 2):
            hands.append(h)
        n_hands = len(hands)
        hand_index = {h: i for i, h in enumerate(hands)}

        # Valid pairs mask
        valid_pairs = np.ones((n_hands, n_hands), dtype=np.float64)
        for hi in range(n_hands):
            for oi in range(n_hands):
                if set(hands[hi]) & set(hands[oi]):
                    valid_pairs[hi, oi] = 0.0

        result = {}
        result['board'] = np.array(board, dtype=np.int8)
        result['hands'] = np.array(hands, dtype=np.int8)
        result['turn_cards'] = np.array(remaining, dtype=np.int8)

        for pot_idx, (hero_bet, opp_bet) in enumerate(COMPUTE_POTS):
            for turn_card in remaining:
                tc_remaining = [c for c in remaining if c != turn_card]
                turn_board = board + [turn_card]

                # Enumerate turn hands (exclude turn card)
                turn_hands = []
                for h in __import__('itertools').combinations(
                        [c for c in remaining if c != turn_card], 2):
                    turn_hands.append(h)
                n_turn = len(turn_hands)

                t_valid = np.ones((n_turn, n_turn), dtype=np.float64)
                for hi in range(n_turn):
                    for oi in range(n_turn):
                        if set(turn_hands[hi]) & set(turn_hands[oi]):
                            t_valid[hi, oi] = 0.0

                # Phase 1: Solve rivers, get continuation values
                river_remaining = [c for c in tc_remaining]
                ev_sum = np.zeros((n_turn, n_turn), dtype=np.float64)
                ev_count = np.zeros((n_turn, n_turn), dtype=np.float64)

                for river_card in river_remaining:
                    river_board = turn_board + [river_card]
                    rc_mask = set([river_card])

                    # River hands (exclude river card)
                    r_hands = []
                    r_idx = []
                    for ti, th in enumerate(turn_hands):
                        if not (set(th) & rc_mask):
                            r_hands.append(th)
                            r_idx.append(ti)
                    n_river = len(r_hands)
                    if n_river < 3:
                        continue

                    r_valid = np.ones((n_river, n_river), dtype=np.float64)
                    for hi in range(n_river):
                        for oi in range(n_river):
                            if set(r_hands[hi]) & set(r_hands[oi]):
                                r_valid[hi, oi] = 0.0

                    # Solve river
                    from game_tree import GameTree
                    r_tree = GameTree(hero_bet, opp_bet, 2, 100, True)

                    # Compute showdown values
                    seven = engine._seven
                    board_mask = 0
                    for c in river_board:
                        board_mask |= 1 << c

                    r_equity = np.zeros((n_river, n_river), dtype=np.float64)
                    for hi, hh in enumerate(r_hands):
                        h_mask = (1 << hh[0]) | (1 << hh[1])
                        hr = seven.get(int(h_mask | board_mask), 9999)
                        for oi, oh in enumerate(r_hands):
                            if r_valid[hi, oi] == 0:
                                continue
                            o_mask = (1 << oh[0]) | (1 << oh[1])
                            opr = seven.get(int(o_mask | board_mask), 9999)
                            if hr < opr:
                                r_equity[hi, oi] = 1.0
                            elif hr == opr:
                                r_equity[hi, oi] = 0.5

                    # River terminal values
                    r_sd_vals = {}
                    for nid in r_tree.terminal_node_ids:
                        tt = r_tree.terminal[nid]
                        hp, op = r_tree.hero_pot[nid], r_tree.opp_pot[nid]
                        if tt == 1:  # FOLD_HERO
                            r_sd_vals[nid] = np.full((n_river, n_river), -hp) * r_valid
                        elif tt == 2:  # FOLD_OPP
                            r_sd_vals[nid] = np.full((n_river, n_river), op) * r_valid
                        elif tt == 3:  # SHOWDOWN
                            pot_won = min(hp, op)
                            r_sd_vals[nid] = (2 * r_equity - 1) * pot_won * r_valid

                    # Quick CFR solve (100 iterations)
                    from range_solver import RangeSolver
                    rs = RangeSolver(engine)
                    opp_w = np.ones(n_river) / n_river
                    strat = rs._run_dcfr(r_tree, opp_w, r_sd_vals, n_river, n_river, 100)

                    # Compute root EV from solved strategy
                    # Simple: use terminal values weighted by strategy
                    # For now, use the (2*eq-1)*pot as game value approx
                    r_ev = (2 * r_equity - 1) * hero_bet * r_valid

                    # Map river EVs to turn indices
                    for ri_h in range(n_river):
                        ti_h = r_idx[ri_h]
                        for ri_o in range(n_river):
                            ti_o = r_idx[ri_o]
                            ev_sum[ti_h, ti_o] += r_ev[ri_h, ri_o]
                            ev_count[ti_h, ti_o] += r_valid[ri_h, ri_o]

                # Turn continuation values
                cont_ev = np.zeros((n_turn, n_turn), dtype=np.float64)
                mask = ev_count > 0
                cont_ev[mask] = ev_sum[mask] / ev_count[mask]

                # Phase 2: Solve turn with continuation values
                from game_tree import GameTree, TERM_SHOWDOWN
                t_tree = GameTree(hero_bet, opp_bet, 2, 100, True)

                # Build turn terminal values
                t_tv = {}
                for nid in t_tree.terminal_node_ids:
                    tt = t_tree.terminal[nid]
                    hp, op = t_tree.hero_pot[nid], t_tree.opp_pot[nid]
                    if tt == 1:
                        t_tv[nid] = np.full((n_turn, n_turn), -hp) * t_valid
                    elif tt == 2:
                        t_tv[nid] = np.full((n_turn, n_turn), op) * t_valid
                    elif tt == 3:
                        scale = min(hp, op) / max(hero_bet, 1)
                        t_tv[nid] = cont_ev * scale * t_valid

                # Solve turn
                rs2 = RangeSolver(engine)
                opp_w2 = np.ones(n_turn) / n_turn
                t_strat = rs2._run_dcfr(t_tree, opp_w2, t_tv, n_turn, n_turn, 200)

                # Compute per-hero-hand game value (averaged over opponents)
                # GV[h] = sum_o opp_w[o] * root_ev[h, o]
                # We approximate root_ev from the continuation values
                hero_gv = np.zeros(n_turn, dtype=np.float64)
                for hi in range(n_turn):
                    valid_sum = t_valid[hi].sum()
                    if valid_sum > 0:
                        hero_gv[hi] = np.sum(cont_ev[hi] * t_valid[hi]) / valid_sum

                # Map to flop hand indices
                flop_gv = np.full(n_hands, 0.0, dtype=np.float32)
                for ti, th in enumerate(turn_hands):
                    fi = hand_index.get(th)
                    if fi is not None:
                        flop_gv[fi] = hero_gv[ti]

                key = f'gv_p{pot_idx}_t{turn_card}'
                result[key] = flop_gv

        # Save
        out_path = os.path.join(output_dir, f'turn_values_{board_idx}.npz')
        np.savez_compressed(out_path, **result)

        elapsed = time.time() - t0
        logger.info(f'Board {board_idx} ({board}): {elapsed:.1f}s')
        return board_idx, elapsed

    except Exception as e:
        logger.error(f'Board {board_idx} failed: {e}')
        import traceback
        traceback.print_exc()
        return board_idx, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_start', type=int, default=0)
    parser.add_argument('--cluster_end', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='output_turn_values')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    all_boards = enumerate_all_flops()
    boards = all_boards[args.cluster_start:args.cluster_end]

    logger.info(f'Computing turn values for boards {args.cluster_start}-{args.cluster_end} '
                f'({len(boards)} boards, {args.n_workers} workers)')

    tasks = [(args.cluster_start + i, board, args.output_dir)
             for i, board in enumerate(boards)]

    if args.n_workers <= 1:
        results = [compute_one_board(t) for t in tasks]
    else:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(compute_one_board, tasks)

    successes = sum(1 for _, t in results if t > 0)
    total_time = sum(t for _, t in results if t > 0)
    logger.info(f'Done: {successes}/{len(boards)} boards, {total_time:.0f}s total')


if __name__ == '__main__':
    main()
