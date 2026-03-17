#!/usr/bin/env python3
"""Precompute river strategies for ALL 80,730 boards with simplified tree.
Split across instances via --start and --end arguments.
"""
import sys, os, time, itertools, pickle, lzma, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submission'))
sys.path.insert(0, os.path.dirname(__file__))

from equity import ExactEquityEngine
from game_tree import GameTree, TERM_SHOWDOWN
from multi_street_solver import (
    _flatten_tree, _build_valid_pairs, _enumerate_hands,
    _fill_river_showdown, _normalize_strategy_numba,
    _cfr_pernode_iterations, warmup_jit
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=80730)
    args = parser.parse_args()

    engine = ExactEquityEngine()
    warmup_jit()

    all_boards = list(itertools.combinations(range(27), 5))
    my_boards = all_boards[args.start:args.end]
    print(f'River boards: {len(my_boards)} (of {len(all_boards)}, range [{args.start}, {args.end}))')

    # Simplified tree: check/bet 65%, fold/call
    # Use small starting pot to get a compact tree
    tree = GameTree(20, 20, 2, 100, True)
    flat = _flatten_tree(tree)
    n_hero_nodes = flat['n_hero_nodes']
    n_opp_nodes = flat['n_opp_nodes']
    max_actions = flat['max_actions']
    n_nodes = flat['n_nodes']

    sd_ids = np.array([nid for nid in tree.terminal_node_ids
                       if tree.terminal[nid] == TERM_SHOWDOWN], dtype=np.int32)
    n_sd = len(sd_ids)
    pot_won_arr = np.array([min(tree.hero_pot[nid], tree.opp_pot[nid])
                            for nid in sd_ids], dtype=np.float64)

    act_types = np.full((n_hero_nodes, max_actions), -1, dtype=np.int8)
    for i, nid in enumerate(tree.hero_node_ids):
        for a, (act_id, _) in enumerate(tree.children[nid]):
            if a < max_actions:
                act_types[i, a] = act_id

    print(f'Tree: {n_nodes} nodes, {n_hero_nodes} hero, {n_sd} SD, {max_actions} max_act')

    n_total = len(my_boards)
    all_strategies = np.zeros((n_total, 231, n_hero_nodes, max_actions), dtype=np.uint8)
    all_hands = np.zeros((n_total, 231, 2), dtype=np.int8)
    all_n_hands = np.zeros(n_total, dtype=np.int16)

    t0 = time.time()
    for bi, board in enumerate(my_boards):
        board = list(board)
        hands = _enumerate_hands(board)
        valid = _build_valid_pairs(hands)
        n = len(hands)
        all_n_hands[bi] = n
        for hi, h in enumerate(hands):
            all_hands[bi, hi] = h

        board_mask = 0
        for c in board:
            board_mask |= 1 << c
        hand_ranks = np.zeros(n, dtype=np.int64)
        for i, hand in enumerate(hands):
            mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
            hand_ranks[i] = engine._seven[mask]

        sd_vals = np.zeros((n_sd, n, n), dtype=np.float64)
        _fill_river_showdown(sd_vals, hand_ranks, valid, pot_won_arr, n, n_sd)

        hero_strat, _, _ = _cfr_pernode_iterations(
            flat['node_player'], flat['node_terminal'],
            flat['node_hero_pot'], flat['node_opp_pot'],
            flat['node_num_actions'], flat['children'],
            flat['hero_node_map'], flat['opp_node_map'],
            sd_vals, valid, n, n,
            5, n_hero_nodes, n_opp_nodes,
            max_actions, n_nodes, sd_ids, n_sd)

        hero_norm = _normalize_strategy_numba(hero_strat, flat['hero_node_num_actions'], n)
        q = np.clip(np.round(hero_norm * 255.0), 0, 255).astype(np.uint8)
        all_strategies[bi, :n] = q

        if bi % 2000 == 0:
            elapsed = time.time() - t0
            rate = (bi + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - bi) / rate if rate > 0 else 0
            print(f'  {bi}/{n_total} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

    elapsed = time.time() - t0
    print(f'Done: {n_total} boards in {elapsed:.0f}s ({elapsed/n_total*1000:.1f}ms/board)')

    data = {
        'strategies': all_strategies,
        'hands': all_hands,
        'n_hands': all_n_hands,
        'action_types': act_types,
        'boards': np.array([list(b) for b in my_boards], dtype=np.int8),
        'start': args.start,
        'end': args.end,
    }

    out_path = f'/opt/blueprint/river_part_{args.start}_{args.end}.pkl.lzma'
    print(f'Compressing...', flush=True)
    compressed = lzma.compress(pickle.dumps(data), preset=3)
    with open(out_path, 'wb') as f:
        f.write(compressed)
    size = os.path.getsize(out_path) / 1024 / 1024
    print(f'Saved: {out_path} ({size:.0f} MB)')

    import subprocess
    subprocess.run(['aws', 's3', 'cp', out_path,
                   f's3://poker-blueprint-2026/river_parts/river_{args.start}_{args.end}.pkl.lzma',
                   '--region', 'us-east-1'])
    print('Uploaded to S3')

if __name__ == '__main__':
    main()
