#!/usr/bin/env python3
"""
Offline blueprint strategy computation for turn (4-card) and river (5-card) boards.

Extends the flop blueprint (compute_blueprint.py) to later streets. River boards
have deterministic equity (no runouts), making them very fast to solve. Turn boards
need only 1 remaining card (~13-17 runouts per matchup), which is also fast.

For each street, boards are clustered by structural features and one representative
board per cluster is solved via full-range CFR+. Multiple pot sizes are solved per
cluster to handle different bet-sizing scenarios.

Usage:
    # Quick test (river only, 1 cluster, 100 iterations):
    python compute_turn_river.py --street river --n_clusters 5 --n_boards 1 \
        --n_iterations 100 --n_workers 1

    # Full river production run (~300 clusters, 2000 iterations):
    python compute_turn_river.py --street river --n_clusters 300 --n_iterations 2000 \
        --n_workers 8

    # Full turn production run (~250 clusters, 2000 iterations):
    python compute_turn_river.py --street turn --n_clusters 250 --n_iterations 2000 \
        --n_workers 8

    # Both streets:
    python compute_turn_river.py --street both --n_workers 8
"""

import os
import sys
import time
import argparse
import multiprocessing as mp

import numpy as np

# Add paths
_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from equity import ExactEquityEngine
from blueprint_cfr import BlueprintCFR
from abstraction import (
    enumerate_all_turns,
    enumerate_all_rivers,
    compute_board_cluster,
    get_representative_boards,
    get_bucket_boundaries,
    compute_board_features,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_N_BUCKETS = 20
DEFAULT_N_ITERATIONS = 2000
DEFAULT_N_WORKERS = 4
DEFAULT_MAX_BET = 100
DEFAULT_MIN_RAISE = 2

DEFAULT_RIVER_CLUSTERS = 300
DEFAULT_TURN_CLUSTERS = 250

# Multiple pot sizes to solve for each board cluster.
# Each entry is (hero_bet, opp_bet) representing the starting pot state.
POT_SIZES = [
    (2, 2),     # SPR 24.5
    (4, 4),     # SPR 12.0
    (10, 10),   # SPR 4.5
    (25, 25),   # SPR 1.5
    (50, 50),   # SPR 0.5
    (75, 75),   # SPR 0.17
    (100, 100), # SPR 0.0
]


# ---------------------------------------------------------------------------
# Worker functions (designed to be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def solve_single_board_multi_pot(args):
    """
    Solve one board cluster at multiple pot sizes.

    Args:
        args: tuple of (cluster_id, board, config_dict)

    Returns:
        dict with cluster results including strategies for each pot size
    """
    cluster_id, board, config = args

    # Each worker creates its own equity engine (not picklable)
    engine = ExactEquityEngine()
    n_buckets = config['n_buckets']
    n_iterations = config['n_iterations']
    pot_sizes = config.get('pot_sizes', POT_SIZES)

    solver = BlueprintCFR(n_buckets, n_buckets, engine)

    start = time.time()

    pot_results = {}

    for pot_idx, (hero_bet, opp_bet) in enumerate(pot_sizes):
        result = solver.solve(
            board=list(board),
            dead_cards=[],
            hero_bet=hero_bet,
            opp_bet=opp_bet,
            hero_first=True,
            n_iterations=n_iterations,
            min_raise=DEFAULT_MIN_RAISE,
            max_bet=DEFAULT_MAX_BET,
        )

        tree = result['tree']

        # Extract action types for hero decision nodes
        hero_action_types = np.array([], dtype=np.int8)
        if tree.hero_node_ids:
            max_act = max(tree.num_actions[nid] for nid in tree.hero_node_ids)
            hero_action_types = np.full(
                (len(tree.hero_node_ids), max_act), -1, dtype=np.int8)
            for i, nid in enumerate(tree.hero_node_ids):
                for a, (act_id, _) in enumerate(tree.children[nid]):
                    hero_action_types[i, a] = act_id

        pot_results[pot_idx] = {
            'hero_strategy': result['hero_strategy'],
            'opp_strategy': result['opp_strategy'],
            'hero_action_types': hero_action_types,
            'n_hero_nodes': len(tree.hero_node_ids),
            'n_opp_nodes': len(tree.opp_node_ids),
            'tree_size': tree.size,
        }

    elapsed = time.time() - start

    return {
        'cluster_id': cluster_id,
        'board': board,
        'pot_results': pot_results,
        'n_iterations': n_iterations,
        'elapsed': elapsed,
    }


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_blueprint(results, output_dir, config, street_name, filename):
    """
    Save the blueprint strategies for a street in the standard .npz format.

    The output format is compatible with the flop blueprint (blueprint_strategies.npz)
    with additional dimensions for pot-size indexing.

    Output arrays:
        strategies:        (n_solved, n_pot_sizes, n_buckets, max_hero_nodes, max_actions)
        cluster_ids:       (n_solved,)
        boards:            (n_solved, n_board_cards)
        board_features:    (n_solved, 12)
        action_types:      (n_solved, n_pot_sizes, max_hero_nodes, max_actions)
        bucket_boundaries: (n_buckets+1,)
        pot_sizes:         (n_pot_sizes, 2)
        config_*:          scalar config values
    """
    os.makedirs(output_dir, exist_ok=True)

    n_buckets = config['n_buckets']
    n_clusters = config['n_clusters']
    pot_sizes = config.get('pot_sizes', POT_SIZES)
    n_pot_sizes = len(pot_sizes)
    n_board_cards = len(results[0]['board'])

    n_solved = len(results)

    # Determine max dimensions across all solved clusters and pot sizes
    max_hero_nodes = 0
    max_actions = 0
    for r in results:
        for pot_idx in r['pot_results']:
            pr = r['pot_results'][pot_idx]
            max_hero_nodes = max(max_hero_nodes, pr['n_hero_nodes'])
            hs = pr['hero_strategy']
            if hs is not None and len(hs) > 0:
                max_actions = max(max_actions, hs.shape[-1])

    if max_hero_nodes == 0:
        max_hero_nodes = 1
    if max_actions == 0:
        max_actions = 1

    # Build tensors
    strategies = np.zeros(
        (n_solved, n_pot_sizes, n_buckets, max_hero_nodes, max_actions),
        dtype=np.float32,
    )
    cluster_ids = np.zeros(n_solved, dtype=np.int32)
    boards_arr = np.zeros((n_solved, n_board_cards), dtype=np.int8)
    action_types = np.full(
        (n_solved, n_pot_sizes, max_hero_nodes, max_actions), -1, dtype=np.int8)

    for i, r in enumerate(results):
        cluster_ids[i] = r['cluster_id']
        board = r['board']
        for j in range(n_board_cards):
            boards_arr[i, j] = board[j]

        for pot_idx in r['pot_results']:
            pr = r['pot_results'][pot_idx]
            hs = pr['hero_strategy']
            if hs is not None and len(hs) > 0:
                nb, nn, na = hs.shape
                strategies[i, pot_idx, :nb, :nn, :na] = hs.astype(np.float32)

            hat = pr['hero_action_types']
            if hat is not None and len(hat) > 0:
                nn2, na2 = hat.shape
                action_types[i, pot_idx, :nn2, :na2] = hat

    # Board feature vectors for cluster lookup at runtime
    board_features = np.zeros((n_solved, 12), dtype=np.float32)
    for i, r in enumerate(results):
        board_features[i] = compute_board_features(list(r['board']))

    path = os.path.join(output_dir, filename)
    np.savez_compressed(
        path,
        strategies=strategies,
        cluster_ids=cluster_ids,
        boards=boards_arr,
        board_features=board_features,
        action_types=action_types,
        bucket_boundaries=get_bucket_boundaries(n_buckets),
        pot_sizes=np.array(pot_sizes, dtype=np.int32),
        config_n_buckets=n_buckets,
        config_n_clusters=n_clusters,
        config_n_iterations=config['n_iterations'],
        config_max_bet=DEFAULT_MAX_BET,
        config_street=street_name,
    )

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n{street_name.capitalize()} blueprint saved: {path} ({size_mb:.1f} MB)")
    print(f"  {n_solved} clusters solved")
    print(f"  {n_pot_sizes} pot sizes x {n_buckets} buckets x "
          f"{max_hero_nodes} nodes x {max_actions} actions")
    return path


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_street_computation(street_name, config):
    """
    Run blueprint computation for a single street (turn or river).

    Args:
        street_name: 'turn' or 'river'
        config: dict with computation parameters

    Returns:
        path to saved blueprint file
    """
    n_buckets = config.get('n_buckets', DEFAULT_N_BUCKETS)
    n_iterations = config.get('n_iterations', DEFAULT_N_ITERATIONS)
    n_workers = config.get('n_workers', DEFAULT_N_WORKERS)
    n_boards = config.get('n_boards', None)
    output_dir = config.get('output_dir', os.path.join(_dir, 'output'))
    pot_sizes = config.get('pot_sizes', POT_SIZES)

    if street_name == 'river':
        n_clusters = config.get('n_clusters', DEFAULT_RIVER_CLUSTERS)
        enumerate_fn = enumerate_all_rivers
        filename = 'river_blueprint.npz'
        total_label = 'C(27,5) = 80,730'
    elif street_name == 'turn':
        n_clusters = config.get('n_clusters', DEFAULT_TURN_CLUSTERS)
        enumerate_fn = enumerate_all_turns
        filename = 'turn_blueprint.npz'
        total_label = 'C(27,4) = 17,550'
    else:
        raise ValueError(f"Unknown street: {street_name}")

    # Update config for downstream use
    config_local = dict(config)
    config_local['n_buckets'] = n_buckets
    config_local['n_clusters'] = n_clusters
    config_local['n_iterations'] = n_iterations
    config_local['pot_sizes'] = pot_sizes

    print("=" * 60)
    print(f"{street_name.upper()} Blueprint Computation")
    print("=" * 60)
    print(f"  Total boards:  {total_label}")
    print(f"  Clusters:      {n_clusters}")
    print(f"  Buckets:       {n_buckets}")
    print(f"  Iterations:    {n_iterations}")
    print(f"  Pot sizes:     {len(pot_sizes)} ({pot_sizes})")
    print(f"  Workers:       {n_workers}")
    print(f"  Output:        {output_dir}")
    print()

    # Enumerate all boards and cluster them
    print(f"Enumerating {street_name} boards...")
    t0 = time.time()
    all_boards = enumerate_fn()
    print(f"  Total {street_name} boards: {len(all_boards)} ({time.time()-t0:.1f}s)")

    print(f"Clustering into {n_clusters} clusters...")
    t0 = time.time()
    representatives = get_representative_boards(all_boards, n_clusters)
    print(f"  Board clusters: {len(representatives)} ({time.time()-t0:.1f}s)")

    if n_boards is not None:
        representatives = representatives[:n_boards]
        print(f"  Limiting to {n_boards} boards for this run")

    n_total = len(representatives)

    # Estimate runtime
    # River is ~instant per board (deterministic equity), turn has ~15 runouts
    if street_name == 'river':
        est_per_board = (n_iterations / 2000) * 0.5 * len(pot_sizes)
    else:
        est_per_board = (n_iterations / 2000) * 2.0 * len(pot_sizes)
    est_total = est_per_board * n_total / max(n_workers, 1)
    print(f"\nEstimated runtime: {est_total / 60:.1f} min "
          f"({est_per_board:.1f}s/board, {n_workers} workers)")
    print()

    # Prepare work items
    work_items = [
        (cid, board, config_local)
        for cid, board in representatives
    ]

    # Run computation
    results = []
    start_time = time.time()

    if n_workers <= 1:
        # Single-process mode
        for i, item in enumerate(work_items):
            cid = item[0]
            board = item[1]
            print(f"[{i+1}/{n_total}] Solving {street_name} cluster {cid} "
                  f"(board: {board})...", end="", flush=True)

            result = solve_single_board_multi_pot(item)
            results.append(result)

            elapsed = time.time() - start_time
            avg_per_board = elapsed / (i + 1)
            remaining = avg_per_board * (n_total - i - 1)

            # Summarize tree sizes across pot sizes
            tree_sizes = [
                result['pot_results'][p]['tree_size']
                for p in result['pot_results']
            ]
            avg_tree = sum(tree_sizes) / len(tree_sizes)

            print(f" done ({result['elapsed']:.1f}s, "
                  f"avg_tree={avg_tree:.0f}, "
                  f"ETA {remaining/60:.1f}m)")
    else:
        # Multi-process mode
        print(f"Starting {n_workers} worker processes...")

        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(solve_single_board_multi_pot, work_items)
            ):
                results.append(result)

                elapsed = time.time() - start_time
                avg_per_board = elapsed / (i + 1)
                remaining = avg_per_board * (n_total - i - 1)

                print(f"[{i+1}/{n_total}] Cluster {result['cluster_id']} done "
                      f"({result['elapsed']:.1f}s, "
                      f"ETA {remaining/60:.1f}m)")

    total_time = time.time() - start_time
    print(f"\n{street_name.capitalize()} computation complete in "
          f"{total_time:.1f}s ({total_time/60:.1f} min)")

    # Sort by cluster_id for consistent output
    results.sort(key=lambda r: r['cluster_id'])

    # Save
    path = save_blueprint(results, output_dir, config_local, street_name, filename)
    return path


def verify_blueprint(path, street_name):
    """
    Quick sanity check on a saved blueprint file.

    Loads the file and prints summary statistics.
    """
    print(f"\nVerifying {street_name} blueprint: {path}")
    data = np.load(path, allow_pickle=True)

    strategies = data['strategies']
    cluster_ids = data['cluster_ids']
    boards = data['boards']
    pot_sizes = data['pot_sizes']
    n_buckets = int(data['config_n_buckets'])
    n_clusters = int(data['config_n_clusters'])
    n_iterations = int(data['config_n_iterations'])
    street = str(data['config_street'])

    print(f"  Street:      {street}")
    print(f"  Shape:       {strategies.shape}")
    print(f"  Clusters:    {len(cluster_ids)} solved / {n_clusters} target")
    print(f"  Buckets:     {n_buckets}")
    print(f"  Iterations:  {n_iterations}")
    print(f"  Pot sizes:   {pot_sizes.tolist()}")
    print(f"  Board cards: {boards.shape[1]}")

    # Check strategy normalization
    # For the first cluster, first pot size, check that strategies sum to ~1
    if len(cluster_ids) > 0:
        s = strategies[0, 0]  # (n_buckets, n_nodes, n_actions)
        # Find a bucket with nonzero strategy
        for b in range(n_buckets):
            row_sum = s[b, 0].sum()
            if row_sum > 0.01:
                print(f"  Sample strategy (cluster={cluster_ids[0]}, "
                      f"pot={pot_sizes[0].tolist()}, bucket={b}, node=0): "
                      f"sum={row_sum:.4f}")
                print(f"    probs = {s[b, 0]}")
                break

    # Check for NaN/Inf
    if np.any(np.isnan(strategies)):
        print("  WARNING: NaN values in strategies!")
    if np.any(np.isinf(strategies)):
        print("  WARNING: Inf values in strategies!")

    n_nonzero = np.count_nonzero(strategies)
    n_total_cells = strategies.size
    print(f"  Nonzero:     {n_nonzero}/{n_total_cells} "
          f"({100*n_nonzero/n_total_cells:.1f}%)")

    print("  OK")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute turn/river blueprint strategies offline")

    parser.add_argument(
        '--street', type=str, default='both',
        choices=['turn', 'river', 'both'],
        help='Which street(s) to compute (default: both)')
    parser.add_argument(
        '--n_iterations', type=int, default=DEFAULT_N_ITERATIONS,
        help=f'CFR iterations per board per pot size (default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument(
        '--n_buckets', type=int, default=DEFAULT_N_BUCKETS,
        help=f'Number of equity buckets (default: {DEFAULT_N_BUCKETS})')
    parser.add_argument(
        '--n_clusters', type=int, default=None,
        help='Number of board clusters (default: 300 river, 250 turn)')
    parser.add_argument(
        '--n_workers', type=int, default=DEFAULT_N_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_N_WORKERS})')
    parser.add_argument(
        '--n_boards', type=int, default=None,
        help='Limit to N boards per street (for testing)')
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(_dir, 'output'),
        help='Output directory for strategy files')
    parser.add_argument(
        '--verify', action='store_true', default=True,
        help='Verify output files after computation (default: True)')
    parser.add_argument(
        '--no-verify', action='store_false', dest='verify',
        help='Skip verification')

    args = parser.parse_args()

    streets = []
    if args.street in ('river', 'both'):
        streets.append('river')
    if args.street in ('turn', 'both'):
        streets.append('turn')

    for street_name in streets:
        config = {
            'n_iterations': args.n_iterations,
            'n_buckets': args.n_buckets,
            'n_workers': args.n_workers,
            'n_boards': args.n_boards,
            'output_dir': args.output_dir,
            'pot_sizes': POT_SIZES,
        }

        # Set street-specific cluster count
        if args.n_clusters is not None:
            config['n_clusters'] = args.n_clusters
        # else: run_street_computation uses the per-street default

        path = run_street_computation(street_name, config)

        if args.verify and path:
            verify_blueprint(path, street_name)

        print()


if __name__ == '__main__':
    main()
