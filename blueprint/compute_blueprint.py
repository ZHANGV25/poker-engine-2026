#!/usr/bin/env python3
"""
Offline blueprint strategy computation.

Precomputes Nash equilibrium strategies for all board clusters by running
full-range CFR+. The output is a compressed .npz file that the runtime
player can load for instant strategy lookup.

Usage:
    # Local testing (1 board, 100 iterations):
    python compute_blueprint.py --n_iterations 100 --n_boards 1 --n_workers 1

    # Full production run (all boards, 10k iterations):
    python compute_blueprint.py --n_iterations 10000 --n_workers 16 --output_dir /data/blueprint

    # Resume from checkpoint:
    python compute_blueprint.py --resume /data/blueprint/checkpoint_5000.npz
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from functools import partial

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
    enumerate_all_flops,
    compute_board_cluster,
    get_representative_boards,
    get_bucket_boundaries,
    compute_board_features,
)


# Default parameters
DEFAULT_N_BUCKETS = 50
DEFAULT_N_CLUSTERS = 200
DEFAULT_N_ITERATIONS = 10000
DEFAULT_N_WORKERS = 4
DEFAULT_CHECKPOINT_INTERVAL = 1000
DEFAULT_MAX_BET = 100
DEFAULT_MIN_RAISE = 2


def solve_single_board(args):
    """
    Solve one board cluster. Designed to run in a worker process.

    Args:
        args: tuple of (cluster_id, board, config_dict)

    Returns:
        dict with cluster results
    """
    cluster_id, board, config = args

    # Each worker creates its own equity engine (not picklable)
    engine = ExactEquityEngine()
    n_buckets = config['n_buckets']
    n_iterations = config['n_iterations']
    checkpoint_interval = config.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL)
    output_dir = config.get('output_dir', '.')

    solver = BlueprintCFR(n_buckets, n_buckets, engine)

    start = time.time()

    # Solve with hero acting first (typical post-flop scenario)
    # In practice, we'd solve both positions; for now, solve hero-first
    # and use symmetry for opp-first.
    result = solver.solve(
        board=list(board),
        dead_cards=[],  # No dead cards for blueprint (pre-discard abstraction)
        hero_bet=1,     # Post-blind: SB=1, BB=2 (simplification)
        opp_bet=2,
        hero_first=True,
        n_iterations=n_iterations,
        min_raise=DEFAULT_MIN_RAISE,
        max_bet=DEFAULT_MAX_BET,
        checkpoint_interval=checkpoint_interval,
    )

    elapsed = time.time() - start

    # Extract tree structure info for storage
    tree = result['tree']
    hero_node_actions = []
    for nid in tree.hero_node_ids:
        actions = [(act_id, child_id) for act_id, child_id in tree.children[nid]]
        hero_node_actions.append(actions)

    opp_node_actions = []
    for nid in tree.opp_node_ids:
        actions = [(act_id, child_id) for act_id, child_id in tree.children[nid]]
        opp_node_actions.append(actions)

    # Build action type array for each node
    hero_action_types = np.zeros(
        (len(tree.hero_node_ids), tree.num_actions[tree.hero_node_ids[0]]
         if tree.hero_node_ids else 0),
        dtype=np.int8) if tree.hero_node_ids else np.array([])

    if tree.hero_node_ids:
        max_act = max(tree.num_actions[nid] for nid in tree.hero_node_ids)
        hero_action_types = np.full(
            (len(tree.hero_node_ids), max_act), -1, dtype=np.int8)
        for i, nid in enumerate(tree.hero_node_ids):
            for a, (act_id, _) in enumerate(tree.children[nid]):
                hero_action_types[i, a] = act_id

    return {
        'cluster_id': cluster_id,
        'board': board,
        'hero_strategy': result['hero_strategy'],
        'opp_strategy': result['opp_strategy'],
        'hero_action_types': hero_action_types,
        'n_hero_nodes': len(tree.hero_node_ids),
        'n_opp_nodes': len(tree.opp_node_ids),
        'n_iterations': n_iterations,
        'elapsed': elapsed,
        'tree_size': tree.size,
    }


def save_checkpoint(results, output_dir, iteration_label, config):
    """Save intermediate results to a checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{iteration_label}.npz")

    # Collect all cluster data
    cluster_ids = []
    boards = []
    hero_strategies = []
    opp_strategies = []
    hero_action_type_list = []

    for r in results:
        cluster_ids.append(r['cluster_id'])
        boards.append(list(r['board']))
        hero_strategies.append(r['hero_strategy'])
        opp_strategies.append(r['opp_strategy'])
        if r['hero_action_types'] is not None and len(r['hero_action_types']) > 0:
            hero_action_type_list.append(r['hero_action_types'])

    np.savez_compressed(
        path,
        cluster_ids=np.array(cluster_ids),
        boards=np.array(boards),
        n_buckets=config['n_buckets'],
        n_clusters=config['n_clusters'],
        n_iterations=config['n_iterations'],
        bucket_boundaries=get_bucket_boundaries(config['n_buckets']),
    )

    print(f"  Checkpoint saved: {path}")
    return path


def save_final(results, output_dir, config):
    """
    Save the final blueprint strategies in the production format.

    Output format:
        blueprint_strategies.npz containing:
        - strategies: array of shape (n_clusters, n_buckets, max_action_seqs, max_actions)
        - cluster_ids: which clusters were solved
        - boards: representative board for each cluster
        - bucket_boundaries: equity boundaries
        - hero_action_types: action type IDs per node
        - config_*: configuration parameters
    """
    os.makedirs(output_dir, exist_ok=True)

    n_buckets = config['n_buckets']
    n_clusters = config['n_clusters']

    # Determine max dimensions across all solved clusters
    max_hero_nodes = max((r['n_hero_nodes'] for r in results), default=0)
    max_opp_nodes = max((r['n_opp_nodes'] for r in results), default=0)

    if max_hero_nodes == 0:
        print("WARNING: No hero decision nodes found. Blueprint may be empty.")
        max_hero_nodes = 1

    # Find max actions across all results
    max_actions = 0
    for r in results:
        if r['hero_strategy'] is not None and len(r['hero_strategy']) > 0:
            max_actions = max(max_actions, r['hero_strategy'].shape[-1])
    if max_actions == 0:
        max_actions = 1

    # Build the strategy tensor: (n_solved_clusters, n_buckets, max_hero_nodes, max_actions)
    n_solved = len(results)
    strategies = np.zeros(
        (n_solved, n_buckets, max_hero_nodes, max_actions), dtype=np.float32)

    cluster_ids = np.zeros(n_solved, dtype=np.int32)
    boards_arr = np.zeros((n_solved, 3), dtype=np.int8)  # flop = 3 cards

    # Also store action types so we know which action each column represents
    action_types = np.full(
        (n_solved, max_hero_nodes, max_actions), -1, dtype=np.int8)

    for i, r in enumerate(results):
        cluster_ids[i] = r['cluster_id']
        board = r['board']
        for j in range(min(3, len(board))):
            boards_arr[i, j] = board[j]

        hs = r['hero_strategy']
        if hs is not None and len(hs) > 0:
            nb, nn, na = hs.shape
            strategies[i, :nb, :nn, :na] = hs.astype(np.float32)

        hat = r['hero_action_types']
        if hat is not None and len(hat) > 0:
            nn2, na2 = hat.shape
            action_types[i, :nn2, :na2] = hat

    # Board feature vectors for cluster lookup at runtime
    board_features = np.zeros((n_solved, 12), dtype=np.float32)
    for i, r in enumerate(results):
        board_features[i] = compute_board_features(list(r['board']))

    path = os.path.join(output_dir, "blueprint_strategies.npz")
    np.savez_compressed(
        path,
        strategies=strategies,
        cluster_ids=cluster_ids,
        boards=boards_arr,
        board_features=board_features,
        action_types=action_types,
        bucket_boundaries=get_bucket_boundaries(n_buckets),
        config_n_buckets=n_buckets,
        config_n_clusters=n_clusters,
        config_n_iterations=config['n_iterations'],
        config_max_bet=DEFAULT_MAX_BET,
    )

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\nBlueprint saved: {path} ({size_mb:.1f} MB)")
    print(f"  {n_solved} clusters solved")
    print(f"  {n_buckets} buckets x {max_hero_nodes} nodes x {max_actions} actions")
    return path


def run_blueprint_computation(config):
    """
    Main computation entry point.

    Args:
        config: dict with computation parameters
    """
    n_buckets = config.get('n_buckets', DEFAULT_N_BUCKETS)
    n_clusters = config.get('n_clusters', DEFAULT_N_CLUSTERS)
    n_iterations = config.get('n_iterations', DEFAULT_N_ITERATIONS)
    n_workers = config.get('n_workers', DEFAULT_N_WORKERS)
    n_boards = config.get('n_boards', None)  # None = all boards
    output_dir = config.get('output_dir', os.path.join(_dir, 'output'))
    checkpoint_interval = config.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL)

    config['n_buckets'] = n_buckets
    config['n_clusters'] = n_clusters
    config['n_iterations'] = n_iterations
    config['checkpoint_interval'] = checkpoint_interval
    config['output_dir'] = output_dir

    print("=" * 60)
    print("Blueprint Strategy Computation")
    print("=" * 60)
    print(f"  Buckets:      {n_buckets}")
    print(f"  Clusters:     {n_clusters}")
    print(f"  Iterations:   {n_iterations}")
    print(f"  Workers:      {n_workers}")
    print(f"  Output:       {output_dir}")
    print()

    # Enumerate all flops and cluster them
    print("Enumerating boards...")
    all_flops = enumerate_all_flops()
    print(f"  Total flops: {len(all_flops)}")

    representatives = get_representative_boards(all_flops, n_clusters)
    print(f"  Board clusters: {len(representatives)}")

    if n_boards is not None:
        representatives = representatives[:n_boards]
        print(f"  Limiting to {n_boards} boards for this run")

    n_total = len(representatives)

    # Estimate runtime
    # Rough estimate: ~100s per board with 10k iterations, scales linearly
    est_per_board = (n_iterations / 10000) * 100  # seconds
    est_total = est_per_board * n_total / max(n_workers, 1)
    print(f"\nEstimated runtime: {est_total / 3600:.1f} hours "
          f"({est_per_board:.0f}s/board, {n_workers} workers)")
    print()

    # Prepare work items
    work_items = [
        (cid, board, config)
        for cid, board in representatives
    ]

    # Run computation
    results = []
    start_time = time.time()

    if n_workers <= 1:
        # Single-process mode (easier to debug)
        for i, item in enumerate(work_items):
            cid = item[0]
            board = item[1]
            print(f"[{i+1}/{n_total}] Solving cluster {cid} "
                  f"(board: {board})...", end="", flush=True)

            result = solve_single_board(item)
            results.append(result)

            elapsed = time.time() - start_time
            avg_per_board = elapsed / (i + 1)
            remaining = avg_per_board * (n_total - i - 1)

            print(f" done ({result['elapsed']:.1f}s, "
                  f"tree={result['tree_size']} nodes, "
                  f"ETA {remaining/60:.1f}m)")

            # Periodic checkpoint
            if (i + 1) % 10 == 0 and output_dir:
                save_checkpoint(results, output_dir, f"boards_{i+1}", config)
    else:
        # Multi-process mode
        print(f"Starting {n_workers} worker processes...")

        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(solve_single_board, work_items)):
                results.append(result)

                elapsed = time.time() - start_time
                avg_per_board = elapsed / (i + 1)
                remaining = avg_per_board * (n_total - i - 1)

                print(f"[{i+1}/{n_total}] Cluster {result['cluster_id']} done "
                      f"({result['elapsed']:.1f}s, "
                      f"tree={result['tree_size']} nodes, "
                      f"ETA {remaining/60:.1f}m)")

                # Periodic checkpoint
                if (i + 1) % max(10, n_total // 10) == 0 and output_dir:
                    save_checkpoint(results, output_dir, f"boards_{i+1}", config)

    total_time = time.time() - start_time
    print(f"\nComputation complete in {total_time/3600:.2f} hours")

    # Sort results by cluster_id for consistent output
    results.sort(key=lambda r: r['cluster_id'])

    # Save final output
    if output_dir:
        final_path = save_final(results, output_dir, config)
        return final_path

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute blueprint poker strategies offline")

    parser.add_argument('--n_iterations', type=int, default=DEFAULT_N_ITERATIONS,
                        help=f'CFR iterations per board (default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument('--n_buckets', type=int, default=DEFAULT_N_BUCKETS,
                        help=f'Number of equity buckets (default: {DEFAULT_N_BUCKETS})')
    parser.add_argument('--n_clusters', type=int, default=DEFAULT_N_CLUSTERS,
                        help=f'Number of board clusters (default: {DEFAULT_N_CLUSTERS})')
    parser.add_argument('--n_workers', type=int, default=DEFAULT_N_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_N_WORKERS})')
    parser.add_argument('--n_boards', type=int, default=None,
                        help='Limit to N boards (for testing)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_dir, 'output'),
                        help='Output directory for strategy files')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=DEFAULT_CHECKPOINT_INTERVAL,
                        help=f'Checkpoint every N iterations (default: {DEFAULT_CHECKPOINT_INTERVAL})')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file (not yet implemented)')

    args = parser.parse_args()

    if args.resume:
        print(f"Resume from {args.resume} not yet implemented.")
        print("Starting fresh computation instead.")

    config = {
        'n_iterations': args.n_iterations,
        'n_buckets': args.n_buckets,
        'n_clusters': args.n_clusters,
        'n_workers': args.n_workers,
        'n_boards': args.n_boards,
        'output_dir': args.output_dir,
        'checkpoint_interval': args.checkpoint_interval,
    }

    run_blueprint_computation(config)


if __name__ == '__main__':
    main()
