#!/usr/bin/env python3
"""
Unbucketed blueprint strategy computation.

Computes Nash equilibrium strategies where every individual hand gets its own
strategy (no equity bucketing). Designed for distributed computation across
multiple instances, with --cluster_start/--cluster_end to split work.

Output format:
    Per-cluster .npz files with strategies of shape:
        (n_pot_sizes, n_hands, n_hero_nodes, n_actions) as uint8

    Strategies are quantized: float probability * 255 -> uint8 to save space.
    To recover probabilities: strategy.astype(float) / 255.0

Usage:
    # Quick local test (1 cluster, 100 iterations):
    python compute_unbucketed.py --street flop --n_clusters 200 --n_iterations 100 \\
        --n_workers 1 --cluster_start 0 --cluster_end 1

    # Full flop run on one instance (clusters 0-99):
    python compute_unbucketed.py --street flop --n_clusters 200 --n_iterations 1000 \\
        --n_workers 8 --cluster_start 0 --cluster_end 100

    # Second instance (clusters 100-199):
    python compute_unbucketed.py --street flop --n_clusters 200 --n_iterations 1000 \\
        --n_workers 8 --cluster_start 100 --cluster_end 200

    # Merge results from multiple instances:
    python merge_results.py --input_dir s3://bucket/results/ --output_dir ./merged/
"""

import os
import sys
import time
import argparse
import logging
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
from abstraction import (
    enumerate_all_flops,
    enumerate_all_turns,
    enumerate_all_rivers,
    get_representative_boards,
    compute_board_features,
)

import blueprint_cfr_unbucketed as unbucketed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_N_CLUSTERS = 200
DEFAULT_N_ITERATIONS = 1000
DEFAULT_N_WORKERS = 4
DEFAULT_MAX_BET = 100
DEFAULT_MIN_RAISE = 2

POT_SIZES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (30, 30),
    (50, 50),
    (100, 100),
]

STREET_ENUMERATORS = {
    'flop': enumerate_all_flops,
    'turn': enumerate_all_turns,
    'river': enumerate_all_rivers,
}

STREET_DEFAULT_CLUSTERS = {
    'flop': 200,
    'turn': 250,
    'river': 300,
}


# ---------------------------------------------------------------------------
# Worker function (designed to be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def solve_single_cluster(args):
    """
    Solve one board cluster at multiple pot sizes using unbucketed CFR.

    Args:
        args: tuple of (cluster_id, board, config_dict)

    Returns:
        dict with cluster results
    """
    cluster_id, board, config = args
    board = list(board)
    n_iterations = config['n_iterations']
    pot_sizes = config.get('pot_sizes', POT_SIZES)
    output_dir = config.get('output_dir')

    # Check if this cluster was already solved (resume support)
    if output_dir:
        out_path = os.path.join(output_dir, f"cluster_{cluster_id}.npz")
        if os.path.exists(out_path):
            try:
                existing = np.load(out_path, allow_pickle=True)
                # Verify it has the expected number of pot sizes
                if existing['hero_strategies'].shape[0] == len(pot_sizes):
                    logger.info("Cluster %d already solved, skipping", cluster_id)
                    return {
                        'cluster_id': cluster_id,
                        'board': board,
                        'elapsed': 0.0,
                        'skipped': True,
                    }
            except Exception:
                pass  # Corrupt file, re-solve

    # Each worker creates its own equity engine (not picklable)
    engine = ExactEquityEngine()

    start = time.time()

    # Solve for each pot size
    pot_results = []
    hero_hands = None

    for pot_idx, (hero_bet, opp_bet) in enumerate(pot_sizes):
        t0 = time.time()

        result = unbucketed.solve(
            board=board,
            dead_cards=[],
            hero_bet=hero_bet,
            opp_bet=opp_bet,
            hero_first=True,
            n_iterations=n_iterations,
            min_raise=DEFAULT_MIN_RAISE,
            max_bet=DEFAULT_MAX_BET,
            equity_engine=engine,
        )

        t_solve = time.time() - t0

        if hero_hands is None:
            hero_hands = result['hero_hands']

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

        pot_results.append({
            'hero_strategy': result['hero_strategy'],
            'opp_strategy': result['opp_strategy'],
            'hero_action_types': hero_action_types,
            'n_hero_nodes': len(tree.hero_node_ids),
            'n_opp_nodes': len(tree.opp_node_ids),
            'tree_size': tree.size,
            'solve_time': t_solve,
        })

    elapsed = time.time() - start

    # Save per-cluster result immediately (enables resumption)
    if output_dir:
        _save_cluster_result(cluster_id, board, hero_hands, pot_results,
                             pot_sizes, n_iterations, output_dir)

    return {
        'cluster_id': cluster_id,
        'board': board,
        'hero_hands': hero_hands,
        'pot_results': pot_results,
        'elapsed': elapsed,
        'skipped': False,
    }


def _save_cluster_result(cluster_id, board, hero_hands, pot_results,
                         pot_sizes, n_iterations, output_dir):
    """Save a single cluster's results as a .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}.npz")

    n_pot_sizes = len(pot_sizes)

    # Determine max dimensions
    max_hero_nodes = 0
    max_opp_nodes = 0
    max_actions = 0
    for pr in pot_results:
        max_hero_nodes = max(max_hero_nodes, pr['n_hero_nodes'])
        max_opp_nodes = max(max_opp_nodes, pr['n_opp_nodes'])
        hs = pr['hero_strategy']
        if hs is not None and len(hs) > 0:
            max_actions = max(max_actions, hs.shape[-1])

    if max_hero_nodes == 0:
        max_hero_nodes = 1
    if max_actions == 0:
        max_actions = 1

    n_hands = len(hero_hands)

    # Shape: (n_pot_sizes, n_hands, max_hero_nodes, max_actions)
    hero_strategies = np.zeros(
        (n_pot_sizes, n_hands, max_hero_nodes, max_actions), dtype=np.uint8)
    opp_strategies = np.zeros(
        (n_pot_sizes, n_hands, max_opp_nodes, max_actions), dtype=np.uint8)

    # Action types: (n_pot_sizes, max_hero_nodes, max_actions)
    action_types = np.full(
        (n_pot_sizes, max_hero_nodes, max_actions), -1, dtype=np.int8)

    for pot_idx, pr in enumerate(pot_results):
        hs = pr['hero_strategy']
        if hs is not None and len(hs) > 0:
            nh, nn, na = hs.shape
            # Quantize to uint8: multiply by 255, round, clip
            quantized = np.clip(np.round(hs * 255.0), 0, 255).astype(np.uint8)
            hero_strategies[pot_idx, :nh, :nn, :na] = quantized

        os_arr = pr['opp_strategy']
        if os_arr is not None and len(os_arr) > 0:
            no, nn2, na2 = os_arr.shape
            quantized_opp = np.clip(np.round(os_arr * 255.0), 0, 255).astype(np.uint8)
            opp_strategies[pot_idx, :no, :nn2, :na2] = quantized_opp

        hat = pr['hero_action_types']
        if hat is not None and len(hat) > 0:
            nn3, na3 = hat.shape
            action_types[pot_idx, :nn3, :na3] = hat

    # Convert hand list to array for storage
    hands_arr = np.array(hero_hands, dtype=np.int8)

    np.savez_compressed(
        out_path,
        hero_strategies=hero_strategies,
        opp_strategies=opp_strategies,
        action_types=action_types,
        hands=hands_arr,
        board=np.array(board, dtype=np.int8),
        pot_sizes=np.array(pot_sizes, dtype=np.int32),
        cluster_id=cluster_id,
        n_iterations=n_iterations,
        board_features=compute_board_features(board),
    )

    size_kb = os.path.getsize(out_path) / 1024
    logger.debug("Saved cluster %d: %s (%.1f KB)", cluster_id, out_path, size_kb)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_unbucketed_computation(config):
    """
    Main computation entry point.

    Args:
        config: dict with computation parameters
    """
    street = config['street']
    n_clusters = config.get('n_clusters', STREET_DEFAULT_CLUSTERS.get(street, DEFAULT_N_CLUSTERS))
    n_iterations = config.get('n_iterations', DEFAULT_N_ITERATIONS)
    n_workers = config.get('n_workers', DEFAULT_N_WORKERS)
    output_dir = config.get('output_dir', os.path.join(_dir, 'output_unbucketed'))
    pot_sizes = config.get('pot_sizes', POT_SIZES)
    cluster_start = config.get('cluster_start', 0)
    cluster_end = config.get('cluster_end', None)

    config['n_clusters'] = n_clusters
    config['n_iterations'] = n_iterations
    config['pot_sizes'] = pot_sizes

    print("=" * 60)
    print(f"Unbucketed Blueprint Computation ({street.upper()})")
    print("=" * 60)
    print(f"  Street:        {street}")
    print(f"  Clusters:      {n_clusters}")
    print(f"  Iterations:    {n_iterations}")
    print(f"  Pot sizes:     {len(pot_sizes)} {pot_sizes}")
    print(f"  Workers:       {n_workers}")
    print(f"  Cluster range: [{cluster_start}, {cluster_end})")
    print(f"  Output:        {output_dir}")
    print()

    # Enumerate boards and cluster them
    if street not in STREET_ENUMERATORS:
        raise ValueError(f"Unknown street: {street}. Must be one of {list(STREET_ENUMERATORS.keys())}")

    print(f"Enumerating {street} boards...")
    t0 = time.time()
    all_boards = STREET_ENUMERATORS[street]()
    print(f"  Total {street} boards: {len(all_boards)} ({time.time()-t0:.1f}s)")

    if config.get('all_boards'):
        # No clustering — every board is its own "cluster"
        representatives = [(i, board) for i, board in enumerate(all_boards)]
        print(f"  All boards mode: {len(representatives)} individual boards (no clustering)")
    else:
        print(f"Clustering into {n_clusters} clusters...")
        t0 = time.time()
        representatives = get_representative_boards(all_boards, n_clusters)
        print(f"  Board clusters: {len(representatives)} ({time.time()-t0:.1f}s)")

    # Apply cluster range filter
    if cluster_end is None:
        cluster_end = len(representatives)
    else:
        cluster_end = min(cluster_end, len(representatives))

    # Filter to the requested range
    filtered = [(cid, board) for cid, board in representatives
                if cluster_start <= cid < cluster_end]

    # Also support index-based if cluster IDs don't align
    if not filtered:
        filtered = representatives[cluster_start:cluster_end]

    n_total = len(filtered)
    if n_total == 0:
        print("No clusters to solve in the specified range.")
        return

    print(f"  Solving {n_total} clusters (range [{cluster_start}, {cluster_end}))")

    # Warmup JIT
    print("\nWarming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    unbucketed.warmup_jit()
    print(f" done ({time.time() - t0:.2f}s)")

    # Prepare work items
    work_config = {
        'n_iterations': n_iterations,
        'pot_sizes': pot_sizes,
        'output_dir': output_dir,
    }
    work_items = [(cid, board, work_config) for cid, board in filtered]

    # Run computation
    results = []
    start_time = time.time()
    n_skipped = 0

    if n_workers <= 1:
        for i, item in enumerate(work_items):
            cid = item[0]
            board = item[1]
            print(f"[{i+1}/{n_total}] Solving {street} cluster {cid} "
                  f"(board: {board})...", end="", flush=True)

            result = solve_single_cluster(item)
            results.append(result)

            if result.get('skipped'):
                n_skipped += 1
                print(" skipped (already solved)")
                continue

            elapsed = time.time() - start_time
            n_done = i + 1 - n_skipped
            if n_done > 0:
                avg_per_board = elapsed / n_done
                remaining = avg_per_board * (n_total - i - 1)
            else:
                remaining = 0

            # Per-pot timing
            pot_times = [pr['solve_time'] for pr in result.get('pot_results', [])]
            pot_time_str = ", ".join(f"{t:.1f}s" for t in pot_times)

            print(f" done ({result['elapsed']:.1f}s [{pot_time_str}], "
                  f"ETA {remaining/60:.1f}m)")
    else:
        print(f"\nStarting {n_workers} worker processes...")

        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(solve_single_cluster, work_items)
            ):
                results.append(result)

                if result.get('skipped'):
                    n_skipped += 1

                elapsed = time.time() - start_time
                n_done = i + 1 - n_skipped
                if n_done > 0:
                    avg_per_board = elapsed / n_done
                    remaining = avg_per_board * (n_total - i - 1)
                else:
                    remaining = 0

                status = "skipped" if result.get('skipped') else f"{result['elapsed']:.1f}s"
                print(f"[{i+1}/{n_total}] Cluster {result['cluster_id']} {status} "
                      f"(ETA {remaining/60:.1f}m)")

    total_time = time.time() - start_time
    n_solved = n_total - n_skipped

    print(f"\nComputation complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Solved: {n_solved}, Skipped (resumed): {n_skipped}")

    # Build combined output file
    if output_dir:
        _save_combined(results, output_dir, config, street)


def _save_combined(results, output_dir, config, street):
    """
    Save a combined index file that references all per-cluster results.

    Also creates a single merged file with all strategies for fast loading.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect metadata from all non-skipped results
    solved_results = [r for r in results if not r.get('skipped')]

    if not solved_results:
        print("No new results to save.")
        return

    # For skipped results, load their data from disk
    all_cluster_ids = []
    all_boards = []

    for r in results:
        all_cluster_ids.append(r['cluster_id'])
        all_boards.append(list(r['board']))

    # Save index file
    index_path = os.path.join(output_dir, f"{street}_index.npz")
    np.savez_compressed(
        index_path,
        cluster_ids=np.array(sorted(all_cluster_ids), dtype=np.int32),
        n_clusters=config['n_clusters'],
        n_iterations=config['n_iterations'],
        pot_sizes=np.array(config['pot_sizes'], dtype=np.int32),
        street=street,
        cluster_start=config.get('cluster_start', 0),
        cluster_end=config.get('cluster_end', len(results)),
    )

    print(f"\nIndex saved: {index_path}")
    print(f"  {len(all_cluster_ids)} clusters indexed")
    print(f"  Per-cluster files in: {output_dir}/cluster_*.npz")

    # Print total disk usage
    total_size = 0
    for f in os.listdir(output_dir):
        if f.startswith('cluster_') and f.endswith('.npz'):
            total_size += os.path.getsize(os.path.join(output_dir, f))
    print(f"  Total cluster data: {total_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute unbucketed blueprint strategies (per-hand, no bucketing)")

    parser.add_argument(
        '--street', type=str, required=True,
        choices=['flop', 'turn', 'river'],
        help='Which street to compute')
    parser.add_argument(
        '--n_clusters', type=int, default=None,
        help='Number of board clusters (default: street-specific)')
    parser.add_argument(
        '--n_iterations', type=int, default=DEFAULT_N_ITERATIONS,
        help=f'CFR iterations per board per pot size (default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument(
        '--n_workers', type=int, default=DEFAULT_N_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_N_WORKERS})')
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(_dir, 'output_unbucketed'),
        help='Output directory for strategy files')
    parser.add_argument(
        '--cluster_start', type=int, default=0,
        help='Start cluster index (inclusive, for splitting across instances)')
    parser.add_argument(
        '--cluster_end', type=int, default=None,
        help='End cluster index (exclusive, for splitting across instances)')
    parser.add_argument(
        '--all_boards', action='store_true',
        help='Solve every board individually (no clustering). Each board is its own cluster.')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    config = {
        'street': args.street,
        'n_iterations': args.n_iterations,
        'n_workers': args.n_workers,
        'output_dir': args.output_dir,
        'cluster_start': args.cluster_start,
        'cluster_end': args.cluster_end,
        'pot_sizes': POT_SIZES,
    }

    if args.all_boards:
        config['all_boards'] = True

    if args.n_clusters is not None:
        config['n_clusters'] = args.n_clusters
    else:
        config['n_clusters'] = STREET_DEFAULT_CLUSTERS.get(
            args.street, DEFAULT_N_CLUSTERS)

    run_unbucketed_computation(config)


if __name__ == '__main__':
    main()
