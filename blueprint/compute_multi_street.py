#!/usr/bin/env python3
"""
Multi-street blueprint strategy computation.

Uses backward induction (river -> turn -> flop) to compute strategies that
account for optimal play on future streets, unlike the single-street solver
which uses raw equity at showdown terminals.

Supports:
    --all_boards     Solve every C(27,3) = 2925 flop board individually
    --n_clusters N   Cluster boards and solve one representative per cluster
    --cluster_start / --cluster_end   Split work across instances
    --n_workers N    Parallel workers (one board per worker)

Output: per-board .npz files in the output directory, plus a merged index.

Usage:
    # Quick test (1 board, 50 iterations):
    python compute_multi_street.py --cluster_start 0 --cluster_end 1 \\
        --n_iterations 50 --n_workers 1

    # Full run (all 2925 boards, 8 workers):
    python compute_multi_street.py --all_boards --n_iterations 500 \\
        --n_workers 8

    # Split across 4 machines:
    python compute_multi_street.py --all_boards --n_iterations 500 \\
        --n_workers 8 --cluster_start 0 --cluster_end 732     # machine 1
    python compute_multi_street.py --all_boards --n_iterations 500 \\
        --n_workers 8 --cluster_start 732 --cluster_end 1464  # machine 2
    # etc.
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

from abstraction import (
    enumerate_all_flops,
    compute_board_features,
    get_representative_boards,
)

import multi_street_solver as solver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_CLUSTERS = 200
DEFAULT_N_ITERATIONS = 500
DEFAULT_N_WORKERS = 4

POT_SIZES = [
    (2, 2), (50, 50),
]

# Which pot index to save turn strategies for (to limit disk/RAM usage).
# Index 1 = (4,4). Turn strategies for other pots are discarded; at runtime,
# all turn lookups round to this pot.
TURN_SAVE_POT_IDX = 1


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def solve_single_board(args):
    """
    Solve one flop board with multi-street backward induction.

    Args:
        args: tuple of (board_id, board, config_dict)

    Returns:
        dict with board results
    """
    board_id, board, config = args
    board = list(board)
    n_iterations = config['n_iterations']
    pot_sizes = config.get('pot_sizes', POT_SIZES)
    turn_save_pot_idx = config.get('turn_save_pot_idx', TURN_SAVE_POT_IDX)
    position_aware = config.get('position_aware', False)
    output_dir = config.get('output_dir')

    # Resume support: skip if already solved
    if output_dir:
        out_path = os.path.join(output_dir, f"board_{board_id}.npz")
        if os.path.exists(out_path):
            try:
                existing = np.load(out_path, allow_pickle=True)
                if existing['flop_strategies'].shape[0] == len(pot_sizes):
                    logger.info("Board %d already solved, skipping", board_id)
                    return {
                        'board_id': board_id,
                        'board': board,
                        'elapsed': 0.0,
                        'skipped': True,
                    }
            except Exception:
                pass

    # Each worker creates its own equity engine (not picklable)
    from equity import ExactEquityEngine
    engine = ExactEquityEngine()

    start = time.time()

    try:
        result = solver.solve_flop_board(
            board, engine,
            n_iterations=n_iterations,
            pot_sizes=pot_sizes,
            turn_save_pot_idx=turn_save_pot_idx,
            position_aware=position_aware,
        )
    except Exception as e:
        logger.error("Board %d (%s) failed: %s", board_id, board, e)
        return {
            'board_id': board_id,
            'board': board,
            'elapsed': time.time() - start,
            'skipped': False,
            'error': str(e),
        }

    elapsed = time.time() - start

    # Save per-board result
    if output_dir:
        _save_board_result(board_id, result, n_iterations, output_dir)

    return {
        'board_id': board_id,
        'board': board,
        'elapsed': elapsed,
        'skipped': False,
        'n_hands': len(result['hands']),
    }


def _save_board_result(board_id, result, n_iterations, output_dir):
    """Save one board's multi-street results as .npz."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"board_{board_id}.npz")

    hands_arr = np.array(result['hands'], dtype=np.int8)

    # Build turn strategy arrays
    # For each pot_size × turn_card: strategy array + hands
    save_dict = {
        'flop_strategies': result['flop_strategies'],
        'action_types': result['action_types'],
        'hands': hands_arr,
        'board': np.array(result['board'], dtype=np.int8),
        'pot_sizes': np.array(result['pot_sizes'], dtype=np.int32),
        'board_features': result['board_features'],
        'board_id': board_id,
        'n_iterations': n_iterations,
    }

    # Save position-aware opp strategies if available
    if 'flop_opp_strategies' in result:
        save_dict['flop_opp_strategies'] = result['flop_opp_strategies']
        save_dict['opp_action_types'] = result['opp_action_types']

    # Save turn strategies if available
    turn_strats = result.get('turn_strategies', {})
    turn_cards_saved = []
    for (pot_idx, turn_card), td in turn_strats.items():
        strat = td['strategy']  # (n_hands, n_nodes, n_actions)
        t_hands = np.array(td['hands'], dtype=np.int8)
        # Quantize to uint8
        q = np.clip(np.round(strat * 255.0), 0, 255).astype(np.uint8)
        key = f'turn_strat_p{pot_idx}_t{turn_card}'
        save_dict[key] = q
        save_dict[f'turn_hands_t{turn_card}'] = t_hands

        # Save turn action types
        tree = td['tree']
        max_hn = len(tree.hero_node_ids)
        max_a = max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1)
        t_act = np.full((max_hn, max_a), -1, dtype=np.int8)
        for i, nid in enumerate(tree.hero_node_ids):
            for a, (act_id, _) in enumerate(tree.children[nid]):
                if a < max_a:
                    t_act[i, a] = act_id
        save_dict[f'turn_actions_p{pot_idx}_t{turn_card}'] = t_act

        # Save opp turn strategy if position-aware
        opp_strat = td.get('opp_strategy')
        if opp_strat is not None:
            oq = np.clip(np.round(opp_strat * 255.0), 0, 255).astype(np.uint8)
            save_dict[f'turn_opp_strat_p{pot_idx}_t{turn_card}'] = oq
            # Opp action types
            max_on = len(tree.opp_node_ids)
            max_oa = max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1)
            t_opp_act = np.full((max_on, max_oa), -1, dtype=np.int8)
            for i, nid in enumerate(tree.opp_node_ids):
                for a, (act_id, _) in enumerate(tree.children[nid]):
                    if a < max_oa:
                        t_opp_act[i, a] = act_id
            save_dict[f'turn_opp_actions_p{pot_idx}_t{turn_card}'] = t_opp_act

        if turn_card not in turn_cards_saved:
            turn_cards_saved.append(turn_card)

    if turn_cards_saved:
        save_dict['turn_cards'] = np.array(sorted(set(turn_cards_saved)), dtype=np.int8)

    np.savez_compressed(out_path, **save_dict)

    size_kb = os.path.getsize(out_path) / 1024
    logger.debug("Saved board %d: %s (%.1f KB)", board_id, out_path, size_kb)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_computation(config):
    """Main entry point for multi-street computation."""
    n_iterations = config.get('n_iterations', DEFAULT_N_ITERATIONS)
    n_workers = config.get('n_workers', DEFAULT_N_WORKERS)
    output_dir = config.get('output_dir',
                            os.path.join(_dir, 'output_multi_street'))
    pot_sizes = config.get('pot_sizes', POT_SIZES)
    cluster_start = config.get('cluster_start', 0)
    cluster_end = config.get('cluster_end', None)
    all_boards = config.get('all_boards', False)
    n_clusters = config.get('n_clusters', DEFAULT_N_CLUSTERS)

    print("=" * 60)
    print("Multi-Street Blueprint Computation")
    print("=" * 60)
    print(f"  Iterations:    {n_iterations}")
    print(f"  Pot sizes:     {len(pot_sizes)} {pot_sizes}")
    print(f"  Workers:       {n_workers}")
    print(f"  Output:        {output_dir}")
    print()

    # Enumerate boards
    print("Enumerating flop boards...")
    t0 = time.time()
    all_flops = enumerate_all_flops()
    print(f"  Total flop boards: {len(all_flops)} ({time.time()-t0:.1f}s)")

    if all_boards:
        boards = [(i, board) for i, board in enumerate(all_flops)]
        print(f"  All-boards mode: {len(boards)} individual boards")
    else:
        print(f"  Clustering into {n_clusters} clusters...")
        t0 = time.time()
        boards = get_representative_boards(all_flops, n_clusters)
        print(f"  Representatives: {len(boards)} ({time.time()-t0:.1f}s)")

    # Apply range filter
    if cluster_end is None:
        cluster_end = len(boards)
    else:
        cluster_end = min(cluster_end, len(boards))

    filtered = [(bid, board) for bid, board in boards
                if cluster_start <= bid < cluster_end]

    if not filtered and boards:
        # Fallback: index-based slice
        filtered = boards[cluster_start:cluster_end]

    n_total = len(filtered)
    if n_total == 0:
        print("No boards to solve in the specified range.")
        return

    print(f"  Solving {n_total} boards (range [{cluster_start}, {cluster_end}))")

    # JIT warmup
    print("\nWarming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    solver.warmup_jit()
    print(f" done ({time.time() - t0:.2f}s)")

    # Prepare work items
    turn_save_pot_idx = config.get('turn_save_pot_idx', TURN_SAVE_POT_IDX)
    work_config = {
        'n_iterations': n_iterations,
        'pot_sizes': pot_sizes,
        'turn_save_pot_idx': turn_save_pot_idx,
        'position_aware': config.get('position_aware', False),
        'output_dir': output_dir,
    }
    work_items = [(bid, board, work_config) for bid, board in filtered]

    # Run
    results = []
    start_time = time.time()
    n_skipped = 0
    n_errors = 0

    if n_workers <= 1:
        for i, item in enumerate(work_items):
            bid = item[0]
            board = item[1]
            print(f"[{i+1}/{n_total}] Board {bid} ({board})...",
                  end="", flush=True)

            result = solve_single_board(item)
            results.append(result)

            if result.get('skipped'):
                n_skipped += 1
                print(" skipped")
                continue
            if result.get('error'):
                n_errors += 1
                print(f" ERROR: {result['error']}")
                continue

            elapsed = time.time() - start_time
            n_done = i + 1 - n_skipped - n_errors
            if n_done > 0:
                avg = elapsed / n_done
                remaining = avg * (n_total - i - 1)
            else:
                remaining = 0

            print(f" {result['elapsed']:.1f}s "
                  f"(ETA {remaining/60:.1f}m)")
    else:
        print(f"\nStarting {n_workers} workers...")
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(solve_single_board, work_items)
            ):
                results.append(result)

                if result.get('skipped'):
                    n_skipped += 1
                if result.get('error'):
                    n_errors += 1

                elapsed = time.time() - start_time
                n_done = i + 1 - n_skipped - n_errors
                if n_done > 0:
                    avg = elapsed / n_done
                    remaining = avg * (n_total - i - 1)
                else:
                    remaining = 0

                status = ("skipped" if result.get('skipped')
                          else f"ERROR" if result.get('error')
                          else f"{result['elapsed']:.1f}s")
                print(f"[{i+1}/{n_total}] Board {result['board_id']} "
                      f"{status} (ETA {remaining/60:.1f}m)")

    total_time = time.time() - start_time
    n_solved = n_total - n_skipped - n_errors

    print(f"\nDone in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Solved: {n_solved}, Skipped: {n_skipped}, Errors: {n_errors}")

    # Build merged index
    if output_dir:
        _save_index(results, output_dir, config)


def _save_index(results, output_dir, config):
    """Save an index file listing all solved boards."""
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "multi_street_index.npz")

    solved = [r for r in results
              if not r.get('skipped') and not r.get('error')]
    if not solved:
        print("No new results to index.")
        return

    board_ids = np.array([r['board_id'] for r in solved], dtype=np.int32)
    boards = np.array([list(r['board']) for r in solved], dtype=np.int8)

    # Compute board features for all boards
    features = np.array([compute_board_features(list(r['board']))
                         for r in solved], dtype=np.float32)

    np.savez_compressed(
        index_path,
        board_ids=board_ids,
        boards=boards,
        board_features=features,
        n_iterations=config.get('n_iterations', DEFAULT_N_ITERATIONS),
        pot_sizes=np.array(config.get('pot_sizes', POT_SIZES), dtype=np.int32),
        cluster_start=config.get('cluster_start', 0),
        cluster_end=config.get('cluster_end', len(results)),
    )

    print(f"\nIndex saved: {index_path}")
    print(f"  {len(solved)} boards indexed")

    # Disk usage
    total_size = 0
    for f in os.listdir(output_dir):
        if f.startswith('board_') and f.endswith('.npz'):
            total_size += os.path.getsize(os.path.join(output_dir, f))
    print(f"  Total data: {total_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute multi-street blueprint strategies via backward induction")

    parser.add_argument(
        '--n_iterations', type=int, default=DEFAULT_N_ITERATIONS,
        help=f'CFR iterations per street (default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument(
        '--n_workers', type=int, default=DEFAULT_N_WORKERS,
        help=f'Parallel workers (default: {DEFAULT_N_WORKERS})')
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(_dir, 'output_multi_street'),
        help='Output directory')
    parser.add_argument(
        '--cluster_start', type=int, default=0,
        help='Start board index (inclusive)')
    parser.add_argument(
        '--cluster_end', type=int, default=None,
        help='End board index (exclusive)')
    parser.add_argument(
        '--all_boards', action='store_true',
        help='Solve every flop board individually (no clustering)')
    parser.add_argument(
        '--n_clusters', type=int, default=DEFAULT_N_CLUSTERS,
        help=f'Number of board clusters (default: {DEFAULT_N_CLUSTERS})')
    parser.add_argument(
        '--position_aware', action='store_true',
        help='Save both P0 (first-to-act) and P1 (second-to-act) strategies')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    config = {
        'n_iterations': args.n_iterations,
        'n_workers': args.n_workers,
        'output_dir': args.output_dir,
        'cluster_start': args.cluster_start,
        'cluster_end': args.cluster_end,
        'pot_sizes': POT_SIZES,
        'turn_save_pot_idx': TURN_SAVE_POT_IDX,
        'position_aware': args.position_aware,
        'all_boards': args.all_boards,
        'n_clusters': args.n_clusters,
    }

    run_computation(config)


if __name__ == '__main__':
    main()
