#!/usr/bin/env python3
"""
Merge v9 per-board .npz files into a single LZMA-compressed blueprint.

Downloads from S3 if needed, merges all 2925 boards into compact format,
compresses with LZMA, and deploys to submission/data/multi_street/.

Output: blueprint.pkl.lzma (~87MB) containing:
  - board_ids, boards, board_features
  - flop_strategies (5 pots × 2925 boards × 276 hands × max_nodes × max_actions)
  - flop_opp_strategies (same, for position-aware P1 play)
  - action_types, opp_action_types
  - pot_sizes

Usage:
    python3 blueprint/merge_v9.py                    # merge from local files
    python3 blueprint/merge_v9.py --download         # download from S3 first
    python3 blueprint/merge_v9.py --download --deploy # download, merge, deploy
"""

import os
import sys
import time
import argparse
import lzma
import pickle
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_project = os.path.dirname(_dir)


def download_from_s3(output_dir, region='us-east-1'):
    """Download all board files from S3."""
    import subprocess
    bucket = 's3://poker-blueprint-2026/multi_street_v9/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading from {bucket} to {output_dir}...")
    subprocess.run([
        'aws', 's3', 'sync', bucket, output_dir,
        '--region', region, '--exclude', '*.log'
    ], check=True)
    count = len([f for f in os.listdir(output_dir) if f.startswith('board_')])
    print(f"Downloaded {count} board files.")
    return count


def merge_boards(board_dir):
    """Merge all per-board .npz files into compact arrays."""
    import glob
    files = sorted(glob.glob(os.path.join(board_dir, 'board_*.npz')))
    print(f"Merging {len(files)} board files...")

    if not files:
        raise FileNotFoundError(f"No board files in {board_dir}")

    # Load first board to get dimensions
    sample = np.load(files[0], allow_pickle=True)
    n_pots = sample['flop_strategies'].shape[0]
    n_hands = sample['flop_strategies'].shape[1]
    max_hero_nodes = sample['flop_strategies'].shape[2]
    max_actions = sample['flop_strategies'].shape[3]
    pot_sizes = sample['pot_sizes']
    has_opp = 'flop_opp_strategies' in sample

    print(f"  Pots: {n_pots} {pot_sizes.tolist()}")
    print(f"  Hands: {n_hands}, Nodes: {max_hero_nodes}, Actions: {max_actions}")
    print(f"  Position-aware: {has_opp}")

    n_boards = len(files)

    # Pre-allocate arrays
    board_ids = np.zeros(n_boards, dtype=np.int32)
    boards = np.zeros((n_boards, 3), dtype=np.int8)
    features = np.zeros((n_boards, 12), dtype=np.float32)
    hands_arr = np.zeros((n_boards, n_hands, 2), dtype=np.int8)
    flop_strats = np.zeros((n_boards, n_pots, n_hands, max_hero_nodes, max_actions),
                           dtype=np.uint8)
    act_types = np.zeros((n_boards, n_pots, max_hero_nodes, max_actions), dtype=np.int8)

    if has_opp:
        opp_strats = np.zeros_like(flop_strats)
        opp_act_types = np.zeros_like(act_types)

    t0 = time.time()
    errors = 0
    for i, fpath in enumerate(files):
        try:
            d = np.load(fpath, allow_pickle=True)
            bid = int(d['board_id'])
            board_ids[i] = bid
            boards[i] = d['board']
            features[i] = d['board_features']
            hands_arr[i] = d['hands']

            s = d['flop_strategies']
            # Handle different node counts per pot (pad to max)
            sp, sh, sn, sa = s.shape
            flop_strats[i, :sp, :sh, :sn, :sa] = s

            a = d['action_types']
            ap, an2, aa = a.shape
            act_types[i, :ap, :an2, :aa] = a

            if has_opp and 'flop_opp_strategies' in d:
                os_ = d['flop_opp_strategies']
                osp, osh, osn, osa = os_.shape
                opp_strats[i, :osp, :osh, :osn, :osa] = os_

                oa = d['opp_action_types']
                oap, oan, oaa = oa.shape
                opp_act_types[i, :oap, :oan, :oaa] = oa

        except Exception as e:
            print(f"  ERROR on {fpath}: {e}")
            errors += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_boards} merged ({time.time()-t0:.0f}s)")

    print(f"  Merged {n_boards - errors} boards in {time.time()-t0:.1f}s ({errors} errors)")

    result = {
        'board_ids': board_ids,
        'boards': boards,
        'board_features': features,
        'flop_strategies': flop_strats,
        'action_types': act_types,
        'pot_sizes': pot_sizes,
        'hands': hands_arr,
    }

    if has_opp:
        result['flop_opp_strategies'] = opp_strats
        result['opp_action_types'] = opp_act_types

    return result


def compress_lzma(data, output_path):
    """Compress merged data with LZMA."""
    print(f"Compressing with LZMA...")
    t0 = time.time()
    pkl = pickle.dumps(data)
    print(f"  Pickled: {len(pkl)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    t0 = time.time()
    compressed = lzma.compress(pkl)
    print(f"  LZMA: {len(compressed)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    with open(output_path, 'wb') as f:
        f.write(compressed)
    print(f"  Saved: {output_path}")
    return len(compressed)


def deploy(lzma_path, submission_dir):
    """Deploy LZMA file to submission/data/multi_street/."""
    deploy_dir = os.path.join(submission_dir, 'data', 'multi_street')
    os.makedirs(deploy_dir, exist_ok=True)

    # Remove old files
    for f in os.listdir(deploy_dir):
        old = os.path.join(deploy_dir, f)
        if os.path.isfile(old):
            print(f"  Removing old: {f}")
            os.remove(old)

    # Copy new LZMA
    import shutil
    dest = os.path.join(deploy_dir, 'blueprint.pkl.lzma')
    shutil.copy2(lzma_path, dest)
    print(f"  Deployed: {dest} ({os.path.getsize(dest)/1024/1024:.1f}MB)")

    # Check total submission size
    total = 0
    for root, dirs, files in os.walk(submission_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.pyc'):
                continue
            total += os.path.getsize(os.path.join(root, f))
    print(f"  Total submission size: {total/1024/1024:.1f}MB")
    if total > 100 * 1024 * 1024:
        print(f"  *** WARNING: exceeds 100MB limit! ***")
    else:
        print(f"  ✓ Under 100MB limit")


def validate(submission_dir):
    """Validate the deployed data loads correctly."""
    print("\nValidating deployment...")
    sys.path.insert(0, submission_dir)

    from equity import ExactEquityEngine
    from multi_street_lookup import MultiStreetLookup

    engine = ExactEquityEngine()
    data_dir = os.path.join(submission_dir, 'data', 'multi_street')

    t0 = time.time()
    ms = MultiStreetLookup(data_dir, equity_engine=engine)
    load_time = time.time() - t0

    print(f"  Loaded {ms.n_boards} boards in {load_time:.1f}s")

    if ms.n_boards < 2900:
        print(f"  *** WARNING: only {ms.n_boards} boards (expected 2925) ***")

    # Test lookups
    test_hands = [[18, 22], [3, 7], [0, 26]]
    test_boards = [[0, 4, 8], [1, 10, 20], [5, 15, 25]]

    for hero, board in zip(test_hands, test_boards):
        for pot in [(2, 2), (10, 10), (50, 50)]:
            for pos in [0, 1]:
                strat = ms.get_strategy(hero, board, pot_state=pot, hero_position=pos)
                if strat is None:
                    print(f"  *** FAIL: no strategy for hero={hero} board={board} pot={pot} pos={pos}")
                    return False

    print(f"  ✓ All lookups successful")
    print(f"  ✓ Load time: {load_time:.1f}s (30s budget: {'OK' if load_time < 25 else 'TIGHT'})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download from S3 first')
    parser.add_argument('--deploy', action='store_true', help='Deploy to submission/')
    parser.add_argument('--board-dir', default=os.path.join(_project, 'submission', 'data', 'multi_street_v9'),
                        help='Directory with board_*.npz files')
    parser.add_argument('--output', default=os.path.join(_project, 'blueprint', 'output', 'blueprint_v9.pkl.lzma'))
    args = parser.parse_args()

    if args.download:
        download_from_s3(args.board_dir)

    # Check board count
    board_files = [f for f in os.listdir(args.board_dir) if f.startswith('board_')] if os.path.isdir(args.board_dir) else []
    print(f"\nBoard files: {len(board_files)}")
    if len(board_files) < 2900:
        print(f"*** Only {len(board_files)} boards — EC2 may not be done yet ***")
        if not args.download:
            print("Try: python3 blueprint/merge_v9.py --download")
        return

    # Merge
    data = merge_boards(args.board_dir)

    # Compress
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    size = compress_lzma(data, args.output)

    if size > 95 * 1024 * 1024:
        print(f"\n*** WARNING: {size/1024/1024:.1f}MB — close to 100MB limit ***")

    # Deploy
    if args.deploy:
        submission_dir = os.path.join(_project, 'submission')
        deploy(args.output, submission_dir)
        validate(submission_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
