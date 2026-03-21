#!/usr/bin/env python3
"""
Merge river compute pickle files into a single LZMA-compressed file for deployment.

Reads 10 pickle files from EC2 (each containing a dict of
board_tuple -> {opp_pb, hero_s, hands}), merges them, and produces a
compact LZMA file for the submission directory.

Input format (per pickle file):
    {
        (c0,c1,c2,c3,c4): {
            'opp_pb': np.array(shape=(5, 231), dtype=uint8),
            'hero_s': np.array(shape=(5, 231, n_actions), dtype=uint8),
            'hands':  np.array(shape=(231, 2), dtype=int8),
        },
        ...
    }

Output format: LZMA-compressed pickle containing:
    {
        'board_index': np.array(shape=(N, 5), dtype=int8),   # sorted board cards
        'opp_pb':      np.array(shape=(N, 5, 231), dtype=uint8),
        'hero_s':      np.array(shape=(N, 5, 231, 3), dtype=uint8),
        'hands':       np.array(shape=(N, 231, 2), dtype=int8),
        'pot_sizes':   [(2,2), (6,6), (14,14), (30,30), (50,50)],
    }

    N = number of boards (up to 80,730 = C(27,5))
    Board lookup: sort 5 board cards, binary search in board_index.

Raw size ~410MB, LZMA ~80-120MB. Loaded once at startup.

Usage:
    python3 merge_river.py --input_dir /path/to/pickles --output_dir submission/data/multi_street
    python3 merge_river.py --input_dir s3://bucket/river_output --output_dir submission/data/multi_street
"""

import os
import sys
import time
import argparse
import glob
import pickle
import lzma

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_project = os.path.dirname(_dir)

POT_SIZES = [(2, 2), (6, 6), (14, 14), (30, 30), (50, 50)]
N_POTS = 5
N_HANDS = 231       # C(22, 2)
HERO_S_ACTIONS = 3  # truncate hero_s to first 3 actions (fold/check, raise_half, raise_pot)
                     # to save space — higher raise levels are rare on river


def load_pickles(input_dir):
    """Load and merge all pickle files from input_dir.

    Supports local directory or S3 prefix (requires boto3).
    Returns merged dict: board_tuple -> {opp_pb, hero_s, hands}.
    """
    merged = {}

    if input_dir.startswith('s3://'):
        # S3 mode
        import boto3
        import io
        parts = input_dir[5:].split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        if not prefix.endswith('/'):
            prefix += '/'

        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        pkl_keys = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pkl'):
                    pkl_keys.append(key)

        print(f"Found {len(pkl_keys)} pickle files in s3://{bucket_name}/{prefix}")
        for i, key in enumerate(sorted(pkl_keys)):
            print(f"  Loading {key}...", end="", flush=True)
            t0 = time.time()
            response = s3.get_object(Bucket=bucket_name, Key=key)
            data = pickle.loads(response['Body'].read())
            n_boards = len(data)
            merged.update(data)
            print(f" {n_boards} boards ({time.time()-t0:.1f}s)")

    else:
        # Local directory
        pkl_files = sorted(glob.glob(os.path.join(input_dir, '*.pkl')))
        if not pkl_files:
            # Try subdirectories
            pkl_files = sorted(glob.glob(os.path.join(input_dir, '**', '*.pkl'),
                                          recursive=True))
        print(f"Found {len(pkl_files)} pickle files in {input_dir}")
        for fpath in pkl_files:
            print(f"  Loading {os.path.basename(fpath)}...", end="", flush=True)
            t0 = time.time()
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            n_boards = len(data)
            merged.update(data)
            print(f" {n_boards} boards ({time.time()-t0:.1f}s)")

    return merged


def build_arrays(merged):
    """Build compact numpy arrays from merged dict.

    Returns dict with:
        board_index: (N, 5) int8
        opp_pb: (N, 5, 231) uint8
        hero_s: (N, 5, 231, 3) uint8
        hands: (N, 231, 2) int8
    """
    # Sort boards for binary search at runtime
    boards_sorted = sorted(merged.keys())
    n_boards = len(boards_sorted)
    print(f"\nBuilding arrays for {n_boards} boards...")

    # Determine dimensions from first entry
    sample = merged[boards_sorted[0]]
    actual_n_hands = sample['hands'].shape[0]
    actual_n_actions = sample['hero_s'].shape[2] if sample['hero_s'].ndim == 3 else 1

    # Use actual hand count (should be 231 for 27-card deck, 5-card board)
    n_hands = actual_n_hands
    # Truncate hero actions to save space
    n_hero_actions = min(actual_n_actions, HERO_S_ACTIONS)

    print(f"  Hands per board: {n_hands}")
    print(f"  Hero actions (raw): {actual_n_actions}, keeping: {n_hero_actions}")
    print(f"  Pot sizes: {N_POTS}")

    # Allocate arrays
    board_index = np.zeros((n_boards, 5), dtype=np.int8)
    opp_pb = np.zeros((n_boards, N_POTS, n_hands), dtype=np.uint8)
    hero_s = np.zeros((n_boards, N_POTS, n_hands, n_hero_actions), dtype=np.uint8)
    hands = np.zeros((n_boards, n_hands, 2), dtype=np.int8)

    t0 = time.time()
    for i, board_tuple in enumerate(boards_sorted):
        entry = merged[board_tuple]

        # Board cards (already sorted since they come from combinations)
        board_index[i] = np.array(board_tuple, dtype=np.int8)

        # Hands array
        entry_hands = entry['hands']
        nh = min(len(entry_hands), n_hands)
        hands[i, :nh] = entry_hands[:nh]

        # opp_pb: (5, n_hands) -> copy directly
        entry_opp = entry['opp_pb']
        opp_pb[i, :, :nh] = entry_opp[:, :nh]

        # hero_s: (5, n_hands, n_actions) -> truncate actions
        entry_hero = entry['hero_s']
        if entry_hero.ndim == 3:
            hero_s[i, :, :nh, :] = entry_hero[:, :nh, :n_hero_actions]
        elif entry_hero.ndim == 2:
            # If only 2D, treat as single action
            hero_s[i, :, :nh, 0] = entry_hero[:, :nh]

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{n_boards} ({time.time()-t0:.0f}s)")

    print(f"  Done building arrays ({time.time()-t0:.1f}s)")

    # Memory stats
    total_bytes = (board_index.nbytes + opp_pb.nbytes +
                   hero_s.nbytes + hands.nbytes)
    print(f"\n  Array sizes:")
    print(f"    board_index: {board_index.shape} = {board_index.nbytes/1024/1024:.1f}MB")
    print(f"    opp_pb:      {opp_pb.shape} = {opp_pb.nbytes/1024/1024:.1f}MB")
    print(f"    hero_s:      {hero_s.shape} = {hero_s.nbytes/1024/1024:.1f}MB")
    print(f"    hands:       {hands.shape} = {hands.nbytes/1024/1024:.1f}MB")
    print(f"    TOTAL RAW:   {total_bytes/1024/1024:.1f}MB")

    return {
        'board_index': board_index,
        'opp_pb': opp_pb,
        'hero_s': hero_s,
        'hands': hands,
        'pot_sizes': POT_SIZES,
        'n_boards': n_boards,
        'n_hands': n_hands,
        'n_hero_actions': n_hero_actions,
    }


def compress_and_save(data, output_dir):
    """LZMA-compress and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'river.pkl.lzma')

    print(f"\nCompressing with LZMA (this may take a few minutes)...")
    t0 = time.time()
    pkl = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_mb = len(pkl) / 1024 / 1024
    print(f"  Pickled: {pkl_mb:.1f}MB ({time.time()-t0:.1f}s)")

    t0 = time.time()
    compressed = lzma.compress(pkl, preset=6)  # preset 6 = good balance
    comp_mb = len(compressed) / 1024 / 1024
    ratio = comp_mb / pkl_mb * 100
    print(f"  LZMA: {comp_mb:.1f}MB ({ratio:.0f}% of raw, {time.time()-t0:.1f}s)")

    t0 = time.time()
    with open(output_path, 'wb') as f:
        f.write(compressed)
    print(f"  Saved: {output_path} ({time.time()-t0:.1f}s)")

    # Check total submission size
    sub_dir = os.path.dirname(output_dir)
    total = 0
    for root, dirs, files in os.walk(sub_dir):
        if '__pycache__' in root:
            continue
        for fn in files:
            if fn.endswith('.pyc'):
                continue
            total += os.path.getsize(os.path.join(root, fn))
    print(f"\n  Total submission size: {total/1024/1024:.1f}MB (limit: 1024MB)")
    if total > 1024 * 1024 * 1024:
        print(f"  *** WARNING: exceeds 1GB limit! ***")
    else:
        print(f"  Under 1GB limit")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge river compute results into deployment format")
    parser.add_argument('--input_dir', required=True,
                        help='Directory (or S3 prefix) containing river_*.pkl files')
    parser.add_argument('--output_dir',
                        default=os.path.join(_project, 'submission', 'data', 'multi_street'),
                        help='Output directory for deployment')
    parser.add_argument('--verify', action='store_true',
                        help='Verify output by loading and spot-checking')
    args = parser.parse_args()

    print("=" * 60)
    print("River Data Merger")
    print("=" * 60)
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    # Step 1: Load all pickle files
    t_total = time.time()
    merged = load_pickles(args.input_dir)
    if not merged:
        print("ERROR: No data loaded!")
        sys.exit(1)
    print(f"\nTotal boards merged: {len(merged)}")

    # Step 2: Build compact arrays
    arrays = build_arrays(merged)

    # Free merged dict to save memory before compression
    del merged

    # Step 3: Compress and save
    output_path = compress_and_save(arrays, args.output_dir)

    # Step 4: Optional verification
    if args.verify:
        print("\nVerifying output...")
        t0 = time.time()
        with open(output_path, 'rb') as f:
            loaded = pickle.loads(lzma.decompress(f.read()))
        print(f"  Load time: {time.time()-t0:.1f}s")
        print(f"  Boards: {loaded['n_boards']}")
        print(f"  board_index shape: {loaded['board_index'].shape}")
        print(f"  opp_pb shape: {loaded['opp_pb'].shape}")
        print(f"  hero_s shape: {loaded['hero_s'].shape}")
        print(f"  hands shape: {loaded['hands'].shape}")

        # Spot check: random board
        idx = np.random.randint(0, loaded['n_boards'])
        board = tuple(loaded['board_index'][idx].tolist())
        opp_mean = loaded['opp_pb'][idx].mean() / 255.0
        hero_mean = loaded['hero_s'][idx].mean() / 255.0
        print(f"\n  Spot check board {board}:")
        print(f"    opp_pb mean: {opp_mean:.3f}")
        print(f"    hero_s mean: {hero_mean:.3f}")
        print(f"    hands[0]: ({loaded['hands'][idx][0][0]}, {loaded['hands'][idx][0][1]})")

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Complete in {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
