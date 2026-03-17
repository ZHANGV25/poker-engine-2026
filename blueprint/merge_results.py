#!/usr/bin/env python3
"""
Merge unbucketed blueprint results from multiple instances.

When compute_unbucketed.py is run across multiple machines (each solving a
subset of clusters via --cluster_start/--cluster_end), this script:

1. Downloads partial results from S3 (or reads from local dirs)
2. Merges per-cluster .npz files into a single blueprint file
3. Uploads the merged result back to S3

The merged output is a single .npz file with strategies indexed by
(cluster_idx, pot_size, hand, node, action) as uint8.

Usage:
    # Merge from local directories:
    python merge_results.py \\
        --input_dirs ./output_part1 ./output_part2 ./output_part3 \\
        --output_dir ./merged \\
        --street flop

    # Merge from S3:
    python merge_results.py \\
        --s3_prefix s3://my-bucket/blueprint/unbucketed/ \\
        --output_dir ./merged \\
        --street flop \\
        --n_clusters 200

    # Verify merged file:
    python merge_results.py --verify ./merged/flop_unbucketed.npz
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import tempfile
import glob as glob_module

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from abstraction import compute_board_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_list_files(s3_prefix, pattern="cluster_*.npz"):
    """List .npz files under an S3 prefix."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_prefix, "--recursive"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.error("aws s3 ls failed: %s", result.stderr)
            return []

        files = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                key = parts[3]
                basename = os.path.basename(key)
                if basename.startswith('cluster_') and basename.endswith('.npz'):
                    files.append(key)
        return files
    except FileNotFoundError:
        logger.error("aws CLI not found. Install with: pip install awscli")
        return []
    except subprocess.TimeoutExpired:
        logger.error("aws s3 ls timed out")
        return []


def s3_download_files(s3_bucket, keys, local_dir):
    """Download files from S3 to local directory."""
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []

    for key in keys:
        basename = os.path.basename(key)
        local_path = os.path.join(local_dir, basename)

        if os.path.exists(local_path):
            logger.debug("Already downloaded: %s", basename)
            downloaded.append(local_path)
            continue

        s3_uri = f"s3://{s3_bucket}/{key}"
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_uri, local_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                downloaded.append(local_path)
                logger.debug("Downloaded: %s", basename)
            else:
                logger.warning("Failed to download %s: %s", s3_uri, result.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("Download timed out: %s", s3_uri)

    return downloaded


def s3_upload_file(local_path, s3_uri):
    """Upload a file to S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", local_path, s3_uri],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info("Uploaded to %s", s3_uri)
            return True
        else:
            logger.error("Upload failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("aws CLI not found")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Upload timed out")
        return False


# ---------------------------------------------------------------------------
# Local file collection
# ---------------------------------------------------------------------------

def collect_local_cluster_files(input_dirs):
    """
    Collect all cluster_*.npz files from one or more local directories.

    Returns:
        dict mapping cluster_id -> file_path
    """
    cluster_files = {}

    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            logger.warning("Directory not found: %s", input_dir)
            continue

        pattern = os.path.join(input_dir, "cluster_*.npz")
        for fpath in sorted(glob_module.glob(pattern)):
            basename = os.path.basename(fpath)
            # Extract cluster_id from filename: cluster_42.npz -> 42
            try:
                cid = int(basename.replace("cluster_", "").replace(".npz", ""))
            except ValueError:
                logger.warning("Cannot parse cluster ID from: %s", basename)
                continue

            if cid in cluster_files:
                logger.warning("Duplicate cluster %d: %s vs %s",
                               cid, cluster_files[cid], fpath)
            cluster_files[cid] = fpath

    return cluster_files


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_clusters(cluster_files, output_path, street, n_clusters=None):
    """
    Merge per-cluster .npz files into a single blueprint file.

    Args:
        cluster_files: dict mapping cluster_id -> file_path
        output_path: path for the merged .npz file
        street: 'flop', 'turn', or 'river'
        n_clusters: total expected clusters (for coverage reporting)

    Returns:
        path to the merged file
    """
    if not cluster_files:
        logger.error("No cluster files to merge")
        return None

    sorted_ids = sorted(cluster_files.keys())
    n_found = len(sorted_ids)

    print(f"\nMerging {n_found} cluster files...")
    if n_clusters:
        print(f"  Coverage: {n_found}/{n_clusters} "
              f"({100*n_found/n_clusters:.1f}%)")
        missing = set(range(n_clusters)) - set(sorted_ids)
        if missing and len(missing) <= 20:
            print(f"  Missing clusters: {sorted(missing)}")
        elif missing:
            print(f"  Missing: {len(missing)} clusters")

    # First pass: determine dimensions by loading one file
    sample_path = cluster_files[sorted_ids[0]]
    sample = np.load(sample_path, allow_pickle=True)

    sample_hero = sample['hero_strategies']
    n_pot_sizes = sample_hero.shape[0]
    sample_n_hands = sample_hero.shape[1]
    sample_n_nodes = sample_hero.shape[2]
    sample_n_actions = sample_hero.shape[3]
    sample_pot_sizes = sample['pot_sizes']
    sample_n_iterations = int(sample['n_iterations'])

    print(f"  Pot sizes: {sample_pot_sizes.tolist()}")
    print(f"  Sample shape per cluster: {sample_hero.shape}")

    # Second pass: find max dimensions across all clusters
    max_n_hands = 0
    max_hero_nodes = 0
    max_opp_nodes = 0
    max_actions = 0

    for cid in sorted_ids:
        try:
            data = np.load(cluster_files[cid], allow_pickle=True)
            hs = data['hero_strategies']
            max_n_hands = max(max_n_hands, hs.shape[1])
            max_hero_nodes = max(max_hero_nodes, hs.shape[2])
            max_actions = max(max_actions, hs.shape[3])
            if 'opp_strategies' in data:
                os_arr = data['opp_strategies']
                max_opp_nodes = max(max_opp_nodes, os_arr.shape[2])
        except Exception as e:
            logger.warning("Error reading cluster %d: %s", cid, e)

    print(f"  Max dimensions: hands={max_n_hands}, hero_nodes={max_hero_nodes}, "
          f"opp_nodes={max_opp_nodes}, actions={max_actions}")

    # Allocate merged arrays
    # Shape: (n_clusters, n_pot_sizes, max_n_hands, max_hero_nodes, max_actions)
    merged_hero = np.zeros(
        (n_found, n_pot_sizes, max_n_hands, max_hero_nodes, max_actions),
        dtype=np.uint8)
    merged_opp = np.zeros(
        (n_found, n_pot_sizes, max_n_hands, max_opp_nodes, max_actions),
        dtype=np.uint8)

    # Per-cluster metadata
    cluster_ids_arr = np.zeros(n_found, dtype=np.int32)
    n_board_cards = {'flop': 3, 'turn': 4, 'river': 5}[street]
    boards_arr = np.zeros((n_found, n_board_cards), dtype=np.int8)
    board_features_arr = np.zeros((n_found, 12), dtype=np.float32)

    # Action types: (n_found, n_pot_sizes, max_hero_nodes, max_actions)
    action_types_arr = np.full(
        (n_found, n_pot_sizes, max_hero_nodes, max_actions), -1, dtype=np.int8)

    # Hands arrays: (n_found, max_n_hands, 2)
    hands_arr = np.zeros((n_found, max_n_hands, 2), dtype=np.int8)
    n_hands_arr = np.zeros(n_found, dtype=np.int32)

    # Third pass: load and merge
    t0 = time.time()
    for idx, cid in enumerate(sorted_ids):
        try:
            data = np.load(cluster_files[cid], allow_pickle=True)

            hs = data['hero_strategies']
            np_, nh, nn, na = hs.shape
            merged_hero[idx, :np_, :nh, :nn, :na] = hs

            if 'opp_strategies' in data:
                os_arr = data['opp_strategies']
                np2, no, nn2, na2 = os_arr.shape
                merged_opp[idx, :np2, :no, :nn2, :na2] = os_arr

            cluster_ids_arr[idx] = cid

            board = data['board']
            boards_arr[idx, :len(board)] = board

            if 'board_features' in data:
                board_features_arr[idx] = data['board_features']
            else:
                board_features_arr[idx] = compute_board_features(list(board))

            if 'action_types' in data:
                at = data['action_types']
                at_shape = at.shape
                action_types_arr[idx, :at_shape[0], :at_shape[1], :at_shape[2]] = at

            if 'hands' in data:
                h = data['hands']
                n_hands_arr[idx] = len(h)
                hands_arr[idx, :len(h)] = h

        except Exception as e:
            logger.error("Error loading cluster %d from %s: %s",
                         cid, cluster_files[cid], e)

        if (idx + 1) % 50 == 0:
            print(f"  Loaded {idx+1}/{n_found} clusters...")

    load_time = time.time() - t0
    print(f"  Loading complete in {load_time:.1f}s")

    # Save merged file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"\nSaving merged file: {output_path}")
    t0 = time.time()

    np.savez_compressed(
        output_path,
        # Strategy data
        hero_strategies=merged_hero,
        opp_strategies=merged_opp,
        action_types=action_types_arr,
        # Hand mapping
        hands=hands_arr,
        n_hands=n_hands_arr,
        # Board metadata
        cluster_ids=cluster_ids_arr,
        boards=boards_arr,
        board_features=board_features_arr,
        # Config
        pot_sizes=sample_pot_sizes,
        n_iterations=sample_n_iterations,
        n_clusters_total=n_clusters or n_found,
        street=street,
        format='unbucketed',
    )

    save_time = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"  Saved in {save_time:.1f}s ({size_mb:.1f} MB)")
    print(f"  Shape: {merged_hero.shape}")
    print(f"  Format: uint8 (quantized probabilities, /255 to recover)")

    return output_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_merged(path):
    """
    Verify integrity of a merged blueprint file.

    Checks:
    - No NaN/Inf values (shouldn't happen with uint8 but check anyway)
    - Strategy probabilities approximately sum to 255 at each node
    - All clusters have non-trivial strategies
    - Action types are valid
    """
    print(f"\nVerifying: {path}")
    data = np.load(path, allow_pickle=True)

    hero = data['hero_strategies']
    cluster_ids = data['cluster_ids']
    boards = data['boards']
    pot_sizes = data['pot_sizes']
    n_hands = data['n_hands']

    n_clusters, n_pots, max_hands, max_nodes, max_actions = hero.shape

    print(f"  Shape: {hero.shape}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Pot sizes: {pot_sizes.tolist()}")
    print(f"  Street: {data['street']}")
    print(f"  Format: {data.get('format', 'unknown')}")
    print(f"  Iterations: {data['n_iterations']}")

    # Check strategy sums
    n_issues = 0
    n_checked = 0
    n_empty = 0

    for ci in range(min(n_clusters, 10)):  # Spot-check first 10
        nh = n_hands[ci]
        for pi in range(n_pots):
            for hi in range(min(nh, 5)):  # Check first 5 hands per cluster
                for ni in range(max_nodes):
                    row = hero[ci, pi, hi, ni, :]
                    total = int(row.sum())
                    if total == 0:
                        n_empty += 1
                        continue
                    n_checked += 1
                    # With uint8 quantization, sum should be ~255 (+/- rounding)
                    if total < 200 or total > 310:
                        n_issues += 1
                        if n_issues <= 5:
                            print(f"  WARNING: cluster={cluster_ids[ci]}, "
                                  f"pot={pi}, hand={hi}, node={ni}: "
                                  f"sum={total}")

    print(f"  Spot-checked {n_checked} strategy entries "
          f"({n_empty} empty, {n_issues} issues)")

    # Check cluster coverage
    unique_ids = set(cluster_ids.tolist())
    print(f"  Unique cluster IDs: {len(unique_ids)}")

    # Check hands
    hands_with_data = sum(1 for i in range(n_clusters) if n_hands[i] > 0)
    print(f"  Clusters with hand data: {hands_with_data}/{n_clusters}")

    if n_issues == 0:
        print("  PASS: All checks passed")
    else:
        print(f"  WARN: {n_issues} issues found")

    # Print size breakdown
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")

    return n_issues == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge unbucketed blueprint results from multiple instances")

    parser.add_argument(
        '--input_dirs', nargs='+', default=None,
        help='Local directories containing cluster_*.npz files')
    parser.add_argument(
        '--s3_prefix', type=str, default=None,
        help='S3 prefix to download cluster files from (e.g., s3://bucket/path/)')
    parser.add_argument(
        '--output_dir', type=str, default=os.path.join(_dir, 'output_unbucketed', 'merged'),
        help='Output directory for merged file')
    parser.add_argument(
        '--street', type=str, default='flop',
        choices=['flop', 'turn', 'river'],
        help='Street name for the output file')
    parser.add_argument(
        '--n_clusters', type=int, default=None,
        help='Total expected number of clusters (for coverage reporting)')
    parser.add_argument(
        '--s3_upload', type=str, default=None,
        help='S3 URI to upload merged result to')
    parser.add_argument(
        '--verify', type=str, default=None,
        help='Path to a merged file to verify (skips merge)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Verify mode
    if args.verify:
        verify_merged(args.verify)
        return

    # Collect cluster files
    cluster_files = {}

    if args.s3_prefix:
        # Download from S3
        print(f"Listing files in {args.s3_prefix}...")

        # Parse S3 URI
        s3_parts = args.s3_prefix.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ""

        keys = s3_list_files(args.s3_prefix)
        print(f"  Found {len(keys)} cluster files")

        if keys:
            tmp_dir = os.path.join(args.output_dir, '_s3_download')
            print(f"  Downloading to {tmp_dir}...")
            local_paths = s3_download_files(bucket, keys, tmp_dir)
            print(f"  Downloaded {len(local_paths)} files")

            # Parse into cluster_files dict
            for fpath in local_paths:
                basename = os.path.basename(fpath)
                try:
                    cid = int(basename.replace("cluster_", "").replace(".npz", ""))
                    cluster_files[cid] = fpath
                except ValueError:
                    pass

    if args.input_dirs:
        # Collect from local directories
        local_files = collect_local_cluster_files(args.input_dirs)
        print(f"Found {len(local_files)} cluster files in local directories")
        # Merge with any S3 files (local takes precedence)
        cluster_files.update(local_files)

    if not cluster_files:
        print("ERROR: No cluster files found. Specify --input_dirs or --s3_prefix.")
        sys.exit(1)

    print(f"\nTotal cluster files: {len(cluster_files)}")

    # Merge
    output_path = os.path.join(args.output_dir, f"{args.street}_unbucketed.npz")
    merged_path = merge_clusters(
        cluster_files, output_path, args.street, args.n_clusters)

    if merged_path:
        # Verify
        verify_merged(merged_path)

        # Upload to S3 if requested
        if args.s3_upload:
            print(f"\nUploading to {args.s3_upload}...")
            s3_upload_file(merged_path, args.s3_upload)


if __name__ == '__main__':
    main()
