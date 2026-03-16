#!/usr/bin/env python3
"""Merge per-board multi-street results into submission data."""
import os, sys, numpy as np

def merge(input_dir, output_dir):
    files = sorted([f for f in os.listdir(input_dir) if f.startswith('board_') and f.endswith('.npz')])
    print(f"Found {len(files)} board files")

    if not files:
        print("No files to merge!")
        return

    # Load all
    boards_data = []
    for f in files:
        d = np.load(os.path.join(input_dir, f), allow_pickle=True)
        if 'flop_strategies' in d:
            boards_data.append(d)

    n = len(boards_data)
    print(f"Valid boards: {n}")

    # Copy to submission multi_street directory
    ms_dir = os.path.join(output_dir, "multi_street")
    os.makedirs(ms_dir, exist_ok=True)

    for f in files:
        src = os.path.join(input_dir, f)
        dst = os.path.join(ms_dir, f)
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    print(f"Copied {len(files)} files to {ms_dir}")
    print(f"Total size: {sum(os.path.getsize(os.path.join(ms_dir, f)) for f in os.listdir(ms_dir)) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    merge("/tmp/multi_street_results", "submission/data")
