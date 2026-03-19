#!/usr/bin/env python3
"""
Merge turn opponent strategies from per-board files into a single LZMA file.

Only extracts opponent strategies (for Bayesian narrowing on turn),
not hero strategies (too large). 4-bit quantization to fit in 1GB limit.

Output: turn_opp.pkl.lzma deployed to submission/data/multi_street/
"""

import os
import sys
import time
import lzma
import pickle
import numpy as np
import glob

_dir = os.path.dirname(os.path.abspath(__file__))
_project = os.path.dirname(_dir)


def merge_turn_opp(board_dir):
    """Merge opponent turn strategies from per-board .npz files."""
    files = sorted(glob.glob(os.path.join(board_dir, 'board_*.npz')))
    print(f"Merging turn opp strategies from {len(files)} boards...")

    result = {}
    t0 = time.time()
    errors = 0

    for i, fpath in enumerate(files):
        try:
            d = np.load(fpath, allow_pickle=True)
            bid = int(d['board_id'])
            board = tuple(d['board'].tolist())

            # Get turn cards
            if 'turn_cards' not in d:
                continue

            turn_cards = d['turn_cards'].tolist()
            pot_sizes = d['pot_sizes'].tolist()
            n_pots = len(pot_sizes)

            board_data = {
                'board': board,
                'pot_sizes': pot_sizes,
                'turn_cards': turn_cards,
            }

            # Extract opp strategies per pot per turn card
            for pi in range(n_pots):
                for tc in turn_cards:
                    opp_key = f'turn_opp_strat_p{pi}_t{tc}'
                    act_key = f'turn_opp_actions_p{pi}_t{tc}'
                    hands_key = f'turn_hands_t{tc}'

                    if opp_key in d and act_key in d:
                        # 4-bit quantize: divide by 16, store as uint8
                        opp_strat = d[opp_key]
                        q = (opp_strat.astype(np.uint16) >> 4).astype(np.uint8)
                        board_data[f'opp_p{pi}_t{tc}'] = q
                        board_data[f'act_p{pi}_t{tc}'] = d[act_key]

                    if hands_key in d and hands_key not in board_data:
                        board_data[f'hands_t{tc}'] = d[hands_key]

            result[bid] = board_data

        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  Error on {fpath}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)")

    print(f"  Merged {len(result)} boards in {time.time()-t0:.1f}s ({errors} errors)")
    return result


def compress_and_deploy(data, output_dir):
    """Compress and deploy turn data."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'turn_opp.pkl.lzma')

    print("Compressing with LZMA...")
    t0 = time.time()
    pkl = pickle.dumps(data)
    print(f"  Pickled: {len(pkl)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    t0 = time.time()
    compressed = lzma.compress(pkl)
    print(f"  LZMA: {len(compressed)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    with open(output_path, 'wb') as f:
        f.write(compressed)
    print(f"  Saved: {output_path}")

    # Check total submission size
    sub_dir = os.path.join(os.path.dirname(output_dir))
    total = 0
    for root, dirs, files in os.walk(sub_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.pyc'):
                continue
            total += os.path.getsize(os.path.join(root, f))
    print(f"  Total submission size: {total/1024/1024:.1f}MB")
    if total > 1024 * 1024 * 1024:
        print(f"  *** WARNING: exceeds 1GB limit! ***")
    else:
        print(f"  ✓ Under 1GB limit")

    return len(compressed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-dir', default='/tmp/v10_turn')
    parser.add_argument('--deploy-dir',
                        default=os.path.join(_project, 'submission', 'data', 'multi_street'))
    args = parser.parse_args()

    data = merge_turn_opp(args.board_dir)
    if data:
        compress_and_deploy(data, args.deploy_dir)
    print("Done.")
