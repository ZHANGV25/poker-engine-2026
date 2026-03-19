#!/usr/bin/env python3
"""
Extract turn HERO strategies from per-board files.
Memory-efficient: one board at a time, 4-bit quantize.
Output: turn_hero.pkl.lzma for blueprint turn decisions.
"""
import os, sys, time, glob, lzma, pickle
import numpy as np

def extract_hero_strats(board_path):
    """Extract hero turn strategies from one board. Returns (bid, data) or None."""
    d = np.load(board_path, allow_pickle=True)
    if 'turn_cards' not in d:
        return None

    bid = int(d['board_id'])
    board = tuple(d['board'].tolist())
    turn_cards = d['turn_cards'].tolist()
    pot_sizes = d['pot_sizes'].tolist()

    result = {'board': board, 'pot_sizes': pot_sizes, 'turn_cards': turn_cards}

    for pi in range(len(pot_sizes)):
        for tc in turn_cards:
            strat_key = f'turn_strat_p{pi}_t{tc}'
            act_key = f'turn_actions_p{pi}_t{tc}'
            hands_key = f'turn_hands_t{tc}'

            if strat_key not in d or act_key not in d:
                continue

            strat = d[strat_key]  # uint8 (n_hands, n_nodes, n_actions)
            # 4-bit quantize: >> 4 (loses bottom 4 bits)
            strat_4bit = (strat >> 4).astype(np.uint8)
            result[f's_{pi}_{tc}'] = strat_4bit
            result[f'a_{pi}_{tc}'] = d[act_key]  # int8 action types

            if hands_key in d and f'h_{tc}' not in result:
                result[f'h_{tc}'] = d[hands_key]

    return bid, result


def main():
    board_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/v10_turn'
    deploy_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'submission', 'data', 'multi_street')

    files = sorted(glob.glob(os.path.join(board_dir, 'board_*.npz')))
    print(f"Processing {len(files)} boards...")

    all_data = {}
    t0 = time.time()
    errors = 0

    for i, fpath in enumerate(files):
        try:
            result = extract_hero_strats(fpath)
            if result is not None:
                bid, data = result
                all_data[bid] = data
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  Error: {fpath}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)")

    print(f"  Extracted {len(all_data)} boards in {time.time()-t0:.1f}s ({errors} errors)")

    print("Compressing...")
    t0 = time.time()
    pkl = pickle.dumps(all_data)
    print(f"  Pickled: {len(pkl)/1024/1024:.1f}MB")
    compressed = lzma.compress(pkl)
    print(f"  LZMA: {len(compressed)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    os.makedirs(deploy_dir, exist_ok=True)
    out_path = os.path.join(deploy_dir, 'turn_hero.pkl.lzma')
    with open(out_path, 'wb') as f:
        f.write(compressed)
    print(f"  Deployed: {out_path}")

    sub_dir = os.path.dirname(deploy_dir)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(sub_dir)
                for f in fs if not f.endswith('.pyc') and '__pycache__' not in r)
    print(f"  Total submission: {total/1024/1024:.1f}MB (limit: 1024MB)")
    print("Done.")

if __name__ == '__main__':
    main()
