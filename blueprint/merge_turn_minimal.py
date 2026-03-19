#!/usr/bin/env python3
"""
Extract minimal turn narrowing data: P(bet|hand) per board/pot/turn_card.

Memory-efficient: processes one board at a time, never holds full strategies.
Output: ~21MB LZMA, contains just the numbers needed for Bayesian narrowing.
"""
import os, sys, time, glob, lzma, pickle
import numpy as np

def extract_p_bet(board_path):
    """Extract P(bet|hand) from one board file. Returns dict or None."""
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
            opp_key = f'turn_opp_strat_p{pi}_t{tc}'
            act_key = f'turn_opp_actions_p{pi}_t{tc}'
            hands_key = f'turn_hands_t{tc}'

            if opp_key not in d or act_key not in d:
                continue

            opp_strat = d[opp_key]    # (n_hands, n_nodes, n_actions)
            act_types = d[act_key]     # (n_nodes, n_actions)

            # Extract P(bet|hand) at root node (node 0)
            if opp_strat.ndim < 3 or act_types.ndim < 2:
                continue

            node_strat = opp_strat[:, 0, :]  # (n_hands, n_actions) at root
            node_acts = act_types[0]           # (n_actions,) at root

            # Sum probability of all raise actions
            p_bet = np.zeros(node_strat.shape[0], dtype=np.float32)
            for a_idx in range(len(node_acts)):
                act = int(node_acts[a_idx])
                if act in (3, 4, 5, 6):  # ACT_RAISE_*
                    p_bet += node_strat[:, a_idx].astype(np.float32) / 255.0

            # Quantize to uint8 (0-255)
            p_bet_q = np.clip(np.round(p_bet * 255), 0, 255).astype(np.uint8)
            result[f'pb_{pi}_{tc}'] = p_bet_q

            # Store hands mapping (once per turn card)
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
            result = extract_p_bet(fpath)
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

    # Compress
    print("Compressing...")
    t0 = time.time()
    pkl = pickle.dumps(all_data)
    print(f"  Pickled: {len(pkl)/1024/1024:.1f}MB")

    compressed = lzma.compress(pkl)
    print(f"  LZMA: {len(compressed)/1024/1024:.1f}MB ({time.time()-t0:.1f}s)")

    # Deploy
    os.makedirs(deploy_dir, exist_ok=True)
    out_path = os.path.join(deploy_dir, 'turn_opp.pkl.lzma')
    with open(out_path, 'wb') as f:
        f.write(compressed)
    print(f"  Deployed: {out_path}")

    # Total size check
    sub_dir = os.path.dirname(deploy_dir)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(sub_dir)
                for f in fs if not f.endswith('.pyc') and '__pycache__' not in r)
    print(f"  Total submission: {total/1024/1024:.1f}MB (limit: 1024MB)")
    print("Done.")


if __name__ == '__main__':
    main()
