"""
Offline script to generate hand rank lookup tables.

Run from the repo root:
    python -m submission.precompute

Generates submission/data/hand_ranks.npz containing:
  - seven_keys: sorted int32 array of 27-bit bitmasks for all C(27,7) = 888,030 seven-card combos
  - seven_vals: int16 array of hand ranks (lower = better) aligned with seven_keys
  - five_keys: sorted int32 array of bitmasks for all C(27,5) = 80,730 five-card combos
  - five_vals: int16 array of hand ranks aligned with five_keys

Lookup: numpy.searchsorted(keys, bitmask) -> index -> vals[index]
"""

import itertools
import os
import numpy as np
from gym_env import PokerEnv, WrappedEval


def cards_to_bitmask(cards):
    """Convert a tuple/list of card ints (0-26) to a 27-bit integer bitmask."""
    mask = 0
    for c in cards:
        mask |= (1 << c)
    return mask


def precompute_seven_card_ranks(evaluator):
    """Compute hand rank for every possible 7-card combination from the 27-card deck."""
    all_cards = list(range(27))
    total = 888030  # C(27,7)
    keys = np.empty(total, dtype=np.int32)
    vals = np.empty(total, dtype=np.int16)

    for idx, combo in enumerate(itertools.combinations(all_cards, 7)):
        # treys evaluate expects (hand=2 cards, board=5 cards)
        # The evaluator internally finds the best 5 of 7, so the split is arbitrary
        hand = [PokerEnv.int_to_card(combo[0]), PokerEnv.int_to_card(combo[1])]
        board = [PokerEnv.int_to_card(c) for c in combo[2:]]
        rank = evaluator.evaluate(hand, board)

        keys[idx] = cards_to_bitmask(combo)
        vals[idx] = rank

        if idx % 100000 == 0:
            print(f"  7-card: {idx}/{total} ({100*idx/total:.1f}%)")

    # Sort by key for binary search lookup
    order = np.argsort(keys)
    keys = keys[order]
    vals = vals[order]
    print(f"  7-card: {total}/{total} (100.0%) - done")
    return keys, vals


def precompute_five_card_ranks(evaluator):
    """Compute hand rank for every possible 5-card combination from the 27-card deck.

    Used for fast discard inference heuristic (evaluate keep_pair + 3-card board).
    We split the 5 cards as hand=first 2, board=last 3. Since there are only 5 cards
    total, the evaluator returns the rank of exactly those 5 cards.
    """
    all_cards = list(range(27))
    total = 80730  # C(27,5)
    keys = np.empty(total, dtype=np.int32)
    vals = np.empty(total, dtype=np.int16)

    for idx, combo in enumerate(itertools.combinations(all_cards, 5)):
        hand = [PokerEnv.int_to_card(combo[0]), PokerEnv.int_to_card(combo[1])]
        board = [PokerEnv.int_to_card(c) for c in combo[2:]]
        rank = evaluator.evaluate(hand, board)

        keys[idx] = cards_to_bitmask(combo)
        vals[idx] = rank

        if idx % 20000 == 0:
            print(f"  5-card: {idx}/{total} ({100*idx/total:.1f}%)")

    order = np.argsort(keys)
    keys = keys[order]
    vals = vals[order]
    print(f"  5-card: {total}/{total} (100.0%) - done")
    return keys, vals


def main():
    print("Initializing evaluator...")
    evaluator = WrappedEval()

    print("Computing 7-card hand ranks (C(27,7) = 888,030)...")
    seven_keys, seven_vals = precompute_seven_card_ranks(evaluator)

    print("Computing 5-card hand ranks (C(27,5) = 80,730)...")
    five_keys, five_vals = precompute_five_card_ranks(evaluator)

    out_path = os.path.join(os.path.dirname(__file__), "data", "hand_ranks.npz")
    np.savez_compressed(
        out_path,
        seven_keys=seven_keys,
        seven_vals=seven_vals,
        five_keys=five_keys,
        five_vals=five_vals,
    )
    file_size = os.path.getsize(out_path)
    print(f"Saved to {out_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  7-card entries: {len(seven_keys)}")
    print(f"  5-card entries: {len(five_keys)}")


if __name__ == "__main__":
    main()
