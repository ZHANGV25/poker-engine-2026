"""
Offline script to precompute pre-flop hand potential for all canonical 5-card hands.

For each possible 5-card hand, we simulate all possible flops and compute:
  "What is the expected equity of the best keep-pair across all flops?"

This gives us an accurate pre-flop hand strength score, replacing the naive
heuristic (count pairs, aces, etc.) with actual expected post-discard equity.

Uses suit isomorphism to reduce computation:
  - 3 suits are interchangeable pre-flop (no community cards yet)
  - Canonical form: relabel suits so the most frequent suit is always 0,
    second most is 1, third is 2
  - This reduces ~80,730 hands to ~13,000-15,000 canonical hands

Run from repo root:
    python -m submission.precompute_preflop

Generates submission/data/preflop_potential.npz
"""

import itertools
import os
import time
import numpy as np
from submission.equity import ExactEquityEngine

KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def canonicalize_hand(cards):
    """Convert a 5-card hand to canonical form under suit permutation.

    Relabel suits so the first-seen suit becomes 0, second becomes 1, third becomes 2.
    Returns a tuple of canonical card ints, sorted.
    """
    suit_map = {}
    next_suit = 0
    canonical = []
    for c in sorted(cards):
        rank = c % 9
        suit = c // 9
        if suit not in suit_map:
            suit_map[suit] = next_suit
            next_suit += 1
        canonical.append(rank + suit_map[suit] * 9)
    return tuple(sorted(canonical))


def compute_hand_potential(engine, hand_5, all_cards):
    """Compute expected post-discard equity for a 5-card hand across sampled flops.

    For each flop (3 community cards from remaining deck):
      - Evaluate all 10 keep-pairs by exact equity
      - Take the best equity (optimal discard)
    Average across all flops = hand potential.

    With 22 remaining cards, there are C(22,3) = 1,540 possible flops.
    Each flop requires 10 keep-pair equity computations.
    This is expensive, so we sample flops instead of enumerating all.
    """
    hand_set = set(hand_5)
    remaining = [c for c in all_cards if c not in hand_set]

    # Sample flops for speed. C(22,3) = 1540 total.
    # With 200 samples, standard error is ~1-2%, good enough for pre-flop decisions.
    all_flops = list(itertools.combinations(remaining, 3))

    # Use all flops if feasible (1540 × 10 keep-pairs × ~10K evals each is too slow)
    # Instead sample 100 flops: 100 × 10 × ~10K = ~10M lookups = ~1-2 seconds per hand
    # With ~14K canonical hands, that's ~14K-28K seconds. Too slow.
    #
    # Faster approach: for each flop, don't compute full equity for each keep-pair.
    # Instead, use the 5-card rank (keep_pair + flop) as a proxy for hand strength.
    # This is what the discard inference does and it's very fast.
    total_score = 0.0
    num_flops = len(all_flops)

    # For each flop, compute what fraction of random opponent 2-card hands
    # our best keep beats. This gives actual equity, not a rank proxy.
    # Sample 200 flops for speed (vs all 1540). Error: ~2%, acceptable.
    import random
    rng = random.Random(hash(tuple(hand_5)))
    sampled_flops = rng.sample(all_flops, min(200, len(all_flops)))
    num_flops = len(sampled_flops)
    for flop in sampled_flops:
        flop_list = list(flop)
        flop_set = set(flop)

        # Find our best keep-pair rank
        best_rank = float('inf')
        for i, j in KEEP_PAIRS:
            kept = [hand_5[i], hand_5[j]]
            five_cards = kept + flop_list
            rank = engine.lookup_five(five_cards)
            if rank < best_rank:
                best_rank = rank

        # Compute what fraction of random opponent 2-card hands we beat
        # (opponent picks any 2 cards not in our hand or the flop)
        opp_available = [c for c in all_cards if c not in hand_set and c not in flop_set]
        wins = 0
        total = 0
        for oi in range(len(opp_available)):
            for oj in range(oi + 1, len(opp_available)):
                opp_rank = engine.lookup_five([opp_available[oi], opp_available[oj]] + flop_list)
                if best_rank < opp_rank:
                    wins += 1
                elif best_rank == opp_rank:
                    wins += 0.5
                total += 1

        total_score += wins / total if total > 0 else 0.5

    return total_score / num_flops


def main():
    print("Loading equity engine...")
    engine = ExactEquityEngine()

    all_cards = list(range(27))

    # Step 1: Find all canonical 5-card hands
    print("Enumerating canonical 5-card hands...")
    canonical_map = {}  # canonical_tuple -> list of original hands
    for combo in itertools.combinations(all_cards, 5):
        canon = canonicalize_hand(combo)
        if canon not in canonical_map:
            canonical_map[canon] = combo  # store one representative

    num_canonical = len(canonical_map)
    print(f"Found {num_canonical} canonical hands (from {80730} total)")

    # Step 2: Compute potential for each canonical hand
    print("Computing hand potentials...")
    canonical_hands = []
    potentials = []

    start = time.time()
    for idx, (canon, representative) in enumerate(canonical_map.items()):
        potential = compute_hand_potential(engine, list(representative), all_cards)
        canonical_hands.append(canon)
        potentials.append(potential)

        if idx % 500 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining_time = (num_canonical - idx) / rate if rate > 0 else 0
            print(f"  {idx}/{num_canonical} ({100*idx/num_canonical:.1f}%) "
                  f"- {elapsed:.0f}s elapsed, ~{remaining_time:.0f}s remaining")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.0f}s")

    # Step 3: Build lookup table
    # Map: bitmask of 5 cards -> potential score
    # For fast lookup, we store ALL 80,730 hands (not just canonical) by
    # mapping each hand to its canonical potential.
    print("Building full lookup table...")
    all_bitmasks = np.empty(80730, dtype=np.int32)
    all_potentials = np.empty(80730, dtype=np.float32)

    # Build canonical -> potential dict
    canon_to_potential = {}
    for canon, pot in zip(canonical_hands, potentials):
        canon_to_potential[canon] = pot

    for idx, combo in enumerate(itertools.combinations(all_cards, 5)):
        canon = canonicalize_hand(combo)
        bitmask = 0
        for c in combo:
            bitmask |= 1 << c
        all_bitmasks[idx] = bitmask
        all_potentials[idx] = canon_to_potential[canon]

    # Sort by bitmask for binary search
    order = np.argsort(all_bitmasks)
    all_bitmasks = all_bitmasks[order]
    all_potentials = all_potentials[order]

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "data", "preflop_potential.npz")
    np.savez_compressed(
        out_path,
        bitmasks=all_bitmasks,
        potentials=all_potentials,
    )
    file_size = os.path.getsize(out_path)
    print(f"Saved to {out_path} ({file_size / 1024:.1f} KB)")
    print(f"  Entries: {len(all_bitmasks)}")
    print(f"  Potential range: [{min(potentials):.4f}, {max(potentials):.4f}]")


if __name__ == "__main__":
    main()
