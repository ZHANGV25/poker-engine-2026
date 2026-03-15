"""
Discard inference: Bayesian opponent range narrowing.

When the opponent discards 3 cards (revealed to us), we infer which 2 cards
they likely kept. For each of the ~120 possible opponent hands, we ask:
"If the opponent originally had {kept1, kept2, disc1, disc2, disc3}, would
keeping {kept1, kept2} have been the rational choice?"

We use Boltzmann weighting: hands that would have been the best or near-best
keep get high weight; hands that would have been a terrible keep get near-zero.
This narrows the opponent's effective range from ~120 hands to ~20-40.
"""

import itertools
import math


class DiscardInference:
    def __init__(self, equity_engine):
        self.engine = equity_engine

    def infer_opponent_weights(self, opp_discards, board, my_cards):
        """
        Compute probability distribution over opponent's possible kept hands.

        Args:
            opp_discards: list of 3 card ints (opponent's revealed discards)
            board: list of 3 card ints (flop community cards)
            my_cards: list of card ints (our cards - 5 if we haven't discarded, 2 if we have)

        Returns:
            dict mapping (card1, card2) tuple (sorted) -> float weight (normalized to sum=1)
        """
        known = set(my_cards) | set(board) | set(opp_discards)
        remaining = [c for c in range(27) if c not in known]

        weights = {}

        for candidate_pair in itertools.combinations(remaining, 2):
            # Reconstruct opponent's hypothetical original 5-card hand
            original_5 = list(candidate_pair) + list(opp_discards)

            # Evaluate all 10 keep-pairs from this 5-card hand using 5-card ranks
            # (quick heuristic: rank of keep_pair + board as a 5-card hand)
            pair_ranks = []
            candidate_rank = None

            for i, j in [(a, b) for a in range(5) for b in range(a + 1, 5)]:
                kept = [original_5[i], original_5[j]]
                five_cards = kept + list(board)
                rank = self.engine.lookup_five(five_cards)
                pair_ranks.append(rank)

                if set(kept) == set(candidate_pair):
                    candidate_rank = rank

            if candidate_rank is None:
                # Shouldn't happen, but safety check
                weights[tuple(sorted(candidate_pair))] = 0.0
                continue

            best_rank = min(pair_ranks)  # lower rank = better hand

            # Boltzmann weight: higher weight for keeps close to optimal.
            # Temperature is derived per-hand from the SPREAD of keep-pair
            # ranks. If all 10 keeps have similar rank (spread is small),
            # any keep is reasonable → high temperature → flat weights.
            # If one keep dominates (spread is large), only the best keep
            # is rational → low temperature → peaked weights.
            #
            # This replaces the old arbitrary temperature=5.0 and ÷500.
            worst_rank = max(pair_ranks)
            spread = worst_rank - best_rank  # how different the keeps are

            delta = candidate_rank - best_rank

            if spread <= 0:
                # All keeps are equal — uniform weight
                weight = 1.0
            else:
                # Normalize delta by the spread so it's in [0, 1]
                # Then apply a fixed temperature to control sharpness
                # Temperature of 3.0 means: a keep that's 1 full spread
                # worse gets weight exp(-3) ≈ 0.05 (very unlikely)
                normalized_delta = delta / spread
                weight = math.exp(-3.0 * normalized_delta)
            weights[tuple(sorted(candidate_pair))] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

