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
        # Temperature controls how "rational" we assume the opponent is.
        # Low temperature = assume perfectly rational (only best keep has weight)
        # High temperature = assume more random (many keeps have weight)
        # Can be calibrated from showdown data during a match.
        self.temperature = 5.0

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

            # Boltzmann weight: higher weight for keeps that are close to optimal
            # delta is how much WORSE this keep is compared to the best keep
            # (positive delta = worse, since lower rank = better)
            delta = candidate_rank - best_rank

            # Normalize delta to a reasonable scale (ranks can range 0-7462 in treys)
            # Divide by a scaling factor so temperature has intuitive meaning
            normalized_delta = delta / 500.0

            weight = math.exp(-normalized_delta * self.temperature)
            weights[tuple(sorted(candidate_pair))] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

    def calibrate_temperature(self, showdown_data):
        """
        Adjust temperature based on observed showdown hands.

        If opponent frequently keeps the optimal pair, lower temperature (more rational).
        If opponent keeps suboptimal pairs often, raise temperature (more random).

        Args:
            showdown_data: list of dicts with 'opp_kept', 'opp_discards', 'board'
        """
        if len(showdown_data) < 10:
            return  # not enough data

        optimal_count = 0
        for sd in showdown_data:
            opp_kept = sd["opp_kept"]
            opp_discards = sd["opp_discards"]
            board = sd["board"]
            original_5 = list(opp_kept) + list(opp_discards)

            # Find the best keep-pair
            best_rank = float("inf")
            kept_rank = None
            for i, j in [(a, b) for a in range(5) for b in range(a + 1, 5)]:
                pair = [original_5[i], original_5[j]]
                rank = self.engine.lookup_five(pair + list(board))
                if rank < best_rank:
                    best_rank = rank
                if set(pair) == set(opp_kept):
                    kept_rank = rank

            if kept_rank is not None and kept_rank == best_rank:
                optimal_count += 1

        optimal_rate = optimal_count / len(showdown_data)

        # If opponent keeps optimal pair >80% of the time, they're very rational
        if optimal_rate > 0.8:
            self.temperature = max(2.0, self.temperature - 1.0)
        # If <50%, they're more random
        elif optimal_rate < 0.5:
            self.temperature = min(15.0, self.temperature + 2.0)
