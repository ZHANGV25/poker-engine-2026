"""
Discard inference: Bayesian opponent range narrowing.

When the opponent discards 3 cards (revealed to us), we infer which 2 cards
they likely kept. For each of the ~120 possible opponent hands, we ask:
"If the opponent originally had {kept1, kept2, disc1, disc2, disc3}, would
keeping {kept1, kept2} have been the rational choice?"

We use Boltzmann weighting: hands that would have been the best or near-best
keep get high weight; hands that would have been a terrible keep get near-zero.
This narrows the opponent's effective range from ~120 hands to ~20-40.

Keep-pair quality is evaluated using made-hand rank PLUS draw potential.
The 5-card rank alone misses flush draws and straight draws that have high
equity but rank as "high card." The draw bonus corrects this so inference
properly weights drawing hands the opponent would rationally keep.
"""

import itertools
import math


class DiscardInference:
    def __init__(self, equity_engine):
        self.engine = equity_engine

    @staticmethod
    def _draw_bonus(kept, board):
        """Estimate drawing potential from card patterns.

        The 5-card rank evaluates made hand strength only. Drawing hands
        (flush draws, straight draws) have high equity but rank as high-card.
        This bonus makes draws rank closer to their true equity value.

        27-card deck: 3 suits × 9 ranks (2-9, A).
        Card encoding: suit = c // 9, rank = c % 9.

        Returns a positive value (higher = stronger draw).
        """
        all_cards = kept + board  # 5 cards
        bonus = 0

        # --- Flush draw detection ---
        # 4 of same suit in 5 cards = 1 card short of flush.
        # With 9 cards per suit and 2 cards to come, ~51% to complete.
        suit_counts = [0, 0, 0]
        kept_suit_counts = [0, 0, 0]
        for c in all_cards:
            suit_counts[c // 9] += 1
        for c in kept:
            kept_suit_counts[c // 9] += 1

        for s in range(3):
            if suit_counts[s] >= 4 and kept_suit_counts[s] >= 1:
                # 4-to-flush with at least one kept card contributing
                bonus += 3000
                break

        # --- Straight draw detection ---
        # Ranks: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=A
        rank_set = set(c % 9 for c in all_cards)
        ranks_sorted = sorted(rank_set)

        # Find longest run of consecutive ranks
        max_run = 1
        run = 1
        for i in range(1, len(ranks_sorted)):
            if ranks_sorted[i] == ranks_sorted[i - 1] + 1:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 1

        # Ace-low wrap: treat A(8) as rank -1 for A-2-3-4-5 straights
        if 8 in rank_set:
            ace_low = sorted([-1] + [r for r in ranks_sorted if r != 8])
            run = 1
            for i in range(1, len(ace_low)):
                if ace_low[i] == ace_low[i - 1] + 1:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 1

        if max_run >= 4:
            bonus += 1500  # OESD (~35% to complete)
        elif max_run == 3:
            bonus += 400  # Some connectivity

        return bonus

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
        board_list = list(board)

        weights = {}

        for candidate_pair in itertools.combinations(remaining, 2):
            # Reconstruct opponent's hypothetical original 5-card hand
            original_5 = list(candidate_pair) + list(opp_discards)

            # Evaluate all 10 keep-pairs using rank + draw potential
            pair_scores = []
            candidate_score = None

            for i, j in [(a, b) for a in range(5) for b in range(a + 1, 5)]:
                kept = [original_5[i], original_5[j]]
                five_cards = kept + board_list

                # Made hand rank (lower = better)
                rank = self.engine.lookup_five(five_cards)

                # Draw bonus: flush draws and straight draws have high
                # equity but rank poorly. This correction makes the
                # inference properly weight drawing hands.
                bonus = self._draw_bonus(kept, board_list)
                score = rank - bonus  # lower = better

                pair_scores.append(score)

                if set(kept) == set(candidate_pair):
                    candidate_score = score

            if candidate_score is None:
                weights[tuple(sorted(candidate_pair))] = 0.0
                continue

            best_score = min(pair_scores)
            worst_score = max(pair_scores)
            spread = worst_score - best_score

            delta = candidate_score - best_score

            if spread <= 0:
                # All keeps are equal — uniform weight
                weight = 1.0
            else:
                # Normalize delta by the spread so it's in [0, 1]
                # Temperature of 3.0: a keep 1 full spread worse
                # gets weight exp(-3) ≈ 0.05 (very unlikely)
                normalized_delta = delta / spread
                weight = math.exp(-3.0 * normalized_delta)
            weights[tuple(sorted(candidate_pair))] = weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        return weights

