"""
Exact equity engine using precomputed hand rank lookup tables.

Instead of Monte Carlo sampling (which has ~2.5% error with 400 samples),
we enumerate ALL possible opponent hands and board runouts exhaustively.
With a 27-card deck, the maximum enumeration is only ~10,920 combinations
(on the flop), which completes in ~20ms with dict lookups.

Performance note: Uses Python dicts for O(1) lookup (~50ns each) instead of
numpy searchsorted (~10μs each). For 218K lookups in the discard evaluation,
this is the difference between ~22ms and ~2.2 seconds.
"""

import os
import itertools
import numpy as np

# All 10 ways to choose 2 cards to keep from 5
KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


class ExactEquityEngine:
    def __init__(self):
        data_path = os.path.join(os.path.dirname(__file__), "data", "hand_ranks.npz")
        data = np.load(data_path)
        # Convert numpy arrays to Python dicts for O(1) lookup.
        # Memory: ~100MB for 888K entries. Fast: ~50ns per lookup vs ~10μs for searchsorted.
        seven_keys = data["seven_keys"]
        seven_vals = data["seven_vals"]
        self._seven = {}
        for i in range(len(seven_keys)):
            self._seven[int(seven_keys[i])] = int(seven_vals[i])

        five_keys = data["five_keys"]
        five_vals = data["five_vals"]
        self._five = {}
        for i in range(len(five_keys)):
            self._five[int(five_keys[i])] = int(five_vals[i])

    def lookup_seven(self, cards_7):
        """Look up the hand rank for exactly 7 cards. Lower = better."""
        mask = 0
        for c in cards_7:
            mask |= 1 << c
        return self._seven[mask]

    def lookup_five(self, cards_5):
        """Look up the hand rank for exactly 5 cards. Lower = better."""
        mask = 0
        for c in cards_5:
            mask |= 1 << c
        return self._five[mask]

    def compute_equity(self, my_cards, board, dead_cards, opp_weights=None):
        """
        Compute exact win probability by enumerating all possible outcomes.

        Args:
            my_cards: list of 2 card ints (our hole cards after discard)
            board: list of 0-5 card ints (visible community cards)
            dead_cards: list of card ints (our discards + opponent discards, removed from play)
            opp_weights: optional dict mapping (card1, card2) tuple -> float weight.
                         If provided, uses weighted equity (Bayesian inference from discards).
                         If None, all opponent hands weighted equally.

        Returns:
            float in [0, 1] representing win probability (ties count as 0.5)
        """
        known = set(my_cards) | set(board) | set(dead_cards)
        remaining = [c for c in range(27) if c not in known]
        board_needed = 5 - len(board)

        wins = 0.0
        total = 0.0

        # Precompute hero's board bitmask base (cards that are always in the 7-card hand)
        my_base = 0
        for c in my_cards:
            my_base |= 1 << c
        board_base = 0
        for c in board:
            board_base |= 1 << c

        seven_lookup = self._seven  # local reference for speed

        if board_needed == 0:
            # River: board is complete, hero rank is fixed
            my_mask = my_base | board_base
            my_rank = seven_lookup[my_mask]

            for opp_pair in itertools.combinations(remaining, 2):
                if opp_weights is not None:
                    key = (opp_pair[0], opp_pair[1]) if opp_pair[0] < opp_pair[1] else (opp_pair[1], opp_pair[0])
                    w = opp_weights.get(key, 0.0)
                    if w <= 0.0:
                        continue
                else:
                    w = 1.0

                opp_mask = (1 << opp_pair[0]) | (1 << opp_pair[1]) | board_base
                opp_rank = seven_lookup[opp_mask]

                if my_rank < opp_rank:
                    wins += w
                elif my_rank == opp_rank:
                    wins += 0.5 * w
                total += w

        elif board_needed == 1:
            # Turn: need 1 more card
            for opp_pair in itertools.combinations(remaining, 2):
                if opp_weights is not None:
                    key = (opp_pair[0], opp_pair[1]) if opp_pair[0] < opp_pair[1] else (opp_pair[1], opp_pair[0])
                    w = opp_weights.get(key, 0.0)
                    if w <= 0.0:
                        continue
                else:
                    w = 1.0

                opp_base = (1 << opp_pair[0]) | (1 << opp_pair[1]) | board_base
                opp0, opp1 = opp_pair

                for c in remaining:
                    if c == opp0 or c == opp1:
                        continue
                    runout_bit = 1 << c
                    my_rank = seven_lookup[my_base | board_base | runout_bit]
                    opp_rank = seven_lookup[opp_base | runout_bit]

                    if my_rank < opp_rank:
                        wins += w
                    elif my_rank == opp_rank:
                        wins += 0.5 * w
                    total += w

        else:
            # Flop: need 2 more cards (turn + river)
            remaining_set = set(remaining)

            for opp_pair in itertools.combinations(remaining, 2):
                if opp_weights is not None:
                    key = (opp_pair[0], opp_pair[1]) if opp_pair[0] < opp_pair[1] else (opp_pair[1], opp_pair[0])
                    w = opp_weights.get(key, 0.0)
                    if w <= 0.0:
                        continue
                else:
                    w = 1.0

                opp_base = (1 << opp_pair[0]) | (1 << opp_pair[1]) | board_base
                # Cards available for runout: remaining minus opp's 2 cards
                runout_cards = [c for c in remaining if c != opp_pair[0] and c != opp_pair[1]]

                for ri in range(len(runout_cards)):
                    r0 = runout_cards[ri]
                    bit0 = 1 << r0
                    for rj in range(ri + 1, len(runout_cards)):
                        r1 = runout_cards[rj]
                        bit1 = bit0 | (1 << r1)
                        my_rank = seven_lookup[my_base | board_base | bit1]
                        opp_rank = seven_lookup[opp_base | bit1]

                        if my_rank < opp_rank:
                            wins += w
                        elif my_rank == opp_rank:
                            wins += 0.5 * w
                        total += w

        return wins / total if total > 0 else 0.5

    def evaluate_all_keep_pairs(self, my_5_cards, board, dead_cards, opp_weights=None):
        """
        For the discard decision: evaluate all 10 possible keep-pairs from 5 hole cards.

        Args:
            my_5_cards: list of 5 card ints (our hole cards before discard)
            board: list of 3 card ints (flop community cards)
            dead_cards: list of card ints (opponent's discards if known, else empty)
            opp_weights: optional Bayesian weights for opponent range

        Returns:
            list of ((keep_idx_1, keep_idx_2), equity) sorted by equity descending
        """
        results = []
        for i, j in KEEP_PAIRS:
            kept = [my_5_cards[i], my_5_cards[j]]
            discarded = [my_5_cards[k] for k in range(5) if k != i and k != j]
            all_dead = list(dead_cards) + discarded
            equity = self.compute_equity(kept, board, all_dead, opp_weights)
            results.append(((i, j), equity))

        results.sort(key=lambda x: -x[1])
        return results
