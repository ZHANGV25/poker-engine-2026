"""
Precomputed river strategy + P(bet|hand) lookup.

Lazy-loads per-board .npz files on demand (like turn boards).
Zero startup cost, ~2ms per board lookup, cached after first load.

Provides:
1. Acting-first strategy lookup (replaces runtime solver, 0ms after load)
2. P(bet|hand) for Bayesian bet-narrowing (replaces heuristic polarized model)

When precomputed data is unavailable for a board, returns None → runtime fallback.
"""

import os
import numpy as np

# Pot sizes matching the precompute
_POT_SIZES = [(4, 4), (16, 16), (40, 40)]


class RiverLookup:
    def __init__(self, data_dir=None):
        """Initialize river lookup.

        Args:
            data_dir: path to directory with river_*.npz files.
                      Files are lazy-loaded on demand, not at startup.
        """
        self._data_dir = data_dir
        self._cache = {}  # board_key -> parsed board data
        self._loaded = data_dir is not None and os.path.isdir(data_dir)
        self._file_count = 0

        if self._loaded:
            # Count files but don't load them
            try:
                self._file_count = sum(1 for f in os.listdir(data_dir)
                                       if f.startswith('river_') and f.endswith('.npz'))
            except Exception:
                self._file_count = 0
                self._loaded = False

    @property
    def loaded(self):
        return self._loaded and self._file_count > 0

    def _board_key(self, board):
        return tuple(sorted(int(c) for c in board))

    def _find_file(self, board):
        """Find the .npz file for a given board.

        Files are named river_NNNNN.npz where NNNNN is the board index
        in the enumeration of all C(27,5) combinations.
        Uses combinatorial number system for O(1) index computation.
        """
        board_sorted = sorted(int(c) for c in board)

        # Compute combinatorial index: the position of this combination
        # in lexicographic order of itertools.combinations(range(27), 5)
        from math import comb
        idx = 0
        for k, card in enumerate(board_sorted):
            # Count combinations that come before this one
            # at position k, with card value 'card'
            start = board_sorted[k - 1] + 1 if k > 0 else 0
            for v in range(start, card):
                idx += comb(26 - v, 4 - k)

        fpath = os.path.join(self._data_dir, f'river_{idx}.npz')
        if os.path.isfile(fpath):
            return fpath
        return None

    def _lazy_load(self, board):
        """Load a single board's data on demand. Cached after first load."""
        key = self._board_key(board)

        if key in self._cache:
            return self._cache[key]

        if not self._loaded:
            return None

        fpath = self._find_file(board)
        if fpath is None or not os.path.isfile(fpath):
            return None

        try:
            data = np.load(fpath, allow_pickle=True)

            # Parse hands
            hands = data['hands']
            hand_map = {}
            for hi in range(len(hands)):
                h = (int(hands[hi][0]), int(hands[hi][1]))
                h_sorted = (min(h), max(h))
                hand_map[h_sorted] = hi

            board_data = {'hand_map': hand_map, 'n_hands': len(hands)}

            for pi in range(len(_POT_SIZES)):
                s_key = f's_p{pi}'
                pb_key = f'pb_p{pi}'
                a_key = f'a_p{pi}'
                if s_key in data.files:
                    board_data[f'strat_{pi}'] = data[s_key]
                    board_data[f'pbet_{pi}'] = data[pb_key]
                    board_data[f'acts_{pi}'] = data[a_key]

            # Cache with LRU eviction
            if len(self._cache) > 200:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = board_data
            return board_data

        except Exception:
            return None

    def _find_pot_idx(self, my_bet, opp_bet):
        pot = my_bet + opp_bet
        best_idx = 0
        best_dist = float('inf')
        for pi, (hb, ob) in enumerate(_POT_SIZES):
            dist = abs(hb + ob - pot)
            if dist < best_dist:
                best_dist = dist
                best_idx = pi
        return best_idx

    def get_strategy(self, hero_cards, board, my_bet, opp_bet):
        """Look up precomputed acting-first strategy.

        Returns dict {action_type: probability} or None.
        """
        bd = self._lazy_load(board)
        if bd is None:
            return None

        hand = (min(hero_cards[0], hero_cards[1]),
                max(hero_cards[0], hero_cards[1]))
        hi = bd['hand_map'].get(hand)
        if hi is None:
            return None

        pi = self._find_pot_idx(my_bet, opp_bet)
        strat_key = f'strat_{pi}'
        acts_key = f'acts_{pi}'

        if strat_key not in bd:
            return None

        strat = bd[strat_key][hi].astype(np.float64) / 255.0
        acts = bd[acts_key]

        total = strat.sum()
        if total > 0:
            strat /= total
        else:
            strat = np.ones(len(strat)) / len(strat)

        result = {}
        for ai in range(len(acts)):
            act = int(acts[ai])
            if act >= 0:
                result[act] = float(strat[ai])

        return result

    def get_p_bet(self, board, pot_idx=None, my_bet=0, opp_bet=0):
        """Get P(bet|hand) for all hands on this board.

        Returns dict {(c1,c2): p_bet} or None.
        """
        bd = self._lazy_load(board)
        if bd is None:
            return None

        if pot_idx is None:
            pot_idx = self._find_pot_idx(my_bet, opp_bet)

        pbet_key = f'pbet_{pot_idx}'
        if pbet_key not in bd:
            return None

        pb = bd[pbet_key].astype(np.float64) / 255.0

        result = {}
        for hand, hi in bd['hand_map'].items():
            result[hand] = float(pb[hi])

        return result
