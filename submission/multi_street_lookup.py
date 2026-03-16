"""
Runtime lookup for multi-street precomputed blueprint strategies.

Loads per-board .npz files (or a merged file) produced by
compute_multi_street.py and provides fast strategy lookups at game time.

Lookup flow:
    1. Find nearest board (exact match by board cards, or feature-distance)
    2. Find hero hand index (direct card-pair lookup, O(1))
    3. Find nearest pot size
    4. Find correct tree node for current bet state
    5. Return strategy as {action_type: probability}

Lookup latency: < 5ms (no equity computation, no tree building at runtime).
"""

import os
import sys
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

ACTION_NAMES = {
    ACT_FOLD: 'fold',
    ACT_CHECK: 'check',
    ACT_CALL: 'call',
    ACT_RAISE_HALF: 'raise_40pct',
    ACT_RAISE_POT: 'raise_70pct',
    ACT_RAISE_ALLIN: 'raise_100pct',
    ACT_RAISE_OVERBET: 'raise_150pct',
}


class MultiStreetLookup:
    """
    Fast runtime lookup for multi-street blueprint strategies.

    Loads strategy data from per-board .npz files or a merged file.
    Each board has strategies indexed by (pot_size, hand, node, action).

    Usage:
        lookup = MultiStreetLookup("output_multi_street/")
        strategy = lookup.get_strategy(
            hero_cards=[0, 8], board=[1, 4, 7], pot_state=(2, 2))
        # strategy = {ACT_CHECK: 0.6, ACT_RAISE_HALF: 0.4}
    """

    def __init__(self, data_path, equity_engine=None):
        """
        Load multi-street blueprint data.

        Args:
            data_path: path to directory with board_*.npz files, or path
                       to a single merged .npz file
            equity_engine: optional ExactEquityEngine for fallback hand
                          matching (when hero cards overlap with stored board)
        """
        self._equity_engine = equity_engine
        self._boards = {}       # board_id -> board data dict
        self._board_list = []   # [(board_id, board_tuple, features)]
        self._hand_maps = {}    # board_id -> {(c1,c2): hand_idx}
        self._node_maps = {}    # (board_id, pot_idx) -> node_info list
        self._pot_sizes = None
        self._loaded = False

        if os.path.isdir(data_path):
            self._load_directory(data_path)
        elif os.path.isfile(data_path):
            self._load_merged_file(data_path)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

    def _load_directory(self, dir_path):
        """Load per-board .npz files from a directory."""
        import glob
        files = sorted(glob.glob(os.path.join(dir_path, "board_*.npz")))

        if not files:
            raise FileNotFoundError(
                f"No board_*.npz files found in {dir_path}")

        for fpath in files:
            try:
                data = np.load(fpath, allow_pickle=True)
                board_id = int(data['board_id'])
                board = tuple(int(c) for c in data['board'])
                hands = data['hands']
                features = data['board_features']
                pot_sizes = data['pot_sizes']

                # Load turn strategies if available
                turn_data = {}
                if 'turn_cards' in data:
                    turn_cards = data['turn_cards']
                    for tc in turn_cards:
                        tc = int(tc)
                        t_hands_key = f'turn_hands_t{tc}'
                        if t_hands_key in data:
                            t_hands = data[t_hands_key]
                            # Build hand map for this turn card
                            t_hmap = {}
                            for hi in range(len(t_hands)):
                                c1, c2 = int(t_hands[hi][0]), int(t_hands[hi][1])
                                t_hmap[(min(c1, c2), max(c1, c2))] = hi
                            turn_data[tc] = {
                                'hands': t_hands,
                                'hand_map': t_hmap,
                            }
                            # Load per-pot strategies and actions
                            for pi in range(len(pot_sizes)):
                                s_key = f'turn_strat_p{pi}_t{tc}'
                                a_key = f'turn_actions_p{pi}_t{tc}'
                                if s_key in data:
                                    if 'pot_strategies' not in turn_data[tc]:
                                        turn_data[tc]['pot_strategies'] = {}
                                        turn_data[tc]['pot_actions'] = {}
                                    turn_data[tc]['pot_strategies'][pi] = data[s_key]
                                    if a_key in data:
                                        turn_data[tc]['pot_actions'][pi] = data[a_key]

                self._boards[board_id] = {
                    'strategies': data['flop_strategies'],
                    'action_types': data['action_types'],
                    'hands': hands,
                    'board': board,
                    'features': features,
                    'pot_sizes': pot_sizes,
                    'turn_data': turn_data,
                }

                self._board_list.append((board_id, board, features))

                # Build hand lookup map
                hmap = {}
                for hi in range(len(hands)):
                    c1, c2 = int(hands[hi][0]), int(hands[hi][1])
                    key = (min(c1, c2), max(c1, c2))
                    hmap[key] = hi
                self._hand_maps[board_id] = hmap

                if self._pot_sizes is None:
                    self._pot_sizes = pot_sizes

            except Exception as e:
                continue

        # Build board lookup index (board tuple -> board_id)
        self._board_to_id = {}
        for bid, board, _ in self._board_list:
            self._board_to_id[board] = bid

        # Build node maps for each (board, pot_idx)
        self._build_all_node_maps()
        self._loaded = True

    def _load_merged_file(self, fpath):
        """Load a single merged .npz file."""
        data = np.load(fpath, allow_pickle=True)

        # Expected format: arrays of board_ids, boards, strategies, etc.
        board_ids = data['board_ids']
        boards = data['boards']
        features = data['board_features']
        pot_sizes = data['pot_sizes']
        self._pot_sizes = pot_sizes

        # Each board's data
        all_strategies = data['all_strategies']   # object array
        all_action_types = data['all_action_types']
        all_hands = data['all_hands']

        for i in range(len(board_ids)):
            bid = int(board_ids[i])
            board = tuple(int(c) for c in boards[i])

            self._boards[bid] = {
                'strategies': all_strategies[i],
                'action_types': all_action_types[i],
                'hands': all_hands[i],
                'board': board,
                'features': features[i],
                'pot_sizes': pot_sizes,
            }

            self._board_list.append((bid, board, features[i]))

            hmap = {}
            hands = all_hands[i]
            for hi in range(len(hands)):
                c1, c2 = int(hands[hi][0]), int(hands[hi][1])
                key = (min(c1, c2), max(c1, c2))
                hmap[key] = hi
            self._hand_maps[bid] = hmap

        self._board_to_id = {}
        for bid, board, _ in self._board_list:
            self._board_to_id[board] = bid

        self._build_all_node_maps()
        self._loaded = True

    def _build_all_node_maps(self):
        """Build game tree node maps for bet-state matching."""
        if self._pot_sizes is None:
            return

        for bid in self._boards:
            bd = self._boards[bid]
            ps = bd['pot_sizes']
            for pi in range(len(ps)):
                hb, ob = int(ps[pi][0]), int(ps[pi][1])
                try:
                    tree = GameTree(hb, ob, 2, 100, True)
                    node_info = []
                    for i, nid in enumerate(tree.hero_node_ids):
                        acts = [a for a, _ in tree.children[nid]]
                        node_info.append((
                            i,
                            tree.hero_pot[nid],
                            tree.opp_pot[nid],
                            ACT_FOLD in acts,
                        ))
                    self._node_maps[(bid, pi)] = node_info
                except Exception:
                    self._node_maps[(bid, pi)] = [(0, hb, ob, False)]

    @property
    def n_boards(self):
        return len(self._boards)

    @property
    def is_loaded(self):
        return self._loaded and len(self._boards) > 0

    # ------------------------------------------------------------------
    # Main lookup
    # ------------------------------------------------------------------

    def get_strategy(self, hero_cards, board, pot_state=None,
                     dead_cards=None, opp_weights=None):
        """
        Look up the blueprint strategy for the current game state.

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints (uses first 3 for flop matching)
            pot_state: (hero_bet, opp_bet) tuple
            dead_cards: optional dead cards
            opp_weights: unused (kept for API compatibility)

        Returns:
            dict {action_type_id: probability}, or None if unavailable
        """
        if not self._loaded:
            return None

        # Find nearest board (use flop cards only)
        flop = tuple(sorted(int(c) for c in board[:3]))
        board_id = self._find_board(flop)
        if board_id is None:
            return None

        bd = self._boards[board_id]

        # Find hand index
        hand_idx = self._find_hand(hero_cards, board_id)
        if hand_idx is None:
            return None

        # Find pot index
        pot_idx = self._find_pot(pot_state, bd['pot_sizes'])

        # Find node index for current bet state
        my_bet = pot_state[0] if pot_state else 2
        opp_bet = pot_state[1] if pot_state else 2
        node_idx = self._find_node(my_bet, opp_bet, board_id, pot_idx)

        # Read strategy
        strats = bd['strategies']
        acts = bd['action_types']

        if pot_idx >= strats.shape[0]:
            pot_idx = 0
        if hand_idx >= strats.shape[1]:
            return None
        if node_idx >= strats.shape[2]:
            node_idx = 0

        raw = strats[pot_idx, hand_idx, node_idx, :]
        act_ids = acts[pot_idx, node_idx, :]

        # Convert uint8 -> float
        probs = raw.astype(np.float64) / 255.0

        result = {}
        total = 0.0
        for a in range(len(probs)):
            if act_ids[a] >= 0 and probs[a] > 0:
                result[int(act_ids[a])] = float(probs[a])
                total += probs[a]

        if not result:
            return None

        # Normalize
        if abs(total - 1.0) > 1e-6 and total > 0:
            for k in result:
                result[k] /= total

        return result

    def get_turn_strategy(self, hero_cards, board, pot_state=None,
                           dead_cards=None, opp_weights=None):
        """Look up turn strategy from multi-street solve.

        Args:
            hero_cards: list of 2 card ints
            board: list of 4+ card ints (flop + turn card)
            pot_state: (hero_bet, opp_bet)

        Returns:
            dict {action_type_id: probability}, or None
        """
        if not self._loaded or len(board) < 4:
            return None

        flop = tuple(sorted(int(c) for c in board[:3]))
        board_id = self._find_board(flop)
        if board_id is None:
            return None

        bd = self._boards[board_id]
        turn_data = bd.get('turn_data', {})
        turn_card = int(board[3])

        if turn_card not in turn_data:
            return None

        td = turn_data[turn_card]
        if 'pot_strategies' not in td:
            return None

        # Find hand index in turn hand list
        c1, c2 = int(hero_cards[0]), int(hero_cards[1])
        key = (min(c1, c2), max(c1, c2))
        hand_idx = td['hand_map'].get(key)
        if hand_idx is None:
            return None

        # Find pot index
        pot_idx = self._find_pot(pot_state, bd['pot_sizes'])
        if pot_idx not in td['pot_strategies']:
            pot_idx = min(td['pot_strategies'].keys()) if td['pot_strategies'] else 0

        strats = td['pot_strategies'].get(pot_idx)
        acts = td['pot_actions'].get(pot_idx)
        if strats is None:
            return None

        # Find node for bet state
        my_bet = pot_state[0] if pot_state else 2
        opp_bet = pot_state[1] if pot_state else 2
        # Use flop node matching (same tree structure)
        node_idx = self._find_node(my_bet, opp_bet, board_id, pot_idx)

        if hand_idx >= strats.shape[0] or node_idx >= strats.shape[1]:
            return None

        raw = strats[hand_idx, node_idx, :]
        probs = raw.astype(np.float64) / 255.0

        if acts is not None and node_idx < acts.shape[0]:
            act_ids = acts[node_idx, :]
        else:
            return None

        result = {}
        total = 0.0
        for a in range(len(probs)):
            if a < len(act_ids) and act_ids[a] >= 0 and probs[a] > 0:
                result[int(act_ids[a])] = float(probs[a])
                total += probs[a]

        if not result:
            return None

        if abs(total - 1.0) > 1e-6 and total > 0:
            for k in result:
                result[k] /= total

        return result

    # ------------------------------------------------------------------
    # Board matching
    # ------------------------------------------------------------------

    def _find_board(self, flop_tuple):
        """Find the board_id for a given flop. Exact match first, then nearest."""
        # Exact match (flop cards in canonical order)
        if flop_tuple in self._board_to_id:
            return self._board_to_id[flop_tuple]

        # Try all orderings (board stored as sorted tuple)
        for perm in _sorted_permutations(flop_tuple):
            if perm in self._board_to_id:
                return self._board_to_id[perm]

        # Feature-distance fallback
        return self._find_nearest_board(list(flop_tuple))

    def _find_nearest_board(self, board):
        """Find nearest board by feature-space distance."""
        try:
            from blueprint_abstraction import compute_board_features
        except ImportError:
            from abstraction import compute_board_features

        features = compute_board_features(board)

        best_dist = float('inf')
        best_bid = None

        for bid, _, bf in self._board_list:
            dist = np.sum((bf - features) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_bid = bid

        return best_bid

    # ------------------------------------------------------------------
    # Hand matching
    # ------------------------------------------------------------------

    def _find_hand(self, hero_cards, board_id):
        """Find hand index by direct card-pair lookup."""
        c1, c2 = int(hero_cards[0]), int(hero_cards[1])
        key = (min(c1, c2), max(c1, c2))

        hmap = self._hand_maps.get(board_id)
        if hmap is None:
            return None

        if key in hmap:
            return hmap[key]

        # Cards overlap with stored board -> equity fallback
        return self._find_nearest_hand(hero_cards, board_id)

    def _find_nearest_hand(self, hero_cards, board_id):
        """
        Fallback: find hand with nearest equity when exact match unavailable.

        This is rare -- only happens when hero's cards overlap with the
        representative board's cards (different from actual board).
        """
        if self._equity_engine is None:
            # Without equity engine, pick middle of range as default
            hmap = self._hand_maps.get(board_id, {})
            if hmap:
                return len(hmap) // 2
            return 0

        bd = self._boards[board_id]
        board = list(bd['board'])
        hands = bd['hands']

        hero_eq = self._equity_engine.compute_equity(
            list(hero_cards), board, [], None)

        best_idx = 0
        best_dist = float('inf')

        for hi in range(len(hands)):
            hc = [int(hands[hi][0]), int(hands[hi][1])]
            if hc[0] in board or hc[1] in board:
                continue
            if hc[0] in hero_cards or hc[1] in hero_cards:
                continue

            eq = self._equity_engine.compute_equity(hc, board, [], None)
            dist = abs(eq - hero_eq)
            if dist < best_dist:
                best_dist = dist
                best_idx = hi

        return best_idx

    # ------------------------------------------------------------------
    # Pot and node matching
    # ------------------------------------------------------------------

    def _find_pot(self, pot_state, pot_sizes):
        """Find nearest pot size index."""
        if pot_state is None:
            return 0

        hero_bet, opp_bet = pot_state
        pot = hero_bet + opp_bet

        best_idx = 0
        best_dist = float('inf')
        for i in range(len(pot_sizes)):
            ps_pot = int(pot_sizes[i][0]) + int(pot_sizes[i][1])
            dist = abs(ps_pot - pot)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _find_node(self, my_bet, opp_bet, board_id, pot_idx):
        """Find hero node index for current bet state."""
        node_info = self._node_maps.get((board_id, pot_idx))
        if not node_info:
            return 0

        facing_bet = opp_bet > my_bet
        pot = my_bet + opp_bet
        if pot <= 0:
            return 0

        bet_ratio = (opp_bet - my_bet) / pot if facing_bet else 0.0

        best_idx = 0
        best_dist = float('inf')

        for strat_idx, hp, op, has_fold in node_info:
            node_pot = hp + op
            if node_pot <= 0:
                continue

            node_facing = op > hp
            if facing_bet and not node_facing:
                continue
            if not facing_bet and node_facing:
                continue

            node_ratio = (op - hp) / node_pot if node_facing else 0.0
            dist = abs(bet_ratio - node_ratio)
            if dist < best_dist:
                best_dist = dist
                best_idx = strat_idx

        return best_idx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_action_probabilities(self, hero_cards, board, pot_state=None):
        """
        Get named action probabilities.

        Returns:
            dict {action_name: probability}, or None
        """
        raw = self.get_strategy(hero_cards, board, pot_state)
        if raw is None:
            return None

        named = {}
        for act_id, prob in raw.items():
            name = ACTION_NAMES.get(act_id, f"action_{act_id}")
            named[name] = prob
        return named

    def sample_action(self, hero_cards, board, pot_state=None):
        """Sample a single action from the strategy."""
        raw = self.get_strategy(hero_cards, board, pot_state)
        if raw is None:
            return None

        actions = list(raw.keys())
        probs = np.array([raw[a] for a in actions])
        probs /= probs.sum()
        return int(np.random.choice(actions, p=probs))


def _sorted_permutations(t):
    """Return the canonical sorted form of a tuple."""
    return [tuple(sorted(t))]
