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
        """Load per-board .npz files from a directory.

        Also checks for merged_blueprint.npz (single file with all boards).
        """
        import glob

        # Check for LZMA-compressed file (best compression, ~5% of raw size)
        lzma_path = os.path.join(dir_path, "blueprint.pkl.lzma")
        if os.path.isfile(lzma_path):
            self._load_lzma(lzma_path)
            return

        # Check for compact merged file (single or split format)
        compact_path = os.path.join(dir_path, "blueprint_v7.npz")
        split_a = os.path.join(dir_path, "bp_a.npz")
        split_flop = os.path.join(dir_path, "blueprint_v7_flop.npz")
        if os.path.isfile(compact_path):
            self._load_compact_merged(compact_path)
            return
        if os.path.isfile(split_a) or os.path.isfile(split_flop):
            self._load_compact_split(dir_path)
            return

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

                board_data = {
                    'strategies': data['flop_strategies'],
                    'action_types': data['action_types'],
                    'hands': hands,
                    'board': board,
                    'features': features,
                    'pot_sizes': pot_sizes,
                    'turn_data': turn_data,
                }

                # Load position-aware opp strategies if available
                if 'flop_opp_strategies' in data:
                    board_data['opp_strategies'] = data['flop_opp_strategies']
                    board_data['opp_action_types'] = data['opp_action_types']

                    # Also load turn opp strategies
                    if 'turn_cards' in data:
                        for tc in data['turn_cards']:
                            tc = int(tc)
                            if tc in turn_data:
                                for pi in range(len(pot_sizes)):
                                    os_key = f'turn_opp_strat_p{pi}_t{tc}'
                                    oa_key = f'turn_opp_actions_p{pi}_t{tc}'
                                    if os_key in data:
                                        if 'opp_pot_strategies' not in turn_data[tc]:
                                            turn_data[tc]['opp_pot_strategies'] = {}
                                            turn_data[tc]['opp_pot_actions'] = {}
                                        turn_data[tc]['opp_pot_strategies'][pi] = data[os_key]
                                        if oa_key in data:
                                            turn_data[tc]['opp_pot_actions'][pi] = data[oa_key]

                self._boards[board_id] = board_data

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

    def _load_lzma(self, fpath):
        """Load from LZMA-compressed pickle (best compression ratio)."""
        import lzma, pickle
        with open(fpath, 'rb') as f:
            data = pickle.loads(lzma.decompress(f.read()))
        # Handle 4-bit quantized strategies: upscale to uint8 range
        if data.get('quantization_bits') == 4:
            for key in ['flop_strategies', 'flop_opp_strategies']:
                if key in data:
                    data[key] = (data[key].astype(np.uint8) << 4)
        self._load_compact_merged_dict(data)
        self._load_turn_from_dict(data)

    def _load_turn_from_dict(self, data):
        """Load turn data from a dict into existing boards."""
        if 'turn_strategies' not in data:
            return
        turn_pot_idx = data.get('turn_pot_idx', 0)
        turn_strats = data['turn_strategies']
        turn_cards = data['turn_cards']
        turn_acts = data.get('turn_action_types')
        turn_hands = data.get('turn_hands')
        t_board_ids = data.get('turn_board_ids', data.get('board_ids'))

        for si in range(len(t_board_ids)):
            bid = int(t_board_ids[si])
            if bid not in self._boards:
                continue
            self._boards[bid]['_turn_raw'] = {
                'strats': turn_strats[si],
                'acts': turn_acts[si] if turn_acts is not None else None,
                'hands': turn_hands[si] if turn_hands is not None else None,
                'cards': turn_cards[si],
                'pot_idx': turn_pot_idx,
            }

    def _load_lzma_deferred(self, dir_path):
        """Load turn and opp data from LZMA files (called after init)."""
        import lzma, pickle, os
        for fname in ['turn.pkl.lzma', 'opp.pkl.lzma']:
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, 'rb') as f:
                data = pickle.loads(lzma.decompress(f.read()))
            if 'turn_strategies' in data:
                self._load_turn_from_dict(data)
            if 'flop_opp_strategies' in data:
                opp_strats = data['flop_opp_strategies']
                opp_acts = data['opp_action_types']
                board_ids = list(self._boards.keys())
                for i, bid in enumerate(sorted(board_ids)):
                    if i < len(opp_strats):
                        self._boards[bid]['opp_strategies'] = opp_strats[i]
                        self._boards[bid]['opp_action_types'] = opp_acts[i]

        # Load turn hero strategies (for blueprint decisions on turn)
        turn_hero_path = os.path.join(dir_path, 'turn_hero.pkl.lzma')
        if os.path.isfile(turn_hero_path):
            with open(turn_hero_path, 'rb') as f:
                self._turn_hero = pickle.loads(lzma.decompress(f.read()))
            self._turn_hero_board_map = {}
            for bid, td in self._turn_hero.items():
                board_key = tuple(sorted(td['board']))
                self._turn_hero_board_map[board_key] = bid

        # Load turn opponent strategies (for Bayesian narrowing on turn)
        turn_opp_path = os.path.join(dir_path, 'turn_opp.pkl.lzma')
        if os.path.isfile(turn_opp_path):
            with open(turn_opp_path, 'rb') as f:
                self._turn_opp = pickle.loads(lzma.decompress(f.read()))
            # Build board→board_id lookup for turn data
            self._turn_opp_board_map = {}
            for bid, td in self._turn_opp.items():
                board_key = tuple(sorted(td['board']))
                self._turn_opp_board_map[board_key] = bid

    def get_turn_strategy(self, hero_cards, board, pot_state, hero_position=0):
        """Get precomputed turn strategy for hero's hand.

        Returns dict {action_type: probability} or None.
        """
        if not hasattr(self, '_turn_hero') or self._turn_hero is None:
            return None

        if len(board) < 4:
            return None

        board_3 = board[:3]
        turn_card = board[3]
        board_key = tuple(sorted(int(c) for c in board_3))
        bid = self._turn_hero_board_map.get(board_key)
        if bid is None:
            return None

        td = self._turn_hero.get(bid)
        if td is None:
            return None

        tc = int(turn_card)
        if tc not in td.get('turn_cards', []):
            return None

        # Find nearest pot
        pot_sizes = td['pot_sizes']
        my_bet, opp_bet = pot_state
        pot = my_bet + opp_bet
        best_pi = 0
        best_dist = float('inf')
        for pi, ps in enumerate(pot_sizes):
            dist = abs(ps[0] + ps[1] - pot)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi

        strat_key = f's_{best_pi}_{tc}'
        act_key = f'a_{best_pi}_{tc}'
        hands_key = f'h_{tc}'

        if strat_key not in td or act_key not in td or hands_key not in td:
            return None

        strats = td[strat_key]    # 4-bit uint8 (n_hands, n_nodes, n_actions)
        act_types = td[act_key]   # int8 (n_nodes, n_actions)
        hands = td[hands_key]     # int8 (n_hands, 2)

        # Find hero's hand in the hands array
        hero_sorted = tuple(sorted(int(c) for c in hero_cards[:2]))
        hand_idx = None
        for hi in range(len(hands)):
            h = (int(hands[hi][0]), int(hands[hi][1]))
            if (min(h), max(h)) == hero_sorted:
                hand_idx = hi
                break

        if hand_idx is None:
            return None

        # Find correct node based on bet state
        # Node 0 is root (first to act). For facing a bet, need to find
        # the node after opponent's bet action.
        # Simple approach: use node 0 for acting first, node 1+ for facing bet
        node_idx = 0
        if opp_bet > my_bet:
            # Facing a bet — find a node where we respond
            # In the tree, after opponent bets, hero has fold/call/raise
            # This is typically node 1 or later
            # Use node that has FOLD as first action (response to bet)
            for ni in range(min(strats.shape[1], act_types.shape[0])):
                if act_types.shape[0] > ni and int(act_types[ni][0]) == 0:  # ACT_FOLD
                    node_idx = ni
                    break

        if node_idx >= strats.shape[1] or node_idx >= act_types.shape[0]:
            return None

        # Extract strategy for this hand at this node
        hand_strat = strats[hand_idx, node_idx, :]  # 4-bit quantized
        node_acts = act_types[node_idx, :]

        # Build {action_type: probability} dict
        result = {}
        total = 0
        for a in range(len(node_acts)):
            act = int(node_acts[a])
            if act < 0:
                continue
            # Upscale 4-bit: multiply by 16 to get uint8 range, then /255
            prob = float(hand_strat[a]) * 16.0 / 255.0
            if prob > 0.001:
                result[act] = prob
                total += prob

        # Normalize
        if total > 0:
            for k in result:
                result[k] /= total

        return result if result else None

    def get_turn_opp_bet_prob(self, board_3, turn_card, pot_state):
        """Get P(bet|hand) for each opponent hand on this turn board.

        Returns dict {(c1,c2): p_bet} or None if no data.
        Used for Bayesian narrowing when opponent bets on turn.
        """
        if not hasattr(self, '_turn_opp') or self._turn_opp is None:
            return None

        board_key = tuple(sorted(int(c) for c in board_3))
        bid = self._turn_opp_board_map.get(board_key)
        if bid is None:
            return None

        td = self._turn_opp.get(bid)
        if td is None:
            return None

        tc = int(turn_card)
        if tc not in td.get('turn_cards', []):
            return None

        # Find nearest pot
        pot_sizes = td['pot_sizes']
        my_bet, opp_bet = pot_state
        pot = my_bet + opp_bet
        best_pi = 0
        best_dist = float('inf')
        for pi, ps in enumerate(pot_sizes):
            dist = abs(ps[0] + ps[1] - pot)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi

        # Minimal format: pb_{pi}_{tc} = uint8 P(bet|hand), h_{tc} = hands
        pb_key = f'pb_{best_pi}_{tc}'
        hands_key = f'h_{tc}'

        if pb_key not in td or hands_key not in td:
            return None

        p_bet_q = td[pb_key]     # uint8 (n_hands,)
        hands = td[hands_key]     # int8 (n_hands, 2)

        if p_bet_q is None or hands is None:
            return None

        # Convert uint8 back to float and build hand→p_bet mapping
        result = {}
        for hi in range(len(hands)):
            h = (int(hands[hi][0]), int(hands[hi][1]))
            h_sorted = (min(h), max(h))
            result[h_sorted] = float(p_bet_q[hi]) / 255.0

        return result

    def _load_compact_split(self, dir_path):
        """Load from split files (bp_a.npz for flop, bp_b.npz for turn, etc.)."""
        import glob

        # Load flop data first
        merged = {}
        for fpath in sorted(glob.glob(os.path.join(dir_path, "bp_a*.npz"))):
            data = np.load(fpath, allow_pickle=True)
            for k in data.files:
                merged[k] = data[k]
        for fpath in sorted(glob.glob(os.path.join(dir_path, "blueprint_v7_flop*.npz"))):
            data = np.load(fpath, allow_pickle=True)
            for k in data.files:
                merged[k] = data[k]

        # Load flop boards
        self._load_compact_merged_dict(merged)

        # Now load turn data separately (may cover partial boards)
        turn_pot_idx = int(merged.get('turn_pot_idx', 0))
        for pattern in ["bp_b*.npz", "bp_c*.npz", "bp_d*.npz", "blueprint_v7_turn*.npz"]:
            for fpath in sorted(glob.glob(os.path.join(dir_path, pattern))):
                td = np.load(fpath, allow_pickle=True)
                if 'turn_strategies' not in td:
                    continue

                turn_strats = td['turn_strategies']
                turn_cards = td['turn_cards']
                turn_acts = td.get('turn_action_types')
                turn_hands = td.get('turn_hands')

                # Determine which boards this file covers
                if 'turn_board_ids' in td:
                    # Partial: explicit board ID mapping
                    t_board_ids = td['turn_board_ids']
                else:
                    # Full: same order as flop board_ids
                    t_board_ids = merged.get('board_ids',
                        np.arange(turn_strats.shape[0], dtype=np.int32))

                # Store raw turn arrays — build hand maps lazily on first access
                for si in range(len(t_board_ids)):
                    bid = int(t_board_ids[si])
                    if bid not in self._boards:
                        continue

                    self._boards[bid]['_turn_raw'] = {
                        'strats': turn_strats[si],
                        'acts': turn_acts[si] if turn_acts is not None else None,
                        'hands': turn_hands[si] if turn_hands is not None else None,
                        'cards': turn_cards[si],
                        'pot_idx': turn_pot_idx,
                    }

    def _load_compact_merged(self, fpath):
        """Load all boards from a compact stacked .npz file."""
        data = np.load(fpath, allow_pickle=True)
        self._load_compact_merged_dict(data)

    def _load_compact_merged_dict(self, data):
        """Load all boards from a dict-like object with stacked arrays."""
        board_ids = data['board_ids']
        boards_arr = data['boards']
        features_arr = data['board_features']
        hands_arr = data['hands']
        pot_sizes = data['pot_sizes']
        flop_strats = data['flop_strategies']
        act_types = data['action_types']

        has_opp = 'flop_opp_strategies' in data
        has_turn = 'turn_strategies' in data
        turn_pot_idx = int(data['turn_pot_idx']) if 'turn_pot_idx' in data else 0

        if has_opp:
            opp_strats = data['flop_opp_strategies']
            opp_acts = data['opp_action_types']
        if has_turn:
            turn_cards_arr = data['turn_cards']
            turn_strats = data['turn_strategies']
            turn_acts = data['turn_action_types']
            turn_hands_arr = data['turn_hands']

        for i in range(len(board_ids)):
            bid = int(board_ids[i])
            board = tuple(int(c) for c in boards_arr[i])
            hands = hands_arr[i]

            # Build hand map
            hmap = {}
            for hi in range(len(hands)):
                c1, c2 = int(hands[hi][0]), int(hands[hi][1])
                hmap[(min(c1, c2), max(c1, c2))] = hi
            self._hand_maps[bid] = hmap

            # Turn data
            turn_data = {}
            if has_turn:
                for ti in range(len(turn_cards_arr[i])):
                    tc = int(turn_cards_arr[i][ti])
                    t_hands = turn_hands_arr[i, ti]
                    t_hmap = {}
                    for hi in range(len(t_hands)):
                        c1, c2 = int(t_hands[hi][0]), int(t_hands[hi][1])
                        if c1 == 0 and c2 == 0 and hi > 0:
                            break  # past valid entries
                        t_hmap[(min(c1, c2), max(c1, c2))] = hi
                    turn_data[tc] = {
                        'hands': t_hands,
                        'hand_map': t_hmap,
                        'pot_strategies': {turn_pot_idx: turn_strats[i, ti]},
                        'pot_actions': {turn_pot_idx: turn_acts[i, ti]},
                    }

            board_data = {
                'strategies': flop_strats[i],
                'action_types': act_types[i],
                'hands': hands,
                'board': board,
                'features': features_arr[i],
                'pot_sizes': pot_sizes,
                'turn_data': turn_data,
            }
            if has_opp:
                board_data['opp_strategies'] = opp_strats[i]
                board_data['opp_action_types'] = opp_acts[i]

            self._boards[bid] = board_data
            self._board_list.append((bid, board, features_arr[i]))

            if self._pot_sizes is None:
                self._pot_sizes = pot_sizes

        self._board_to_id = {}
        for bid, board, _ in self._board_list:
            self._board_to_id[board] = bid

        self._build_all_node_maps()
        self._loaded = True

    def _load_merged_boards(self, fpath):
        """Load all boards from a single merged .npz file.

        The merged file has keys like 'b{id}_{field}' for each board,
        plus 'board_id_list' listing all board IDs.
        """
        data = np.load(fpath, allow_pickle=True)
        board_ids = data['board_id_list']

        for bid in board_ids:
            bid = int(bid)
            prefix = f'b{bid}_'

            board = tuple(int(c) for c in data[f'{prefix}board'])
            hands = data[f'{prefix}hands']
            features = data[f'{prefix}board_features']
            pot_sizes = data[f'{prefix}pot_sizes']

            # Load turn data
            turn_data = {}
            tc_key = f'{prefix}turn_cards'
            if tc_key in data:
                turn_cards = data[tc_key]
                for tc in turn_cards:
                    tc = int(tc)
                    t_hands_key = f'{prefix}turn_hands_t{tc}'
                    if t_hands_key in data:
                        t_hands = data[t_hands_key]
                        t_hmap = {}
                        for hi in range(len(t_hands)):
                            c1, c2 = int(t_hands[hi][0]), int(t_hands[hi][1])
                            t_hmap[(min(c1, c2), max(c1, c2))] = hi
                        turn_data[tc] = {
                            'hands': t_hands,
                            'hand_map': t_hmap,
                        }
                        for pi in range(len(pot_sizes)):
                            s_key = f'{prefix}turn_strat_p{pi}_t{tc}'
                            a_key = f'{prefix}turn_actions_p{pi}_t{tc}'
                            if s_key in data:
                                if 'pot_strategies' not in turn_data[tc]:
                                    turn_data[tc]['pot_strategies'] = {}
                                    turn_data[tc]['pot_actions'] = {}
                                turn_data[tc]['pot_strategies'][pi] = data[s_key]
                                if a_key in data:
                                    turn_data[tc]['pot_actions'][pi] = data[a_key]

            board_data = {
                'strategies': data[f'{prefix}flop_strategies'],
                'action_types': data[f'{prefix}action_types'],
                'hands': hands,
                'board': board,
                'features': features,
                'pot_sizes': pot_sizes,
                'turn_data': turn_data,
            }

            # Position-aware opp strategies
            opp_key = f'{prefix}flop_opp_strategies'
            if opp_key in data:
                board_data['opp_strategies'] = data[opp_key]
                board_data['opp_action_types'] = data[f'{prefix}opp_action_types']

                if tc_key in data:
                    for tc in data[tc_key]:
                        tc = int(tc)
                        if tc in turn_data:
                            for pi in range(len(pot_sizes)):
                                os_key = f'{prefix}turn_opp_strat_p{pi}_t{tc}'
                                oa_key = f'{prefix}turn_opp_actions_p{pi}_t{tc}'
                                if os_key in data:
                                    if 'opp_pot_strategies' not in turn_data[tc]:
                                        turn_data[tc]['opp_pot_strategies'] = {}
                                        turn_data[tc]['opp_pot_actions'] = {}
                                    turn_data[tc]['opp_pot_strategies'][pi] = data[os_key]
                                    if oa_key in data:
                                        turn_data[tc]['opp_pot_actions'][pi] = data[oa_key]

            self._boards[bid] = board_data
            self._board_list.append((bid, board, features))

            hmap = {}
            for hi in range(len(hands)):
                c1, c2 = int(hands[hi][0]), int(hands[hi][1])
                hmap[(min(c1, c2), max(c1, c2))] = hi
            self._hand_maps[bid] = hmap

            if self._pot_sizes is None:
                self._pot_sizes = pot_sizes

        self._board_to_id = {}
        for bid, board, _ in self._board_list:
            self._board_to_id[board] = bid

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
        """Build game tree node maps lazily — only build per-pot template once."""
        if self._pot_sizes is None:
            return

        # Build ONE node map per pot size (same for all boards)
        # instead of per-board × per-pot (20,475 trees → 7 trees)
        sample_bid = next(iter(self._boards))
        has_opp = 'opp_strategies' in self._boards[sample_bid]
        ps = self._pot_sizes

        self._pot_node_maps = {}
        self._pot_opp_node_maps = {}

        for pi in range(len(ps)):
            hb, ob = int(ps[pi][0]), int(ps[pi][1])
            try:
                tree = GameTree(hb, ob, 2, 100, True)
                node_info = []
                for i, nid in enumerate(tree.hero_node_ids):
                    acts = [a for a, _ in tree.children[nid]]
                    node_info.append((i, tree.hero_pot[nid], tree.opp_pot[nid], ACT_FOLD in acts))
                self._pot_node_maps[pi] = node_info

                if has_opp:
                    opp_info = []
                    for i, nid in enumerate(tree.opp_node_ids):
                        acts = [a for a, _ in tree.children[nid]]
                        opp_info.append((i, tree.hero_pot[nid], tree.opp_pot[nid], ACT_FOLD in acts))
                    self._pot_opp_node_maps[pi] = opp_info
            except Exception:
                self._pot_node_maps[pi] = [(0, hb, ob, False)]

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
                     dead_cards=None, opp_weights=None, hero_position=0):
        """
        Look up the blueprint strategy for the current game state.

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints (uses first 3 for flop matching)
            pot_state: (hero_bet, opp_bet) tuple
            dead_cards: optional dead cards
            opp_weights: unused (kept for API compatibility)
            hero_position: 0 = hero acts first (P0), 1 = hero acts second (P1)

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

        # Position-aware: use opp strategies when hero is P1 (acting second)
        use_opp = (hero_position == 1 and 'opp_strategies' in bd)

        if use_opp:
            strats = bd['opp_strategies']
            acts = bd['opp_action_types']
            node_idx = self._find_opp_node(my_bet, opp_bet, board_id, pot_idx)
        else:
            strats = bd['strategies']
            acts = bd['action_types']
            node_idx = self._find_node(my_bet, opp_bet, board_id, pot_idx)

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
                           dead_cards=None, opp_weights=None, hero_position=0):
        """Look up turn strategy from multi-street solve.

        Args:
            hero_cards: list of 2 card ints
            board: list of 4+ card ints (flop + turn card)
            pot_state: (hero_bet, opp_bet)
            hero_position: 0 = first to act, 1 = second to act

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

        # Lazy build turn data from raw arrays on first access
        if '_turn_raw' in bd and not bd.get('turn_data'):
            raw = bd['_turn_raw']
            turn_data = {}
            for ti in range(len(raw['cards'])):
                tc = int(raw['cards'][ti])
                t_h = raw['hands'][ti] if raw['hands'] is not None else None
                t_hmap = {}
                if t_h is not None:
                    for hi in range(len(t_h)):
                        c1, c2 = int(t_h[hi][0]), int(t_h[hi][1])
                        if c1 == 0 and c2 == 0 and hi > 0:
                            break
                        t_hmap[(min(c1, c2), max(c1, c2))] = hi
                turn_data[tc] = {
                    'hands': t_h,
                    'hand_map': t_hmap,
                    'pot_strategies': {raw['pot_idx']: raw['strats'][ti]},
                    'pot_actions': {raw['pot_idx']: raw['acts'][ti]} if raw['acts'] is not None else {},
                }
            bd['turn_data'] = turn_data

        turn_data = bd.get('turn_data', {})
        turn_card = int(board[3])

        if turn_card not in turn_data:
            return None

        td = turn_data[turn_card]

        # Position-aware: use opp strategies when hero is P1
        use_opp = (hero_position == 1 and 'opp_pot_strategies' in td)
        strat_key = 'opp_pot_strategies' if use_opp else 'pot_strategies'
        act_key = 'opp_pot_actions' if use_opp else 'pot_actions'

        if strat_key not in td:
            # Fall back to hero strategies
            strat_key = 'pot_strategies'
            act_key = 'pot_actions'
            use_opp = False

        if strat_key not in td:
            return None

        # Find hand index in turn hand list
        c1, c2 = int(hero_cards[0]), int(hero_cards[1])
        key = (min(c1, c2), max(c1, c2))
        hand_idx = td['hand_map'].get(key)
        if hand_idx is None:
            return None

        # Find pot index
        pot_idx = self._find_pot(pot_state, bd['pot_sizes'])
        if pot_idx not in td[strat_key]:
            pot_idx = min(td[strat_key].keys()) if td[strat_key] else 0

        strats = td[strat_key].get(pot_idx)
        acts = td[act_key].get(pot_idx)
        if strats is None:
            return None

        # Find node for bet state
        my_bet = pot_state[0] if pot_state else 2
        opp_bet = pot_state[1] if pot_state else 2
        if use_opp:
            node_idx = self._find_opp_node(my_bet, opp_bet, board_id, pot_idx)
        else:
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
        node_info = self._pot_node_maps.get(pot_idx)
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

    def _find_opp_node(self, my_bet, opp_bet, board_id, pot_idx):
        """Find opp (P1) node index for current bet state.

        When hero is P1, hero's bet state maps to opp nodes in the
        solved tree (where P1 = opp). Note: from hero-P1's perspective,
        'my_bet' is what P1 has bet, 'opp_bet' is what P0 has bet.
        In the tree, P1 nodes have hero_pot=P0's bet, opp_pot=P1's bet.
        So we swap: match hp against opp_bet, op against my_bet.
        """
        node_info = self._pot_opp_node_maps.get(pot_idx)
        if not node_info:
            # Fall back to hero node matching
            return self._find_node(my_bet, opp_bet, board_id, pot_idx)

        # From P1's view: facing_bet means P0 bet more than P1
        facing_bet = opp_bet > my_bet
        pot = my_bet + opp_bet
        if pot <= 0:
            return 0

        bet_ratio = (opp_bet - my_bet) / pot if facing_bet else 0.0

        best_idx = 0
        best_dist = float('inf')

        for strat_idx, hp, op, has_fold in node_info:
            # hp = hero_pot (P0's bet), op = opp_pot (P1's bet) in tree
            # For P1 hero: P0's bet = opp_bet, P1's bet = my_bet
            # node "facing bet" from P1 view: hp > op (P0 bet more)
            node_pot = hp + op
            if node_pot <= 0:
                continue

            node_facing = hp > op  # P0 bet more than P1
            if facing_bet and not node_facing:
                continue
            if not facing_bet and node_facing:
                continue

            node_ratio = (hp - op) / node_pot if node_facing else 0.0
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
