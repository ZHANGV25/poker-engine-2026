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

        # Lazy-load turn data: per-board files loaded on demand.
        # Zero startup memory, ~1-2ms per board lookup, full quality
        # (16 nodes, 5 pots, hero + opp strategies).
        turn_boards_dir = os.path.join(dir_path, 'turn_boards')
        if os.path.isdir(turn_boards_dir):
            self._turn_boards_dir = turn_boards_dir
            self._turn_board_cache = {}  # bid → loaded data
            # Build board_key → bid mapping from flop data (already loaded)
            self._turn_board_id_map = {}
            for bid in sorted(self._boards.keys()):
                bd = self._boards[bid]
                board_key = tuple(sorted(bd['board']))
                fpath = os.path.join(turn_boards_dir, f'{bid}.lzma')
                if os.path.isfile(fpath):
                    self._turn_board_id_map[board_key] = bid
            self._turn_opp = True  # signal that turn opp data is available
            self._turn_hero = True  # signal that turn hero data is available

        # Legacy: single-file turn data (fallback)
        if not hasattr(self, '_turn_hero') or self._turn_hero is None:
            turn_hero_path = os.path.join(dir_path, 'turn_hero.pkl.lzma')
            if os.path.isfile(turn_hero_path):
                with lzma.open(turn_hero_path, 'rb') as f:
                    self._turn_hero = pickle.load(f)
                self._turn_hero_board_map = {}
                for bid, td in self._turn_hero.items():
                    board_key = tuple(sorted(td['board']))
                    self._turn_hero_board_map[board_key] = bid

        if not hasattr(self, '_turn_opp') or self._turn_opp is None:
            turn_opp_path = os.path.join(dir_path, 'turn_opp.pkl.lzma')
            if os.path.isfile(turn_opp_path):
                with lzma.open(turn_opp_path, 'rb') as f:
                    self._turn_opp = pickle.load(f)
                self._turn_opp_board_map = {}
                for bid, td in self._turn_opp.items():
                    board_key = tuple(sorted(td['board']))
                    self._turn_opp_board_map[board_key] = bid

    def _lazy_load_turn_board(self, board_3):
        """Load a single turn board's data on demand. Cached after first load."""
        board_key = tuple(sorted(int(c) for c in board_3))

        # Check cache first
        if board_key in self._turn_board_cache:
            return self._turn_board_cache[board_key]

        # Find board ID
        bid = self._turn_board_id_map.get(board_key)
        if bid is None:
            return None

        # Load from disk
        fpath = os.path.join(self._turn_boards_dir, f'{bid}.lzma')
        if not os.path.isfile(fpath):
            return None

        import lzma, pickle
        with lzma.open(fpath, 'rb') as f:
            data = pickle.load(f)

        # Cache (limit cache to ~50 boards to control memory)
        if len(self._turn_board_cache) > 50:
            # Remove oldest entry
            oldest = next(iter(self._turn_board_cache))
            del self._turn_board_cache[oldest]
        self._turn_board_cache[board_key] = data
        return data

    def get_turn_strategy(self, hero_cards, board, pot_state=None, hero_position=0):
        """Get precomputed turn strategy for hero's hand.

        Returns dict {action_type: probability} or None.
        """
        try:
            if not hasattr(self, '_turn_hero') or self._turn_hero is None:
                return None
            if len(board) < 4 or pot_state is None:
                return None

            board_3 = board[:3]
            turn_card = board[3]

            # Lazy load
            if hasattr(self, '_turn_boards_dir'):
                td = self._lazy_load_turn_board(board_3)
            elif hasattr(self, '_turn_hero_board_map'):
                board_key = tuple(sorted(int(c) for c in board_3))
                bid = self._turn_hero_board_map.get(board_key)
                td = self._turn_hero.get(bid) if bid is not None else None
            else:
                return None

            if td is None:
                return None

            tc = int(turn_card)
            if tc not in td.get('turn_cards', []):
                return None

            my_bet, opp_bet = pot_state
            pot = my_bet + opp_bet
            pot_sizes = td['pot_sizes']
            best_pi = min(range(len(pot_sizes)),
                         key=lambda i: abs(pot_sizes[i][0] + pot_sizes[i][1] - pot))

            sk = f's_{best_pi}_{tc}'
            ak = f'a_{best_pi}_{tc}'
            hk = f'h_{tc}'
            if sk not in td or ak not in td or hk not in td:
                return None

            strats = td[sk]
            act_types = td[ak]
            hands = td[hk]

            # Find hero hand
            c0, c1 = int(hero_cards[0]), int(hero_cards[1])
            hero_key = (min(c0, c1), max(c0, c1))
            hand_idx = None
            for hi in range(len(hands)):
                h0, h1 = int(hands[hi][0]), int(hands[hi][1])
                if (min(h0, h1), max(h0, h1)) == hero_key:
                    hand_idx = hi
                    break
            if hand_idx is None:
                return None

            # Node selection: 0 = acting first, find FOLD node for facing bet
            node_idx = 0
            if opp_bet > my_bet:
                for ni in range(act_types.shape[0]):
                    if int(act_types[ni][0]) == 0:  # ACT_FOLD = facing bet
                        node_idx = ni
                        break

            if node_idx >= strats.shape[1]:
                return None

            hand_strat = strats[hand_idx, node_idx, :]
            node_acts = act_types[node_idx, :]

            result = {}
            for a in range(len(node_acts)):
                act = int(node_acts[a])
                if act < 0:
                    continue
                prob = float(hand_strat[a]) * 16.0 / 255.0
                if prob > 0.005:
                    result[act] = prob

            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
                return result
            return None
        except Exception:
            return None

    def get_turn_opp_bet_prob(self, board_3, turn_card, pot_state):
        """Get P(bet|hand) for each opponent hand on this turn board.

        Returns dict {(c1,c2): p_bet} or None if no data.
        Used for Bayesian narrowing when opponent bets on turn.
        """
        if not hasattr(self, '_turn_opp') or self._turn_opp is None:
            return None

        # Lazy load: get board data from per-board files
        if hasattr(self, '_turn_boards_dir'):
            td = self._lazy_load_turn_board(board_3)
        else:
            # Legacy single-file mode
            board_key = tuple(sorted(int(c) for c in board_3))
            bid = self._turn_opp_board_map.get(board_key)
            td = self._turn_opp.get(bid) if bid is not None else None

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


# ======================================================================
# River Blueprint Lookup
# ======================================================================

# River pot sizes used during compute (hero_bet, opp_bet)
_RIVER_POT_SIZES = [(2, 2), (6, 6), (14, 14), (30, 30), (50, 50)]


class RiverBlueprintLookup:
    """
    Fast runtime lookup for precomputed river strategies and opponent P(bet|hand).

    Loaded from a single LZMA-compressed pickle file produced by merge_river.py.
    Provides:
        - get_river_opp_bet_prob(board, pot_state): P(bet|hand) for Bayesian narrowing
        - get_river_hero_strategy(board, pot_state): hero acting-first strategy

    All arrays are uint8 (0-255), converted to float on lookup.
    Board lookup: sort 5 cards, binary search in board_index.
    """

    def __init__(self, data_path):
        """Load river data from LZMA-compressed pickle.

        Args:
            data_path: path to river.pkl.lzma file
        """
        self._loaded = False
        self._board_index = None  # (N, 5) int8 — sorted board cards
        self._opp_pb = None       # (N, 5, 231) uint8 — P(bet|hand) per pot
        self._hero_s = None       # (N, 5, 231, 3) uint8 — hero strategy per pot
        self._hands = None        # (N, 231, 2) int8 — hand card pairs per board
        self._pot_sizes = _RIVER_POT_SIZES
        self._n_boards = 0

        # Hand map cache: board_idx -> {(c1,c2): hand_idx}
        self._hand_map_cache = {}

        if not os.path.isfile(data_path):
            return

        import lzma
        import pickle as _pickle

        with open(data_path, 'rb') as f:
            data = _pickle.loads(lzma.decompress(f.read()))

        self._board_index = data['board_index']
        self._opp_pb = data['opp_pb']
        self._hero_s = data['hero_s']
        self._hands = data['hands']
        self._n_boards = data['n_boards']
        if 'pot_sizes' in data:
            self._pot_sizes = data['pot_sizes']
        self._loaded = True

    @property
    def is_loaded(self):
        return self._loaded and self._n_boards > 0

    def _find_board_idx(self, board):
        """Find the row index for a sorted 5-card board via binary search.

        Args:
            board: tuple/list of 5 ints (will be sorted)

        Returns:
            int index into board_index, or None if not found
        """
        if not self._loaded:
            return None

        key = np.array(sorted(int(c) for c in board), dtype=np.int8)

        # Binary search on first card, then narrow
        lo, hi = 0, self._n_boards
        bi = self._board_index

        # Multi-column binary search: compare lexicographically
        for col in range(5):
            # Find range where column matches
            col_vals = bi[lo:hi, col]
            target = key[col]

            # Find leftmost occurrence
            left = np.searchsorted(col_vals, target, side='left')
            right = np.searchsorted(col_vals, target, side='right')

            if left >= right:
                return None  # not found

            lo = lo + left
            hi = lo + (right - left)

        if lo < self._n_boards:
            return lo
        return None

    def _get_hand_map(self, board_idx):
        """Build or retrieve hand map for a board.

        Returns dict: (c1, c2) sorted -> hand_index
        """
        if board_idx in self._hand_map_cache:
            return self._hand_map_cache[board_idx]

        hands = self._hands[board_idx]  # (231, 2) int8
        hmap = {}
        for hi in range(len(hands)):
            c1, c2 = int(hands[hi][0]), int(hands[hi][1])
            if c1 == 0 and c2 == 0 and hi > 0:
                # Past valid entries (shouldn't happen with 231 hands)
                break
            hmap[(min(c1, c2), max(c1, c2))] = hi
        self._hand_map_cache[board_idx] = hmap

        # Limit cache size
        if len(self._hand_map_cache) > 100:
            oldest = next(iter(self._hand_map_cache))
            del self._hand_map_cache[oldest]

        return hmap

    def _find_pot_idx(self, pot_state):
        """Find nearest pot index from the 5 precomputed pot sizes.

        Args:
            pot_state: (my_bet, opp_bet) tuple

        Returns:
            int pot index (0-4)
        """
        my_bet, opp_bet = pot_state
        pot = my_bet + opp_bet
        best_idx = 0
        best_dist = float('inf')
        for i, (hb, ob) in enumerate(self._pot_sizes):
            dist = abs(hb + ob - pot)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def get_river_opp_bet_prob(self, board, pot_state):
        """Get P(bet|hand) for each opponent hand on this river board.

        Used for Bayesian narrowing when opponent bets on river.
        Replaces heuristic _polarized_narrow_range with equilibrium data.

        Args:
            board: list/tuple of 5 card ints
            pot_state: (my_bet, opp_bet)

        Returns:
            dict {(c1, c2) sorted: float P(bet|hand)} or None
        """
        if not self._loaded or len(board) != 5:
            return None

        board_idx = self._find_board_idx(board)
        if board_idx is None:
            return None

        pot_idx = self._find_pot_idx(pot_state)
        hmap = self._get_hand_map(board_idx)

        # opp_pb[board_idx, pot_idx, :] -> (231,) uint8
        pb_arr = self._opp_pb[board_idx, pot_idx]

        result = {}
        for hand_pair, hi in hmap.items():
            result[hand_pair] = float(pb_arr[hi]) / 255.0

        return result

    def get_river_hero_strategy(self, board, pot_state):
        """Get hero acting-first strategy for this river board.

        Returns strategy as dict {(c1,c2) sorted: np.array([p_check, p_raise_half, p_raise_pot])}.
        Probabilities are floats summing to ~1.0.

        Used when hero acts first on river, replacing equity threshold heuristics.

        Args:
            board: list/tuple of 5 card ints
            pot_state: (my_bet, opp_bet)

        Returns:
            dict {(c1, c2) sorted: np.array of strategy} or None
        """
        if not self._loaded or len(board) != 5:
            return None

        board_idx = self._find_board_idx(board)
        if board_idx is None:
            return None

        pot_idx = self._find_pot_idx(pot_state)
        hmap = self._get_hand_map(board_idx)

        # hero_s[board_idx, pot_idx, :, :] -> (231, n_actions) uint8
        strat_arr = self._hero_s[board_idx, pot_idx]

        result = {}
        for hand_pair, hi in hmap.items():
            raw = strat_arr[hi].astype(np.float64) / 255.0
            total = raw.sum()
            if total > 0:
                raw /= total
            else:
                # Default: check
                raw = np.zeros_like(raw)
                raw[0] = 1.0
            result[hand_pair] = raw

        return result
