"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Strategy hierarchy:
    1. Lead protection (fold/check when winning by enough)
    2. Preflop: precomputed GTO strategy tree
    3. Discard: best keep-pair by exact equity with Bayesian inference
    4. Postflop:
       a. Multi-street blueprint (backward induction, flop-aware)
       b. Single-street blueprint (unbucketed or bucketed)
       c. Range solver (river) or one-hand solver (flop/turn) as fallback
       d. Opponent range narrowing when facing bets
"""

import os
import sys
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from agents.agent import Agent
from gym_env import PokerEnv

from equity import ExactEquityEngine
from inference import DiscardInference
from solver import SubgameSolver
from range_solver import RangeSolver
from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

# Preload at module import time.
# Phase 1 (synchronous): load flop from npz (fast, ~4s)
# Phase 2 (background): decompress turn+opp from LZMA (slow, ~10s)
import threading
_PRELOAD = {'multi_street': None, 'done': False, 'deferred_done': False}
_data_dir = os.path.join(_dir, "data", "multi_street")
try:
    _pre_engine = ExactEquityEngine()
    from multi_street_lookup import MultiStreetLookup
    if os.path.isdir(_data_dir):
        _PRELOAD['multi_street'] = MultiStreetLookup(
            _data_dir, equity_engine=_pre_engine)
    _PRELOAD['done'] = True
except Exception:
    _PRELOAD['done'] = True

def _deferred_load():
    """Load LZMA turn+opp+river data in background after server starts."""
    import lzma, pickle
    try:
        if _PRELOAD['multi_street'] and os.path.isdir(_data_dir):
            _PRELOAD['multi_street']._load_lzma_deferred(_data_dir)
            _PRELOAD['multi_street']._build_all_node_maps()
    except Exception:
        pass
    _PRELOAD['deferred_done'] = True

threading.Thread(target=_deferred_load, daemon=True).start()

# Graceful imports for optional modules
try:
    from multi_street_lookup import MultiStreetLookup
    _MULTI_STREET_AVAILABLE = True
except Exception:
    _MULTI_STREET_AVAILABLE = False

try:
    from blueprint_lookup_unbucketed import BlueprintLookupUnbucketed
    _BLUEPRINT_UNBUCKETED_AVAILABLE = True
except Exception:
    _BLUEPRINT_UNBUCKETED_AVAILABLE = False

try:
    from blueprint_lookup import BlueprintLookup
    _BLUEPRINT_AVAILABLE = True
except Exception:
    _BLUEPRINT_AVAILABLE = False

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CALL = PokerEnv.ActionType.CALL.value
CHECK = PokerEnv.ActionType.CHECK.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_BLUEPRINT_FILES = {1: "flop_blueprint.npz", 2: "turn_blueprint.npz",
                    3: "river_blueprint.npz"}

_POT_FRACTIONS = {
    ACT_RAISE_HALF: 0.40, ACT_RAISE_POT: 0.70,
    ACT_RAISE_ALLIN: 1.00, ACT_RAISE_OVERBET: 1.50,
}


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)
        self.range_solver = RangeSolver(self.engine)

        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()
        # Use preloaded data from module-level (loaded before server starts).
        self._multi_street = _PRELOAD.get('multi_street')
        self._multi_street_loaded = _PRELOAD.get('done', False)
        self._blueprints = self._load_blueprints() if self._multi_street_loaded else {}

        self._current_hand = -1
        self._opp_weights = None
        self._bankroll = 0
        self._hand_reward = 0  # accumulates within a hand
        self._total_hands = 1000
        self._blind_pos = 0  # detected from bets each hand

        # Pot control: don't raise on 3+ streets without near-nuts
        self._streets_raised = 0
        self._current_street = -1
        self._raised_this_street = False
        self._last_street_seen = -1
        self._last_hero_bet = 0     # track our bet size for MDF narrowing
        self._last_pot_before = 0   # pot before our bet
        self._opp_bet_at_raise = 0  # opponent's bet when we last raised (for re-raise MDF)

        # Track opponent betting patterns per street for adaptive narrowing
        self._opp_bets_by_street = {1: 0, 2: 0, 3: 0}   # flop, turn, river
        self._opp_actions_by_street = {1: 0, 2: 0, 3: 0}
        # Keep old names for backward compat with adaptive showdown tracking
        self._opp_river_bets = 0
        self._opp_river_actions = 0
        self._opp_bet_showdown_wins = 0   # times they bet, went to showdown, won
        self._opp_bet_showdown_total = 0  # times they bet and went to showdown
        self._opp_bet_this_hand = False   # did opponent bet this hand
        self._we_folded_this_hand = False

        # Track time ourselves — observation doesn't include time_left
        import time as _time
        self._match_start = _time.time()
        self._time_budget = 900  # conservative: 900 of 1000s to leave margin

        # Decision path counters (for verifying no fallbacks fire)
        self._path_counts = {
            'ms_flop': 0, 'ms_turn': 0, 'range_solver': 0,
            'ss_blueprint': 0, 'one_hand_solver': 0, 'emergency': 0,
        }

    def __name__(self):
        return "PlayerAgent"

    # ----------------------------------------------------------------
    #  INIT HELPERS
    # ----------------------------------------------------------------

    def _load_preflop_table(self):
        path = os.path.join(_dir, "data", "preflop_potential.npz")
        if not os.path.exists(path):
            return None
        data = np.load(path)
        return dict(zip(data["bitmasks"].tolist(), data["potentials"].tolist()))

    def _load_preflop_strategy(self):
        strat_path = os.path.join(_dir, "data", "preflop_strategy.npz")
        tree_path = os.path.join(_dir, "data", "preflop_tree.pkl")
        if not os.path.exists(strat_path) or not os.path.exists(tree_path):
            return None
        import pickle
        data = np.load(strat_path)
        with open(tree_path, 'rb') as f:
            children_map = pickle.load(f)
        return {
            'strategies': data['strategies'],
            'pot_min': float(data['pot_min']),
            'pot_max': float(data['pot_max']),
            'n_buckets': int(data['n_buckets']),
            'raise_levels': data['raise_levels'].tolist(),
            'node_players': data['node_players'],
            'node_bet_sb': data['node_bet_sb'],
            'node_bet_bb': data['node_bet_bb'],
            'children_map': children_map,
        }

    def _load_multi_street(self):
        """Load multi-street backward-induction blueprint."""
        if not _MULTI_STREET_AVAILABLE:
            return None
        data_dir = os.path.join(_dir, "data", "multi_street")
        if not os.path.isdir(data_dir):
            # Try single merged file
            merged = os.path.join(_dir, "data", "multi_street_blueprint.npz")
            if os.path.isfile(merged):
                try:
                    return MultiStreetLookup(merged, equity_engine=self.engine)
                except Exception:
                    return None
            return None
        try:
            return MultiStreetLookup(data_dir, equity_engine=self.engine)
        except Exception:
            return None

    def _load_blueprints(self):
        """Load single-street blueprint files (fallback after multi-street)."""
        blueprints = {}
        if not _BLUEPRINT_AVAILABLE and not _BLUEPRINT_UNBUCKETED_AVAILABLE:
            return blueprints

        data_dir = os.path.join(_dir, "data")
        for street, filename in _BLUEPRINT_FILES.items():
            loaded = None
            if _BLUEPRINT_UNBUCKETED_AVAILABLE:
                base, ext = os.path.splitext(filename)
                for candidate in [f"{base}_unbucketed{ext}", filename]:
                    fpath = os.path.join(data_dir, candidate)
                    if os.path.exists(fpath):
                        try:
                            loaded = BlueprintLookupUnbucketed(
                                fpath, equity_engine=self.engine)
                            break
                        except Exception:
                            pass
            if loaded is None and _BLUEPRINT_AVAILABLE:
                fpath = os.path.join(data_dir, filename)
                if os.path.exists(fpath):
                    try:
                        loaded = BlueprintLookup(fpath, equity_engine=self.engine)
                    except Exception:
                        pass
            blueprints[street] = loaded
        return blueprints

    # ----------------------------------------------------------------
    #  UTILITY
    # ----------------------------------------------------------------

    def _preflop_potential(self, my_cards):
        if self._preflop_table is None:
            return None
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask)

    def _get_preflop_bucket(self, my_cards):
        if self._preflop_strategy is None or self._preflop_table is None:
            return None
        potential = self._preflop_potential(my_cards)
        if potential is None:
            return None
        ps = self._preflop_strategy
        frac = (potential - ps['pot_min']) / (ps['pot_max'] - ps['pot_min'])
        frac = max(0.0, min(1.0 - 1e-9, frac))
        return int(frac * ps['n_buckets'])

    def _find_preflop_node(self, observation):
        if self._preflop_strategy is None:
            return None
        ps = self._preflop_strategy
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        blind_pos = self._blind_pos
        bet_sb = my_bet if blind_pos == 0 else opp_bet
        bet_bb = my_bet if blind_pos == 1 else opp_bet

        best_nid, best_dist = None, float('inf')
        for nid in range(len(ps['node_players'])):
            if ps['node_players'][nid] != blind_pos:
                continue
            if ps['node_players'][nid] == -1:
                continue
            d = abs(ps['node_bet_sb'][nid] - bet_sb) + abs(ps['node_bet_bb'][nid] - bet_bb)
            if d == 0:
                return nid
            if d < best_dist:
                best_dist = d
                best_nid = nid
        return best_nid

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            # Track opponent bet showdown results from previous hand.
            # Only count actual showdowns (river reached, we didn't fold).
            # Excludes our folds (main contamination source in old code).
            if (self._opp_bet_this_hand and not self._we_folded_this_hand
                    and self._current_street == 3 and self._hand_reward != 0):
                self._opp_bet_showdown_total += 1
                if self._hand_reward < 0:  # we lost at showdown = they won
                    self._opp_bet_showdown_wins += 1

            # Apply accumulated reward from previous hand
            self._bankroll += self._hand_reward
            self._hand_reward = 0
            self._current_hand = hand_number
            self._opp_weights = None
            self._streets_raised = 0
            self._current_street = -1
            self._raised_this_street = False
            self._last_street_seen = -1
            self._opp_bet_this_hand = False
            self._we_folded_this_hand = False
            self._narrowed_this_street = False
            self._opp_bet_at_raise = 0

    def _parse_cards(self, observation):
        my = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_d = [c for c in observation["my_discarded_cards"] if c != -1]
        return my, board, opp_d, my_d

    def _pot_size(self, obs):
        return obs.get("pot_size", obs["my_bet"] + obs["opp_bet"])

    # ----------------------------------------------------------------
    #  DISCARD
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards)
        results = self.engine.evaluate_all_keep_pairs(
            my_cards, board, opp_discards, self._opp_weights)
        best = results[0][0]
        return (DISCARD, 0, best[0], best[1])

    # ----------------------------------------------------------------
    #  PREFLOP
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        valid = observation["valid_actions"]
        my_bet, opp_bet = observation["my_bet"], observation["opp_bet"]

        # Detect blind position: SB posted 1, BB posted 2
        # Use observation field if available, otherwise detect from bets
        if 'blind_position' in observation:
            self._blind_pos = observation['blind_position']
        elif my_bet <= 1 and opp_bet <= 2:
            # First preflop action: SB has bet 1, BB has bet 2
            self._blind_pos = 0 if my_bet <= opp_bet else 1

        bucket = self._get_preflop_bucket(my_cards)
        node_id = self._find_preflop_node(observation)

        if bucket is not None and node_id is not None and self._preflop_strategy:
            ps = self._preflop_strategy
            strategy = ps['strategies'][node_id, bucket]
            children = ps['children_map'].get(node_id, {})

            if children:
                vm = np.zeros(len(strategy))
                for aid in children:
                    if aid == 0 and valid[FOLD]:
                        vm[aid] = strategy[aid]
                    elif aid == 1 and (valid[CALL] or valid[CHECK]):
                        vm[aid] = strategy[aid]
                    elif aid >= 2 and valid[RAISE]:
                        vm[aid] = strategy[aid]

                total = vm.sum()
                if total > 0:
                    vm /= total
                    chosen = int(np.random.choice(len(vm), p=vm))
                    if chosen == 0:
                        return (FOLD, 0, 0, 0)
                    elif chosen == 1:
                        return (CALL, 0, 0, 0) if valid[CALL] else (CHECK, 0, 0, 0)
                    else:
                        amt = ps['raise_levels'][chosen - 2] - opp_bet
                        amt = max(amt, observation["min_raise"])
                        amt = min(amt, observation["max_raise"])
                        # Cap preflop raises to small sizes — get to postflop
                        # where our blueprint + solver dominate.
                        amt = min(amt, observation["min_raise"])
                        if amt > 0 and valid[RAISE]:
                            return (RAISE, amt, 0, 0)
                        return (CALL, 0, 0, 0) if valid[CALL] else (CHECK, 0, 0, 0)

        # Fallback: pot odds
        potential = self._preflop_potential(my_cards) or 0.5
        if valid[CALL]:
            cost = opp_bet - my_bet
            if cost <= 1:
                return (CALL, 0, 0, 0)
            pot = self._pot_size(observation)
            if potential >= cost / (cost + pot):
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        return (CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  POSTFLOP
    # ----------------------------------------------------------------

    def _blueprint_to_engine(self, action_type, observation):
        """Map abstract blueprint action to concrete engine action."""
        valid = observation["valid_actions"]
        pot = self._pot_size(observation)
        min_r, max_r = observation["min_raise"], observation["max_raise"]

        if action_type == ACT_FOLD:
            return (FOLD, 0, 0, 0) if valid[FOLD] else (
                (CHECK, 0, 0, 0) if valid[CHECK] else None)
        if action_type == ACT_CHECK:
            return (CHECK, 0, 0, 0) if valid[CHECK] else (
                (CALL, 0, 0, 0) if valid[CALL] else None)
        if action_type == ACT_CALL:
            # Safety: don't blindly call huge bets (shoves) from blueprint
            # Blueprint doesn't know the actual bet size — it just says "call"
            if valid[CALL]:
                my_bet = observation["my_bet"]
                opp_bet = observation["opp_bet"]
                cost = opp_bet - my_bet
                if cost > 50:
                    # Large call — verify equity justifies it
                    return None  # fall through to equity threshold check
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0) if valid[CHECK] else None
        if action_type in _POT_FRACTIONS:
            amt = max(int(pot * _POT_FRACTIONS[action_type]), min_r)
            amt = min(amt, max_r)
            if valid[RAISE] and amt > 0:
                return (RAISE, amt, 0, 0)
            return (CALL, 0, 0, 0) if valid[CALL] else (
                (CHECK, 0, 0, 0) if valid[CHECK] else None)
        return None

    def _try_strategy(self, strategy, observation):
        """Sample from a strategy dict and return an engine action, or None."""
        if not strategy:
            return None

        aids = list(strategy.keys())
        p = np.array([strategy[a] for a in aids])
        p /= p.sum()
        chosen = int(np.random.choice(aids, p=p))
        return self._blueprint_to_engine(chosen, observation)

    def _bayesian_range_update(self, board, my_bet, opp_bet, street):
        """Update opponent range using P(action|hand) from blueprint.

        When the opponent bets, we look up the blueprint's opponent strategy
        at the matching node and multiply each hand's weight by P(bet|hand).
        Hands the equilibrium says would bet get full weight; hands it says
        wouldn't bet get near-zero weight.

        Returns True if update was applied, False if no blueprint data available.
        """
        if self._multi_street is None or self._opp_weights is None:
            return False

        # Only works for flop (street 1) currently — need precomputed turn for street 2
        if street != 1:
            return False

        try:
            ms = self._multi_street
            flop = tuple(sorted(int(c) for c in board[:3]))
            board_id = ms._find_board(flop)
            if board_id is None:
                return False

            bd = ms._boards[board_id]
            if 'opp_strategies' not in bd:
                return False

            # Find pot index
            pot_idx = ms._find_pot((my_bet, opp_bet), bd['pot_sizes'])

            # Find the opp node matching current bet state
            # The opponent just bet, so we need the node BEFORE their bet
            # (where they decided to bet vs check)
            # Use _find_opp_node with the state before the bet
            # Before opp bet: both bets were equal at my_bet
            opp_node_idx = ms._find_opp_node(my_bet, my_bet, board_id, pot_idx)

            opp_strats = bd['opp_strategies']  # (n_pots, n_hands, n_opp_nodes, n_actions)
            opp_acts = bd['opp_action_types']   # (n_pots, n_opp_nodes, n_actions)

            if pot_idx >= opp_strats.shape[0]:
                return False
            if opp_node_idx >= opp_strats.shape[2]:
                return False

            # Get P(action|hand) for each hand at this node
            node_strats = opp_strats[pot_idx, :, opp_node_idx, :]  # (n_hands, n_actions)
            node_acts = opp_acts[pot_idx, opp_node_idx, :]          # (n_actions,)

            # Map the opponent's actual bet to the closest abstract action
            raise_amt = opp_bet - my_bet
            pot = my_bet * 2
            if pot > 0:
                bet_frac = raise_amt / pot
            else:
                bet_frac = 1.0

            # Find the raise action closest to the actual bet fraction
            best_act_idx = None
            best_dist = float('inf')
            raise_fracs = {3: 0.40, 4: 0.70, 5: 1.00, 6: 1.50}  # ACT_RAISE_*

            for a_idx in range(len(node_acts)):
                act = int(node_acts[a_idx])
                if act < 0:
                    continue
                if act in raise_fracs:
                    dist = abs(bet_frac - raise_fracs[act])
                    if dist < best_dist:
                        best_dist = dist
                        best_act_idx = a_idx

            if best_act_idx is None:
                return False

            # Extract P(bet|hand) for this action
            p_bet_per_hand = node_strats[:, best_act_idx].astype(np.float64) / 255.0

            # Apply Bayesian update to opponent range
            hands = bd['hands']
            hand_map = ms._hand_maps.get(board_id, {})

            updated = False
            for pair, weight in list(self._opp_weights.items()):
                if weight <= 0:
                    continue
                key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                hand_idx = hand_map.get(key)

                if hand_idx is not None and hand_idx < len(p_bet_per_hand):
                    p_bet = p_bet_per_hand[hand_idx]
                    # Multiply weight by P(bet|hand), with a high floor to protect
                    # against opponents who deviate from equilibrium (bet with
                    # hands the blueprint says wouldn't bet)
                    self._opp_weights[pair] *= max(p_bet, 0.20)
                    updated = True

            if not updated:
                return False

            # Renormalize
            total = sum(self._opp_weights.values())
            if total > 0:
                for k in self._opp_weights:
                    self._opp_weights[k] /= total

            return True

        except Exception:
            return False

    def _cfr_bayesian_narrow(self, board, dead_cards, my_bet, opp_bet, street):
        """Narrow opponent range using real-time CFR-computed P(bet|hand).

        Solves the game from opponent's perspective: for each hand in their
        range, computes the equilibrium probability of betting. Hands that
        would bet get full weight, hands that wouldn't get reduced weight.

        This replaces heuristic polarized narrowing with game-theoretic
        Bayesian narrowing. ~400ms per call, well within budget.

        Returns True if narrowing was applied, False on failure.
        """
        if self._opp_weights is None:
            return False

        try:
            # Build hero range (what opponent thinks we have)
            # Use uniform over non-conflicting hands
            known = set(board) | set(dead_cards)
            hero_range = {}
            import itertools as _it
            remaining = [c for c in range(27) if c not in known]
            for h in _it.combinations(remaining, 2):
                hero_range[h] = 1.0

            # Skip if ranges are too large (would be too slow)
            n_opp = sum(1 for w in self._opp_weights.values() if w > 0.001)
            n_hero = len(hero_range)
            if n_opp * n_hero > 10000:
                return False  # fall back to heuristic

            p_bet = self.solver.compute_opponent_bet_probs(
                board=board, dead_cards=dead_cards,
                opp_range=self._opp_weights,
                hero_range=hero_range,
                my_bet=my_bet, opp_bet=opp_bet,
                street=street,
                min_raise=max(2, opp_bet - my_bet),
                iterations=50)

            if p_bet is None or len(p_bet) < 3:
                return False

            # Bayesian update: multiply each hand's weight by P(bet|hand)
            # Floor at 0.05 to avoid zeroing out hands that deviate from GTO
            for hand in list(self._opp_weights.keys()):
                if hand in p_bet:
                    self._opp_weights[hand] *= max(p_bet[hand], 0.05)

            # Renormalize
            total = sum(self._opp_weights.values())
            if total > 0:
                for k in self._opp_weights:
                    self._opp_weights[k] /= total

            return True

        except Exception:
            return False

    def _polarized_narrow_range(self, board, dead_cards, my_bet, opp_bet):
        """Construct polarized opponent range when facing a river bet.

        GTO: opponent bets with VALUE (top) + BLUFFS (bottom), checks MIDDLE.
        Bluff ratio = bet / (bet + pot), derived from bet size.
        Total bet frequency tracked during the match.
        """
        if self._opp_weights is None:
            return

        raise_amt = opp_bet - my_bet
        pot_before = my_bet * 2
        if raise_amt <= 0 or pot_before <= 0:
            return

        bet_frac = raise_amt / pot_before
        bluff_ratio = bet_frac / (1 + bet_frac)

        # Estimate total bet frequency from per-street tracked data.
        # Defaults from field median across 12 matches:
        #   Turn: 31% (range 14-77%, median 31%)
        #   River: 23% (range 8-56%, median 23%)
        n_board = len(board)  # 4 = turn, 5 = river
        default_freq = 0.31 if n_board <= 4 else 0.23
        obs_street = 2 if n_board <= 4 else 3  # map board length to street number

        if self._opp_actions_by_street.get(obs_street, 0) > 15:
            total_bet_freq = (self._opp_bets_by_street[obs_street] /
                              self._opp_actions_by_street[obs_street])
        else:
            total_bet_freq = default_freq

        total_bet_freq = max(0.10, min(0.70, total_bet_freq))  # clamp

        value_pct = total_bet_freq / (1 + bluff_ratio)
        bluff_pct = total_bet_freq - value_pct

        # Rank all hands by strength
        board_set = set(board)
        strengths = {}
        for pair in self._opp_weights:
            if self._opp_weights[pair] <= 0 or set(pair) & board_set:
                continue
            try:
                if len(board) >= 5:
                    r = self.engine.lookup_seven(list(pair) + list(board))
                elif len(board) == 4:
                    import itertools as _it
                    cards_6 = list(pair) + list(board)
                    r = min(self.engine.lookup_five(list(c)) for c in _it.combinations(cards_6, 5))
                elif len(board) >= 3:
                    r = self.engine.lookup_five(list(pair) + list(board[:3]))
                else:
                    continue
            except Exception:
                continue
            strengths[pair] = r

        if not strengths:
            return

        ranked = sorted(strengths.items(), key=lambda x: x[1])  # best first
        n = len(ranked)

        # Keep top value_pct (value bets) + bottom bluff_pct (bluffs)
        # Remove the middle
        n_value = int(n * value_pct)
        n_bluff = int(n * bluff_pct)
        n_value = max(1, n_value)

        keep_indices = set(range(n_value))  # top
        keep_indices.update(range(n - n_bluff, n))  # bottom

        for i, (hand, _) in enumerate(ranked):
            if i not in keep_indices:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _reraise_narrow_range(self, board, dead_cards, my_bet, opp_bet):
        """Two-phase game-theoretic narrowing for opponent re-raises.

        Phase 1 — MDF fold removal:
            When opponent faces our raise, the weakest hands in their range
            fold. Fold fraction = 1 - MDF, where:
                MDF = pot_faced / (pot_faced + call_cost)
                pot_faced = my_bet + opp_bet_before_reraise
                call_cost = my_bet - opp_bet_before_reraise

            Simplifies to: MDF = (my_bet + O) / (2 × my_bet)
            where O = opponent's bet when we raised.

        Phase 2 — Polarized re-raise selection:
            Of the defending hands (those that didn't fold), the re-raisers
            form a polarized sub-range: top value + bottom bluffs.

            Bluff ratio α = rr_size / (rr_size + pot_if_called)
            where rr_size = opp_bet - my_bet, pot_if_called = 2 × my_bet.

            Re-raise frequency within defending range: same GTO proportion
            as initial bet (geometric/self-similar model). Each raise level
            selects ~f of the acting range, where f = 0.17 (river) / 0.13 (turn)
            from solved equilibrium.
        """
        if self._opp_weights is None:
            return

        # Phase 1: MDF fold removal
        O = self._opp_bet_at_raise  # opponent's bet when we raised
        if my_bet > O and my_bet > 0:
            # MDF = (my_bet + O) / (2 * my_bet)
            mdf = (my_bet + O) / (2.0 * my_bet)
            mdf = max(0.3, min(0.95, mdf))  # safety clamp
            self._mdf_narrow_range(board, dead_cards, mdf)

        # Phase 2: Polarized re-raise within defending range
        rr_size = opp_bet - my_bet
        pot_if_called = 2.0 * my_bet  # pot if they had just called our raise
        if rr_size <= 0 or pot_if_called <= 0:
            return

        # Bluff ratio: α = rr_size / (rr_size + pot_if_called)
        bluff_ratio = rr_size / (rr_size + pot_if_called)

        # GTO re-raise frequency within defending range (geometric model)
        n_board = len(board)  # 4 = turn, 5 = river
        default_freq = 0.13 if n_board <= 4 else 0.17
        obs_street = 2 if n_board <= 4 else 3

        # Use per-street tracked opponent data if available
        if self._opp_actions_by_street.get(obs_street, 0) > 15:
            total_bet_freq = (self._opp_bets_by_street[obs_street] /
                              self._opp_actions_by_street[obs_street])
        else:
            total_bet_freq = default_freq
        total_bet_freq = max(0.08, min(0.60, total_bet_freq))

        value_pct = total_bet_freq / (1.0 + bluff_ratio)
        bluff_pct = total_bet_freq - value_pct

        # Rank surviving hands by strength and apply polarized selection
        board_set = set(board)
        strengths = {}
        for pair in self._opp_weights:
            if self._opp_weights[pair] <= 0 or set(pair) & board_set:
                continue
            try:
                if len(board) >= 5:
                    r = self.engine.lookup_seven(list(pair) + list(board))
                elif len(board) == 4:
                    import itertools as _it
                    cards_6 = list(pair) + list(board)
                    r = min(self.engine.lookup_five(list(c))
                            for c in _it.combinations(cards_6, 5))
                elif len(board) >= 3:
                    r = self.engine.lookup_five(list(pair) + list(board[:3]))
                else:
                    continue
            except Exception:
                continue
            strengths[pair] = r

        if not strengths:
            return

        ranked = sorted(strengths.items(), key=lambda x: x[1])  # best first
        n = len(ranked)
        n_value = max(1, int(n * value_pct))
        n_bluff = int(n * bluff_pct)

        keep_indices = set(range(n_value))
        keep_indices.update(range(n - n_bluff, n))

        for i, (hand, _) in enumerate(ranked):
            if i not in keep_indices:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _mdf_narrow_range(self, board, dead_cards, keep_fraction):
        """Narrow opponent range by keeping top X% of hands.

        keep_fraction is derived from MDF (minimum defense frequency):
        MDF = pot / (pot + bet) = 1 / (1 + bet/pot)

        When we bet and opponent calls, they defend with top MDF% of range.
        """
        if self._opp_weights is None:
            return

        board_set = set(board)
        strengths = {}
        for pair in self._opp_weights:
            if self._opp_weights[pair] <= 0 or set(pair) & board_set:
                continue
            try:
                if len(board) >= 5:
                    r = self.engine.lookup_seven(list(pair) + list(board))
                elif len(board) == 4:
                    import itertools as _it
                    cards_6 = list(pair) + list(board)
                    r = min(self.engine.lookup_five(list(c)) for c in _it.combinations(cards_6, 5))
                elif len(board) >= 3:
                    r = self.engine.lookup_five(list(pair) + list(board[:3]))
                else:
                    continue
            except Exception:
                continue
            strengths[pair] = r

        if not strengths:
            return

        ranked = sorted(strengths.items(), key=lambda x: x[1])
        cutoff = int(len(ranked) * keep_fraction)
        if cutoff < len(ranked):
            for hand, _ in ranked[cutoff:]:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _soft_narrow_range(self, my_bet, opp_bet, board, dead_cards):
        """Soft Bayesian range adjustment when opponent bets.

        Instead of removing weak hands (exploitable), we reduce their
        weight proportionally to hand strength. Strong hands keep full
        weight, weak hands get reduced but never eliminated.

        This corrects the one-hand solver's tendency to overcall:
        the solver computes equity against the full range, but the
        opponent's betting range is stronger than average.
        """
        if self._opp_weights is None:
            return

        raise_amt = opp_bet - my_bet
        pot_before = my_bet * 2
        if raise_amt <= 0 or pot_before <= 0:
            return

        # Bigger bets → opponent range is more polarized (stronger)
        # Proven on server: 0.3x turned river from -1393 to +1974 chips
        ratio = raise_amt / max(pot_before, 1)
        strength = min(0.5, ratio * 0.3)

        board_set = set(board)
        strengths = {}
        for pair in self._opp_weights:
            if self._opp_weights[pair] <= 0 or set(pair) & board_set:
                continue
            try:
                if len(board) >= 5:
                    r = self.engine.lookup_seven(list(pair) + list(board))
                elif len(board) == 4:
                    # Turn: evaluate best 5 of 6 cards (2 hole + 4 board)
                    import itertools as _it
                    cards_6 = list(pair) + list(board)
                    r = min(self.engine.lookup_five(list(c)) for c in _it.combinations(cards_6, 5))
                elif len(board) >= 3:
                    r = self.engine.lookup_five(list(pair) + list(board[:3]))
                else:
                    continue
            except Exception:
                continue
            strengths[pair] = r

        if not strengths:
            return

        # Rank hands by strength (lower rank = better)
        ranked = sorted(strengths.items(), key=lambda x: x[1])
        n = len(ranked)

        # Apply soft weight adjustment: top hands keep weight,
        # bottom hands get weight reduced (but not to zero)
        for i, (hand, _) in enumerate(ranked):
            position = i / n  # 0.0 = strongest, 1.0 = weakest
            # Weight multiplier: strong hands → 1.0, weak hands → (1 - strength)
            multiplier = 1.0 - strength * position
            self._opp_weights[hand] *= max(multiplier, 0.1)  # floor at 10%

        # Renormalize
        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _narrow_range_by_bet(self, my_bet, opp_bet, board, dead_cards):
        """Narrow opponent range based on bet-to-pot ratio."""
        if self._opp_weights is None:
            return
        raise_amt = opp_bet - my_bet
        pot_before = my_bet * 2
        if raise_amt <= 0 or pot_before <= 0:
            return

        ratio = raise_amt / max(pot_before, 1)
        if ratio <= 0.5:
            keep = 0.85
        elif ratio <= 1.0:
            keep = 0.70
        elif ratio <= 3.0:
            keep = 0.50
        else:
            keep = 0.30

        board_set = set(board)
        strengths = {}
        for pair in self._opp_weights:
            if self._opp_weights[pair] <= 0 or set(pair) & board_set:
                continue
            try:
                if len(board) >= 5:
                    r = self.engine.lookup_seven(list(pair) + list(board))
                elif len(board) >= 3:
                    r = self.engine.lookup_five(list(pair) + list(board[:3]))
                else:
                    r = 999999
            except Exception:
                r = 999999
            strengths[pair] = r

        if not strengths:
            return
        ranked = sorted(strengths.items(), key=lambda x: x[1])
        cutoff = int(len(ranked) * keep)
        if cutoff < len(ranked):
            for hand, _ in ranked[cutoff:]:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _handle_postflop(self, observation, my_cards, board,
                         opp_discards, my_discards, info):
        """Postflop decision.

        Priority:
            1. Compute/narrow opponent range
            2. Flop: multi-street blueprint (backward induction, best quality)
            3. Turn: one-hand CFR solver (200 iters, equity-based)
            4. River: one-hand CFR solver (200 iters, deterministic)
        """

        dead = my_discards + opp_discards
        my_bet, opp_bet = observation["my_bet"], observation["opp_bet"]
        street = observation["street"]
        valid = observation["valid_actions"]

        # Track time ourselves — observation may not include time_left
        import time as _time
        time_left = max(0, self._time_budget - (_time.time() - self._match_start))

        # Pot control: track streets raised
        if street != self._current_street:
            # New street — opponent continued (didn't fold)
            if self._current_street >= 1 and self._opp_weights is not None:
                # MDF-based narrowing: use our last bet size to compute
                # how tight the opponent's calling range should be.
                # Use PREVIOUS street's board (what opponent saw when deciding
                # to call). Current board already has the new card dealt.
                prev_board = board[:len(board) - 1] if len(board) > 3 else board
                if self._last_hero_bet > 0 and self._last_pot_before > 0:
                    bet_frac = self._last_hero_bet / self._last_pot_before
                    mdf = 1.0 / (1.0 + bet_frac)  # game theory MDF
                    self._mdf_narrow_range(prev_board, dead, mdf)
                # else: check-check — both showed weakness, range stays wide.
                # Don't narrow: opponent chose not to bet strong hands.
            self._current_street = street
            self._raised_this_street = False
            self._narrowed_this_street = False
            self._last_hero_bet = 0
            self._last_pot_before = 0
            self._opp_bet_at_raise = 0

        # Opponent range inference
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards)

        # Range update when opponent bets
        if opp_bet > my_bet:
            self._opp_bet_this_hand = True
        if opp_bet > my_bet and self._opp_weights is not None:
            if street == 1:
                if not self._narrowed_this_street:
                    self._narrowed_this_street = True
                    # Flop: Bayesian update from blueprint P(action|hand)
                    if not self._bayesian_range_update(board, my_bet, opp_bet, street):
                        self._soft_narrow_range(my_bet, opp_bet, board, dead)
            elif street in (2, 3):
                if not self._narrowed_this_street:
                    self._narrowed_this_street = True
                    # Polarized narrowing with per-street tracked frequency.
                    # Tested: more accurate than CFR Bayesian (54% vs 40%
                    # at predicting actual opponent hand across 35 scenarios).
                    self._polarized_narrow_range(board, dead, my_bet, opp_bet)
                else:
                    # Re-raise: two-phase game-theoretic narrowing
                    self._reraise_narrow_range(board, dead, my_bet, opp_bet)

        pot_state = (my_bet, opp_bet)

        # Determine hero's position from cached blind_pos (detected at preflop)
        blind_pos = self._blind_pos
        hero_position = 1 if blind_pos == 0 else 0  # SB=second(1), BB=first(0)
        hero_is_first = (hero_position == 0)

        # 1. Flop: multi-street blueprint (backward induction)
        if street == 1 and self._multi_street is not None:
            try:
                strat = self._multi_street.get_strategy(
                    my_cards, board, pot_state=pot_state,
                    hero_position=hero_position)
                action = self._try_strategy(strat, observation)
                if action is not None:
                    self._path_counts['ms_flop'] += 1
                    # Track our bet for MDF narrowing on next street
                    if action[0] == RAISE:
                        self._last_hero_bet = action[1]
                        self._last_pot_before = pot_state[0] + pot_state[1]
                        self._opp_bet_at_raise = opp_bet
                    return action
            except Exception:
                pass

        # 2. Turn: equity thresholds (proven 133% recovery rate in field)
        #    Turn lookahead overcalled — 32% WR on invested hands vs 56% on checks.
        #    One-hand solver for facing bets to avoid raise-then-fold pattern.
        if street == 2:
            self._path_counts['ms_turn'] += 1
            facing_bet = opp_bet > my_bet

            if facing_bet and time_left > 100:
                result = self.solver.solve_and_act(
                    hero_cards=my_cards, board=board,
                    opp_range=self._opp_weights, dead_cards=dead,
                    my_bet=my_bet, opp_bet=opp_bet, street=street,
                    min_raise=observation["min_raise"],
                    max_raise=observation["max_raise"],
                    valid_actions=valid, hero_is_first=True,
                    time_remaining=time_left)
                if result is not None:
                    if result[0] == RAISE:
                        self._raised_this_street = True
                        self._streets_raised += 1
                        self._last_hero_bet = result[1]
                        self._last_pot_before = my_bet + opp_bet
                        self._opp_bet_at_raise = opp_bet
                    return result

            return self._equity_threshold_play(
                my_cards, board, dead, observation, valid, street)

        # 3. River: equity thresholds acting first (bets 20-25%, includes bluffs)
        #    range solver facing bets (range-balanced call/fold/raise)
        if street == 3:
            facing_bet = opp_bet > my_bet

            if facing_bet and time_left > 50:
                self._path_counts['range_solver'] += 1
                # Range solver for call/fold: solves ALL hero hands for
                # coordinated calling strategy (can't be exploited by
                # opponent varying bet sizing).
                result = self.range_solver.solve_and_act(
                    hero_cards=my_cards, board=board,
                    opp_range=self._opp_weights, dead_cards=dead,
                    my_bet=my_bet, opp_bet=opp_bet, street=street,
                    min_raise=observation["min_raise"],
                    max_raise=observation["max_raise"],
                    valid_actions=valid, time_remaining=time_left)
                if result is not None:
                    return result

            # Acting first OR fallback: equity thresholds
            # Bets more aggressively than the range solver (20-25% vs 14%)
            # and includes bluffs — better for exploiting this field.
            return self._equity_threshold_play(
                my_cards, board, dead, observation, valid, street)

        # 4. Flop fallback: equity thresholds (if blueprint missed)
        if street == 1:
            return self._equity_threshold_play(
                my_cards, board, dead, observation, valid, street)

        # 5. Fallback
        if time_left > 30:
            self._path_counts['one_hand_solver'] += 1
            return self._solve_street(
                my_cards, board, dead, my_bet, opp_bet, street,
                observation, valid, hero_is_first, time_left)

        # 5. Emergency: check/fold
        self._path_counts['emergency'] += 1
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _equity_threshold_play(self, my_cards, board, dead, observation, valid, street):
        """Equity thresholds + pot control for turn and river.

        Simple, consistent, no solver convergence issues.
        Proven to beat the complex solver approach by +3464 chips in self-play.
        """
        import random as _random

        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot = my_bet + opp_bet
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        continue_cost = opp_bet - my_bet
        pot_odds = continue_cost / (continue_cost + pot) if continue_cost > 0 else 0

        equity = self.engine.compute_equity(my_cards, board, dead, self._opp_weights)

        # For BETTING decisions: compute equity against opponent's CALLING range.
        # When we bet, weak hands fold — only strong hands call.
        # MDF = 1/(1+bet/pot). If we bet 50% pot, they call with top 67%.
        # Our equity vs calling range is LOWER than vs full range.
        bet_eq = equity  # default: same as full range equity
        if self._opp_weights and valid[RAISE]:
            # Estimate equity vs calling range for a typical bet size (50% pot)
            bet_frac = 0.5
            call_mdf = 1.0 / (1.0 + bet_frac)  # ~67% of hands call
            # Temporarily narrow to calling range and compute equity
            import copy
            temp_weights = copy.copy(self._opp_weights)
            self._mdf_narrow_range(board, dead, call_mdf)
            bet_eq = self.engine.compute_equity(my_cards, board, dead, self._opp_weights)
            self._opp_weights = temp_weights  # restore

        # Pot control: can only raise on 2 streets max unless near-nuts
        can_raise = self._streets_raised < 2 or bet_eq > 0.92

        # All-in with near-nuts (vs calling range)
        if bet_eq > 0.92 and valid[RAISE]:
            self._raised_this_street = True
            self._streets_raised += 1
            self._last_hero_bet = max_raise
            self._last_pot_before = pot
            self._opp_bet_at_raise = opp_bet
            return (RAISE, max_raise, 0, 0)

        # Strong value raise (vs calling range)
        if bet_eq > 0.82 and valid[RAISE] and can_raise:
            self._raised_this_street = True
            self._streets_raised += 1
            amt = max(int(pot * 0.65), min_raise)
            amt = min(amt, max_raise)
            self._last_hero_bet = amt
            self._last_pot_before = pot
            self._opp_bet_at_raise = opp_bet
            return (RAISE, amt, 0, 0)

        # Medium value raise (vs calling range)
        if bet_eq > 0.72 and valid[RAISE] and can_raise:
            self._raised_this_street = True
            self._streets_raised += 1
            amt = max(int(pot * 0.4), min_raise)
            amt = min(amt, max_raise)
            self._last_hero_bet = amt
            self._last_pot_before = pot
            self._opp_bet_at_raise = opp_bet
            return (RAISE, amt, 0, 0)

        # GTO bluff with near-zero equity hands
        # Bluff frequency = bet/(bet+pot) scaled by street
        # Computed from equilibrium: turn bets at 78% of river frequency.
        street_bluff_scale = {2: 0.78, 3: 1.0}.get(street, 0)
        if equity < 0.15 and valid[RAISE] and can_raise and street_bluff_scale > 0:
            bet_size = max(int(pot * 0.6), min_raise)
            bluff_freq = bet_size / (bet_size + pot) if pot > 0 else 0.1
            if _random.random() < bluff_freq * street_bluff_scale:
                self._raised_this_street = True
                self._streets_raised += 1
                self._opp_bet_at_raise = opp_bet
                return (RAISE, min(bet_size, max_raise), 0, 0)

        # Call if equity justifies (against polarized-narrowed range)
        if valid[CALL] and equity >= pot_odds:
            return (CALL, 0, 0, 0)

        # Check
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    def _solve_turn_with_river(self, my_cards, board, dead, my_bet, opp_bet,
                               observation, valid, hero_is_first, time_left):
        """Solve turn with backward induction from river sub-games.

        For each possible river card, solve the river betting game to get
        correct continuation values. This accounts for river betting —
        weak hands get lower EV (face bets, fold), strong hands get higher
        EV (value bet, get called).
        """
        from game_tree import GameTree, TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN

        if self._opp_weights is None:
            return self._solve_street(
                my_cards, board, dead, my_bet, opp_bet, 2,
                observation, valid, hero_is_first, time_left)

        known = set(my_cards) | set(board) | set(dead)
        opp_hands = []
        opp_weights_list = []
        for hand, weight in self._opp_weights.items():
            if weight > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights_list.append(weight)

        if not opp_hands:
            return self._solve_street(
                my_cards, board, dead, my_bet, opp_bet, 2,
                observation, valid, hero_is_first, time_left)

        opp_w = np.array(opp_weights_list, dtype=np.float64)
        opp_w /= opp_w.sum()
        n_opp = len(opp_hands)

        # Build river tree (cached by pot)
        river_tree = self.solver._get_tree(my_bet, opp_bet, 2, 100, hero_is_first)

        # Set up CFR arrays for river tree
        hero_idx = {nid: i for i, nid in enumerate(river_tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(river_tree.opp_node_ids)}
        max_act = max(
            max((river_tree.num_actions[nid] for nid in river_tree.hero_node_ids), default=1),
            max((river_tree.num_actions[nid] for nid in river_tree.opp_node_ids), default=1), 1)
        n_hero_r = len(river_tree.hero_node_ids)
        n_opp_r = len(river_tree.opp_node_ids)

        # Solve each river card
        remaining = [c for c in range(27) if c not in known]
        river_cards = [c for c in remaining if c not in my_cards]
        river_ev_sum = np.zeros(n_opp, dtype=np.float64)
        n_river = 0

        for rc in river_cards:
            board_5 = list(board) + [rc]

            # Compute river equity (deterministic)
            hero_rank = self.engine.lookup_seven(list(my_cards) + board_5)
            equity_vec = np.zeros(n_opp, dtype=np.float64)
            for i, oh in enumerate(opp_hands):
                if rc in oh:
                    continue
                opp_rank = self.engine.lookup_seven(list(oh) + board_5)
                if hero_rank < opp_rank:
                    equity_vec[i] = 1.0
                elif hero_rank == opp_rank:
                    equity_vec[i] = 0.5

            # Build terminal values
            tv = {}
            for nid in river_tree.terminal_node_ids:
                tt = river_tree.terminal[nid]
                hp, op = river_tree.hero_pot[nid], river_tree.opp_pot[nid]
                if tt == TERM_FOLD_HERO:
                    tv[nid] = np.full(n_opp, -hp, dtype=np.float64)
                elif tt == TERM_FOLD_OPP:
                    tv[nid] = np.full(n_opp, op, dtype=np.float64)
                elif tt == TERM_SHOWDOWN:
                    pot_won = min(hp, op)
                    tv[nid] = (2.0 * equity_vec - 1.0) * pot_won

            # Run 75 CFR iterations for each river card — decent convergence
            # while staying under 5s per-action ARM limit.
            hero_reg = np.zeros((n_hero_r, max_act), dtype=np.float64)
            hero_ss = np.zeros((n_hero_r, max_act), dtype=np.float64)
            opp_reg = np.zeros((n_opp_r, n_opp, max_act), dtype=np.float64)

            root_val = None
            for t in range(75):
                root_val = self.solver._cfr_traverse(
                    river_tree, 0, 1.0, opp_w,
                    hero_reg, hero_ss, opp_reg,
                    hero_idx, opp_idx, tv, n_opp, max_act, t)

            river_ev_sum += root_val
            n_river += 1

        # Average → turn continuation values
        continuation_ev = river_ev_sum / max(n_river, 1)

        # Solve turn with continuation values at showdown terminals
        turn_tree = self.solver._get_tree(my_bet, opp_bet, 2, 100, hero_is_first)

        # Find reference pot (check-check = min showdown pot)
        sd_pots = [turn_tree.hero_pot[nid] + turn_tree.opp_pot[nid]
                   for nid in turn_tree.terminal_node_ids
                   if turn_tree.terminal[nid] == TERM_SHOWDOWN]
        ref_pot = min(sd_pots) if sd_pots else (my_bet + opp_bet)

        turn_tv = {}
        for nid in turn_tree.terminal_node_ids:
            tt = turn_tree.terminal[nid]
            hp, op = turn_tree.hero_pot[nid], turn_tree.opp_pot[nid]
            if tt == TERM_FOLD_HERO:
                turn_tv[nid] = np.full(n_opp, -hp, dtype=np.float64)
            elif tt == TERM_FOLD_OPP:
                turn_tv[nid] = np.full(n_opp, op, dtype=np.float64)
            elif tt == TERM_SHOWDOWN:
                actual_pot = hp + op
                scale = actual_pot / ref_pot if ref_pot > 0 else 1.0
                turn_tv[nid] = continuation_ev * scale

        # Determine iterations based on time
        if time_left > 800:
            turn_iters = 400
        elif time_left > 500:
            turn_iters = 200
        else:
            turn_iters = 100

        turn_strategy = self.solver._run_cfr(
            turn_tree, opp_w, turn_tv, n_opp, turn_iters)

        return self.solver._strategy_to_action(
            turn_tree, turn_strategy, my_bet, opp_bet,
            observation["min_raise"], observation["max_raise"],
            valid)

    def _solve_street(self, my_cards, board, dead, my_bet, opp_bet,
                      street, observation, valid, hero_is_first, time_left):
        """Run one-hand CFR solver for the current street."""
        result = self.solver.solve_and_act(
            hero_cards=my_cards, board=board,
            opp_range=self._opp_weights, dead_cards=dead,
            my_bet=my_bet, opp_bet=opp_bet, street=street,
            min_raise=observation["min_raise"],
            max_raise=observation["max_raise"],
            valid_actions=valid, hero_is_first=hero_is_first,
            time_remaining=time_left)
        if result is not None:
            return result
        # Solver returned None (no opp range) — use equity fallback
        equity = self.engine.compute_equity(
            my_cards, board, dead, self._opp_weights)
        if opp_bet > my_bet and valid[CALL]:
            cost = opp_bet - my_bet
            pot = my_bet + opp_bet
            threshold = cost / (pot + cost) if (pot + cost) > 0 else 0.5
            if equity >= threshold:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0) if valid[FOLD] else (CHECK, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  MAIN
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        if reward != 0:
            self._hand_reward += reward

        # Check if preload finished
        if not self._multi_street_loaded and _PRELOAD.get('done'):
            self._multi_street = _PRELOAD.get('multi_street')
            self._multi_street_loaded = True




        # Lead protection — if we can survive on blinds alone, fold everything
        # Binary ELO: winning by 1 chip = winning by 1000. Protect guaranteed wins.
        actual_bankroll = self._bankroll
        hands_remaining = self._total_hands - hand_number
        blind_cost = hands_remaining * 1.5
        if actual_bankroll > blind_cost + 10:
            valid = observation["valid_actions"]
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[FOLD]:
                self._we_folded_this_hand = True
                return (FOLD, 0, 0, 0)
            if valid[DISCARD]:
                my, board, opp_d, _ = self._parse_cards(observation)
                return self._handle_discard(observation, my, board, opp_d)
            self._we_folded_this_hand = True
            return (FOLD, 0, 0, 0)

        my_cards, board, opp_d, my_d = self._parse_cards(observation)
        valid = observation["valid_actions"]

        if valid[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_d)
        if observation["street"] == 0:
            action = self._handle_preflop(observation, my_cards)
        else:
            action = self._handle_postflop(observation, my_cards, board,
                                           opp_d, my_d, info)

        if action[0] == FOLD:
            self._we_folded_this_hand = True
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)
        if reward != 0:
            self._hand_reward += reward

        # Track opponent aggression per street
        street = observation["street"]
        if street in (1, 2, 3):
            opp_bet = observation["opp_bet"]
            my_bet = observation["my_bet"]
            self._opp_actions_by_street[street] += 1
            if opp_bet > my_bet:
                self._opp_bets_by_street[street] += 1
            # Keep legacy river counters for showdown tracking
            if street == 3:
                if opp_bet > my_bet:
                    self._opp_river_bets += 1
                self._opp_river_actions += 1

        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_d) == 3 and self._opp_weights is None:
            my = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_d, board, my)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
