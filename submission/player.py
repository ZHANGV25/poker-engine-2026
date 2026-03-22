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
from depth_limited_solver import DepthLimitedSolver
from river_lookup import RiverLookup
from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

# Preload at module import time.
# Phase 1 (synchronous): load flop from npz (fast, ~4s)
# Phase 2 (background): decompress turn+opp from LZMA (slow, ~10s)
import threading
_PRELOAD = {'multi_street': None, 'river_data': None, 'done': False, 'deferred_done': False}
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
    # Extract river.tar.lzma to /tmp if needed
    try:
        _river_archive = os.path.join(_dir, "data", "river.tar.lzma")
        _river_extract = "/tmp/river_extract"
        _river_dir_check = os.path.join(_dir, "data", "river")
        if os.path.isfile(_river_archive) and not os.path.isdir(_river_dir_check):
            import tarfile
            os.makedirs(_river_extract, exist_ok=True)
            with lzma.open(_river_archive) as lz:
                with tarfile.open(fileobj=lz) as tar:
                    tar.extractall(_river_extract)
            _PRELOAD['river_extracted'] = os.path.join(_river_extract, "river")
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
    VERSION = "5.0"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        import logging
        _log = logging.getLogger(__name__)
        _log.info(f"PlayerAgent {self.VERSION} starting")
        # Log C solver and flop DL status after init
        try:
            from range_solver import _USE_C_SOLVER
            _log.info(f"C solver: {'ACTIVE' if _USE_C_SOLVER else 'UNAVAILABLE (Python fallback)'}")
        except:
            _log.info("C solver: UNAVAILABLE")

        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)
        self.range_solver = RangeSolver(self.engine)
        self.depth_limited_solver = DepthLimitedSolver(self.engine, self.range_solver)

        # Precomputed river strategies (from EC2, loaded if available)
        # Check: 1) direct files in data/river/, 2) extracted from tar.lzma
        _river_dir = os.path.join(_dir, "data", "river")
        _river_extracted = _PRELOAD.get('river_extracted')
        if os.path.isdir(_river_dir):
            self.river_lookup = RiverLookup(_river_dir)
        elif _river_extracted and os.path.isdir(_river_extracted):
            self.river_lookup = RiverLookup(_river_extracted)
        else:
            self.river_lookup = RiverLookup()  # empty, will use runtime solver

        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()
        # Use preloaded data from module-level (loaded before server starts).
        self._multi_street = _PRELOAD.get('multi_street')
        self._multi_street_loaded = _PRELOAD.get('done', False)
        self._river_data = _PRELOAD.get('river_data')  # precomputed P(bet|hand)
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
        self._street_start_bet = 0  # both players' bet level at start of current street

        # Track opponent betting patterns per street for adaptive narrowing
        self._opp_bets_by_street = {1: 0, 2: 0, 3: 0}   # flop, turn, river
        self._opp_actions_by_street = {1: 0, 2: 0, 3: 0}
        # Track opponent folds to our bets (for adaptive bluffing)
        self._opp_folds_to_bet = {1: 0, 2: 0, 3: 0}
        self._opp_faces_bet = {1: 0, 2: 0, 3: 0}
        # Keep old names for backward compat with adaptive showdown tracking
        self._opp_river_bets = 0
        self._opp_river_actions = 0
        self._opp_bet_showdown_wins = 0   # times they bet, went to showdown, won
        self._opp_bet_showdown_total = 0  # times they bet and went to showdown
        self._opp_bet_this_hand = False   # did opponent bet this hand
        self._we_folded_this_hand = False
        # Track our bet-called WR: when we bet and opponent calls, did we win?
        # Used to detect selective callers who only call with better.
        self._hero_bet_called_wins = 0
        self._hero_bet_called_total = 0
        self._hero_bet_this_hand = False  # did we bet this hand

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

            # Track our bet-called WR (when we bet and opponent didn't fold)
            if (self._hero_bet_this_hand and not self._we_folded_this_hand
                    and self._current_street == 3 and self._hand_reward != 0):
                self._hero_bet_called_total += 1
                if self._hand_reward > 0:
                    self._hero_bet_called_wins += 1

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
            self._hero_bet_this_hand = False
            self._river_reweighted = False
            self._narrowed_this_street = False
            self._opp_bet_at_raise = 0
            self._street_start_bet = 0

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

            # Compute blueprint's average bet frequency for this board/pot
            blueprint_avg_bet = float(p_bet_per_hand.mean())

            # Compare with tracked opponent bet frequency
            obs_street = street  # 1=flop, 2=turn
            tracked_freq = None
            if self._opp_actions_by_street.get(obs_street, 0) > 15:
                tracked_freq = (self._opp_bets_by_street[obs_street] /
                                self._opp_actions_by_street[obs_street])

            # Principled mixture model:
            # Opponent = α × GTO_player + (1-α) × random_bettor
            # α = how closely they match GTO:
            #   α = 1 - |f_tracked - f_blueprint| / max(f_tracked, f_blueprint)
            # Random bettor bets each hand at f_tracked (their actual rate).
            if tracked_freq is not None and max(tracked_freq, blueprint_avg_bet) > 0.01:
                alpha = max(0.1, 1.0 - abs(tracked_freq - blueprint_avg_bet) /
                            max(tracked_freq, blueprint_avg_bet, 0.01))
                uniform_bet = tracked_freq
            else:
                alpha = 1.0  # no data yet, trust blueprint
                uniform_bet = blueprint_avg_bet

            # Apply Bayesian update with blended P(bet|hand)
            hands = bd['hands']
            hand_map = ms._hand_maps.get(board_id, {})

            updated = False
            for pair, weight in list(self._opp_weights.items()):
                if weight <= 0:
                    continue
                key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                hand_idx = hand_map.get(key)

                if hand_idx is not None and hand_idx < len(p_bet_per_hand):
                    p_blueprint = float(p_bet_per_hand[hand_idx])
                    # P(bet|hand) = α × P_blueprint + (1-α) × f_tracked
                    p_blended = alpha * p_blueprint + (1.0 - alpha) * uniform_bet
                    self._opp_weights[pair] *= max(p_blended, 0.005)
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

    def _turn_check_bayesian_narrow(self, board, street):
        """Bayesian narrowing when opponent CHECKS, using blueprint P(check|hand).

        P(check|hand) = 1 - P(bet|hand). Hands the blueprint says would
        bet get downweighted (they checked but "shouldn't" have). Hands
        the blueprint says would check keep full weight.

        Board-specific — accounts for slow-playing. On trap-heavy boards,
        strong hands have high P(check) and keep weight. On value-heavy
        boards, strong hands that check get downweighted.

        Uses same trust blending as bet narrowing.
        """
        if self._opp_weights is None:
            return

        try:
            if street == 2 and self._multi_street is not None:
                # Turn: use turn opp data
                if len(board) < 4:
                    return
                p_bet_map = self._multi_street.get_turn_opp_bet_prob(
                    board[:3], board[3], (2, 2))  # use small pot for check scenario
                if p_bet_map is None:
                    return
            elif street == 1 and self._multi_street is not None:
                # Flop: use flop blueprint opp data
                # Reuse the existing _bayesian_range_update logic but for checks
                # Get P(bet|hand) from flop blueprint
                ms = self._multi_street
                flop = tuple(sorted(int(c) for c in board[:3]))
                board_id = ms._find_board(flop)
                if board_id is None:
                    return
                bd = ms._boards[board_id]
                if 'opp_strategies' not in bd:
                    return
                pot_idx = ms._find_pot((2, 2), bd['pot_sizes'])
                opp_node_idx = ms._find_opp_node(2, 2, board_id, pot_idx)
                opp_strats = bd['opp_strategies']
                if pot_idx >= opp_strats.shape[0] or opp_node_idx >= opp_strats.shape[2]:
                    return
                node_strats = opp_strats[pot_idx, :, opp_node_idx, :]
                node_acts = bd['opp_action_types'][pot_idx, opp_node_idx, :]
                # Sum P(bet|hand) across raise actions
                p_bet_arr = np.zeros(node_strats.shape[0], dtype=np.float64)
                for a_idx in range(len(node_acts)):
                    act = int(node_acts[a_idx])
                    if act in (3, 4, 5, 6):
                        p_bet_arr += node_strats[:, a_idx].astype(np.float64) / 255.0
                # Build hand map
                hand_map = ms._hand_maps.get(board_id, {})
                p_bet_map = {}
                for pair in self._opp_weights:
                    key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                    hand_idx = hand_map.get(key)
                    if hand_idx is not None and hand_idx < len(p_bet_arr):
                        p_bet_map[key] = float(p_bet_arr[hand_idx])
                if not p_bet_map:
                    return
            else:
                return

            # Compute blueprint average for trust blending
            p_values = list(p_bet_map.values())
            blueprint_avg = sum(p_values) / len(p_values) if p_values else 0.2

            # Trust blending with tracked frequency
            obs_street = street
            tracked_freq = None
            if self._opp_actions_by_street.get(obs_street, 0) > 15:
                tracked_freq = (self._opp_bets_by_street[obs_street] /
                                self._opp_actions_by_street[obs_street])

            if tracked_freq is not None and max(tracked_freq, blueprint_avg) > 0.01:
                alpha = max(0.1, 1.0 - abs(tracked_freq - blueprint_avg) /
                            max(tracked_freq, blueprint_avg, 0.01))
            else:
                alpha = 1.0

            # Apply P(check|hand) = 1 - P(bet|hand)
            for pair, weight in list(self._opp_weights.items()):
                if weight <= 0:
                    continue
                key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                p_bet = p_bet_map.get(key)
                if p_bet is not None:
                    # Blend: P(check) from blueprint vs uniform
                    p_check_blueprint = 1.0 - p_bet
                    # If opponent checks more than GTO, trust check less
                    # (everyone checks, so check is less informative)
                    p_check_uniform = 1.0 - (tracked_freq if tracked_freq else blueprint_avg)
                    p_check = alpha * p_check_blueprint + (1.0 - alpha) * p_check_uniform
                    self._opp_weights[pair] *= max(p_check, 0.005)

            total = sum(self._opp_weights.values())
            if total > 0:
                for k in self._opp_weights:
                    self._opp_weights[k] /= total

        except Exception:
            pass

    def _turn_bayesian_narrow(self, board, my_bet, opp_bet):
        """Bayesian narrowing on turn using precomputed P(bet|hand).

        Same principle as flop Bayesian: multiply each hand's weight by
        P(bet|hand) from the precomputed turn blueprint. Board-specific,
        pot-specific, equilibrium-derived.

        Returns True if applied, False if no turn data available.
        """
        if self._opp_weights is None or self._multi_street is None:
            return False

        try:
            flop = board[:3]
            turn_card = board[3] if len(board) >= 4 else None
            if turn_card is None:
                return False

            p_bet_map = self._multi_street.get_turn_opp_bet_prob(
                flop, turn_card, (my_bet, opp_bet))

            if p_bet_map is None or len(p_bet_map) < 3:
                return False

            # Compute blueprint average for trust blending
            p_values = list(p_bet_map.values())
            blueprint_avg = sum(p_values) / len(p_values) if p_values else 0.2

            # Track opponent turn bet frequency for trust blending
            obs_street = 2
            tracked_freq = None
            if self._opp_actions_by_street.get(obs_street, 0) > 15:
                tracked_freq = (self._opp_bets_by_street[obs_street] /
                                self._opp_actions_by_street[obs_street])

            # Trust blending (same formula as flop)
            if tracked_freq is not None and max(tracked_freq, blueprint_avg) > 0.01:
                alpha = max(0.1, 1.0 - abs(tracked_freq - blueprint_avg) /
                            max(tracked_freq, blueprint_avg, 0.01))
                uniform_bet = tracked_freq
            else:
                alpha = 1.0
                uniform_bet = blueprint_avg

            # Apply Bayesian update
            updated = False
            for pair, weight in list(self._opp_weights.items()):
                if weight <= 0:
                    continue
                key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                p_blueprint = p_bet_map.get(key)
                if p_blueprint is not None:
                    p_blended = alpha * p_blueprint + (1.0 - alpha) * uniform_bet
                    self._opp_weights[pair] *= max(p_blended, 0.005)
                    updated = True

            if not updated:
                return False

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
        pot_before = my_bet + self._street_start_bet
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
                self._opp_weights[hand] *= 0.005

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
                self._opp_weights[hand] *= 0.005

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    def _bayesian_call_narrow(self, board, hero_bet, pot_before, prev_street, dead_cards=None):
        """Bayesian narrowing when opponent CALLS our bet at street transition.

        Uses P(call|hand) from the blueprint instead of crude MDF.
        Blueprint knows board-specific calling patterns — draws call,
        dominated hands fold, trapping hands sometimes raise.

        Falls back to MDF if blueprint data unavailable.

        Args:
            board: the PREVIOUS street's board (before new card dealt)
            hero_bet: our bet amount that opponent called
            pot_before: pot before our bet
            prev_street: which street the call happened on (1=flop, 2=turn)
        """
        if self._opp_weights is None:
            return self._mdf_narrow_range(board, [], 1.0 / (1.0 + hero_bet / max(pot_before, 1)))

        try:
            if prev_street == 1 and self._multi_street is not None:
                ms = self._multi_street
                flop = tuple(sorted(int(c) for c in board[:3]))
                board_id = ms._find_board(flop)
                if board_id is None:
                    return self._mdf_narrow_range(board, [], 1.0 / (1.0 + hero_bet / max(pot_before, 1)))

                bd = ms._boards[board_id]
                # Flop call: use flop opp_strategies
                if 'opp_strategies' not in bd:
                    return self._mdf_narrow_range(board, [], 1.0 / (1.0 + hero_bet / max(pot_before, 1)))

                # The opponent faced our bet. Find the node where they face it.
                # Our bet put us at (opp_bet_at_raise + hero_bet) vs opp_bet_at_raise
                # From opp's view: they see hero_pot > opp_pot
                facing_bet = self._opp_bet_at_raise  # their bet level when we raised
                our_total = facing_bet + hero_bet  # our bet after raising
                pot_idx = ms._find_pot((our_total, facing_bet), bd['pot_sizes'])

                # Find opp node where they face our bet
                opp_node_idx = ms._find_opp_node(facing_bet, our_total, board_id, pot_idx)

                opp_strats = bd['opp_strategies']
                opp_acts = bd['opp_action_types']

                if pot_idx >= opp_strats.shape[0] or opp_node_idx >= opp_strats.shape[2]:
                    return self._mdf_narrow_range(board, [], 1.0 / (1.0 + hero_bet / max(pot_before, 1)))

                node_strats = opp_strats[pot_idx, :, opp_node_idx, :]
                node_acts = opp_acts[pot_idx, opp_node_idx, :]

                # Find the CALL action index
                call_act_idx = None
                for a_idx in range(len(node_acts)):
                    if int(node_acts[a_idx]) == 2:  # ACT_CALL = 2
                        call_act_idx = a_idx
                        break

                if call_act_idx is None:
                    return self._mdf_narrow_range(board, [], 1.0 / (1.0 + hero_bet / max(pot_before, 1)))

                p_call_per_hand = node_strats[:, call_act_idx].astype(np.float64) / 255.0

                # Also include raise as "continuing" (they didn't fold)
                # P(continue|hand) = P(call|hand) + P(raise|hand)
                for a_idx in range(len(node_acts)):
                    act = int(node_acts[a_idx])
                    if act in (3, 4, 5, 6):  # ACT_RAISE_*
                        p_call_per_hand += node_strats[:, a_idx].astype(np.float64) / 255.0

                # Clamp to [0, 1]
                p_call_per_hand = np.minimum(p_call_per_hand, 1.0)

                # Apply Bayesian update
                hand_map = ms._hand_maps.get(board_id, {})
                updated = False
                for pair, weight in list(self._opp_weights.items()):
                    if weight <= 0:
                        continue
                    key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                    hand_idx = hand_map.get(key)
                    if hand_idx is not None and hand_idx < len(p_call_per_hand):
                        p_cont = float(p_call_per_hand[hand_idx])
                        self._opp_weights[pair] *= max(p_cont, 0.005)
                        updated = True

                if updated:
                    total = sum(self._opp_weights.values())
                    if total > 0:
                        for k in self._opp_weights:
                            self._opp_weights[k] /= total
                    return

            if prev_street == 2:
                # Turn→River: runtime solve P(continue|hand) at facing-bet node.
                # Uses compute_opp_call_probs which builds the correct tree
                # (opponent faces our bet, decides fold/call/raise).
                import itertools as _it3
                _dead = dead_cards or []
                _known = set(board) | set(_dead)
                _hero_range = {}
                for _h in _it3.combinations(range(27), 2):
                    if not (set(_h) & _known):
                        _hero_range[_h] = 1.0
                _hr_total = sum(_hero_range.values())
                for _k in _hero_range:
                    _hero_range[_k] /= _hr_total

                facing_bet = self._opp_bet_at_raise
                our_total = facing_bet + hero_bet
                _p_cont = self.range_solver.compute_opp_call_probs(
                    board=board, opp_range=self._opp_weights,
                    hero_range=_hero_range, dead_cards=_dead,
                    hero_bet=our_total, opp_bet_before=facing_bet,
                    street=2, min_raise=2, iterations=200)

                if _p_cont and len(_p_cont) >= 3:
                    updated = False
                    for _pair, _w in list(self._opp_weights.items()):
                        if _w <= 0:
                            continue
                        _key = (min(_pair[0], _pair[1]), max(_pair[0], _pair[1]))
                        _p = _p_cont.get(_key)
                        if _p is not None:
                            self._opp_weights[_pair] *= max(float(_p), 0.005)
                            updated = True
                    if updated:
                        total = sum(self._opp_weights.values())
                        if total > 0:
                            for k in self._opp_weights:
                                self._opp_weights[k] /= total
                        return

            # Fallback: MDF
            mdf = 1.0 / (1.0 + hero_bet / max(pot_before, 1))
            self._mdf_narrow_range(board, [], mdf)

        except Exception:
            mdf = 1.0 / (1.0 + hero_bet / max(pot_before, 1))
            self._mdf_narrow_range(board, [], mdf)

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
                self._opp_weights[hand] *= 0.005

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
        pot_before = my_bet + self._street_start_bet
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
            self._opp_weights[hand] *= max(multiplier, 0.005)  # 0.5% floor

        # Renormalize
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
                # Bayesian call narrowing: use P(call|hand) from blueprint
                # instead of crude MDF. Blueprint knows draws, traps, etc.
                # Use PREVIOUS street's board (what opponent saw when deciding
                # to call). Current board already has the new card dealt.
                prev_board = board[:len(board) - 1] if len(board) > 3 else board
                if self._last_hero_bet > 0 and self._last_pot_before > 0:
                    self._bayesian_call_narrow(
                        prev_board, self._last_hero_bet,
                        self._last_pot_before, self._current_street, dead)
                # else: check-check — both showed weakness, range stays wide.
            self._current_street = street
            self._raised_this_street = False
            self._narrowed_this_street = False
            self._last_hero_bet = 0
            self._last_pot_before = 0
            self._opp_bet_at_raise = 0
            self._street_start_bet = my_bet  # both bets equal at street start

        # Determine hero's position from cached blind_pos (detected at preflop)
        blind_pos = self._blind_pos
        hero_position = 1 if blind_pos == 0 else 0  # SB=second(1), BB=first(0)
        hero_is_first = (hero_position == 0)

        # Opponent range inference
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards)

        # Range update from opponent actions
        if opp_bet > my_bet:
            self._opp_bet_this_hand = True

        if self._opp_weights is not None and not self._narrowed_this_street:
            if opp_bet > my_bet:
                # Opponent BET: narrow range toward betting hands
                self._narrowed_this_street = True
                if street == 1:
                    # Flop: runtime P(bet|hand) when C solver available,
                    # else blueprint Bayesian, else soft heuristic
                    flop_narrowed = False
                    if _USE_C_SOLVER and time_left > 300:
                        try:
                            remaining = [c for c in range(27)
                                         if c not in set(board) | set(dead)]
                            hero_range = {}
                            n_pairs = len(remaining) * (len(remaining)-1) // 2
                            for h in itertools.combinations(remaining, 2):
                                hero_range[h] = 1.0 / max(n_pairs, 1)
                            p_bet_rt = self.range_solver.compute_opp_bet_probs(
                                board, self._opp_weights, hero_range, dead,
                                my_bet, opp_bet, street, min_raise=2,
                                iterations=150)
                            if p_bet_rt:
                                for pair, w in list(self._opp_weights.items()):
                                    if w <= 0: continue
                                    key = (min(pair[0],pair[1]),max(pair[0],pair[1]))
                                    pb = p_bet_rt.get(key, 0.5)
                                    self._opp_weights[pair] = w * max(pb, 0.005)
                                total_w = sum(self._opp_weights.values())
                                if total_w > 0:
                                    for k in self._opp_weights:
                                        self._opp_weights[k] /= total_w
                                flop_narrowed = True
                        except Exception:
                            pass
                    if not flop_narrowed:
                        if not self._bayesian_range_update(board, my_bet, opp_bet, street):
                            self._soft_narrow_range(my_bet, opp_bet, board, dead)
                elif street == 2:
                    # Turn: runtime P(bet|hand) when C solver available,
                    # else blueprint Bayesian, else polarized heuristic
                    turn_narrowed = False
                    if _USE_C_SOLVER and time_left > 200:
                        try:
                            remaining = [c for c in range(27)
                                         if c not in set(board) | set(dead)]
                            hero_range = {}
                            n_pairs = len(remaining) * (len(remaining)-1) // 2
                            for h in itertools.combinations(remaining, 2):
                                hero_range[h] = 1.0 / max(n_pairs, 1)
                            p_bet_rt = self.range_solver.compute_opp_bet_probs(
                                board, self._opp_weights, hero_range, dead,
                                my_bet, opp_bet, street, min_raise=2,
                                iterations=150)
                            if p_bet_rt:
                                for pair, w in list(self._opp_weights.items()):
                                    if w <= 0: continue
                                    key = (min(pair[0],pair[1]),max(pair[0],pair[1]))
                                    pb = p_bet_rt.get(key, 0.5)
                                    self._opp_weights[pair] = w * max(pb, 0.005)
                                total_w = sum(self._opp_weights.values())
                                if total_w > 0:
                                    for k in self._opp_weights:
                                        self._opp_weights[k] /= total_w
                                turn_narrowed = True
                        except Exception:
                            pass
                    if not turn_narrowed:
                        if (self._multi_street is not None and
                                not self._turn_bayesian_narrow(board, my_bet, opp_bet)):
                            self._polarized_narrow_range(board, dead, my_bet, opp_bet)
                else:
                    # River: narrowing happens later in the river-specific
                    # code block via runtime compute_opp_bet_probs().
                    pass
            elif opp_bet == my_bet and not hero_is_first and street in (1, 2):
                # Opponent CHECKED: Bayesian update with P(check|hand) from
                # blueprint. Board-specific — accounts for slow-playing
                # (strong hands that GTO checks sometimes to trap).
                # Only on flop/turn where we have blueprint data.
                self._narrowed_this_street = True
                self._turn_check_bayesian_narrow(board, street)

        elif opp_bet > my_bet and self._opp_weights is not None and self._narrowed_this_street:
            # Re-raise: two-phase game-theoretic narrowing
            self._reraise_narrow_range(board, dead, my_bet, opp_bet)

        pot_state = (my_bet, opp_bet)

        # Check C solver availability (used by flop DL and turn DL)
        try:
            from range_solver import _USE_C_SOLVER
        except ImportError:
            _USE_C_SOLVER = False

        # 1. Flop: multi-street blueprint for ALL decisions.
        #    Blueprint uses backward induction (river→turn→flop) with
        #    continuation values — accounts for future street betting.
        #    Runtime solver only sees current-street equity (no future streets)
        #    which caused turn to regress from +6.4 to 0.0/hand.
        if street == 1 and self._multi_street is not None:
            # Flop depth-limited solver: solve against narrowed range with
            # turn+river continuation values. Works for BOTH AF and FB.
            # Replaces blueprint entirely when C solver is available.
            facing_flop_bet = opp_bet > my_bet
            if (_USE_C_SOLVER
                    and self._opp_weights is not None
                    and time_left > 200):
                try:
                    result = self.depth_limited_solver.solve_flop_facing_bet(
                        my_cards, board, self._opp_weights, dead,
                        my_bet, opp_bet,
                        observation["min_raise"], observation["max_raise"],
                        valid, time_left, hero_position=hero_position)
                    if result is not None:
                        self._path_counts['flop_dl'] = self._path_counts.get('flop_dl', 0) + 1
                        if result[0] == RAISE:
                            self._raised_this_street = True
                            self._streets_raised += 1
                            self._last_hero_bet = result[1]
                            self._last_pot_before = pot_state[0] + pot_state[1]
                            self._opp_bet_at_raise = opp_bet
                        if result[0] == FOLD:
                            self._we_folded_this_hand = True
                        return result
                except Exception:
                    pass  # fall through to blueprint

            try:
                strat = self._multi_street.get_strategy(
                    my_cards, board, pot_state=pot_state,
                    hero_position=hero_position)
                action = self._try_strategy(strat, observation)
                if action is not None:
                    # Equity gate: blueprint was solved against uniform range.
                    if (action[0] == CALL and self._opp_weights is not None
                            and opp_bet > my_bet):
                        continue_cost = opp_bet - my_bet
                        pot = my_bet + opp_bet
                        pot_odds = continue_cost / (continue_cost + pot)
                        eq = self.engine.compute_equity(
                            my_cards, board, dead, self._opp_weights)
                        if eq < pot_odds:
                            self._we_folded_this_hand = True
                            return (FOLD, 0, 0, 0)

                    self._path_counts['ms_flop'] += 1
                    if action[0] == RAISE:
                        self._last_hero_bet = action[1]
                        self._last_pot_before = pot_state[0] + pot_state[1]
                        self._opp_bet_at_raise = opp_bet
                    return action
            except Exception:
                pass

        # 2. Turn: blueprint for ALL decisions.
        #    Same reason as flop: backward induction accounts for river
        #    continuation values. Runtime solver regressed turn from +6.4 to 0.0/hand.
        if street == 2:
            self._path_counts['ms_turn'] += 1

            facing_bet = opp_bet > my_bet

            # Facing a bet: depth-limited solver (narrowed range + continuation values)
            # v32 architecture (70% WR). The solver accounts for the narrowed
            # opponent range that the blueprint ignores.
            if facing_bet and time_left > 100 and self._opp_weights is not None:
                self._path_counts['range_solver'] += 1
                result = self.depth_limited_solver.solve_turn_facing_bet(
                    hero_cards=my_cards, board=board,
                    opp_range=self._opp_weights, dead_cards=dead,
                    my_bet=my_bet, opp_bet=opp_bet,
                    min_raise=observation["min_raise"],
                    max_raise=observation["max_raise"],
                    valid_actions=valid, time_remaining=time_left,
                    hero_position=hero_position)
                if result is not None:
                    if result[0] == RAISE:
                        self._raised_this_street = True
                        self._streets_raised += 1
                        self._last_hero_bet = result[1]
                        self._last_pot_before = pot_state[0] + pot_state[1]
                        self._opp_bet_at_raise = opp_bet
                    return result

            # Acting first: depth-limited solver when C solver available.
            # Previous AF solver regression (v33/v34) was SINGLE-STREET.
            # Depth-limited uses river continuation values (multi-street).
            if (not facing_bet and _USE_C_SOLVER
                    and self._opp_weights is not None
                    and time_left > 200):
                try:
                    result = self.depth_limited_solver.solve_turn_facing_bet(
                        hero_cards=my_cards, board=board,
                        opp_range=self._opp_weights, dead_cards=dead,
                        my_bet=my_bet, opp_bet=opp_bet,
                        min_raise=observation["min_raise"],
                        max_raise=observation["max_raise"],
                        valid_actions=valid, time_remaining=time_left,
                        hero_position=hero_position)
                    if result is not None:
                        self._path_counts['turn_dl_af'] = self._path_counts.get('turn_dl_af', 0) + 1
                        if result[0] == RAISE:
                            self._raised_this_street = True
                            self._streets_raised += 1
                            self._last_hero_bet = result[1]
                            self._last_pot_before = pot_state[0] + pot_state[1]
                            self._opp_bet_at_raise = opp_bet
                        return result
                except Exception:
                    pass

            # Fallback: blueprint (backward induction, multi-street)
            if self._multi_street is not None:
                try:
                    strat = self._multi_street.get_turn_strategy(
                        my_cards, board, pot_state=pot_state,
                        hero_position=hero_position)
                    action = self._try_strategy(strat, observation)
                    if action is not None:
                        # Turn equity gate: fold when equity < pot odds
                        if (action[0] == CALL and self._opp_weights is not None
                                and opp_bet > my_bet):
                            continue_cost = opp_bet - my_bet
                            pot = my_bet + opp_bet
                            pot_odds = continue_cost / (continue_cost + pot)
                            eq = self.engine.compute_equity(
                                my_cards, board, dead, self._opp_weights)
                            if eq < pot_odds:
                                self._we_folded_this_hand = True
                                return (FOLD, 0, 0, 0)

                        if action[0] == RAISE:
                            self._raised_this_street = True
                            self._streets_raised += 1
                            self._last_hero_bet = action[1]
                            self._last_pot_before = pot_state[0] + pot_state[1]
                            self._opp_bet_at_raise = opp_bet
                        return action
                except Exception:
                    pass

            # Fallback: equity thresholds
            result = self._equity_threshold_play(
                my_cards, board, dead, observation, valid, street)
            return result

        # 3. River decisions.
        #    Acting first: precomputed lookup (0ms, 500-iter, if available)
        #                  fallback: runtime range solver
        #    Facing bet: Bayesian P(bet|hand) narrowing (precomputed if available,
        #                else adaptive polarized heuristic) + range solver + equity gate
        if street == 3:
            # Late-bind river lookup if extracted after init
            if not self.river_lookup.loaded:
                _rx = _PRELOAD.get('river_extracted')
                if _rx and os.path.isdir(_rx):
                    self.river_lookup = RiverLookup(_rx)

            facing_bet = opp_bet > my_bet

            # River board-aware range reweighting.
            # The discard inference evaluated keep-pairs on the FLOP (3 cards).
            # By the river, turn+river cards complete draws — straights and
            # flushes that weren't present on the flop are now made hands.
            # In check-check pots (no betting to narrow), the range is stuck
            # at flop-era weights: 53% pair/high, when reality is 52% straight+.
            # Fix: boost hands that are strong on the FULL board, reduce weak ones.
            # Only for acting-first (CC pots). Facing-bet already has P(bet|hand).
            # Acting first: use runtime solver with narrowed range (not precomputed).
            # Precomputed strategies were solved against uniform range and miss
            # value bets against narrowed opponents. Runtime solver sees actual range.

            if time_left > 50 and self._opp_weights is not None:
                self._path_counts['range_solver'] += 1

                # Bet-narrowing for facing bets.
                # Runtime P(bet|hand): solve from opponent's perspective to get
                # TRUE betting probabilities against our actual range.
                # Falls back to precomputed/heuristic if runtime fails.
                solve_range = self._opp_weights
                used_runtime_narrow = False

                if opp_bet > my_bet and _USE_C_SOLVER and time_left > 150:
                    # Runtime solve: compute P(bet|hand) from opponent's perspective
                    hero_range = {}
                    remaining = [c for c in range(27) if c not in set(board) | set(dead)]
                    for h in itertools.combinations(remaining, 2):
                        hero_range[h] = 1.0 / max(len(remaining) * (len(remaining)-1) // 2, 1)
                    p_bet_runtime = self.range_solver.compute_opp_bet_probs(
                        board, self._opp_weights, hero_range, dead,
                        my_bet, opp_bet, street, min_raise=2, iterations=200)
                    if p_bet_runtime:
                        narrowed = {}
                        for pair, weight in self._opp_weights.items():
                            if weight <= 0:
                                continue
                            key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                            pb = p_bet_runtime.get(key, 0.5)
                            narrowed[pair] = weight * max(pb, 0.001)
                        total_w = sum(narrowed.values())
                        if total_w > 0:
                            solve_range = {k: v / total_w
                                           for k, v in narrowed.items()}
                            used_runtime_narrow = True

                if opp_bet > my_bet and not used_runtime_narrow:
                    board_set = set(board)
                    hand_strengths = []
                    for oh, w in self._opp_weights.items():
                        if w > 0.001 and not (set(oh) & board_set):
                            try:
                                rank = self.engine.lookup_seven(
                                    list(oh) + list(board))
                                hand_strengths.append((oh, w, rank))
                            except Exception:
                                pass

                    if hand_strengths:
                        hand_strengths.sort(key=lambda x: x[2])
                        n = len(hand_strengths)

                        # Adaptive: shift based on tracked showdown WR
                        opp_wr = (self._opp_bet_showdown_wins /
                                  max(self._opp_bet_showdown_total, 10))
                        value_pct = min(0.50, 0.30 + max(0, (opp_wr - 0.40)) * 0.5)
                        bluff_pct = max(0.02, 0.10 - max(0, (opp_wr - 0.40)) * 0.2)

                        narrowed = {}
                        for i, (oh, w, r) in enumerate(hand_strengths):
                            if i < n * value_pct or i >= n * (1 - bluff_pct):
                                narrowed[oh] = w
                            else:
                                narrowed[oh] = w * 0.15
                        total_w = sum(narrowed.values())
                        if total_w > 0:
                            narrowed = {k: v / total_w
                                        for k, v in narrowed.items()}

                        # Blend with uniform by confidence
                        alpha = min(1.0, self._opp_bet_showdown_total / 30.0)
                        if alpha < 1.0:
                            known = set(board) | set(dead)
                            for oh in narrowed:
                                if not (set(oh) & known):
                                    uniform_w = 1.0  # uniform weight
                                    narrowed[oh] = (alpha * narrowed[oh] +
                                                    (1 - alpha) * uniform_w)
                            total_w = sum(narrowed.values())
                            if total_w > 0:
                                narrowed = {k: v / total_w
                                            for k, v in narrowed.items()}

                        solve_range = narrowed

                result = self.range_solver.solve_and_act(
                    hero_cards=my_cards, board=board,
                    opp_range=solve_range, dead_cards=dead,
                    my_bet=my_bet, opp_bet=opp_bet, street=street,
                    min_raise=observation["min_raise"],
                    max_raise=observation["max_raise"],
                    valid_actions=valid, time_remaining=time_left)
                if result is not None:
                    # Floor override and bluff injection REMOVED —
                    # with full tree C solver + fixed inference, the solver
                    # makes informed bet/check decisions. Overriding with
                    # less-informed precomputed data was a conflict.

                    # River equity gate: fold when equity < pot odds + margin.
                    # Margin of 0.12 trims marginal calls that are technically
                    # above pot odds but lose in practice (97.8% accuracy
                    # across 2625 calls — folds 179 losers, only 4 winners).
                    if (result[0] == CALL and opp_bet > my_bet
                            and self._opp_weights is not None):
                        continue_cost = opp_bet - my_bet
                        pot = my_bet + opp_bet
                        pot_odds = continue_cost / (continue_cost + pot)
                        eq = self.engine.compute_equity(
                            my_cards, board, dead, solve_range)
                        if eq < pot_odds + 0.12:
                            self._we_folded_this_hand = True
                            return (FOLD, 0, 0, 0)

                    # Note: selective caller, overbet, floor override, bluff injection
                    # all removed — with full tree C solver + DL solving on all streets,
                    # active) already includes 150% pot as an action. The solver
                    # picks the optimal size from 40/70/100/150%. Overriding its
                    # choice was conflicting with the equilibrium strategy.

                    # Track raise for re-raise narrowing and pot control
                    if result[0] == RAISE:
                        self._raised_this_street = True
                        self._streets_raised += 1
                        self._last_hero_bet = result[1]
                        self._last_pot_before = my_bet + opp_bet
                        self._opp_bet_at_raise = opp_bet
                        self._hero_bet_this_hand = True
                    return result

            # Fallback: equity thresholds if range solver fails or time is low
            result = self._equity_threshold_play(
                my_cards, board, dead, observation, valid, street)
            return result

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
        # Bluff frequency = bet/(bet+pot) scaled by street and opponent tendency.
        # Suppress bluffs against calling stations (they call >50% → bluffs are -EV).
        street_bluff_scale = {2: 0.78, 3: 1.0}.get(street, 0)
        if equity < 0.15 and valid[RAISE] and can_raise and street_bluff_scale > 0:
            # Check opponent fold rate — don't bluff if they call too much
            obs_street = street
            faces = self._opp_faces_bet.get(obs_street, 0)
            if faces > 15:
                fold_rate = self._opp_folds_to_bet[obs_street] / faces
                if fold_rate < 0.35:
                    street_bluff_scale = 0  # calling station — never bluff
                elif fold_rate < 0.50:
                    street_bluff_scale *= 0.3  # tight caller — bluff rarely

            if street_bluff_scale > 0:
                bet_size = max(int(pot * 0.6), min_raise)
                bluff_freq = bet_size / (bet_size + pot) if pot > 0 else 0.1
                if _random.random() < bluff_freq * street_bluff_scale:
                    self._raised_this_street = True
                    self._streets_raised += 1
                    self._opp_bet_at_raise = opp_bet
                    self._last_hero_bet = min(bet_size, max_raise)
                    self._last_pot_before = pot
                    return (RAISE, min(bet_size, max_raise), 0, 0)

        # Call if equity justifies (against polarized-narrowed range)
        if valid[CALL] and equity >= pot_odds:
            return (CALL, 0, 0, 0)

        # Check
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

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
        if self._river_data is None and _PRELOAD.get('river_data') is not None:
            self._river_data = _PRELOAD['river_data']

        # Turn data loaded lazily per-board — no stalling needed.




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

        # Track opponent folds to our bets (for adaptive bluffing)
        # If we had bet (my_bet > opp_bet before their action) and
        # the hand terminates with positive reward, opponent folded.
        if terminated and reward > 0 and street in (1, 2, 3):
            my_bet = observation["my_bet"]
            opp_bet = observation["opp_bet"]
            if my_bet > opp_bet:  # we had bet, they folded
                self._opp_folds_to_bet[street] += 1
                self._opp_faces_bet[street] += 1
            elif my_bet == opp_bet and my_bet > 2:
                # They called and we won at showdown — they faced our bet
                self._opp_faces_bet[street] += 1

        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_d) == 3 and self._opp_weights is None:
            my = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_d, board, my)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
