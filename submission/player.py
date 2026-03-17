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
        # Load multi-street blueprint in background thread
        # (avoids blocking __init__ while still being ready for postflop)
        self._multi_street = None
        self._multi_street_loaded = False
        self._blueprints = {}
        import threading
        self._load_thread = threading.Thread(target=self._background_load, daemon=True)
        self._load_thread.start()

        self._current_hand = -1
        self._opp_weights = None
        self._bankroll = 0
        self._hand_reward = 0  # accumulates within a hand
        self._total_hands = 1000

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

    def _background_load(self):
        """Load heavy data in background thread."""
        self._multi_street = self._load_multi_street()
        self._blueprints = self._load_blueprints()
        self._multi_street_loaded = True

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
        blind_pos = observation.get("blind_position", 0)
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
            # Apply accumulated reward from previous hand
            self._bankroll += self._hand_reward
            self._hand_reward = 0
            self._current_hand = hand_number
            self._opp_weights = None

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
                        # Cap total preflop bet to 16. In this 27-card deck,
                        # equities are compressed — big preflop pots are
                        # coinflips. Win postflop where we have real edges.
                        max_total_bet = 16
                        if my_bet + amt > max_total_bet:
                            # Don't raise above cap — call or fold instead
                            if valid[CALL]:
                                return (CALL, 0, 0, 0)
                            return (CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0)
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
            return (CALL, 0, 0, 0) if valid[CALL] else (
                (CHECK, 0, 0, 0) if valid[CHECK] else None)
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
        """Postflop decision with multi-street blueprint priority.

        Priority:
            1. Compute/narrow opponent range
            2. Multi-street blueprint (backward induction, best quality)
            3. Single-street blueprint
            4. Range solver (river) or one-hand solver (flop/turn)
        """
        # If background load hasn't finished, skip blueprint (use fallback solvers)
        # Don't block — the 5s per-action timeout would kill us

        dead = my_discards + opp_discards
        my_bet, opp_bet = observation["my_bet"], observation["opp_bet"]
        street = observation["street"]
        valid = observation["valid_actions"]
        time_left = observation.get("time_left", 400)

        # Opponent range inference
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards)

        # Narrow range on bet/raise
        if opp_bet > my_bet and self._opp_weights is not None:
            self._narrow_range_by_bet(my_bet, opp_bet, board, dead)

        pot_state = (my_bet, opp_bet)

        # Determine hero's position: BB acts first postflop, SB acts second
        # blind_position=0 means SB, blind_position=1 means BB
        blind_pos = observation.get("blind_position", 0)
        hero_position = 1 if blind_pos == 0 else 0  # SB=second(1), BB=first(0)

        # 1. Multi-street blueprint (flop: backward induction)
        if street == 1 and self._multi_street is not None:
            try:
                strat = self._multi_street.get_strategy(
                    my_cards, board, pot_state=pot_state,
                    hero_position=hero_position)
                action = self._try_strategy(strat, observation)
                if action is not None:
                    self._path_counts['ms_flop'] += 1
                    return action
            except Exception:
                pass

        # 2. Multi-street turn strategies (backward induction from river)
        if street == 2 and self._multi_street is not None:
            try:
                strat = self._multi_street.get_turn_strategy(
                    my_cards, board, pot_state=pot_state,
                    hero_position=hero_position)
                action = self._try_strategy(strat, observation)
                if action is not None:
                    self._path_counts['ms_turn'] += 1
                    return action
            except Exception:
                pass

        # 3. Range solver (river — adapts to narrowed opponent range)
        # Only use if enough time (ARM64 is ~10-20x slower than x86)
        if street == 3 and self._opp_weights and time_left > 200:
            action = self.range_solver.solve_and_act(
                hero_cards=my_cards, board=board,
                opp_range=self._opp_weights, dead_cards=dead,
                my_bet=my_bet, opp_bet=opp_bet, street=street,
                min_raise=observation["min_raise"],
                max_raise=observation["max_raise"],
                valid_actions=valid, time_remaining=time_left)
            if action is not None:
                self._path_counts['range_solver'] += 1
                return action

        # 4. Single-street blueprint (FALLBACK — should not fire if above work)
        bp = self._blueprints.get(street)
        if bp is not None:
            try:
                strat = bp.get_strategy(
                    hero_cards=my_cards, board=board, pot_state=pot_state,
                    dead_cards=dead, opp_weights=self._opp_weights)
                action = self._try_strategy(strat, observation)
                if action is not None:
                    self._path_counts['ss_blueprint'] += 1
                    return action
            except Exception:
                pass

        # 5. One-hand solver (FALLBACK — should not fire)
        if time_left > 30:
            self._path_counts['one_hand_solver'] += 1
            return self.solver.solve_and_act(
                hero_cards=my_cards, board=board,
                opp_range=self._opp_weights, dead_cards=dead,
                my_bet=my_bet, opp_bet=opp_bet, street=street,
                min_raise=observation["min_raise"],
                max_raise=observation["max_raise"],
                valid_actions=valid, hero_is_first=True,
                time_remaining=time_left)

        # 6. Emergency: check/fold (FALLBACK — should NEVER fire)
        self._path_counts['emergency'] += 1
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
                return (FOLD, 0, 0, 0)
            if valid[DISCARD]:
                my, board, opp_d, _ = self._parse_cards(observation)
                return self._handle_discard(observation, my, board, opp_d)
            return (FOLD, 0, 0, 0)

        my_cards, board, opp_d, my_d = self._parse_cards(observation)
        valid = observation["valid_actions"]

        if valid[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_d)
        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)
        return self._handle_postflop(observation, my_cards, board,
                                     opp_d, my_d, info)

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)
        if reward != 0:
            self._hand_reward += reward

        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_d) == 3 and self._opp_weights is None:
            my = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_d, board, my)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
