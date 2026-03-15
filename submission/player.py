"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Hybrid strategy: precomputed blueprint strategies (range-balanced Nash
equilibria) for post-flop play when available, with real-time CFR solver
as fallback. Pre-flop uses a separate precomputed strategy table.

Blueprint strategies are loaded from submission/data/ at startup.
If blueprint files don't exist, the bot seamlessly falls back to the
real-time CFR solver for that street.
"""

import os
import sys
import random
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from agents.agent import Agent
from gym_env import PokerEnv

from equity import ExactEquityEngine
from inference import DiscardInference
from solver import SubgameSolver

# Try to import blueprint lookup module (graceful fallback if unavailable)
try:
    from blueprint_lookup import BlueprintLookup
    _BLUEPRINT_AVAILABLE = True
except Exception:
    _BLUEPRINT_AVAILABLE = False

# Import action type constants for blueprint action mapping
from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

FOLD = PokerEnv.ActionType.FOLD.value      # 0
RAISE = PokerEnv.ActionType.RAISE.value    # 1
CHECK = PokerEnv.ActionType.CHECK.value    # 2
CALL = PokerEnv.ActionType.CALL.value      # 3
DISCARD = PokerEnv.ActionType.DISCARD.value  # 4

# Blueprint file names per street (street 1=flop, 2=turn, 3=river)
_BLUEPRINT_FILES = {
    1: "flop_blueprint.npz",
    2: "turn_blueprint.npz",
    3: "river_blueprint.npz",
}


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)

        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()

        # Load blueprint strategies for post-flop streets
        self._blueprints = self._load_blueprints()

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None
        self._last_seen_action = None

    def __name__(self):
        return "PlayerAgent"

    # ----------------------------------------------------------------
    #  INIT HELPERS
    # ----------------------------------------------------------------

    def _load_preflop_table(self):
        data_path = os.path.join(_dir, "data", "preflop_potential.npz")
        if not os.path.exists(data_path):
            return None
        data = np.load(data_path)
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
        """Load blueprint strategy files for each post-flop street.

        Returns a dict mapping street (1,2,3) -> BlueprintLookup or None.
        If the blueprint module isn't available or a file doesn't exist,
        that street maps to None (will use real-time solver instead).
        """
        blueprints = {}
        if not _BLUEPRINT_AVAILABLE:
            return blueprints

        data_dir = os.path.join(_dir, "data")
        for street, filename in _BLUEPRINT_FILES.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    blueprints[street] = BlueprintLookup(
                        filepath, equity_engine=self.engine
                    )
                except Exception:
                    # File exists but failed to load — skip silently
                    blueprints[street] = None
            else:
                blueprints[street] = None

        return blueprints

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
        blind_pos = self._get_blind_position(observation)
        my_player = blind_pos
        bet_sb = my_bet if my_player == 0 else opp_bet
        bet_bb = my_bet if my_player == 1 else opp_bet

        # Exact match
        for nid in range(len(ps['node_players'])):
            if (ps['node_players'][nid] == my_player and
                    ps['node_bet_sb'][nid] == bet_sb and
                    ps['node_bet_bb'][nid] == bet_bb):
                return nid

        # Round to nearest node
        best_nid = None
        best_dist = float('inf')
        for nid in range(len(ps['node_players'])):
            if ps['node_players'][nid] != my_player or ps['node_players'][nid] == -1:
                continue
            dist = abs(ps['node_bet_sb'][nid] - bet_sb) + abs(ps['node_bet_bb'][nid] - bet_bb)
            if dist < best_dist:
                best_dist = dist
                best_nid = nid
        return best_nid

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None
            self._last_seen_action = None

    def _parse_cards(self, observation):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        return my_cards, board, opp_discards, my_discards

    def _get_pot_size(self, observation):
        return observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])

    def _get_blind_position(self, observation):
        return observation.get("blind_position", 0)

    def _compute_raise_amount(self, observation, pot_fraction):
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        amount = max(int(pot_size * pot_fraction), min_raise)
        return min(amount, max_raise)

    # ----------------------------------------------------------------
    #  DISCARD
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )
        results = self.engine.evaluate_all_keep_pairs(
            my_cards, board, opp_discards, self._opp_weights
        )
        best_keep = results[0][0]
        return (DISCARD, 0, best_keep[0], best_keep[1])

    # ----------------------------------------------------------------
    #  PRE-FLOP
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        valid_actions = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        continue_cost = opp_bet - my_bet

        # Try precomputed GTO strategy
        bucket = self._get_preflop_bucket(my_cards)
        node_id = self._find_preflop_node(observation)

        if bucket is not None and node_id is not None and self._preflop_strategy is not None:
            ps = self._preflop_strategy
            strategy = ps['strategies'][node_id, bucket]
            children = ps['children_map'].get(node_id, {})

            if len(children) > 0:
                valid_mask = np.zeros(len(strategy))
                for act_id in children:
                    if act_id == 0 and valid_actions[FOLD]:
                        valid_mask[act_id] = strategy[act_id]
                    elif act_id == 1:
                        if valid_actions[CALL] or valid_actions[CHECK]:
                            valid_mask[act_id] = strategy[act_id]
                    elif act_id >= 2 and valid_actions[RAISE]:
                        valid_mask[act_id] = strategy[act_id]

                total = valid_mask.sum()
                if total > 0:
                    valid_mask /= total
                    chosen = int(np.random.choice(len(valid_mask), p=valid_mask))

                    if chosen == 0:
                        return (FOLD, 0, 0, 0)
                    elif chosen == 1:
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)
                    elif chosen >= 2:
                        raise_to = ps['raise_levels'][chosen - 2]
                        raise_amount = raise_to - opp_bet
                        raise_amount = max(raise_amount, observation["min_raise"])
                        raise_amount = min(raise_amount, observation["max_raise"])
                        if raise_amount > 0 and valid_actions[RAISE]:
                            return (RAISE, raise_amount, 0, 0)
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)

        # Fallback: pot odds
        potential = self._preflop_potential(my_cards)
        if potential is None:
            potential = 0.5

        if valid_actions[CALL]:
            if continue_cost <= 1:
                return (CALL, 0, 0, 0)
            pot_size = self._get_pot_size(observation)
            required_equity = continue_cost / (continue_cost + pot_size)
            if potential >= required_equity:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  POST-FLOP (blueprint + CFR solver fallback)
    # ----------------------------------------------------------------

    def _try_blueprint(self, observation, my_cards, board, dead_cards=None):
        """Try to get an action from the blueprint strategy.

        Returns a concrete (action_type, amount, 0, 0) tuple if the
        blueprint has a strategy for this state, or None to signal
        that the caller should fall back to the real-time solver.
        """
        street = observation["street"]
        blueprint = self._blueprints.get(street)
        if blueprint is None:
            return None

        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        facing_bet = opp_bet > my_bet

        # The flop blueprint was computed with (1,2) starting bets which
        # doesn't match runtime flop state (always equal bets). Non-root
        # nodes only have fold/call (no raise). Solver handles flop better.
        if street == 1:
            return None

        # Get blueprint strategy (action type -> probability)
        try:
            pot_state = (my_bet, opp_bet)
            strategy = blueprint.get_strategy(
                hero_cards=my_cards, board=board, pot_state=pot_state,
                dead_cards=dead_cards)
        except Exception:
            return None

        if strategy is None:
            return None

        # Verify the strategy makes sense for our situation
        if facing_bet:
            has_fold = ACT_FOLD in strategy and strategy[ACT_FOLD] > 0.001
            if not has_fold:
                # Strategy has no fold option but we're facing a bet →
                # wrong node was returned, fall back to solver
                return None

        # Sample an action type from the blueprint strategy
        action_ids = list(strategy.keys())
        probs = np.array([strategy[a] for a in action_ids])
        probs /= probs.sum()  # safety normalization
        chosen_action = int(np.random.choice(action_ids, p=probs))

        # Map the abstract action type to a concrete engine action
        return self._blueprint_action_to_engine(
            chosen_action, observation
        )

    def _blueprint_action_to_engine(self, action_type, observation):
        """Convert a blueprint action type ID to a concrete engine action.

        Maps abstract blueprint actions (ACT_FOLD, ACT_RAISE_HALF, etc.)
        to the engine's (action, amount, 0, 0) format.

        Args:
            action_type: int, one of ACT_FOLD..ACT_RAISE_OVERBET
            observation: the current game observation

        Returns:
            (action, amount, 0, 0) tuple, or None if the action is invalid
            and no safe fallback exists.
        """
        valid_actions = observation["valid_actions"]
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        if action_type == ACT_FOLD:
            if valid_actions[FOLD]:
                return (FOLD, 0, 0, 0)
            # Can't fold (shouldn't happen, but be safe)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        elif action_type == ACT_CHECK:
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            # Blueprint says check but we can't — try call
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            return None

        elif action_type == ACT_CALL:
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        elif action_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                             ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            # Map to pot fraction
            pot_fractions = {
                ACT_RAISE_HALF: 0.40,
                ACT_RAISE_POT: 0.70,
                ACT_RAISE_ALLIN: 1.00,
                ACT_RAISE_OVERBET: 1.50,
            }
            fraction = pot_fractions[action_type]
            raise_amount = max(int(pot_size * fraction), min_raise)
            raise_amount = min(raise_amount, max_raise)

            if valid_actions[RAISE] and raise_amount > 0:
                return (RAISE, raise_amount, 0, 0)

            # Raise not valid — fall back to call or check
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        # Unknown action type — signal fallback
        return None

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Post-flop decision: try blueprint first, then CFR solver."""
        dead_cards = my_discards + opp_discards

        # Try blueprint strategy first
        blueprint_action = self._try_blueprint(observation, my_cards, board, dead_cards)
        if blueprint_action is not None:
            return blueprint_action

        # Fall back to real-time CFR solver
        dead_cards = my_discards + opp_discards
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        max_raise = observation["max_raise"]

        return self.solver.solve_and_act(
            hero_cards=my_cards,
            board=board,
            opp_range=self._opp_weights,
            dead_cards=dead_cards,
            my_bet=my_bet,
            opp_bet=opp_bet,
            street=observation["street"],
            min_raise=observation["min_raise"],
            max_raise=max_raise,
            valid_actions=observation["valid_actions"],
            hero_is_first=True,
            time_remaining=observation.get("time_left", 400),
        )

    # ----------------------------------------------------------------
    #  MAIN
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
        valid_actions = observation["valid_actions"]

        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

        return self._handle_postflop(observation, my_cards, board, opp_discards, my_discards, info)

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
