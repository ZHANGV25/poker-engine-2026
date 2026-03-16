"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Clean version: multi-street blueprint ONLY. No fallbacks.
If the blueprint doesn't cover a situation, we fold/check.
This gives clean signal on blueprint quality.
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
from multi_street_lookup import MultiStreetLookup
from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CALL = PokerEnv.ActionType.CALL.value
CHECK = PokerEnv.ActionType.CHECK.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_POT_FRACTIONS = {
    ACT_RAISE_HALF: 0.40, ACT_RAISE_POT: 0.70,
    ACT_RAISE_ALLIN: 1.00, ACT_RAISE_OVERBET: 1.50,
}


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)

        # Preflop
        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()

        # Multi-street blueprint (the ONLY postflop strategy)
        self._multi_street = self._load_multi_street()
        if self._multi_street is None:
            raise RuntimeError("Multi-street blueprint not found! Cannot run without it.")

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None
        self._bankroll = 0
        self._total_hands = 1000

    def __name__(self):
        return "PlayerAgent"

    # ----------------------------------------------------------------
    #  INIT
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
        data_dir = os.path.join(_dir, "data", "multi_street")
        if os.path.isdir(data_dir):
            try:
                return MultiStreetLookup(data_dir, equity_engine=self.engine)
            except Exception:
                pass
        merged = os.path.join(_dir, "data", "multi_street_blueprint.npz")
        if os.path.isfile(merged):
            try:
                return MultiStreetLookup(merged, equity_engine=self.engine)
            except Exception:
                pass
        return None

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
        my_bet, opp_bet = observation["my_bet"], observation["opp_bet"]
        blind_pos = observation.get("blind_position", 0)
        bet_sb = my_bet if blind_pos == 0 else opp_bet
        bet_bb = my_bet if blind_pos == 1 else opp_bet

        best_nid, best_dist = None, float('inf')
        for nid in range(len(ps['node_players'])):
            if ps['node_players'][nid] != blind_pos or ps['node_players'][nid] == -1:
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
            self._current_hand = hand_number
            self._opp_weights = None

    def _parse_cards(self, observation):
        my = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_d = [c for c in observation["my_discarded_cards"] if c != -1]
        return my, board, opp_d, my_d

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
                        if amt > 0 and valid[RAISE]:
                            return (RAISE, amt, 0, 0)
                        return (CALL, 0, 0, 0) if valid[CALL] else (CHECK, 0, 0, 0)

        # Fallback: pot odds
        potential = self._preflop_potential(my_cards) or 0.5
        if valid[CALL]:
            cost = opp_bet - my_bet
            if cost <= 1:
                return (CALL, 0, 0, 0)
            pot = observation.get("pot_size", my_bet + opp_bet)
            if potential >= cost / (cost + pot):
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        return (CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  POSTFLOP — multi-street blueprint ONLY
    # ----------------------------------------------------------------

    def _blueprint_to_engine(self, action_type, observation):
        valid = observation["valid_actions"]
        pot = observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])
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

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Pure multi-street blueprint. No fallbacks."""
        dead = my_discards + opp_discards

        # Opponent range inference
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards)

        # Multi-street blueprint lookup
        pot_state = (observation["my_bet"], observation["opp_bet"])
        strategy = self._multi_street.get_strategy(
            my_cards, board, pot_state=pot_state)

        if strategy:
            # Check validity
            probs = list(strategy.values())
            facing_bet = observation["opp_bet"] > observation["my_bet"]

            # Skip unconverged
            if len(probs) > 2 and max(probs) - min(probs) < 0.05:
                strategy = None

            # Skip wrong node (facing bet but no fold)
            if strategy and facing_bet:
                if ACT_FOLD not in strategy or strategy[ACT_FOLD] < 0.001:
                    strategy = None

        if strategy:
            aids = list(strategy.keys())
            p = np.array([strategy[a] for a in aids])
            p /= p.sum()
            chosen = int(np.random.choice(aids, p=p))
            action = self._blueprint_to_engine(chosen, observation)
            if action is not None:
                return action

        # NO FALLBACK — just check or fold
        valid = observation["valid_actions"]
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
            self._bankroll += reward

        # Lead protection
        hands_remaining = self._total_hands - hand_number
        blind_cost = hands_remaining * 1.5
        if self._bankroll > blind_cost + 10:
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
            self._bankroll += reward

        opp_d = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_d) == 3 and self._opp_weights is None:
            my = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_d, board, my)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
