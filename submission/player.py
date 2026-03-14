"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Strategy: GTO via real-time CFR subgame solving. No exploitation.

Architecture:
  1. ExactEquityEngine: Exact win probability via full enumeration (~5ms)
  2. DiscardInference: Bayesian opponent range narrowing from revealed discards
  3. SubgameSolver: CFR+ solver computes Nash equilibrium for post-flop betting
     in real-time (~90-130ms per decision)

Key edges:
  - Exact equity (zero error) vs Monte Carlo (2.5% error)
  - Discard inference narrows opponent range from ~120 to ~20-40 hands
  - CFR solver computes provably optimal bet/check/fold/call frequencies
    instead of heuristic equity thresholds
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

FOLD = PokerEnv.ActionType.FOLD.value      # 0
RAISE = PokerEnv.ActionType.RAISE.value    # 1
CHECK = PokerEnv.ActionType.CHECK.value    # 2
CALL = PokerEnv.ActionType.CALL.value      # 3
DISCARD = PokerEnv.ActionType.DISCARD.value  # 4


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        # Core components
        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)

        # Pre-flop hand potential lookup table
        self._preflop_table = self._load_preflop_table()

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
        bitmasks = data["bitmasks"]
        potentials = data["potentials"]
        table = {}
        for i in range(len(bitmasks)):
            table[int(bitmasks[i])] = float(potentials[i])
        return table

    def _preflop_potential(self, my_cards):
        if self._preflop_table is None:
            return None
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask)

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None
            self._last_seen_action = None
            self._streets_raised = 0
            self._current_street = -1
            self._raised_this_street = False

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
        amount = min(amount, max_raise)
        return amount

    # ----------------------------------------------------------------
    #  ACTION-BASED RANGE NARROWING
    # ----------------------------------------------------------------

    def _narrow_range_by_action(self, opp_action, my_cards, board, dead_cards):
        """Update opponent weights based on their betting action.

        When opponent takes an aggressive action, they likely have a stronger
        hand. Filter the bottom portion of their range and recompute weights.

        This is not exploitation — it's Bayesian inference. Their action IS
        evidence about their hand strength.
        """
        if self._opp_weights is None or len(board) < 3:
            return

        hand_equities = {}
        for opp_pair in self._opp_weights:
            if self._opp_weights[opp_pair] <= 0:
                continue
            if len(board) >= 3:
                five = list(opp_pair) + list(board[:3])
                rank = self.engine.lookup_five(five)
                hand_equities[opp_pair] = rank

        if not hand_equities:
            return

        sorted_hands = sorted(hand_equities.items(), key=lambda x: x[1])
        n = len(sorted_hands)

        if opp_action == "RAISE":
            cutoff_idx = int(n * 0.4)
        elif opp_action == "CALL":
            cutoff_idx = int(n * 0.7)
        else:
            return

        if cutoff_idx < n:
            weak_hands = {h for h, _ in sorted_hands[cutoff_idx:]}
            for hand in weak_hands:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    # ----------------------------------------------------------------
    #  DISCARD DECISION
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        """Choose which 2 of 5 cards to keep on the flop."""
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
    #  PRE-FLOP STRATEGY
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        """Pre-flop strategy using precomputed hand potential table."""
        valid_actions = observation["valid_actions"]

        potential = self._preflop_potential(my_cards)
        if potential is not None:
            strength = max(0.0, min(10.0, (potential - 0.37) / 0.028))
        else:
            strength = self._preflop_heuristic(my_cards)

        continue_cost = observation["opp_bet"] - observation["my_bet"]

        if valid_actions[CALL]:
            if strength >= 7.0 and valid_actions[RAISE]:
                raise_amt = self._compute_raise_amount(observation, 0.65)
                return (RAISE, raise_amt, 0, 0)
            if continue_cost <= 1:
                return (CALL, 0, 0, 0)
            if strength >= 3.0:
                return (CALL, 0, 0, 0)
            if continue_cost > 6 and strength < 2.0:
                return (FOLD, 0, 0, 0)
            return (CALL, 0, 0, 0)

        if valid_actions[CHECK]:
            if strength >= 8.0 and valid_actions[RAISE]:
                raise_amt = self._compute_raise_amount(observation, 0.65)
                return (RAISE, raise_amt, 0, 0)
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    def _preflop_heuristic(self, my_cards):
        from collections import Counter
        ranks = [c % 9 for c in my_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(c // 9 for c in my_cards)
        strength = 0.0
        if any(count >= 3 for count in rank_counts.values()):
            strength += 5
        pair_count = sum(1 for count in rank_counts.values() if count >= 2)
        if pair_count >= 2:
            strength += 4
        elif pair_count == 1:
            strength += 2
        if 8 in ranks:
            strength += 1
        if max(suit_counts.values()) >= 3:
            strength += 1
        sorted_ranks = sorted(set(ranks))
        run = 1
        max_run = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= 3:
            strength += 1
        return strength

    # ----------------------------------------------------------------
    #  POST-FLOP BETTING (CFR Solver)
    # ----------------------------------------------------------------

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Post-flop decisions via real-time CFR subgame solving.

        Instead of equity thresholds, we solve a small game tree (~100-200 nodes)
        using CFR+ for 75-150 iterations. This computes the Nash equilibrium
        action frequencies for our specific hand against the opponent's range.

        The solver handles bluffing, pot control, check-raising, and bet sizing
        implicitly — these all emerge from the equilibrium computation.
        """
        dead_cards = my_discards + opp_discards
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]

        # Determine if we're initiating action or responding to a bet.
        # This determines the tree shape: initiating = CHECK/BET options,
        # responding = FOLD/CALL/RAISE options.
        # We're "first" if bets are equal (no outstanding bet to respond to).
        hero_is_first = (my_bet == opp_bet)

        return self.solver.solve_and_act(
            hero_cards=my_cards,
            board=board,
            opp_range=self._opp_weights,
            dead_cards=dead_cards,
            my_bet=my_bet,
            opp_bet=opp_bet,
            street=observation["street"],
            min_raise=observation["min_raise"],
            max_raise=observation["max_raise"],
            valid_actions=observation["valid_actions"],
            hero_is_first=hero_is_first,
            time_remaining=observation.get("time_left", 400),
        )

    # ----------------------------------------------------------------
    #  MAIN ACT / OBSERVE
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
        valid_actions = observation["valid_actions"]

        # Narrow opponent range based on their action (if we have weights)
        opp_action = observation.get("opp_last_action", "None")
        if opp_action in ("RAISE", "CALL") and self._opp_weights is not None:
            self._narrow_range_by_action(opp_action, my_cards, board, my_discards + opp_discards)

        # Discard phase
        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        # Pre-flop
        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

        # Post-flop
        return self._handle_postflop(observation, my_cards, board, opp_discards, my_discards, info)

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        # Infer opponent range when we first see their discards
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )

        # Narrow range based on opponent's action
        opp_action = observation.get("opp_last_action", "None")
        if opp_action in ("RAISE", "CALL") and self._opp_weights is not None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
            opp_disc = [c for c in observation["opp_discarded_cards"] if c != -1]
            self._narrow_range_by_action(opp_action, my_cards, board, my_discards + opp_disc)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
