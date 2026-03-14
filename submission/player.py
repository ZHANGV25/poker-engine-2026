"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Strategy: GTO-first approach. No exploitation — play solid, unexploitable
poker and let opponents make mistakes.

Architecture:
  1. ExactEquityEngine: Exact win probability via full enumeration (~5ms)
  2. DiscardInference: Bayesian opponent range narrowing from revealed discards
  3. Action-based range narrowing: update opponent range when they bet/call/raise
  4. Pot control: don't build big pots without near-nut hands

Key edges:
  - Exact equity (zero error) vs Monte Carlo (2.5% error)
  - Discard + action inference narrows opponent range from ~120 to ~20-40 hands
  - Pot control prevents value-owning ourselves with medium hands
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

        # Pre-flop hand potential lookup table
        self._preflop_table = self._load_preflop_table()

        # GTO betting thresholds (fixed, not adjusted by opponent behavior)
        self.RAISE_THRESHOLD = 0.72       # equity needed to raise
        self.STRONG_RAISE_THRESHOLD = 0.82  # equity for large raises
        self.ALL_IN_THRESHOLD = 0.92      # equity for all-in
        self.MAX_RAISE_STREETS = 2        # max streets we raise on per hand without nuts

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None
        self._last_seen_action = None
        self._streets_raised = 0          # how many different streets we've raised on this hand
        self._current_street = -1
        self._raised_this_street = False

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

        # Compute equity for each possible opponent hand against the board
        hand_equities = {}
        known = set(my_cards) | set(board) | set(dead_cards)
        remaining = [c for c in range(27) if c not in known]

        for opp_pair in self._opp_weights:
            if self._opp_weights[opp_pair] <= 0:
                continue
            # Quick equity proxy: 5-card rank of opponent's hand + 3 board cards
            if len(board) >= 3:
                five = list(opp_pair) + list(board[:3])
                rank = self.engine.lookup_five(five)
                hand_equities[opp_pair] = rank  # lower = better

        if not hand_equities:
            return

        # Sort hands by strength (lower rank = stronger)
        sorted_hands = sorted(hand_equities.items(), key=lambda x: x[1])
        n = len(sorted_hands)

        # Determine cutoff based on action type
        if opp_action == "RAISE":
            # Keep top 40% of range (raises signal strength)
            cutoff_idx = int(n * 0.4)
        elif opp_action == "CALL":
            # Keep top 70% of range (calls signal some strength)
            cutoff_idx = int(n * 0.7)
        else:
            # CHECK — no narrowing (consistent with any hand)
            return

        # Zero out hands below cutoff
        if cutoff_idx < n:
            weak_hands = {h for h, _ in sorted_hands[cutoff_idx:]}
            for hand in weak_hands:
                self._opp_weights[hand] = 0.0

        # Renormalize
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
    #  POST-FLOP BETTING (GTO-focused)
    # ----------------------------------------------------------------

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """GTO-focused post-flop betting with pot control.

        Core principles:
        1. Compute equity against narrowed opponent range
        2. Only raise with strong hands, check/call with medium hands
        3. Don't raise on more than 2 streets without near-nuts equity
        4. Bluff at a fixed GTO-balanced frequency
        """
        dead_cards = my_discards + opp_discards
        equity = self.engine.compute_equity(my_cards, board, dead_cards, self._opp_weights)

        valid_actions = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        street = observation["street"]

        continue_cost = opp_bet - my_bet
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        # Track which streets we've raised on (for pot control)
        if street != self._current_street:
            self._current_street = street
            self._raised_this_street = False

        # --- Opponent re-raised us ---
        opp_action = observation.get("opp_last_action", "None")
        if opp_action == "RAISE" and self._raised_this_street:
            # Opponent re-raised after we raised. Only continue with near-nuts.
            if equity > self.ALL_IN_THRESHOLD and valid_actions[RAISE]:
                return (RAISE, max_raise, 0, 0)
            elif equity >= pot_odds and valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            elif valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # --- Pot control: cap raising streets ---
        # Don't raise on 3+ streets unless we have nuts-level equity.
        # This prevents building 100-chip pots with two pair / trips.
        can_raise_for_value = (
            self._streets_raised < self.MAX_RAISE_STREETS
            or equity > self.ALL_IN_THRESHOLD
        )

        # --- Check-raise opportunity ---
        # As first to act with a monster hand, sometimes check to induce a bet
        is_first_to_act = (my_bet == opp_bet)  # no bet to respond to
        if (is_first_to_act
                and equity > self.STRONG_RAISE_THRESHOLD
                and valid_actions[CHECK]
                and random.random() < 0.3):  # 30% check-raise frequency
            return (CHECK, 0, 0, 0)

        # --- Make decision ---

        # All-in with near-certain hands
        if equity > self.ALL_IN_THRESHOLD and valid_actions[RAISE]:
            self._raised_this_street = True
            self._streets_raised += 1
            return (RAISE, max_raise, 0, 0)

        # Strong value raise
        if equity > self.STRONG_RAISE_THRESHOLD and valid_actions[RAISE] and can_raise_for_value:
            self._raised_this_street = True
            self._streets_raised += 1
            frac = 0.65 + 0.15 * (equity - self.STRONG_RAISE_THRESHOLD) / (1.0 - self.STRONG_RAISE_THRESHOLD)
            raise_amt = self._compute_raise_amount(observation, frac)
            return (RAISE, raise_amt, 0, 0)

        # Medium value raise (only if we haven't raised too many streets)
        if equity > self.RAISE_THRESHOLD and valid_actions[RAISE] and can_raise_for_value:
            self._raised_this_street = True
            self._streets_raised += 1
            frac = 0.5 + 0.15 * (equity - self.RAISE_THRESHOLD) / (self.STRONG_RAISE_THRESHOLD - self.RAISE_THRESHOLD)
            raise_amt = self._compute_raise_amount(observation, frac)
            return (RAISE, raise_amt, 0, 0)

        # GTO-balanced bluff (fixed frequency based on bet sizing)
        # Bluff frequency = bet_size / (bet_size + pot) scaled down for earlier streets
        if equity < 0.25 and valid_actions[RAISE] and can_raise_for_value:
            bet_size = max(int(pot_size * 0.6), min_raise)
            gto_bluff_freq = bet_size / (bet_size + pot_size) if pot_size > 0 else 0.1
            # Scale down: full frequency on river, 40% on turn, 20% on flop
            street_scale = {1: 0.20, 2: 0.40, 3: 1.0}.get(street, 0.1)
            effective_bluff_freq = gto_bluff_freq * street_scale * 0.5  # conservative

            if random.random() < effective_bluff_freq:
                self._raised_this_street = True
                self._streets_raised += 1
                raise_amt = self._compute_raise_amount(observation, 0.6)
                return (RAISE, raise_amt, 0, 0)

        # Call if equity justifies it
        if valid_actions[CALL] and equity >= pot_odds:
            return (CALL, 0, 0, 0)

        # Check if possible
        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    def _compute_raise_amount(self, observation, pot_fraction):
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        amount = max(int(pot_size * pot_fraction), min_raise)
        amount = min(amount, max_raise)
        return amount

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
