"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Strategy: Pure GTO via real-time CFR subgame solving. Zero exploitation.

All decisions are derived from:
  1. Nash equilibrium computation (CFR+ solver for post-flop)
  2. Precomputed Nash equilibrium (CFR for pre-flop)
  3. Bayesian inference (discard inference — math, not exploitation)
  4. Exact enumeration (equity engine — math, not exploitation)

Nothing in this bot assumes anything about opponent tendencies.
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

        # Precomputed tables
        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None

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
        # Vectorized dict construction (avoids slow Python loop over 80K entries)
        return dict(zip(bitmasks.tolist(), potentials.tolist()))

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
        """Find matching node in precomputed preflop game tree."""
        if self._preflop_strategy is None:
            return None
        ps = self._preflop_strategy
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        blind_pos = self._get_blind_position(observation)
        my_player = blind_pos
        bet_sb = my_bet if my_player == 0 else opp_bet
        bet_bb = my_bet if my_player == 1 else opp_bet

        for nid in range(len(ps['node_players'])):
            if (ps['node_players'][nid] == my_player and
                    ps['node_bet_sb'][nid] == bet_sb and
                    ps['node_bet_bb'][nid] == bet_bb):
                return nid
        return None

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None

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
    #  RANGE NARROWING
    # ----------------------------------------------------------------

    def _narrow_range_by_raise(self, observation, my_cards, board, dead_cards):
        """Narrow opponent range when they raise, using pot-odds math.

        This is Bayesian inference, not exploitation: a raise of X into
        pot P is only a +EV value bet if the raiser's equity exceeds a
        threshold. Filter the range to hands meeting that threshold.

        The key insight: a 98-chip raise into a 4-chip pot is ONLY
        rational with a very strong hand. A 2-chip raise into a 4-chip
        pot is rational with many hands. The narrowing scales with the
        bet-to-pot ratio.
        """
        if self._opp_weights is None:
            return

        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        raise_amount = opp_bet - my_bet
        pot_before_raise = my_bet + my_bet  # pot before opponent raised (both had my_bet)

        if raise_amount <= 0 or pot_before_raise <= 0:
            return

        # How aggressive is this raise relative to the pot?
        # A GTO player bets with value hands that have equity > some threshold
        # and bluffs at a rate proportional to bet/(bet+pot).
        # But a very large bet (e.g., 98 into 4) implies very strong hand
        # because the risk/reward ratio only makes sense with high equity.
        bet_to_pot = raise_amount / max(pot_before_raise, 1)

        # For large overbets (>3x pot), narrow aggressively
        # For small bets (<1x pot), narrow mildly
        # This scales smoothly: bigger bet = tighter range
        if bet_to_pot <= 0.5:
            keep_fraction = 0.85  # small bet: keep top 85%
        elif bet_to_pot <= 1.0:
            keep_fraction = 0.70  # pot-sized: keep top 70%
        elif bet_to_pot <= 3.0:
            keep_fraction = 0.50  # overbet: keep top 50%
        else:
            keep_fraction = 0.30  # massive overbet (98 into 4): keep top 30%

        # Compute hand strength for each opponent hand
        hand_strengths = {}
        for opp_pair in self._opp_weights:
            if self._opp_weights[opp_pair] <= 0:
                continue
            five = list(opp_pair) + list(board[:3])
            rank = self.engine.lookup_five(five)
            hand_strengths[opp_pair] = rank  # lower = stronger

        if not hand_strengths:
            return

        # Sort by strength, keep top fraction
        sorted_hands = sorted(hand_strengths.items(), key=lambda x: x[1])
        n = len(sorted_hands)
        cutoff = int(n * keep_fraction)

        if cutoff < n:
            for hand, _ in sorted_hands[cutoff:]:
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
        """Choose which 2 of 5 cards to keep. Uses exact equity over all
        10 keep-pairs. As SB, uses Bayesian inference on opponent's discards."""
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
    #  PRE-FLOP STRATEGY (Precomputed GTO)
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        """Pre-flop using precomputed GTO mixed strategies from CFR.

        Looks up our hand strength bucket and current tree node, then
        samples from the Nash equilibrium distribution.

        For bet levels not in the precomputed tree, uses pot odds with
        raw hand potential (no opponent range assumption — just math).
        """
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

        # Fallback: pure pot odds using hand potential (no exploitation)
        # Hand potential is our average equity across all flops against
        # a random opponent — the GTO-correct value when we have no
        # information about the opponent's range.
        potential = self._preflop_potential(my_cards)
        if potential is None:
            potential = 0.5  # default when table unavailable

        if valid_actions[CALL]:
            if continue_cost <= 1:
                # SB completing: pot odds = 33%, all hands have potential > 37%
                return (CALL, 0, 0, 0)

            # Pot odds comparison with raw potential (no range assumption)
            pot_size = self._get_pot_size(observation)
            required_equity = continue_cost / (continue_cost + pot_size)
            if potential >= required_equity:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  POST-FLOP BETTING (Real-Time CFR Solver)
    # ----------------------------------------------------------------

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Post-flop via real-time CFR+ subgame solving.

        Solves the betting game tree (~130 nodes) for 60-100 iterations
        to find the Nash equilibrium. The solver's opponent model uses
        the discard-inferred range weights — no exploitative adjustments.
        """
        dead_cards = my_discards + opp_discards
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        # Hero is always root of the solver tree — the solver is only
        # called when it's our turn. The tree builder uses bet state
        # (equal vs unequal) to determine actions (CHECK/BET vs FOLD/CALL/RAISE).
        # BUG FIX: previously set hero_is_first = (my_bet == opp_bet), which
        # made the tree root the OPPONENT when facing a bet. The solver then
        # returned the opponent's strategy instead of ours — causing us to
        # fold 100% equity hands to min-raises.
        hero_is_first = True

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

        # Narrow opponent range based on their action (Bayesian, not exploitative).
        # When opponent bets/raises, filter range to hands strong enough to
        # justify that bet size given pot odds. This is math, not assumption.
        opp_action = observation.get("opp_last_action", "None")
        if opp_action == "RAISE" and self._opp_weights is not None and len(board) >= 3:
            self._narrow_range_by_raise(observation, my_cards, board, my_discards + opp_discards)

        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

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


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
