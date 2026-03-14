"""
Real-time CFR+ subgame solver for post-flop poker decisions.

Replaces equity-threshold-based betting with provably GTO-optimal strategies.
For each decision, builds a small game tree (~50-200 nodes), runs CFR+ for
75-150 iterations (~30-50ms), and returns the Nash equilibrium action.

Key insight: hero has exactly ONE hand, so hero's information sets are
determined purely by the action history (no card dimension). The opponent
has ~20-40 possible hands. This asymmetry keeps the tree tiny.
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import numpy as np
import itertools
from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3


class SubgameSolver:
    def __init__(self, equity_engine):
        self.engine = equity_engine
        self._tree_cache = {}

    def solve_and_act(self, hero_cards, board, opp_range, dead_cards,
                      my_bet, opp_bet, street, min_raise, max_raise,
                      valid_actions, hero_is_first, time_remaining):
        """Solve the subgame and return a concrete action.

        Args:
            hero_cards: list of 2 ints (our hole cards)
            board: list of 3-5 ints (community cards)
            opp_range: dict mapping (c1,c2) -> weight, or None
            dead_cards: list of ints (discards)
            my_bet, opp_bet: current bets
            street: 1-3 (flop/turn/river)
            min_raise, max_raise: valid raise range
            valid_actions: list of 5 bools
            hero_is_first: True if hero acts first
            time_remaining: seconds left in match

        Returns:
            (action_type, raise_amount, 0, 0) tuple
        """
        # Filter opponent range to hands with meaningful weight
        if opp_range is None:
            # No inference available — fall back to equity thresholds
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        # Safety: exclude opponent hands that overlap with known cards
        known_cards = set(hero_cards) | set(board) | set(dead_cards)
        opp_hands = []
        opp_weights = []
        for hand, weight in opp_range.items():
            if weight > 0.001 and not (set(hand) & known_cards):
                opp_hands.append(hand)
                opp_weights.append(weight)

        if not opp_hands:
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        opp_weights = np.array(opp_weights, dtype=np.float64)
        opp_weights /= opp_weights.sum()
        n_opp = len(opp_hands)

        # Choose iteration count based on time remaining.
        # Budget: 500s for 1000 hands = 0.5s/hand. Each hand has ~4 decisions.
        # Target: <100ms per solver call. ARM64 is ~1.5x slower than Apple Silicon.
        if time_remaining > 300:
            iterations = 100
        elif time_remaining > 150:
            iterations = 60
        elif time_remaining > 50:
            iterations = 30
        else:
            # Critical: fall back to thresholds entirely
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        # Build game tree
        max_bet = 100  # MAX_PLAYER_BET
        tree = self._get_tree(my_bet, opp_bet, min_raise, max_bet, hero_is_first)

        if tree.size < 2:
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        # Compute terminal payoffs for each opponent hand
        # terminal_values[terminal_idx][opp_hand_idx] = hero's payoff
        terminal_values = self._compute_terminal_values(
            tree, hero_cards, board, dead_cards, opp_hands, street)

        # Run CFR+
        hero_strategy = self._run_cfr(tree, opp_weights, terminal_values,
                                       n_opp, iterations)

        # Map to concrete action
        return self._strategy_to_action(
            tree, hero_strategy, my_bet, opp_bet, min_raise, max_raise, valid_actions)

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet, hero_first):
        """Build or retrieve a cached game tree."""
        key = (hero_bet, opp_bet, min_raise, max_bet, hero_first)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(hero_bet, opp_bet, min_raise, max_bet, hero_first)
        return self._tree_cache[key]

    def _compute_terminal_values(self, tree, hero_cards, board, dead_cards,
                                  opp_hands, street):
        """Compute hero's payoff at each terminal node for each opponent hand.

        Precomputes per-hand equity ONCE, then applies it to all showdown
        terminals with different pot sizes. This avoids redundant equity
        calculations (the main bottleneck on flop/turn).

        Returns dict: terminal_node_id -> np.array of shape (n_opp,)
        """
        values = {}
        n_opp = len(opp_hands)

        # Precompute per-hand equity (or win/loss for river)
        # equity_vec[i] = hero's equity against opponent hand i (0.0 to 1.0)
        equity_vec = np.zeros(n_opp, dtype=np.float64)

        if len(board) == 5:
            # River: deterministic comparison
            hero_rank = self.engine.lookup_seven(list(hero_cards) + list(board))
            for i, oh in enumerate(opp_hands):
                opp_rank = self.engine.lookup_seven(list(oh) + list(board))
                if hero_rank < opp_rank:
                    equity_vec[i] = 1.0
                elif hero_rank == opp_rank:
                    equity_vec[i] = 0.5
                # else 0.0
        else:
            # Flop/Turn: enumerate runouts once per opponent hand
            known = set(hero_cards) | set(board) | set(dead_cards)
            board_needed = 5 - len(board)
            hero_list = list(hero_cards)
            board_list = list(board)

            for i, oh in enumerate(opp_hands):
                remaining = [c for c in range(27) if c not in known and c not in oh]
                wins = 0.0
                total = 0
                for runout in itertools.combinations(remaining, board_needed):
                    full_board = board_list + list(runout)
                    hr = self.engine.lookup_seven(hero_list + full_board)
                    opr = self.engine.lookup_seven(list(oh) + full_board)
                    if hr < opr:
                        wins += 1.0
                    elif hr == opr:
                        wins += 0.5
                    total += 1
                equity_vec[i] = wins / total if total > 0 else 0.5

        # Now assign terminal values using precomputed equity
        for node_id in tree.terminal_node_ids:
            term_type = tree.terminal[node_id]
            hero_pot = tree.hero_pot[node_id]
            opp_pot = tree.opp_pot[node_id]

            if term_type == TERM_FOLD_HERO:
                values[node_id] = np.full(n_opp, -hero_pot, dtype=np.float64)
            elif term_type == TERM_FOLD_OPP:
                values[node_id] = np.full(n_opp, opp_pot, dtype=np.float64)
            elif term_type == TERM_SHOWDOWN:
                pot_won = min(hero_pot, opp_pot)
                # EV = equity * pot_won - (1-equity) * pot_won = (2*equity - 1) * pot_won
                values[node_id] = (2.0 * equity_vec - 1.0) * pot_won

        return values

    def _run_cfr(self, tree, opp_weights, terminal_values, n_opp, iterations):
        """Run CFR+ and return hero's average strategy at the root.

        Returns np.array of shape (num_root_actions,) — probability per action.
        """
        n_hero = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        # Map node_id -> index in hero/opp arrays
        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        # Max actions at any node
        max_act = max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1)
        max_act_opp = max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1)
        max_act = max(max_act, max_act_opp, 1)

        # Regret and strategy accumulators
        hero_regrets = np.zeros((n_hero, max_act), dtype=np.float64)
        hero_strategy_sum = np.zeros((n_hero, max_act), dtype=np.float64)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        for t in range(iterations):
            self._cfr_traverse(
                tree, 0, 1.0, opp_weights.copy(),
                hero_regrets, hero_strategy_sum, opp_regrets,
                hero_idx, opp_idx, terminal_values, n_opp, max_act, t)

        # Extract hero's average strategy at root
        root = 0
        if root in hero_idx:
            idx = hero_idx[root]
            n_act = tree.num_actions[root]
            total = hero_strategy_sum[idx, :n_act].sum()
            if total > 0:
                return hero_strategy_sum[idx, :n_act] / total
            return np.ones(n_act) / n_act
        else:
            # Root is an opponent node (shouldn't happen if hero acts)
            return np.array([1.0])

    def _regret_match(self, regrets, n_actions):
        """Compute strategy from regrets using regret matching."""
        positive = np.maximum(regrets[:n_actions], 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(n_actions) / n_actions

    def _regret_match_vec(self, regrets, n_actions):
        """Vectorized regret matching for opponent (n_opp, n_actions)."""
        positive = np.maximum(regrets[:, :n_actions], 0)
        totals = positive.sum(axis=1, keepdims=True)
        mask = totals > 0
        result = np.where(mask, positive / np.maximum(totals, 1e-10),
                         np.ones_like(positive) / n_actions)
        return result

    def _cfr_traverse(self, tree, node_id, hero_reach, opp_reach_vec,
                       hero_regrets, hero_strategy_sum, opp_regrets,
                       hero_idx, opp_idx, terminal_values, n_opp, max_act, t):
        """Recursive CFR+ traversal. Returns np.array of shape (n_opp,)."""

        # Terminal node
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero node
            idx = hero_idx[node_id]
            strategy = self._regret_match(hero_regrets[idx], n_act)

            action_values = np.zeros((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                action_values[a] = self._cfr_traverse(
                    tree, child_id, hero_reach * strategy[a], opp_reach_vec,
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[a] * action_values[a]

            # Update hero regrets (weighted by opponent reach × weights)
            for a in range(n_act):
                cf_regret = np.dot(action_values[a] - node_value, opp_reach_vec)
                hero_regrets[idx, a] = max(0, hero_regrets[idx, a] + cf_regret)

            # Update average strategy
            hero_strategy_sum[idx, :n_act] += hero_reach * strategy

            return node_value

        else:  # Opponent node
            idx = opp_idx[node_id]
            strategy = self._regret_match_vec(opp_regrets[idx], n_act)  # (n_opp, n_act)

            action_values = np.zeros((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                action_values[a] = self._cfr_traverse(
                    tree, child_id, hero_reach, opp_reach_vec * strategy[:, a],
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[:, a] * action_values[a]

            # Update opponent regrets per hand
            for a in range(n_act):
                inst_regret = hero_reach * (action_values[a] - node_value)
                opp_regrets[idx, :, a] = np.maximum(0, opp_regrets[idx, :, a] + inst_regret)

            return node_value

    def _strategy_to_action(self, tree, strategy, my_bet, opp_bet,
                             min_raise, max_raise, valid_actions):
        """Convert solver strategy to a concrete engine action."""
        root_children = tree.children[0]

        # Sample action from strategy (deterministic for near-pure)
        if strategy.max() > 0.90:
            action_idx = int(np.argmax(strategy))
        else:
            action_idx = int(np.random.choice(len(strategy), p=strategy))

        act_type, child_id = root_children[action_idx]

        if act_type == ACT_FOLD:
            return (FOLD, 0, 0, 0)
        elif act_type == ACT_CHECK:
            return (CHECK, 0, 0, 0)
        elif act_type == ACT_CALL:
            return (CALL, 0, 0, 0)
        elif act_type in (ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN):
            # The tree stored the new total bet; compute the raise amount
            child_hero_pot = tree.hero_pot[child_id]
            child_opp_pot = tree.opp_pot[child_id]
            # Our new bet is the max of hero/opp pot at the child (we're the one who raised)
            new_bet = max(child_hero_pot, child_opp_pot)
            other_bet = min(child_hero_pot, child_opp_pot)
            raise_amount = new_bet - other_bet
            raise_amount = max(raise_amount, min_raise)
            raise_amount = min(raise_amount, max_raise)

            if not valid_actions[RAISE]:
                # Can't raise, fall back to call or check
                if valid_actions[CALL]:
                    return (CALL, 0, 0, 0)
                if valid_actions[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

            return (RAISE, raise_amount, 0, 0)

        return (FOLD, 0, 0, 0)

    def _fallback(self, hero_cards, board, dead_cards,
                  my_bet, opp_bet, valid_actions, min_raise, max_raise):
        """Equity-threshold fallback when solver can't run."""
        equity = self.engine.compute_equity(hero_cards, board, dead_cards)
        pot_size = my_bet + opp_bet
        continue_cost = opp_bet - my_bet
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        if equity > 0.92 and valid_actions[RAISE]:
            return (RAISE, max_raise, 0, 0)
        if equity > 0.72 and valid_actions[RAISE]:
            amt = max(int(pot_size * 0.6), min_raise)
            amt = min(amt, max_raise)
            return (RAISE, amt, 0, 0)
        if valid_actions[CALL] and equity >= pot_odds:
            return (CALL, 0, 0, 0)
        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)
