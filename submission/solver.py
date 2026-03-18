"""
Real-time CFR+ subgame solver for post-flop poker decisions.

One-hand solver: hero has exactly ONE hand, opponent has ~20-90 hands.
Hero's information sets are determined purely by action history (no card
dimension). Opponent has ~20-90 possible hands. This asymmetry keeps
the tree small enough for real-time solving.
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
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
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
        if opp_range is None:
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

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

        # Iteration count scales with time remaining.
        # Phase 2 (1000s): 200 iters ≈ 500ms ARM64 → safe.
        # Phase 3 (1500s): 400 iters → better convergence.
        if time_remaining > 800:
            iterations = 400
        elif time_remaining > 400:
            iterations = 200
        elif time_remaining > 100:
            iterations = 100
        elif time_remaining > 30:
            iterations = 50
        else:
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        max_bet = 100
        # Always build tree with hero acting first. At the point solve_and_act
        # is called, it's hero's turn — the tree should start with hero's
        # decision (fold/call/raise if facing bet, check/bet if not).
        # hero_is_first=False would put opponent at root, causing _run_cfr
        # to fail to find hero's strategy.
        tree = self._get_tree(my_bet, opp_bet, min_raise, max_bet, True)

        if tree.size < 2:
            return self._fallback(hero_cards, board, dead_cards,
                                  my_bet, opp_bet, valid_actions, min_raise, max_raise)

        terminal_values = self._compute_terminal_values(
            tree, hero_cards, board, dead_cards, opp_hands, street)

        hero_strategy = self._run_cfr(tree, opp_weights, terminal_values,
                                       n_opp, iterations)

        return self._strategy_to_action(
            tree, hero_strategy, my_bet, opp_bet, min_raise, max_raise,
            valid_actions)

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet, hero_first):
        key = (hero_bet, opp_bet, min_raise, max_bet, hero_first)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(hero_bet, opp_bet, min_raise, max_bet, hero_first)
        return self._tree_cache[key]

    def _compute_terminal_values(self, tree, hero_cards, board, dead_cards,
                                  opp_hands, street):
        values = {}
        n_opp = len(opp_hands)
        equity_vec = np.zeros(n_opp, dtype=np.float64)

        if len(board) == 5:
            hero_rank = self.engine.lookup_seven(list(hero_cards) + list(board))
            for i, oh in enumerate(opp_hands):
                opp_rank = self.engine.lookup_seven(list(oh) + list(board))
                if hero_rank < opp_rank:
                    equity_vec[i] = 1.0
                elif hero_rank == opp_rank:
                    equity_vec[i] = 0.5
        else:
            known = set(hero_cards) | set(board) | set(dead_cards)
            board_needed = 5 - len(board)
            hero_list = list(hero_cards)
            board_list = list(board)
            seven_lookup = self.engine._seven

            for i, oh in enumerate(opp_hands):
                remaining = [c for c in range(27) if c not in known and c not in oh]
                wins = 0.0
                total = 0
                hero_base = (1 << hero_list[0]) | (1 << hero_list[1])
                opp_base = (1 << oh[0]) | (1 << oh[1])
                board_mask = 0
                for c in board_list:
                    board_mask |= 1 << c

                for runout in itertools.combinations(remaining, board_needed):
                    runout_mask = 0
                    for c in runout:
                        runout_mask |= 1 << c
                    full_mask = board_mask | runout_mask
                    hr = seven_lookup[hero_base | full_mask]
                    opr = seven_lookup[opp_base | full_mask]
                    if hr < opr:
                        wins += 1.0
                    elif hr == opr:
                        wins += 0.5
                    total += 1
                equity_vec[i] = wins / total if total > 0 else 0.5

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
                values[node_id] = (2.0 * equity_vec - 1.0) * pot_won

        return values

    def _run_cfr(self, tree, opp_weights, terminal_values, n_opp, iterations):
        n_hero = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1),
            1)

        hero_regrets = np.zeros((n_hero, max_act), dtype=np.float64)
        hero_strategy_sum = np.zeros((n_hero, max_act), dtype=np.float64)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        for t in range(iterations):
            self._cfr_traverse(
                tree, 0, 1.0, opp_weights,
                hero_regrets, hero_strategy_sum, opp_regrets,
                hero_idx, opp_idx, terminal_values, n_opp, max_act, t)

        # Extract hero's strategy at first hero decision node.
        # If hero acts first, root (node 0) is a hero node.
        # If opponent acts first, hero's first decision is a child of root.
        root = 0
        hero_node = None
        if root in hero_idx:
            hero_node = root
        else:
            # Root is opponent node — find hero's first decision node.
            # All children of the opponent root lead to hero decision nodes
            # (or terminals). We need the one that matches the current game
            # state (opponent already bet, so hero faces fold/call/raise).
            # When facing a bet, the relevant child is after opponent's
            # actual bet action. Since we don't know which action the opp
            # took, find ANY hero child — they all share the same hero
            # strategy in a one-hand solver (hero has one info set per node).
            for _, child_id in tree.children[root]:
                if child_id in hero_idx:
                    hero_node = child_id
                    break

        if hero_node is not None:
            idx = hero_idx[hero_node]
            n_act = tree.num_actions[hero_node]
            total = hero_strategy_sum[idx, :n_act].sum()
            if total > 0:
                return hero_strategy_sum[idx, :n_act] / total
            return np.ones(n_act) / n_act
        return np.array([1.0])

    def _cfr_traverse(self, tree, node_id, hero_reach, opp_reach_vec,
                       hero_regrets, hero_strategy_sum, opp_regrets,
                       hero_idx, opp_idx, terminal_values, n_opp, max_act, t):

        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero
            idx = hero_idx[node_id]
            reg = hero_regrets[idx, :n_act]
            pos = np.maximum(reg, 0.0)
            total = pos.sum()
            strategy = pos / total if total > 0 else np.full(n_act, 1.0 / n_act)

            action_values = np.empty((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._cfr_traverse(
                    tree, children[a][1], hero_reach * strategy[a], opp_reach_vec,
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[a] * action_values[a]

            for a in range(n_act):
                cf_regret = np.dot(action_values[a] - node_value, opp_reach_vec)
                hero_regrets[idx, a] = max(0.0, hero_regrets[idx, a] + cf_regret)

            hero_strategy_sum[idx, :n_act] += hero_reach * strategy
            return node_value

        else:  # Opponent
            idx = opp_idx[node_id]
            reg = opp_regrets[idx, :, :n_act]
            pos = np.maximum(reg, 0.0)
            totals = pos.sum(axis=1, keepdims=True)
            strategy = np.where(totals > 0, pos / np.maximum(totals, 1e-10),
                               np.full_like(pos, 1.0 / n_act))

            action_values = np.empty((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._cfr_traverse(
                    tree, children[a][1], hero_reach, opp_reach_vec * strategy[:, a],
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[:, a] * action_values[a]

            hr = hero_reach
            for a in range(n_act):
                inst_regret = hr * (node_value - action_values[a])
                opp_regrets[idx, :, a] = np.maximum(0.0, opp_regrets[idx, :, a] + inst_regret)

            return node_value

    def _strategy_to_action(self, tree, strategy, my_bet, opp_bet,
                             min_raise, max_raise, valid_actions):
        root_children = tree.children[0]
        action_idx = int(np.random.choice(len(strategy), p=strategy))
        act_type, child_id = root_children[action_idx]

        if act_type == ACT_FOLD:
            return (FOLD, 0, 0, 0)
        elif act_type == ACT_CHECK:
            return (CHECK, 0, 0, 0)
        elif act_type == ACT_CALL:
            return (CALL, 0, 0, 0)
        elif act_type in (ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            child_hero_pot = tree.hero_pot[child_id]
            child_opp_pot = tree.opp_pot[child_id]
            new_bet = max(child_hero_pot, child_opp_pot)
            other_bet = min(child_hero_pot, child_opp_pot)
            raise_amount = new_bet - other_bet
            raise_amount = max(raise_amount, min_raise)
            raise_amount = min(raise_amount, max_raise)

            if not valid_actions[RAISE]:
                if valid_actions[CALL]:
                    return (CALL, 0, 0, 0)
                if valid_actions[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

            return (RAISE, raise_amount, 0, 0)

        return (FOLD, 0, 0, 0)

    def _fallback(self, hero_cards, board, dead_cards,
                  my_bet, opp_bet, valid_actions, min_raise, max_raise):
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
