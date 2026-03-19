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

    def compute_opponent_bet_probs(self, board, dead_cards, opp_range,
                                    hero_range, my_bet, opp_bet, street,
                                    min_raise, iterations=50):
        """Compute P(bet|hand) for each opponent hand via CFR.

        Solves the game from opponent's perspective: opponent is "hero" with
        a range of hands, and we are "opponent" with hero_range. Returns
        the probability that each opponent hand would bet/raise (any
        aggressive action) at their first decision.

        This replaces heuristic polarized narrowing with game-theoretic
        Bayesian narrowing. Board-specific, pot-specific, equilibrium-derived.

        Args:
            board: list of community cards
            dead_cards: discards
            opp_range: dict {(c1,c2): weight} — opponent's current range
            hero_range: dict {(c1,c2): weight} — our range (from opp's view)
            my_bet, opp_bet: current bets (from OUR perspective)
            street: 2 or 3
            min_raise: minimum raise
            iterations: CFR iterations (50 is enough for P(bet|hand))

        Returns:
            dict {(c1,c2): p_bet} or None
        """
        known = set(board) | set(dead_cards)

        # Opponent hands = "hero" in the solver
        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

        # Our hands = "opponent" in the solver
        hero_hands = []
        hero_weights = []
        for hand, w in hero_range.items():
            if w > 0.001 and not (set(hand) & known):
                hero_hands.append(hand)
                hero_weights.append(w)

        if len(opp_hands) < 3 or len(hero_hands) < 3:
            return None

        opp_w = np.array(opp_weights, dtype=np.float64)
        opp_w /= opp_w.sum()
        hero_w = np.array(hero_weights, dtype=np.float64)
        hero_w /= hero_w.sum()

        n_opp = len(opp_hands)   # "hero" in solver
        n_hero = len(hero_hands)  # "opponent" in solver

        # Build tree from OPPONENT's perspective: they act first (deciding
        # to bet/check), we respond. Swap bets: opponent's bet = opp_bet,
        # their "opp" bet = my_bet.
        max_bet = 100
        tree = self._get_tree(opp_bet, my_bet, min_raise, max_bet, True)

        if tree.size < 2:
            return None

        # Compute equity matrix: opp_hand (hero) vs hero_hand (opp)
        # Terminal values from opponent's perspective
        n_terms = len(tree.terminal_node_ids)
        equity_matrix = np.zeros((n_opp, n_hero), dtype=np.float64)

        board_list = list(board)
        if len(board) == 5:
            for oi, oh in enumerate(opp_hands):
                oh_rank = self.engine.lookup_seven(list(oh) + board_list)
                for hi, hh in enumerate(hero_hands):
                    if set(hh) & set(oh):
                        continue
                    hh_rank = self.engine.lookup_seven(list(hh) + board_list)
                    if oh_rank < hh_rank:
                        equity_matrix[oi, hi] = 1.0
                    elif oh_rank == hh_rank:
                        equity_matrix[oi, hi] = 0.5
        else:
            seven_lookup = self.engine._seven
            board_mask = 0
            for c in board_list:
                board_mask |= 1 << c
            board_needed = 5 - len(board)
            for oi, oh in enumerate(opp_hands):
                oh_set = set(oh)
                opp_base = (1 << oh[0]) | (1 << oh[1])
                for hi, hh in enumerate(hero_hands):
                    if set(hh) & oh_set:
                        continue
                    hero_base = (1 << hh[0]) | (1 << hh[1])
                    remaining = [c for c in range(27)
                                 if c not in known and c not in oh and c not in hh]
                    wins = total = 0
                    for runout in itertools.combinations(remaining, board_needed):
                        rm = 0
                        for c in runout:
                            rm |= 1 << c
                        fm = board_mask | rm
                        or_ = seven_lookup[opp_base | fm]
                        hr_ = seven_lookup[hero_base | fm]
                        if or_ < hr_:
                            wins += 1
                        elif or_ == hr_:
                            wins += 0.5
                        total += 1
                    equity_matrix[oi, hi] = wins / total if total > 0 else 0.5

        # Build terminal values for each (opp_hand, hero_hand) pair
        # From opponent's perspective: they are "hero" in the tree
        terminal_values = {}
        for node_id in tree.terminal_node_ids:
            tt = tree.terminal[node_id]
            hp = tree.hero_pot[node_id]   # opponent's pot (they're "hero")
            op = tree.opp_pot[node_id]    # our pot (we're "opp")
            tv = np.zeros((n_opp, n_hero), dtype=np.float64)

            if tt == TERM_FOLD_HERO:
                tv[:, :] = -hp  # opponent folds, loses hp
            elif tt == TERM_FOLD_OPP:
                tv[:, :] = op   # we fold, opponent wins op
            elif tt == TERM_SHOWDOWN:
                pot_won = min(hp, op)
                tv = (2.0 * equity_matrix - 1.0) * pot_won

            terminal_values[node_id] = tv

        # Run CFR — simplified for extracting opponent strategy
        # "hero" = opponent (has n_opp hands), "opp" = us (has n_hero hands)
        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1), 1)

        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        # Hero (=opponent) has per-hand regrets and strategy sums
        hero_regrets = np.zeros((n_hero_nodes, n_opp, max_act), dtype=np.float64)
        hero_strat_sum = np.zeros((n_hero_nodes, n_opp, max_act), dtype=np.float64)
        # Opp (=us) has per-hand regrets
        opp_regrets = np.zeros((n_opp_nodes, n_hero, max_act), dtype=np.float64)

        def regret_match(reg, n_act):
            pos = np.maximum(reg[:n_act], 0)
            t = pos.sum()
            return pos / t if t > 0 else np.full(n_act, 1.0 / n_act)

        def traverse(nid, hero_reach, opp_reach):
            if tree.terminal[nid] != TERM_NONE:
                return terminal_values[nid]

            n_act = tree.num_actions[nid]
            children = tree.children[nid]
            player = tree.player[nid]

            if player == 0:  # "hero" = opponent, per-hand strategy
                idx = hero_idx[nid]
                strats = np.zeros((n_opp, n_act), dtype=np.float64)
                for oi in range(n_opp):
                    strats[oi] = regret_match(hero_regrets[idx, oi], n_act)

                act_vals = np.zeros((n_act, n_opp, n_hero), dtype=np.float64)
                node_val = np.zeros((n_opp, n_hero), dtype=np.float64)
                for a in range(n_act):
                    new_reach = hero_reach * strats[:, a]
                    act_vals[a] = traverse(children[a][1], new_reach, opp_reach)
                    node_val += strats[:, a:a+1] * act_vals[a]

                for oi in range(n_opp):
                    for a in range(n_act):
                        cf = np.dot(act_vals[a, oi] - node_val[oi], opp_reach)
                        hero_regrets[idx, oi, a] = max(0, hero_regrets[idx, oi, a] + cf)
                    hero_strat_sum[idx, oi, :n_act] += hero_reach[oi] * strats[oi]

                return node_val

            else:  # "opp" = us, per-hand strategy
                idx = opp_idx[nid]
                strats = np.zeros((n_hero, n_act), dtype=np.float64)
                for hi in range(n_hero):
                    strats[hi] = regret_match(opp_regrets[idx, hi], n_act)

                act_vals = np.zeros((n_act, n_opp, n_hero), dtype=np.float64)
                node_val = np.zeros((n_opp, n_hero), dtype=np.float64)
                for a in range(n_act):
                    new_reach = opp_reach * strats[:, a]
                    act_vals[a] = traverse(children[a][1], hero_reach, new_reach)
                    node_val += strats[:, a:a+1].T * act_vals[a]

                for hi in range(n_hero):
                    for a in range(n_act):
                        cf = np.dot(hero_reach, node_val[:, hi] - act_vals[a, :, hi])
                        opp_regrets[idx, hi, a] = max(0, opp_regrets[idx, hi, a] + cf)

                return node_val

        hero_init = opp_w.copy()
        opp_init = hero_w.copy()
        for _ in range(iterations):
            traverse(0, hero_init.copy(), opp_init.copy())

        # Extract P(bet|hand) at root for each opponent hand
        root = 0
        if root not in hero_idx:
            return None

        idx = hero_idx[root]
        n_act = tree.num_actions[root]
        root_children = tree.children[root]

        # Identify which actions are aggressive (bet/raise)
        aggressive = set()
        for a, (act_type, _) in enumerate(root_children):
            if act_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                           ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
                aggressive.add(a)

        result = {}
        for oi, hand in enumerate(opp_hands):
            total = hero_strat_sum[idx, oi, :n_act].sum()
            if total > 0:
                strat = hero_strat_sum[idx, oi, :n_act] / total
            else:
                strat = np.ones(n_act) / n_act
            p_bet = sum(strat[a] for a in aggressive)
            result[hand] = p_bet

        return result

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
