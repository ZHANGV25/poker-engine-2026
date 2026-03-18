"""
Range-based real-time subgame re-solver.

Unlike the one-hand solver (solver.py), this solves for ALL hero hands
simultaneously, producing range-balanced strategies. Used when the
opponent's range has been narrowed by their betting actions, making
the precomputed blueprint inaccurate.

Key differences from one-hand solver:
- Hero has N_HERO possible hands, each with its own strategy
- Terminal values: (n_terminals, n_hero, n_opp) matrix
- Hero regrets: (n_hero_nodes, n_hero, max_act)
- Output: strategy for each hero hand (not just one hand)

This is what Pluribus calls "depth-limited subgame solving."
"""

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


class RangeSolver:
    def __init__(self, equity_engine):
        self.engine = equity_engine
        self._tree_cache = {}

    def solve_and_act(self, hero_cards, board, opp_range, dead_cards,
                      my_bet, opp_bet, street, min_raise, max_raise,
                      valid_actions, time_remaining):
        """Re-solve the subgame with full hero range and narrowed opponent range.

        Args:
            hero_cards: list of 2 ints (our actual hand)
            board: list of 3-5 ints
            opp_range: dict (c1,c2) -> weight (narrowed)
            dead_cards: list of ints
            my_bet, opp_bet: current bets
            street: 1-3
            min_raise, max_raise: raise bounds
            valid_actions: list of 5 bools
            time_remaining: seconds left

        Returns:
            (action_type, raise_amount, 0, 0) tuple
        """
        if opp_range is None:
            return None  # can't re-solve without opponent range

        # Build hero and opponent hand lists
        known = set(board) | set(dead_cards)
        hero_hands = []
        opp_hands = []
        opp_weights = []

        for hand, weight in opp_range.items():
            if weight > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(weight)

        if not opp_hands:
            return None

        opp_weights = np.array(opp_weights, dtype=np.float64)
        opp_weights /= opp_weights.sum()

        # Enumerate all possible hero hands
        remaining = [c for c in range(27) if c not in known]
        for h in itertools.combinations(remaining, 2):
            hero_hands.append(h)

        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        if n_hero == 0:
            return None

        # Find our hand's index
        hero_tuple = tuple(sorted(hero_cards))
        hero_idx_in_list = None
        for i, h in enumerate(hero_hands):
            if tuple(sorted(h)) == hero_tuple:
                hero_idx_in_list = i
                break

        if hero_idx_in_list is None:
            return None

        # Used ONLY for river facing-bet decisions (~50 calls/match).
        # Turn uses equity thresholds (0 compute) so 94% of budget is free.
        # 200 iters at ~50 calls = ~500s ARM = 33% of 1500s budget.
        if time_remaining > 800:
            iterations = 200
        elif time_remaining > 400:
            iterations = 100
        elif time_remaining > 200:
            iterations = 50
        else:
            iterations = 20

        # Build game tree
        max_bet = 100
        tree = self._get_tree(my_bet, opp_bet, min_raise, max_bet)

        if tree.size < 2:
            return None

        # Compute pairwise equity matrix: (n_hero, n_opp)
        equity_matrix = self._compute_equity_matrix(
            hero_hands, opp_hands, board, dead_cards, street)

        # Compute terminal values for all (hero, opp) pairs
        terminal_values = self._compute_terminal_values(
            tree, equity_matrix, n_hero, n_opp)

        # Run range-based CFR
        hero_strategy = self._run_range_cfr(
            tree, opp_weights, terminal_values,
            n_hero, n_opp, iterations)

        # Extract strategy for our specific hand
        our_strategy = hero_strategy[hero_idx_in_list]

        # Convert to action
        return self._strategy_to_action(
            tree, our_strategy, my_bet, opp_bet, min_raise, max_raise,
            valid_actions)

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet):
        key = (hero_bet, opp_bet, min_raise, max_bet, True)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(hero_bet, opp_bet, min_raise, max_bet, True)
        return self._tree_cache[key]

    def _compute_equity_matrix(self, hero_hands, opp_hands, board, dead_cards, street):
        """Compute equity[h][o] = hero hand h's equity vs opp hand o."""
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)
        equity = np.zeros((n_hero, n_opp), dtype=np.float64)

        known_base = set(board) | set(dead_cards)
        board_list = list(board)

        if len(board) == 5:
            # River: deterministic
            for hi, hh in enumerate(hero_hands):
                if set(hh) & known_base:
                    continue
                hr = self.engine.lookup_seven(list(hh) + board_list)
                for oi, oh in enumerate(opp_hands):
                    if set(oh) & set(hh):
                        continue
                    opr = self.engine.lookup_seven(list(oh) + board_list)
                    if hr < opr:
                        equity[hi, oi] = 1.0
                    elif hr == opr:
                        equity[hi, oi] = 0.5
        else:
            # Turn: enumerate 1 remaining card
            board_needed = 5 - len(board)
            for hi, hh in enumerate(hero_hands):
                hh_set = set(hh)
                if hh_set & known_base:
                    continue
                known_h = known_base | hh_set
                for oi, oh in enumerate(opp_hands):
                    oh_set = set(oh)
                    if oh_set & hh_set:
                        continue
                    known_ho = known_h | oh_set
                    remaining = [c for c in range(27) if c not in known_ho]
                    wins = 0.0
                    total = 0
                    for runout in itertools.combinations(remaining, board_needed):
                        full_board = board_list + list(runout)
                        hr = self.engine.lookup_seven(list(hh) + full_board)
                        opr = self.engine.lookup_seven(list(oh) + full_board)
                        if hr < opr:
                            wins += 1.0
                        elif hr == opr:
                            wins += 0.5
                        total += 1
                    equity[hi, oi] = wins / total if total > 0 else 0.5

        return equity

    def _compute_terminal_values(self, tree, equity_matrix, n_hero, n_opp):
        """Compute terminal values for all (hero, opp) pairs.

        Returns dict: node_id -> np.array of shape (n_hero, n_opp)
        """
        values = {}
        for node_id in tree.terminal_node_ids:
            term_type = tree.terminal[node_id]
            hero_pot = tree.hero_pot[node_id]
            opp_pot = tree.opp_pot[node_id]

            if term_type == TERM_FOLD_HERO:
                values[node_id] = np.full((n_hero, n_opp), -hero_pot, dtype=np.float64)
            elif term_type == TERM_FOLD_OPP:
                values[node_id] = np.full((n_hero, n_opp), opp_pot, dtype=np.float64)
            elif term_type == TERM_SHOWDOWN:
                pot_won = min(hero_pot, opp_pot)
                values[node_id] = (2.0 * equity_matrix - 1.0) * pot_won

        return values

    def _run_range_cfr(self, tree, opp_weights, terminal_values,
                        n_hero, n_opp, iterations):
        """Run CFR+ with full hero range.

        Returns np.array of shape (n_hero, n_root_actions) — strategy per hand.
        """
        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        hero_node_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_node_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1),
            1
        )

        # Hero: regrets and strategy sum per hand per node
        hero_regrets = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)
        hero_strat_sum = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)

        # Opponent: regrets per hand per node (same as one-hand solver)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        # Initial hero reach: uniform (each hand equally likely)
        hero_reach_init = np.ones(n_hero, dtype=np.float64) / n_hero

        for t in range(iterations):
            self._range_cfr_traverse(
                tree, 0, hero_reach_init.copy(), opp_weights.copy(),
                hero_regrets, hero_strat_sum, opp_regrets,
                hero_node_idx, opp_node_idx, terminal_values,
                n_hero, n_opp, max_act)

        # Extract average strategy at root for each hero hand
        root = 0
        if root not in hero_node_idx:
            return np.ones((n_hero, 1)) / 1

        idx = hero_node_idx[root]
        n_act = tree.num_actions[root]
        result = np.zeros((n_hero, n_act), dtype=np.float64)

        for hi in range(n_hero):
            total = hero_strat_sum[idx, hi, :n_act].sum()
            if total > 0:
                result[hi] = hero_strat_sum[idx, hi, :n_act] / total
            else:
                result[hi] = np.ones(n_act) / n_act

        return result

    def _regret_match_single(self, regrets, n_act):
        """Regret matching for a single information set."""
        pos = np.maximum(regrets[:n_act], 0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(n_act) / n_act

    def _range_cfr_traverse(self, tree, node_id, hero_reach, opp_reach,
                             hero_regrets, hero_strat_sum, opp_regrets,
                             hero_node_idx, opp_node_idx, terminal_values,
                             n_hero, n_opp, max_act):
        """Range-based CFR traversal.

        Returns np.array of shape (n_hero, n_opp) — utility matrix.
        """
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero node
            idx = hero_node_idx[node_id]

            # Each hero hand has its own strategy
            strategies = np.zeros((n_hero, n_act), dtype=np.float64)
            for hi in range(n_hero):
                strategies[hi] = self._regret_match_single(hero_regrets[idx, hi], n_act)

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float64)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                new_hero_reach = hero_reach * strategies[:, a]
                action_values[a] = self._range_cfr_traverse(
                    tree, child_id, new_hero_reach, opp_reach,
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_node_idx, opp_node_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1] * action_values[a]

            # Update regrets per hero hand
            for hi in range(n_hero):
                for a in range(n_act):
                    cf_regret = np.dot(action_values[a, hi] - node_value[hi], opp_reach)
                    hero_regrets[idx, hi, a] = max(0, hero_regrets[idx, hi, a] + cf_regret)

            # Update strategy sum
            for hi in range(n_hero):
                hero_strat_sum[idx, hi, :n_act] += hero_reach[hi] * strategies[hi]

            return node_value

        else:  # Opponent node
            idx = opp_node_idx[node_id]

            # Each opp hand has its own strategy
            strategies = np.zeros((n_opp, n_act), dtype=np.float64)
            for oi in range(n_opp):
                strategies[oi] = self._regret_match_single(opp_regrets[idx, oi], n_act)

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float64)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                new_opp_reach = opp_reach * strategies[:, a]
                action_values[a] = self._range_cfr_traverse(
                    tree, child_id, hero_reach, new_opp_reach,
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_node_idx, opp_node_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1].T * action_values[a]

            # Update opp regrets
            for oi in range(n_opp):
                for a in range(n_act):
                    # Opp utility = -hero utility
                    cf_regret = np.dot(hero_reach, node_value[:, oi] - action_values[a, :, oi])
                    opp_regrets[idx, oi, a] = max(0, opp_regrets[idx, oi, a] + cf_regret)

            return node_value

    def _strategy_to_action(self, tree, strategy, my_bet, opp_bet,
                             min_raise, max_raise, valid_actions):
        """Convert strategy to concrete engine action."""
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
