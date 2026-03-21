"""
Range-based real-time subgame re-solver with DCFR.

Solves for ALL hero hands simultaneously, producing range-balanced
strategies. Full tree (4 bet sizes, 2 raises max) accounts for
re-raise threat, giving realistic ~17% bet frequency acting first
with proper value+bluff polarization.

Key features:
- Full tree with re-raises (not compact) for solve_and_act
- DCFR (Noam Brown's params: alpha=1.5, beta=0, gamma=2)
- Proper card blocking in terminal values (blocked pairs = 0)
- Vectorized traversal with numpy tensordot
"""

import numpy as np
import itertools
from math import pow as fpow
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

        Uses full tree (4 bet sizes, 2 raises) + DCFR for balanced strategies.
        """
        if opp_range is None:
            return None

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

        remaining = [c for c in range(27) if c not in known]
        for h in itertools.combinations(remaining, 2):
            hero_hands.append(h)

        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        if n_hero == 0:
            return None

        hero_tuple = tuple(sorted(hero_cards))
        hero_idx_in_list = None
        for i, h in enumerate(hero_hands):
            if tuple(sorted(h)) == hero_tuple:
                hero_idx_in_list = i
                break

        if hero_idx_in_list is None:
            return None

        # Full tree + DCFR: ~1.2s per 500 iters Mac, ~3s ARM.
        if time_remaining > 600:
            iterations = 500
        elif time_remaining > 300:
            iterations = 300
        elif time_remaining > 100:
            iterations = 200
        else:
            iterations = 50

        # Full tree: 4 bet sizes, 2 raises max
        max_bet = 100
        tree = self._get_tree(my_bet, opp_bet, min_raise, max_bet,
                              compact=False)

        if tree.size < 2:
            return None

        # Compute equity matrix AND not-blocked mask
        equity_matrix, not_blocked = self._compute_equity_and_mask(
            hero_hands, opp_hands, board, dead_cards, street)

        # Terminal values with proper card blocking
        terminal_values = self._compute_terminal_values(
            tree, equity_matrix, not_blocked)

        # Run DCFR
        hero_strategy = self._run_dcfr(
            tree, opp_weights, terminal_values,
            n_hero, n_opp, iterations)

        our_strategy = hero_strategy[hero_idx_in_list]

        return self._strategy_to_action(
            tree, our_strategy, my_bet, opp_bet, min_raise, max_raise,
            valid_actions)

    def compute_opp_bet_probs(self, board, opp_range, hero_range,
                               dead_cards, my_bet, opp_bet, street,
                               min_raise, iterations=1000):
        """Compute P(bet|hand) for each opponent hand.

        Uses compact tree (lighter weight, adequate for narrowing).
        """
        known = set(board) | set(dead_cards)

        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

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

        n_opp = len(opp_hands)
        n_hero = len(hero_hands)

        # Compact tree for narrowing (lighter weight)
        pre_bet = min(my_bet, opp_bet)
        tree = self._get_tree(pre_bet, pre_bet, min_raise, 100,
                              compact=True)
        if tree.size < 2:
            return None

        eq_matrix, nb_mask = self._compute_equity_and_mask(
            opp_hands, hero_hands, board, dead_cards, street)

        tv = self._compute_terminal_values(tree, eq_matrix, nb_mask)

        strat = self._run_dcfr(tree, hero_w, tv, n_opp, n_hero, iterations)

        children = tree.children[0]
        bet_indices = [a for a, (act_type, _) in enumerate(children)
                       if act_type != ACT_CHECK]

        result = {}
        for oi, hand in enumerate(opp_hands):
            p_bet = float(strat[oi, bet_indices].sum()) if bet_indices else 0.0
            key = (min(hand[0], hand[1]), max(hand[0], hand[1]))
            result[key] = p_bet

        return result

    def compute_opp_call_probs(self, board, opp_range, hero_range,
                                dead_cards, hero_bet, opp_bet_before,
                                street, min_raise=2, iterations=200):
        """Compute P(continue|hand) for each opponent hand when FACING our bet.

        Unlike compute_opp_bet_probs (which solves acting-first check/bet),
        this solves from the opponent's perspective at the FACING-BET node:
        they see our raise and decide fold/call/raise.

        P(continue|hand) = P(call|hand) + P(raise|hand) = 1 - P(fold|hand).

        Args:
            board: community cards
            opp_range: opponent's current narrowed range
            hero_range: our range (from opponent's perspective)
            dead_cards: discards
            hero_bet: our total bet (the bet opponent faces)
            opp_bet_before: opponent's bet before our raise
            street: 1-3
            min_raise: minimum raise increment
            iterations: DCFR iterations

        Returns:
            dict {(c1,c2): p_continue} or None
        """
        known = set(board) | set(dead_cards)

        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

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

        n_opp = len(opp_hands)
        n_hero = len(hero_hands)

        # Build tree from OPPONENT's perspective facing our bet.
        # Opponent is "hero" in the tree with opp_bet_before chips in.
        # They face our raise (hero_bet > opp_bet_before).
        # Tree root: opponent decides fold/call/raise.
        tree = self._get_tree(opp_bet_before, hero_bet, min_raise, 100,
                              compact=True)
        if tree.size < 2:
            return None

        # Equity from opponent's perspective
        eq_matrix, nb_mask = self._compute_equity_and_mask(
            opp_hands, hero_hands, board, dead_cards, street)

        tv = self._compute_terminal_values(tree, eq_matrix, nb_mask)

        # Solve: opponent is "hero", we are "opp"
        strat = self._run_dcfr(tree, hero_w, tv, n_opp, n_hero, iterations)

        # Extract P(continue|hand) = 1 - P(fold|hand)
        # At root, opponent faces our bet: actions are fold/call/raise
        children = tree.children[0]
        fold_idx = None
        for a, (act_type, _) in enumerate(children):
            if act_type == ACT_FOLD:
                fold_idx = a
                break

        result = {}
        for oi, hand in enumerate(opp_hands):
            if fold_idx is not None and fold_idx < strat.shape[1]:
                p_fold = float(strat[oi, fold_idx])
                p_continue = 1.0 - p_fold
            else:
                p_continue = 1.0  # no fold action → always continues
            key = (min(hand[0], hand[1]), max(hand[0], hand[1]))
            result[key] = p_continue

        return result

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet, compact):
        key = (hero_bet, opp_bet, min_raise, max_bet, True, compact)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(
                hero_bet, opp_bet, min_raise, max_bet, True, compact=compact)
        return self._tree_cache[key]

    def _compute_equity_and_mask(self, hero_hands, opp_hands, board,
                                  dead_cards, street):
        """Compute equity matrix and not-blocked mask using vectorized numpy.

        Returns (equity, not_blocked) where:
        - equity[h][o] = hero hand h's equity vs opp hand o (0 if blocked)
        - not_blocked[h][o] = 1.0 if hands don't share cards, 0.0 if they do

        Uses numpy broadcasting instead of nested Python loops:
        - River: 272 lookups + 1 broadcast vs 18,496 loop iterations
        - Turn: 17 × 272 lookups + 17 broadcasts vs 258,944 loop iterations
        - Flop: 91 × 272 lookups + 91 broadcasts vs ~1M loop iterations
        """
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)
        known_base = set(board) | set(dead_cards)
        board_list = list(board)
        seven = self.engine._seven

        # Build not-blocked mask: 1 where hands don't share cards
        # Use bitmask representation for fast overlap detection
        hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands],
                              dtype=np.int64)
        opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands],
                             dtype=np.int64)
        # Blocked if any bit overlaps: (hero_mask & opp_mask) != 0
        overlap = hero_masks[:, None] & opp_masks[None, :]  # (n_hero, n_opp)
        not_blocked = (overlap == 0).astype(np.float64)

        # Also zero out hero hands that overlap with known cards
        known_mask = 0
        for c in known_base:
            known_mask |= 1 << c
        for hi in range(n_hero):
            if hero_masks[hi] & known_mask:
                not_blocked[hi, :] = 0.0

        board_mask = 0
        for c in board_list:
            board_mask |= 1 << c

        if len(board) == 5:
            # River: deterministic. Precompute all ranks, then broadcast.
            hero_keys = np.array([hero_masks[hi] | board_mask
                                  for hi in range(n_hero)], dtype=np.int64)
            opp_keys = np.array([opp_masks[oi] | board_mask
                                 for oi in range(n_opp)], dtype=np.int64)
            hero_ranks = np.array([seven.get(int(k), 9999)
                                   for k in hero_keys], dtype=np.int32)
            opp_ranks = np.array([seven.get(int(k), 9999)
                                  for k in opp_keys], dtype=np.int32)

            # Broadcasting: (n_hero, 1) vs (1, n_opp)
            equity = np.where(
                hero_ranks[:, None] < opp_ranks[None, :], 1.0,
                np.where(hero_ranks[:, None] == opp_ranks[None, :], 0.5, 0.0))
            equity *= not_blocked

        else:
            # Turn/Flop: enumerate runout cards, vectorize per-card comparison.
            # For each possible runout, compute ranks for all hands at once.
            remaining_cards = [c for c in range(27) if c not in known_base]
            board_needed = 5 - len(board)
            equity = np.zeros((n_hero, n_opp), dtype=np.float64)
            count = np.zeros((n_hero, n_opp), dtype=np.float64)

            for runout in itertools.combinations(remaining_cards, board_needed):
                runout_mask = 0
                for c in runout:
                    runout_mask |= 1 << c
                full_board_mask = board_mask | runout_mask

                # Mask out hands that contain a runout card
                runout_set_mask = runout_mask
                hero_valid = np.array(
                    [(hero_masks[hi] & runout_set_mask) == 0
                     for hi in range(n_hero)], dtype=bool)
                opp_valid = np.array(
                    [(opp_masks[oi] & runout_set_mask) == 0
                     for oi in range(n_opp)], dtype=bool)

                # Valid pair matrix
                valid_pairs = (hero_valid[:, None] & opp_valid[None, :]).astype(
                    np.float64) * not_blocked

                # Compute ranks for all hands on this full board
                hero_ranks = np.array(
                    [seven.get(int(hero_masks[hi] | full_board_mask), 9999)
                     if hero_valid[hi] else 9999
                     for hi in range(n_hero)], dtype=np.int32)
                opp_ranks = np.array(
                    [seven.get(int(opp_masks[oi] | full_board_mask), 9999)
                     if opp_valid[oi] else 9999
                     for oi in range(n_opp)], dtype=np.int32)

                # Vectorized comparison
                wins = (hero_ranks[:, None] < opp_ranks[None, :]).astype(
                    np.float64) * valid_pairs
                ties = (hero_ranks[:, None] == opp_ranks[None, :]).astype(
                    np.float64) * valid_pairs

                equity += wins + 0.5 * ties
                count += valid_pairs

            # Average equity across runouts
            equity = np.where(count > 0, equity / np.maximum(count, 1), 0.0)

        return equity, not_blocked

    def _compute_terminal_values(self, tree, equity_matrix, not_blocked):
        """Terminal values with proper card blocking.

        All terminal values are multiplied by not_blocked so that
        impossible hand matchups (sharing cards) contribute 0.
        """
        values = {}
        for node_id in tree.terminal_node_ids:
            term_type = tree.terminal[node_id]
            hero_pot = tree.hero_pot[node_id]
            opp_pot = tree.opp_pot[node_id]

            if term_type == TERM_FOLD_HERO:
                values[node_id] = -hero_pot * not_blocked
            elif term_type == TERM_FOLD_OPP:
                values[node_id] = opp_pot * not_blocked
            elif term_type == TERM_SHOWDOWN:
                pot_won = min(hero_pot, opp_pot)
                values[node_id] = (2.0 * equity_matrix - 1.0) * pot_won * not_blocked

        return values

    def _run_dcfr(self, tree, opp_weights, terminal_values,
                   n_hero, n_opp, iterations):
        """Run DCFR with Noam Brown's parameters.

        DCFR discounts early iterations so the average strategy
        converges faster. Parameters: alpha=1.5, beta=0, gamma=2.
        """
        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        hero_node_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_node_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1),
            1)

        hero_regrets = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)
        hero_strat_sum = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        hero_reach_init = np.ones(n_hero, dtype=np.float64) / n_hero

        # DCFR parameters (from Noam Brown's poker solver)
        alpha, beta, gamma = 1.5, 0.0, 2.0

        for t in range(1, iterations + 1):
            if t > 1:
                pos_w = fpow(t - 1, alpha) / (fpow(t - 1, alpha) + 1.0)
                neg_w = fpow(t - 1, beta) / (fpow(t - 1, beta) + 1.0)
                strat_w = fpow((t - 1) / t, gamma)

                hero_regrets *= np.where(hero_regrets > 0, pos_w, neg_w)
                opp_regrets *= np.where(opp_regrets > 0, pos_w, neg_w)
                hero_strat_sum *= strat_w

            self._range_cfr_traverse(
                tree, 0, hero_reach_init.copy(), opp_weights.copy(),
                hero_regrets, hero_strat_sum, opp_regrets,
                hero_node_idx, opp_node_idx, terminal_values,
                n_hero, n_opp, max_act)

        # Extract average strategy at root
        root = 0
        if root not in hero_node_idx:
            return np.ones((n_hero, 1)) / 1

        idx = hero_node_idx[root]
        n_act = tree.num_actions[root]
        strat_slice = hero_strat_sum[idx, :, :n_act]
        totals = strat_slice.sum(axis=1, keepdims=True)
        result = np.where(totals > 0, strat_slice / np.maximum(totals, 1e-10),
                         np.full_like(strat_slice, 1.0 / n_act))
        return result

    @staticmethod
    def _regret_match_batch(regrets, n_act):
        pos = np.maximum(regrets[:, :n_act], 0)
        totals = pos.sum(axis=1, keepdims=True)
        return np.where(totals > 0, pos / np.maximum(totals, 1e-10),
                       np.full_like(pos, 1.0 / n_act))

    def _range_cfr_traverse(self, tree, node_id, hero_reach, opp_reach,
                             hero_regrets, hero_strat_sum, opp_regrets,
                             hero_node_idx, opp_node_idx, terminal_values,
                             n_hero, n_opp, max_act):
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero
            idx = hero_node_idx[node_id]
            strategies = self._regret_match_batch(hero_regrets[idx], n_act)

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

            diff = action_values - node_value[np.newaxis, :, :]
            cf_regrets = np.tensordot(diff, opp_reach, axes=([2], [0]))
            hero_regrets[idx, :, :n_act] += cf_regrets.T

            hero_strat_sum[idx, :, :n_act] += hero_reach[:, np.newaxis] * strategies
            return node_value

        else:  # Opponent
            idx = opp_node_idx[node_id]
            strategies = self._regret_match_batch(opp_regrets[idx], n_act)

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

            diff = node_value[np.newaxis, :, :] - action_values
            cf_regrets = np.tensordot(diff, hero_reach, axes=([1], [0]))
            opp_regrets[idx, :, :n_act] += cf_regrets.T

            return node_value

    def _strategy_to_action(self, tree, strategy, my_bet, opp_bet,
                             min_raise, max_raise, valid_actions):
        root_children = tree.children[0]

        strategy = np.maximum(strategy, 0)
        total = strategy.sum()
        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.ones(len(strategy)) / len(strategy)

        action_idx = int(np.random.choice(len(strategy), p=strategy))
        act_type, child_id = root_children[action_idx]

        if act_type == ACT_FOLD:
            return (FOLD, 0, 0, 0)
        elif act_type == ACT_CHECK:
            return (CHECK, 0, 0, 0)
        elif act_type == ACT_CALL:
            return (CALL, 0, 0, 0)
        elif act_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                          ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
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
