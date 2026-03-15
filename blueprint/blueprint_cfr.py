"""
Full-range CFR engine for blueprint strategy computation.

Unlike the real-time SubgameSolver (which assumes hero has ONE specific hand),
this solver treats BOTH players as having a RANGE of hands. Each player's
information set is indexed by (hand_bucket, action_history).

The regret tables are:
    hero_regrets:  (n_hero_buckets, n_hero_nodes, n_actions)
    opp_regrets:   (n_opp_buckets, n_opp_nodes, n_actions)

After convergence, the average strategy is the Nash equilibrium.
"""

import os
import sys
import numpy as np
import itertools

_submission_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission")
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)


class BlueprintCFR:
    """
    Full-range CFR+ solver for precomputing blueprint strategies.

    Solves a game where both players have a range of hands (bucketed by equity).
    The resulting strategy maps (bucket, decision_point) -> action probabilities.
    """

    def __init__(self, n_hero_buckets, n_opp_buckets, equity_engine):
        """
        Args:
            n_hero_buckets: number of equity buckets for hero hands
            n_opp_buckets: number of equity buckets for opponent hands
            equity_engine: ExactEquityEngine instance for hand evaluation
        """
        self.n_hero_buckets = n_hero_buckets
        self.n_opp_buckets = n_opp_buckets
        self.engine = equity_engine

    def solve(self, board, dead_cards, hero_bet, opp_bet, hero_first,
              n_iterations, min_raise=2, max_bet=100,
              checkpoint_callback=None, checkpoint_interval=1000):
        """
        Run full-range CFR+ and return the converged strategy.

        Args:
            board: list of card ints (3-5 community cards)
            dead_cards: list of card ints (discards, removed from play)
            hero_bet: hero's current cumulative bet
            opp_bet: opponent's current cumulative bet
            hero_first: True if hero acts first
            n_iterations: number of CFR iterations to run
            min_raise: minimum raise increment
            max_bet: maximum total bet per player
            checkpoint_callback: optional function(iteration, hero_strategy_sum, opp_strategy_sum)
                                 called every checkpoint_interval iterations
            checkpoint_interval: how often to call checkpoint_callback

        Returns:
            dict with:
                'hero_strategy': np.array (n_hero_buckets, n_hero_nodes, max_actions)
                    Average strategy probabilities for hero
                'opp_strategy': np.array (n_opp_buckets, n_opp_nodes, max_actions)
                    Average strategy probabilities for opponent
                'tree': GameTree instance
                'hand_matchups': precomputed matchup data
                'n_iterations': number of iterations completed
        """
        # Build game tree
        tree = GameTree(hero_bet, opp_bet, min_raise, max_bet, hero_first)

        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        if n_hero_nodes == 0 and n_opp_nodes == 0:
            return {
                'hero_strategy': np.array([]),
                'opp_strategy': np.array([]),
                'tree': tree,
                'hand_matchups': None,
                'n_iterations': 0,
            }

        # Index maps for decision nodes
        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        # Max actions at any node
        max_act = 1
        for nid in tree.hero_node_ids:
            max_act = max(max_act, tree.num_actions[nid])
        for nid in tree.opp_node_ids:
            max_act = max(max_act, tree.num_actions[nid])

        # Enumerate all possible hands for both players
        all_hands = list(itertools.combinations(
            [c for c in range(27) if c not in set(board) | set(dead_cards)], 2
        ))

        # Bucket each hand
        hand_equities = {}
        for hand in all_hands:
            eq = self.engine.compute_equity(list(hand), board, dead_cards)
            hand_equities[hand] = eq

        def equity_to_bucket(eq, n_buckets):
            b = int(eq * n_buckets)
            return min(b, n_buckets - 1)

        hero_hand_buckets = {h: equity_to_bucket(hand_equities[h], self.n_hero_buckets)
                             for h in all_hands}
        opp_hand_buckets = {h: equity_to_bucket(hand_equities[h], self.n_opp_buckets)
                            for h in all_hands}

        # Group hands by bucket
        hero_bucket_hands = [[] for _ in range(self.n_hero_buckets)]
        opp_bucket_hands = [[] for _ in range(self.n_opp_buckets)]
        for h in all_hands:
            hero_bucket_hands[hero_hand_buckets[h]].append(h)
            opp_bucket_hands[opp_hand_buckets[h]].append(h)

        # Precompute terminal payoffs for all (hero_hand, opp_hand) matchups.
        # We store equity for each matchup, then derive terminal values on the fly.
        # matchup_equity[(h, o)] = hero's equity (win probability) vs opp hand o
        matchup_equity = self._precompute_matchup_equities(
            all_hands, board, dead_cards)

        # Regret and strategy sum tables
        # Indexed by (bucket, node_index, action)
        hero_regrets = np.zeros(
            (self.n_hero_buckets, n_hero_nodes, max_act), dtype=np.float64)
        hero_strategy_sum = np.zeros(
            (self.n_hero_buckets, n_hero_nodes, max_act), dtype=np.float64)
        opp_regrets = np.zeros(
            (self.n_opp_buckets, n_opp_nodes, max_act), dtype=np.float64)
        opp_strategy_sum = np.zeros(
            (self.n_opp_buckets, n_opp_nodes, max_act), dtype=np.float64)

        # CFR+ iterations
        for t in range(n_iterations):
            # For each hero bucket × opp bucket combination with valid hands
            for hb in range(self.n_hero_buckets):
                if not hero_bucket_hands[hb]:
                    continue
                for ob in range(self.n_opp_buckets):
                    if not opp_bucket_hands[ob]:
                        continue

                    # Find non-overlapping hand pairs and their combined weight
                    # Weight = number of valid (hero_hand, opp_hand) combos
                    n_valid = 0
                    total_equity = 0.0

                    for hh in hero_bucket_hands[hb]:
                        for oh in opp_bucket_hands[ob]:
                            if set(hh) & set(oh):
                                continue  # hands overlap
                            n_valid += 1
                            key = (hh, oh) if hh < oh else (oh, hh)
                            if hh < oh:
                                total_equity += matchup_equity.get(key, 0.5)
                            else:
                                total_equity += 1.0 - matchup_equity.get(key, 0.5)

                    if n_valid == 0:
                        continue

                    avg_equity = total_equity / n_valid

                    # Traverse with average equity for this bucket pair
                    self._cfr_traverse_bucketed(
                        tree, 0,
                        hb, ob,
                        1.0, 1.0,  # reach probabilities
                        avg_equity,
                        hero_regrets, hero_strategy_sum,
                        opp_regrets, opp_strategy_sum,
                        hero_idx, opp_idx,
                        max_act, t,
                    )

            # Checkpoint
            if checkpoint_callback and (t + 1) % checkpoint_interval == 0:
                checkpoint_callback(t + 1, hero_strategy_sum, opp_strategy_sum)

        # Normalize strategy sums to get average strategies
        hero_strategy = self._normalize_strategy(hero_strategy_sum, tree, tree.hero_node_ids)
        opp_strategy = self._normalize_strategy(opp_strategy_sum, tree, tree.opp_node_ids)

        return {
            'hero_strategy': hero_strategy,
            'opp_strategy': opp_strategy,
            'tree': tree,
            'hand_matchups': matchup_equity,
            'n_iterations': n_iterations,
        }

    def _precompute_matchup_equities(self, all_hands, board, dead_cards):
        """
        Precompute hero equity for all non-overlapping hand matchups.

        For river boards (5 cards), this is exact (deterministic comparison).
        For flop/turn, we enumerate runouts.

        Returns:
            dict mapping (hand_a, hand_b) with a < b -> float equity of hand_a
        """
        matchups = {}
        known_base = set(board) | set(dead_cards)
        board_list = list(board)
        board_needed = 5 - len(board)

        seven_lookup = self.engine._seven

        if board_needed == 0:
            # River: direct rank comparison
            board_mask = 0
            for c in board:
                board_mask |= 1 << c

            # Precompute rank for every hand
            hand_ranks = {}
            for hand in all_hands:
                mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
                hand_ranks[hand] = seven_lookup[mask]

            for i in range(len(all_hands)):
                ha = all_hands[i]
                for j in range(i + 1, len(all_hands)):
                    hb = all_hands[j]
                    if set(ha) & set(hb):
                        continue
                    ra = hand_ranks[ha]
                    rb = hand_ranks[hb]
                    if ra < rb:
                        matchups[(ha, hb)] = 1.0
                    elif ra == rb:
                        matchups[(ha, hb)] = 0.5
                    else:
                        matchups[(ha, hb)] = 0.0
        else:
            # Flop/Turn: enumerate runouts
            board_mask = 0
            for c in board:
                board_mask |= 1 << c

            remaining_base = [c for c in range(27) if c not in known_base]

            for i in range(len(all_hands)):
                ha = all_hands[i]
                ha_set = set(ha)
                ha_mask = (1 << ha[0]) | (1 << ha[1])

                for j in range(i + 1, len(all_hands)):
                    hb = all_hands[j]
                    if ha_set & set(hb):
                        continue

                    hb_mask = (1 << hb[0]) | (1 << hb[1])

                    # Cards available for runout
                    runout_pool = [c for c in remaining_base
                                   if c != ha[0] and c != ha[1]
                                   and c != hb[0] and c != hb[1]]

                    wins_a = 0.0
                    total = 0

                    for runout in itertools.combinations(runout_pool, board_needed):
                        runout_mask = 0
                        for c in runout:
                            runout_mask |= 1 << c
                        full = board_mask | runout_mask

                        ra = seven_lookup[ha_mask | full]
                        rb = seven_lookup[hb_mask | full]

                        if ra < rb:
                            wins_a += 1.0
                        elif ra == rb:
                            wins_a += 0.5
                        total += 1

                    matchups[(ha, hb)] = wins_a / total if total > 0 else 0.5

        return matchups

    def _regret_match(self, regrets, n_actions):
        """Standard regret matching: positive regrets -> strategy."""
        pos = np.maximum(regrets[:n_actions], 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(n_actions) / n_actions

    def _cfr_traverse_bucketed(self, tree, node_id,
                                hero_bucket, opp_bucket,
                                hero_reach, opp_reach,
                                matchup_equity,
                                hero_regrets, hero_strategy_sum,
                                opp_regrets, opp_strategy_sum,
                                hero_idx, opp_idx,
                                max_act, iteration):
        """
        Recursive CFR+ traversal for a single (hero_bucket, opp_bucket) pair.

        Uses the average equity for this bucket pair at showdown terminals.

        Returns:
            float: hero's expected value at this node for this bucket matchup
        """
        # Terminal node
        if tree.terminal[node_id] != TERM_NONE:
            hero_pot = tree.hero_pot[node_id]
            opp_pot = tree.opp_pot[node_id]
            term = tree.terminal[node_id]

            if term == TERM_FOLD_HERO:
                return -float(hero_pot)
            elif term == TERM_FOLD_OPP:
                return float(opp_pot)
            elif term == TERM_SHOWDOWN:
                pot_won = min(hero_pot, opp_pot)
                # EV = (2 * equity - 1) * pot_won
                return (2.0 * matchup_equity - 1.0) * pot_won

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero decision node
            idx = hero_idx[node_id]
            strategy = self._regret_match(hero_regrets[hero_bucket, idx], n_act)

            action_values = np.zeros(n_act, dtype=np.float64)
            node_value = 0.0

            for a in range(n_act):
                _, child_id = children[a]
                action_values[a] = self._cfr_traverse_bucketed(
                    tree, child_id,
                    hero_bucket, opp_bucket,
                    hero_reach * strategy[a], opp_reach,
                    matchup_equity,
                    hero_regrets, hero_strategy_sum,
                    opp_regrets, opp_strategy_sum,
                    hero_idx, opp_idx,
                    max_act, iteration,
                )
                node_value += strategy[a] * action_values[a]

            # Update hero regrets (weighted by opponent reach)
            for a in range(n_act):
                regret = opp_reach * (action_values[a] - node_value)
                # CFR+: clamp regrets to non-negative
                hero_regrets[hero_bucket, idx, a] = max(
                    0.0, hero_regrets[hero_bucket, idx, a] + regret)

            # Accumulate average strategy (weighted by hero reach)
            hero_strategy_sum[hero_bucket, idx, :n_act] += hero_reach * strategy

            return node_value

        else:  # Opponent decision node
            idx = opp_idx[node_id]
            strategy = self._regret_match(opp_regrets[opp_bucket, idx], n_act)

            action_values = np.zeros(n_act, dtype=np.float64)
            node_value = 0.0

            for a in range(n_act):
                _, child_id = children[a]
                action_values[a] = self._cfr_traverse_bucketed(
                    tree, child_id,
                    hero_bucket, opp_bucket,
                    hero_reach, opp_reach * strategy[a],
                    matchup_equity,
                    hero_regrets, hero_strategy_sum,
                    opp_regrets, opp_strategy_sum,
                    hero_idx, opp_idx,
                    max_act, iteration,
                )
                node_value += strategy[a] * action_values[a]

            # Update opponent regrets
            # Opponent wants to minimize hero's value (zero-sum)
            for a in range(n_act):
                # Opp regret = opp_value(action) - opp_value(current)
                #            = -hero_value(action) - (-hero_value(current))
                #            = node_value - action_values[a]  (from hero's perspective inverted)
                regret = hero_reach * (node_value - action_values[a])
                opp_regrets[opp_bucket, idx, a] = max(
                    0.0, opp_regrets[opp_bucket, idx, a] + regret)

            # Accumulate opponent average strategy
            opp_strategy_sum[opp_bucket, idx, :n_act] += opp_reach * strategy

            return node_value

    def _normalize_strategy(self, strategy_sum, tree, node_ids):
        """
        Normalize strategy sums to probability distributions.

        Args:
            strategy_sum: array of shape (n_buckets, n_nodes, max_act)
            tree: GameTree
            node_ids: list of node IDs corresponding to the node dimension

        Returns:
            Normalized strategy array (same shape)
        """
        result = np.copy(strategy_sum)
        n_buckets = result.shape[0]

        for b in range(n_buckets):
            for i, nid in enumerate(node_ids):
                n_act = tree.num_actions[nid]
                total = result[b, i, :n_act].sum()
                if total > 0:
                    result[b, i, :n_act] /= total
                    result[b, i, n_act:] = 0.0
                else:
                    # Uniform default
                    result[b, i, :n_act] = 1.0 / n_act
                    result[b, i, n_act:] = 0.0

        return result


class BlueprintCFRFast:
    """
    Optimized full-range CFR+ that vectorizes across hand matchups.

    Instead of iterating over individual (hero_bucket, opp_bucket) pairs,
    this version processes all hands simultaneously using vectorized equity
    lookups at terminals.
    """

    def __init__(self, n_buckets, equity_engine):
        """
        Args:
            n_buckets: number of equity buckets (same for hero and opp)
            equity_engine: ExactEquityEngine instance
        """
        self.n_buckets = n_buckets
        self.engine = equity_engine

    def solve(self, board, dead_cards, hero_bet, opp_bet, hero_first,
              n_iterations, min_raise=2, max_bet=100,
              checkpoint_callback=None, checkpoint_interval=1000):
        """
        Run vectorized full-range CFR+.

        This version creates a matrix of terminal payoffs indexed by
        (hero_hand_index, opp_hand_index) and uses matrix operations
        for faster traversal.

        Returns same format as BlueprintCFR.solve().
        """
        # Build game tree
        tree = GameTree(hero_bet, opp_bet, min_raise, max_bet, hero_first)

        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        if n_hero_nodes == 0 and n_opp_nodes == 0:
            return {
                'hero_strategy': np.array([]),
                'opp_strategy': np.array([]),
                'tree': tree,
                'hand_matchups': None,
                'n_iterations': 0,
            }

        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = 1
        for nid in tree.hero_node_ids + tree.opp_node_ids:
            max_act = max(max_act, tree.num_actions[nid])

        # Enumerate hands and compute equities
        known = set(board) | set(dead_cards)
        all_hands = list(itertools.combinations(
            [c for c in range(27) if c not in known], 2))
        n_hands = len(all_hands)
        hand_to_idx = {h: i for i, h in enumerate(all_hands)}

        # Compute equity of each hand vs uniform range
        equities = np.zeros(n_hands, dtype=np.float64)
        for i, hand in enumerate(all_hands):
            equities[i] = self.engine.compute_equity(list(hand), board, dead_cards)

        # Assign buckets
        buckets = np.minimum((equities * self.n_buckets).astype(int),
                             self.n_buckets - 1)

        # Build overlap mask: valid_matchups[i,j] = True if hands don't overlap
        valid_matchups = np.ones((n_hands, n_hands), dtype=bool)
        for i in range(n_hands):
            hi = set(all_hands[i])
            for j in range(n_hands):
                if hi & set(all_hands[j]):
                    valid_matchups[i, j] = False

        # Precompute pairwise equities for showdown evaluation
        # matchup_eq[i,j] = hero equity when hero=hand_i vs opp=hand_j
        matchup_eq = self._compute_pairwise_equities(all_hands, board, dead_cards)

        # Precompute terminal payoff matrices
        # For each terminal node: payoff_matrix[i,j] = hero's payoff
        terminal_payoffs = {}
        for nid in tree.terminal_node_ids:
            term = tree.terminal[nid]
            hp = tree.hero_pot[nid]
            op = tree.opp_pot[nid]

            if term == TERM_FOLD_HERO:
                terminal_payoffs[nid] = np.full(
                    (n_hands, n_hands), -float(hp), dtype=np.float64)
            elif term == TERM_FOLD_OPP:
                terminal_payoffs[nid] = np.full(
                    (n_hands, n_hands), float(op), dtype=np.float64)
            elif term == TERM_SHOWDOWN:
                pot_won = min(hp, op)
                terminal_payoffs[nid] = (2.0 * matchup_eq - 1.0) * pot_won

            # Zero out invalid matchups
            terminal_payoffs[nid] *= valid_matchups

        # Regret and strategy tables indexed by bucket
        hero_regrets = np.zeros(
            (self.n_buckets, n_hero_nodes, max_act), dtype=np.float64)
        hero_strategy_sum = np.zeros(
            (self.n_buckets, n_hero_nodes, max_act), dtype=np.float64)
        opp_regrets = np.zeros(
            (self.n_buckets, n_opp_nodes, max_act), dtype=np.float64)
        opp_strategy_sum = np.zeros(
            (self.n_buckets, n_opp_nodes, max_act), dtype=np.float64)

        # Build bucket->hand_indices mapping
        bucket_hand_indices = [[] for _ in range(self.n_buckets)]
        for i in range(n_hands):
            bucket_hand_indices[buckets[i]].append(i)

        # CFR+ iterations
        for t in range(n_iterations):
            # Traverse with full hand matrices
            self._cfr_traverse_matrix(
                tree, 0,
                np.ones(n_hands, dtype=np.float64),  # hero reach per hand
                np.ones(n_hands, dtype=np.float64),  # opp reach per hand
                buckets, bucket_hand_indices,
                hero_regrets, hero_strategy_sum,
                opp_regrets, opp_strategy_sum,
                hero_idx, opp_idx,
                terminal_payoffs, valid_matchups,
                n_hands, max_act, t,
            )

            if checkpoint_callback and (t + 1) % checkpoint_interval == 0:
                checkpoint_callback(t + 1, hero_strategy_sum, opp_strategy_sum)

        # Normalize
        hero_strategy = self._normalize_strategy(
            hero_strategy_sum, tree, tree.hero_node_ids)
        opp_strategy = self._normalize_strategy(
            opp_strategy_sum, tree, tree.opp_node_ids)

        return {
            'hero_strategy': hero_strategy,
            'opp_strategy': opp_strategy,
            'tree': tree,
            'hand_matchups': matchup_eq,
            'n_iterations': n_iterations,
        }

    def _compute_pairwise_equities(self, all_hands, board, dead_cards):
        """
        Compute pairwise equity matrix.

        Returns:
            np.array of shape (n_hands, n_hands) where [i,j] = equity of hand i vs hand j
        """
        n = len(all_hands)
        eq = np.full((n, n), 0.5, dtype=np.float64)

        board_mask = 0
        for c in board:
            board_mask |= 1 << c

        board_needed = 5 - len(board)
        known = set(board) | set(dead_cards)
        remaining_base = [c for c in range(27) if c not in known]
        seven_lookup = self.engine._seven

        if board_needed == 0:
            # River: direct comparison
            hand_ranks = np.zeros(n, dtype=np.int64)
            for i, hand in enumerate(all_hands):
                mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
                hand_ranks[i] = seven_lookup[mask]

            for i in range(n):
                for j in range(i + 1, n):
                    if set(all_hands[i]) & set(all_hands[j]):
                        continue
                    if hand_ranks[i] < hand_ranks[j]:
                        eq[i, j] = 1.0
                        eq[j, i] = 0.0
                    elif hand_ranks[i] == hand_ranks[j]:
                        eq[i, j] = 0.5
                        eq[j, i] = 0.5
                    else:
                        eq[i, j] = 0.0
                        eq[j, i] = 1.0
        else:
            # Flop/Turn: enumerate runouts
            for i in range(n):
                ha = all_hands[i]
                ha_set = set(ha)
                ha_mask = (1 << ha[0]) | (1 << ha[1])

                for j in range(i + 1, n):
                    hb = all_hands[j]
                    if ha_set & set(hb):
                        continue
                    hb_mask = (1 << hb[0]) | (1 << hb[1])

                    runout_pool = [c for c in remaining_base
                                   if c not in ha_set and c != hb[0] and c != hb[1]]

                    wins_a = 0.0
                    total = 0
                    for runout in itertools.combinations(runout_pool, board_needed):
                        rm = 0
                        for c in runout:
                            rm |= 1 << c
                        full = board_mask | rm
                        ra = seven_lookup[ha_mask | full]
                        rb = seven_lookup[hb_mask | full]
                        if ra < rb:
                            wins_a += 1.0
                        elif ra == rb:
                            wins_a += 0.5
                        total += 1

                    e = wins_a / total if total > 0 else 0.5
                    eq[i, j] = e
                    eq[j, i] = 1.0 - e

        return eq

    def _get_bucket_strategy(self, regrets, bucket, node_idx, n_actions,
                             bucket_hand_indices):
        """Get the strategy for a bucket at a node (regret matching)."""
        pos = np.maximum(regrets[bucket, node_idx, :n_actions], 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(n_actions) / n_actions

    def _cfr_traverse_matrix(self, tree, node_id,
                              hero_reach, opp_reach,
                              buckets, bucket_hand_indices,
                              hero_regrets, hero_strategy_sum,
                              opp_regrets, opp_strategy_sum,
                              hero_idx, opp_idx,
                              terminal_payoffs, valid_matchups,
                              n_hands, max_act, iteration):
        """
        Matrix-based CFR traversal.

        hero_reach: array of shape (n_hands,) - hero's reach prob per hand
        opp_reach: array of shape (n_hands,) - opp's reach prob per hand

        Returns:
            np.array of shape (n_hands, n_hands) - hero's counterfactual values
            [i,j] = value when hero has hand i, opp has hand j
        """
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_payoffs[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero node
            idx = hero_idx[node_id]

            # Get strategy per hero hand (based on bucket)
            strategy_per_hand = np.zeros((n_hands, n_act), dtype=np.float64)
            for b in range(self.n_buckets):
                if not bucket_hand_indices[b]:
                    continue
                strat = self._get_bucket_strategy(
                    hero_regrets, b, idx, n_act, bucket_hand_indices)
                for hi in bucket_hand_indices[b]:
                    strategy_per_hand[hi] = strat

            # Compute action values: (n_act, n_hands, n_hands)
            action_values = []
            node_value = np.zeros((n_hands, n_hands), dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                # Update hero reach: hero_reach * strategy[a] for each hero hand
                new_hero_reach = hero_reach * strategy_per_hand[:, a]
                av = self._cfr_traverse_matrix(
                    tree, child_id,
                    new_hero_reach, opp_reach,
                    buckets, bucket_hand_indices,
                    hero_regrets, hero_strategy_sum,
                    opp_regrets, opp_strategy_sum,
                    hero_idx, opp_idx,
                    terminal_payoffs, valid_matchups,
                    n_hands, max_act, iteration,
                )
                action_values.append(av)
                # node_value[i,j] += strategy[i,a] * av[i,j]
                node_value += strategy_per_hand[:, a:a+1] * av

            # Update regrets per bucket
            for b in range(self.n_buckets):
                if not bucket_hand_indices[b]:
                    continue
                indices = bucket_hand_indices[b]

                for a in range(n_act):
                    # Counterfactual regret = sum over opp hands of
                    #   opp_reach[j] * (action_value[i,j] - node_value[i,j])
                    # averaged over hero hands in this bucket
                    total_regret = 0.0
                    for hi in indices:
                        diff = action_values[a][hi] - node_value[hi]
                        cf_regret = np.dot(diff, opp_reach)
                        total_regret += cf_regret

                    avg_regret = total_regret / len(indices)
                    hero_regrets[b, idx, a] = max(
                        0.0, hero_regrets[b, idx, a] + avg_regret)

                # Update strategy sum
                strat = self._get_bucket_strategy(
                    hero_regrets, b, idx, n_act, bucket_hand_indices)
                avg_reach = sum(hero_reach[hi] for hi in indices) / len(indices)
                hero_strategy_sum[b, idx, :n_act] += avg_reach * strat

            return node_value

        else:  # Opponent node
            idx = opp_idx[node_id]

            # Get strategy per opp hand (based on bucket)
            strategy_per_hand = np.zeros((n_hands, n_act), dtype=np.float64)
            for b in range(self.n_buckets):
                if not bucket_hand_indices[b]:
                    continue
                strat = self._get_bucket_strategy(
                    opp_regrets, b, idx, n_act, bucket_hand_indices)
                for oi in bucket_hand_indices[b]:
                    strategy_per_hand[oi] = strat

            action_values = []
            node_value = np.zeros((n_hands, n_hands), dtype=np.float64)

            for a in range(n_act):
                _, child_id = children[a]
                new_opp_reach = opp_reach * strategy_per_hand[:, a]
                av = self._cfr_traverse_matrix(
                    tree, child_id,
                    hero_reach, new_opp_reach,
                    buckets, bucket_hand_indices,
                    hero_regrets, hero_strategy_sum,
                    opp_regrets, opp_strategy_sum,
                    hero_idx, opp_idx,
                    terminal_payoffs, valid_matchups,
                    n_hands, max_act, iteration,
                )
                action_values.append(av)
                # node_value[i,j] += strategy[j,a] * av[i,j]
                node_value += strategy_per_hand[:, a] * av

            # Update opp regrets per bucket
            for b in range(self.n_buckets):
                if not bucket_hand_indices[b]:
                    continue
                indices = bucket_hand_indices[b]

                for a in range(n_act):
                    total_regret = 0.0
                    for oi in indices:
                        # Opp wants to minimize hero value
                        diff = node_value[:, oi] - action_values[a][:, oi]
                        cf_regret = np.dot(diff, hero_reach)
                        total_regret += cf_regret

                    avg_regret = total_regret / len(indices)
                    opp_regrets[b, idx, a] = max(
                        0.0, opp_regrets[b, idx, a] + avg_regret)

                strat = self._get_bucket_strategy(
                    opp_regrets, b, idx, n_act, bucket_hand_indices)
                avg_reach = sum(opp_reach[oi] for oi in indices) / len(indices)
                opp_strategy_sum[b, idx, :n_act] += avg_reach * strat

            return node_value

    def _normalize_strategy(self, strategy_sum, tree, node_ids):
        """Normalize strategy sums to probability distributions."""
        result = np.copy(strategy_sum)
        n_buckets = result.shape[0]

        for b in range(n_buckets):
            for i, nid in enumerate(node_ids):
                n_act = tree.num_actions[nid]
                total = result[b, i, :n_act].sum()
                if total > 0:
                    result[b, i, :n_act] /= total
                    result[b, i, n_act:] = 0.0
                else:
                    result[b, i, :n_act] = 1.0 / n_act
                    result[b, i, n_act:] = 0.0

        return result
