"""
Runtime lookup for precomputed blueprint strategies.

This module is imported by the runtime player to get action probabilities
without running CFR in real-time. The lookup flow:

1. Load the precomputed .npz file (once at startup)
2. For a given game state: compute board cluster, hand bucket
3. Look up the strategy and return action probabilities

The lookup itself is O(1) -- just array indexing. The only computation
is the initial equity calculation for hand bucketing (~5-10ms).

(Copied from blueprint/lookup.py for self-contained deployment.)
"""

import os
import sys
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)


class BlueprintLookup:
    """
    Fast runtime lookup for precomputed blueprint strategies.

    Usage:
        lookup = BlueprintLookup("blueprint_strategies.npz")
        probs = lookup.get_strategy(hero_cards=[0, 8], board=[1, 4, 7],
                                     pot_state=(1, 2), action_history=[])
        # probs is a dict: {action_type: probability}
    """

    # Map action type IDs to human-readable names
    ACTION_NAMES = {
        ACT_FOLD: 'fold',
        ACT_CHECK: 'check',
        ACT_CALL: 'call',
        ACT_RAISE_HALF: 'raise_40pct',
        ACT_RAISE_POT: 'raise_70pct',
        ACT_RAISE_ALLIN: 'raise_100pct',
        ACT_RAISE_OVERBET: 'raise_150pct',
    }

    def __init__(self, strategy_file, equity_engine=None):
        """
        Load precomputed blueprint strategies.

        Args:
            strategy_file: path to blueprint_strategies.npz
            equity_engine: optional ExactEquityEngine (created if not provided)
        """
        if not os.path.exists(strategy_file):
            raise FileNotFoundError(f"Blueprint file not found: {strategy_file}")

        data = np.load(strategy_file, allow_pickle=True)

        self.strategies = data['strategies']          # (n_solved, n_buckets, n_nodes, n_actions)
        self.cluster_ids = data['cluster_ids']        # (n_solved,)
        self.boards = data['boards']                  # (n_solved, 3)
        self.board_features = data['board_features']  # (n_solved, 12)
        self.action_types = data['action_types']      # (n_solved, n_nodes, n_actions)
        self.bucket_boundaries = data['bucket_boundaries']  # (n_buckets+1,)

        self.n_buckets = int(data['config_n_buckets'])
        self.n_clusters = int(data['config_n_clusters'])
        self.max_bet = int(data['config_max_bet'])

        # Build cluster_id -> index map
        self._cluster_to_idx = {}
        for i, cid in enumerate(self.cluster_ids):
            self._cluster_to_idx[int(cid)] = i

        # Equity engine for hand bucketing at runtime
        if equity_engine is not None:
            self.engine = equity_engine
        else:
            from equity import ExactEquityEngine
            self.engine = ExactEquityEngine()

        self._loaded = True

    @property
    def n_solved_clusters(self):
        return len(self.cluster_ids)

    def get_strategy(self, hero_cards, board, pot_state=None, action_history=None):
        """
        Look up the blueprint strategy for the current game state.

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints
            pot_state: tuple (hero_bet, opp_bet) or None for default
            action_history: list of action IDs taken so far, or None

        Returns:
            dict mapping action_type_id -> probability
            Returns None if no strategy is available for this state.
        """
        # 1. Compute board cluster
        cluster_id = self._find_nearest_cluster(board)
        if cluster_id is None:
            return None

        cluster_idx = self._cluster_to_idx[cluster_id]

        # 2. Compute hand bucket
        bucket = self._compute_bucket(hero_cards, board)

        # 3. Determine which node in the tree corresponds to the action history
        node_idx = self._action_history_to_node(action_history)

        # 4. Look up strategy
        strat = self.strategies[cluster_idx, bucket, node_idx, :]
        act_types = self.action_types[cluster_idx, node_idx, :]

        # Build result dict
        result = {}
        total = 0.0
        for a in range(len(strat)):
            if act_types[a] >= 0 and strat[a] > 0:
                result[int(act_types[a])] = float(strat[a])
                total += strat[a]

        if not result:
            return None

        # Normalize (should already be normalized, but safety)
        if abs(total - 1.0) > 1e-6:
            for k in result:
                result[k] /= total

        return result

    def get_action_probabilities(self, hero_cards, board, dead_cards=None):
        """
        Simplified interface: get action probabilities for the root node.

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints
            dead_cards: optional list of dead card ints (unused for lookup)

        Returns:
            dict mapping action_name -> probability, or None
        """
        raw = self.get_strategy(hero_cards, board)
        if raw is None:
            return None

        named = {}
        for act_id, prob in raw.items():
            name = self.ACTION_NAMES.get(act_id, f"action_{act_id}")
            named[name] = prob

        return named

    def _find_nearest_cluster(self, board):
        """
        Find the nearest solved cluster for a given board.

        Uses feature-space distance to the representative boards.
        """
        from blueprint_abstraction import compute_board_features, compute_board_cluster

        # First, try exact cluster match
        cid = compute_board_cluster(list(board), self.n_clusters)
        if cid in self._cluster_to_idx:
            return cid

        # If exact cluster wasn't solved, find nearest by feature distance
        features = compute_board_features(list(board))

        best_dist = float('inf')
        best_cid = None

        for i in range(len(self.cluster_ids)):
            dist = np.sum((self.board_features[i] - features) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_cid = int(self.cluster_ids[i])

        return best_cid

    def _compute_bucket(self, hero_cards, board):
        """
        Compute the equity bucket for hero's hand.

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints

        Returns:
            int bucket_id
        """
        equity = self.engine.compute_equity(list(hero_cards), list(board), [])
        bucket = int(equity * self.n_buckets)
        return min(bucket, self.n_buckets - 1)

    def _action_history_to_node(self, action_history):
        """
        Map an action history to a node index in the stored strategy.

        For the initial version, we only store root-node strategies (node 0).
        A full implementation would walk the game tree to find the correct node.

        Args:
            action_history: list of action IDs, or None

        Returns:
            int node index (0 for root)
        """
        if action_history is None or len(action_history) == 0:
            return 0

        # For now, return root node. Full tree navigation would be:
        # Walk the stored tree structure following the action sequence.
        # This is a TODO for the full implementation.
        return 0

    def sample_action(self, hero_cards, board, dead_cards=None):
        """
        Sample a single action from the blueprint strategy.

        Returns:
            int action_type_id, or None if no strategy available
        """
        raw = self.get_strategy(hero_cards, board)
        if raw is None:
            return None

        actions = list(raw.keys())
        probs = [raw[a] for a in actions]
        probs = np.array(probs)
        probs /= probs.sum()  # safety normalization

        return int(np.random.choice(actions, p=probs))

    def get_strategy_for_bucket(self, cluster_id, bucket_id, node_idx=0):
        """
        Direct access to strategy by cluster and bucket.
        Useful for analysis and debugging.

        Args:
            cluster_id: int cluster ID
            bucket_id: int bucket (0 = weakest, n_buckets-1 = strongest)
            node_idx: int node index in the tree

        Returns:
            dict mapping action_type_id -> probability, or None
        """
        if cluster_id not in self._cluster_to_idx:
            return None

        idx = self._cluster_to_idx[cluster_id]
        strat = self.strategies[idx, bucket_id, node_idx, :]
        act_types = self.action_types[idx, node_idx, :]

        result = {}
        for a in range(len(strat)):
            if act_types[a] >= 0 and strat[a] > 0:
                result[int(act_types[a])] = float(strat[a])

        if not result:
            return None

        total = sum(result.values())
        if total > 0:
            for k in result:
                result[k] /= total

        return result

    def describe_strategy(self, hero_cards, board, dead_cards=None):
        """
        Human-readable description of the blueprint strategy.
        Useful for debugging and analysis.
        """
        from blueprint_abstraction import compute_board_cluster, compute_board_features

        cluster_id = self._find_nearest_cluster(board)
        bucket = self._compute_bucket(hero_cards, board)
        equity = self.engine.compute_equity(list(hero_cards), list(board),
                                             list(dead_cards or []))

        lines = []
        lines.append(f"Hero cards: {hero_cards}")
        lines.append(f"Board: {board}")
        lines.append(f"Equity: {equity:.3f}")
        lines.append(f"Bucket: {bucket}/{self.n_buckets}")
        lines.append(f"Cluster: {cluster_id}")

        probs = self.get_action_probabilities(hero_cards, board)
        if probs:
            lines.append("Strategy:")
            for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {prob:.3f}")
        else:
            lines.append("No strategy available")

        return "\n".join(lines)
