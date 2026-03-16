"""
Runtime lookup for unbucketed precomputed blueprint strategies.

Like blueprint_lookup.py, but instead of grouping hands into equity buckets,
each individual hand (card pair) has its own strategy. This eliminates the
need for runtime equity computation -- just look up the hand directly.

Strategies are stored as uint8 (quantized probabilities 0-255) to save space.

The class also handles legacy bucketed .npz files transparently: if the loaded
file contains 'bucket_boundaries' but not 'hand_lists', it delegates to the
original bucketed lookup path.

Lookup flow (unbucketed):
1. Load the precomputed .npz file (once at startup)
2. For a given game state: find nearest board cluster (feature distance)
3. Look up hero's hand index directly in the cluster's hand_list (no equity!)
4. Look up the strategy and return action probabilities

The lookup itself is O(n_hands) for the hand scan, but n_hands is small (~276)
and it's a simple array comparison -- much faster than equity computation.
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


class BlueprintLookupUnbucketed:
    """
    Fast runtime lookup for unbucketed precomputed blueprint strategies.

    Each hand (card pair) has its own strategy entry instead of being grouped
    into equity buckets. Strategies are stored as uint8 and converted to
    float at lookup time.

    Also supports legacy bucketed files (auto-detected on load).

    Usage:
        lookup = BlueprintLookupUnbucketed("blueprint_strategies.npz")
        probs = lookup.get_strategy(hero_cards=[0, 8], board=[1, 4, 7],
                                     pot_state=(1, 2))
        # probs is a dict: {action_type: probability}
    """

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

        Auto-detects whether the file is unbucketed (has 'hand_lists') or
        bucketed (has 'bucket_boundaries'), and sets up accordingly.

        Args:
            strategy_file: path to .npz blueprint file
            equity_engine: optional ExactEquityEngine (only needed for
                           bucketed fallback; created lazily if required)
        """
        if not os.path.exists(strategy_file):
            raise FileNotFoundError(f"Blueprint file not found: {strategy_file}")

        data = np.load(strategy_file, allow_pickle=True)

        # Detect file format
        self._unbucketed = 'hand_lists' in data

        # Common fields
        self.strategies = data['strategies']
        self.cluster_ids = data['cluster_ids']
        self.boards = data['boards']
        self.board_features = data['board_features']
        self.action_types = data['action_types']

        self.n_clusters = int(data['config_n_clusters'])
        self.max_bet = int(data['config_max_bet'])

        # Multi-pot-size support
        if 'pot_sizes' in data:
            self.pot_sizes = data['pot_sizes']
            self._has_pot_sizes = True
        else:
            self.pot_sizes = None
            self._has_pot_sizes = False

        if self._unbucketed:
            # Unbucketed: hand_lists maps cluster_idx -> (n_hands, 2)
            self.hand_lists = data['hand_lists']
            self.n_buckets = None
            self.bucket_boundaries = None

            # Precompute hand lookup dicts: cluster_idx -> {(c1,c2): hand_idx}
            self._hand_maps = self._build_hand_maps()

            # Equity engine only needed for fallback (hand not in list)
            self._equity_engine = equity_engine
        else:
            # Bucketed fallback
            self.hand_lists = None
            self.bucket_boundaries = data['bucket_boundaries']
            self.n_buckets = int(data['config_n_buckets'])

            # Equity engine required for bucketed mode
            if equity_engine is not None:
                self._equity_engine = equity_engine
            else:
                from equity import ExactEquityEngine
                self._equity_engine = ExactEquityEngine()

        # Build cluster_id -> index map
        self._cluster_to_idx = {}
        for i, cid in enumerate(self.cluster_ids):
            self._cluster_to_idx[int(cid)] = i

        # Precompute hero node bet-state mapping
        self._node_maps = self._build_node_maps()

        self._loaded = True

    @property
    def n_solved_clusters(self):
        return len(self.cluster_ids)

    @property
    def is_unbucketed(self):
        return self._unbucketed

    def _build_hand_maps(self):
        """Build per-cluster hand lookup dicts for O(1) hand matching.

        Returns:
            dict: cluster_idx -> {(c1, c2): hand_idx}
                  where (c1, c2) is stored with c1 <= c2 (canonical order)
        """
        hand_maps = {}
        for ci in range(len(self.hand_lists)):
            hands = self.hand_lists[ci]
            hmap = {}
            for hi in range(len(hands)):
                c1, c2 = int(hands[hi][0]), int(hands[hi][1])
                # Store in canonical order (smaller first)
                key = (min(c1, c2), max(c1, c2))
                hmap[key] = hi
            hand_maps[ci] = hmap
        return hand_maps

    def get_strategy(self, hero_cards, board, pot_state=None, action_history=None,
                     dead_cards=None, opp_weights=None):
        """
        Look up the blueprint strategy for the current game state.

        For unbucketed blueprints: directly matches hero's cards to the
        cluster's hand list (no equity computation). Falls back to nearest
        equity match only if hero's cards overlap with the cluster's
        representative board.

        For bucketed blueprints: computes equity and maps to bucket (same
        as original BlueprintLookup).

        Args:
            hero_cards: list of 2 card ints
            board: list of 3-5 card ints
            pot_state: tuple (hero_bet, opp_bet) or None for default
            action_history: unused (kept for API compatibility)
            dead_cards: optional list of dead card ints
            opp_weights: optional opponent range weights

        Returns:
            dict mapping action_type_id -> probability
            Returns None if no strategy is available for this state.
        """
        # 1. Find nearest board cluster
        cluster_id = self._find_nearest_cluster(board)
        if cluster_id is None:
            return None

        cluster_idx = self._cluster_to_idx[cluster_id]

        # 2. Find hand index (unbucketed) or bucket (bucketed)
        if self._unbucketed:
            hand_idx = self._find_hand_index(hero_cards, cluster_idx, board,
                                             dead_cards, opp_weights)
            if hand_idx is None:
                return None
        else:
            hand_idx = self._compute_bucket(hero_cards, board, dead_cards,
                                            opp_weights)

        # 3. Find the right hero node for current bet state
        my_bet = pot_state[0] if pot_state else 0
        opp_bet = pot_state[1] if pot_state else 0

        if self._has_pot_sizes:
            pot_idx = self._find_nearest_pot(pot_state)
            node_idx = self._find_node_for_state(my_bet, opp_bet, pot_idx)
            raw_strat = self.strategies[cluster_idx, pot_idx, hand_idx, node_idx, :]
            act_types = self.action_types[cluster_idx, pot_idx, node_idx, :]
        else:
            node_idx = self._find_node_for_state(my_bet, opp_bet, 0)
            raw_strat = self.strategies[cluster_idx, hand_idx, node_idx, :]
            act_types = self.action_types[cluster_idx, node_idx, :]

        # 4. Convert uint8 -> float (unbucketed stores as uint8)
        if self._unbucketed and raw_strat.dtype == np.uint8:
            strat = raw_strat.astype(np.float64) / 255.0
        else:
            strat = raw_strat.astype(np.float64)

        # Build result dict
        result = {}
        total = 0.0
        for a in range(len(strat)):
            if act_types[a] >= 0 and strat[a] > 0:
                result[int(act_types[a])] = float(strat[a])
                total += strat[a]

        if not result:
            return None

        # Normalize
        if abs(total - 1.0) > 1e-6:
            for k in result:
                result[k] /= total

        return result

    def get_action_probabilities(self, hero_cards, board, dead_cards=None):
        """
        Simplified interface: get action probabilities for the root node.

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

    # ------------------------------------------------------------------
    #  Hand lookup (unbucketed)
    # ------------------------------------------------------------------

    def _find_hand_index(self, hero_cards, cluster_idx, board,
                         dead_cards=None, opp_weights=None):
        """
        Find the strategy index for hero's hand in an unbucketed cluster.

        Fast path: exact card-pair match in the cluster's hand list (O(1)
        dict lookup, no equity computation).

        Slow path (rare): hero's cards overlap with the cluster's
        representative board, so the exact pair won't be in the hand list.
        Falls back to finding the hand with nearest equity.

        Args:
            hero_cards: list of 2 card ints
            cluster_idx: int, index into strategies array
            board: list of board card ints (for equity fallback)
            dead_cards: optional list of dead card ints
            opp_weights: optional opponent range weights

        Returns:
            int hand index, or None if lookup fails
        """
        c1, c2 = int(hero_cards[0]), int(hero_cards[1])
        key = (min(c1, c2), max(c1, c2))

        hmap = self._hand_maps.get(cluster_idx)
        if hmap is None:
            return None

        # Fast path: exact match
        if key in hmap:
            return hmap[key]

        # Slow path: hero cards overlap with representative board.
        # Find nearest hand by equity similarity.
        return self._find_nearest_hand_by_equity(
            hero_cards, cluster_idx, board, dead_cards, opp_weights
        )

    def _find_nearest_hand_by_equity(self, hero_cards, cluster_idx, board,
                                     dead_cards=None, opp_weights=None):
        """
        Fallback: find the hand in the cluster with most similar equity.

        This is only called when hero's cards aren't in the cluster's hand
        list (because the representative board used different cards). It
        computes equity once for hero's hand and compares against cached
        equities for all hands in the list.

        Returns:
            int hand index, or None
        """
        # Lazy-load equity engine
        if self._equity_engine is None:
            from equity import ExactEquityEngine
            self._equity_engine = ExactEquityEngine()

        hero_equity = self._equity_engine.compute_equity(
            list(hero_cards), list(board),
            list(dead_cards or []), opp_weights
        )

        hands = self.hand_lists[cluster_idx]
        n_hands = len(hands)
        if n_hands == 0:
            return None

        # Compute equity for each hand in the list and find closest
        best_idx = 0
        best_dist = float('inf')

        for hi in range(n_hands):
            hc = [int(hands[hi][0]), int(hands[hi][1])]
            # Skip hands that conflict with the actual board
            if hc[0] in board or hc[1] in board:
                continue
            if dead_cards and (hc[0] in dead_cards or hc[1] in dead_cards):
                continue
            # Skip hands that conflict with hero's cards
            if hc[0] in hero_cards or hc[1] in hero_cards:
                continue

            eq = self._equity_engine.compute_equity(
                hc, list(board), list(dead_cards or []), opp_weights
            )
            dist = abs(eq - hero_equity)
            if dist < best_dist:
                best_dist = dist
                best_idx = hi

        return best_idx

    # ------------------------------------------------------------------
    #  Board cluster lookup (same as bucketed version)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    #  Node and pot matching (same as bucketed version)
    # ------------------------------------------------------------------

    def _build_node_maps(self):
        """Rebuild game trees and map hero nodes to bet states."""
        from game_tree import GameTree, ACT_FOLD

        maps = {}

        if self._has_pot_sizes:
            for pi, ps in enumerate(self.pot_sizes):
                hb, ob = int(ps[0]), int(ps[1])
                try:
                    tree = GameTree(hb, ob, 2, self.max_bet, True)
                    node_info = []
                    for i, nid in enumerate(tree.hero_node_ids):
                        acts = [a for a, _ in tree.children[nid]]
                        node_info.append((
                            i,
                            tree.hero_pot[nid],
                            tree.opp_pot[nid],
                            ACT_FOLD in acts,
                        ))
                    maps[pi] = node_info
                except Exception:
                    maps[pi] = [(0, hb, ob, False)]
        else:
            root_acts = self.action_types[0, 0, :]
            has_fold_root = 0 in root_acts[root_acts >= 0]
            if has_fold_root:
                hb, ob = 1, 2
            else:
                hb, ob = 2, 2
            try:
                tree = GameTree(hb, ob, 2, self.max_bet, True)
                node_info = []
                for i, nid in enumerate(tree.hero_node_ids):
                    acts = [a for a, _ in tree.children[nid]]
                    node_info.append((
                        i,
                        tree.hero_pot[nid],
                        tree.opp_pot[nid],
                        ACT_FOLD in acts,
                    ))
                maps[0] = node_info
            except Exception:
                maps[0] = [(0, hb, ob, False)]

        return maps

    def _find_node_for_state(self, my_bet, opp_bet, pot_idx=0):
        """Find the hero node index matching the current bet state."""
        node_info = self._node_maps.get(pot_idx, [(0, 0, 0, False)])
        if not node_info:
            return 0

        facing_bet = opp_bet > my_bet
        pot = my_bet + opp_bet
        if pot <= 0:
            return 0

        bet_ratio = (opp_bet - my_bet) / pot if facing_bet else 0.0

        best_idx = 0
        best_dist = float('inf')

        for strategy_idx, hp, op, has_fold in node_info:
            node_pot = hp + op
            if node_pot <= 0:
                continue

            node_facing = op > hp

            if facing_bet and not node_facing:
                continue
            if not facing_bet and node_facing:
                continue

            node_ratio = (op - hp) / node_pot if node_facing else 0.0
            dist = abs(bet_ratio - node_ratio)

            if dist < best_dist:
                best_dist = dist
                best_idx = strategy_idx

        return best_idx

    def _find_nearest_pot(self, pot_state):
        """Find the nearest pot size index for multi-pot blueprints."""
        if self.pot_sizes is None or pot_state is None:
            return 0

        hero_bet, opp_bet = pot_state if pot_state else (2, 2)
        pot = hero_bet + opp_bet

        best_idx = 0
        best_dist = float('inf')
        for i, ps in enumerate(self.pot_sizes):
            dist = abs(ps[0] + ps[1] - pot)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    #  Bucketed fallback (same as original BlueprintLookup)
    # ------------------------------------------------------------------

    def _compute_bucket(self, hero_cards, board, dead_cards=None,
                        opp_weights=None):
        """
        Compute the equity bucket for hero's hand (bucketed mode only).
        """
        equity = self._equity_engine.compute_equity(
            list(hero_cards), list(board), list(dead_cards or []),
            opp_weights)
        bucket = int(equity * self.n_buckets)
        return min(bucket, self.n_buckets - 1)

    # ------------------------------------------------------------------
    #  Convenience methods
    # ------------------------------------------------------------------

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
        probs /= probs.sum()

        return int(np.random.choice(actions, p=probs))

    def get_strategy_for_hand(self, cluster_id, hand_idx, node_idx=0,
                              pot_idx=0):
        """
        Direct access to strategy by cluster and hand index.
        Useful for analysis and debugging.

        Args:
            cluster_id: int cluster ID
            hand_idx: int hand index (or bucket index for bucketed mode)
            node_idx: int node index in the tree
            pot_idx: int pot size index (0 if no multi-pot)

        Returns:
            dict mapping action_type_id -> probability, or None
        """
        if cluster_id not in self._cluster_to_idx:
            return None

        idx = self._cluster_to_idx[cluster_id]

        if self._has_pot_sizes:
            raw_strat = self.strategies[idx, pot_idx, hand_idx, node_idx, :]
            act_types = self.action_types[idx, pot_idx, node_idx, :]
        else:
            raw_strat = self.strategies[idx, hand_idx, node_idx, :]
            act_types = self.action_types[idx, node_idx, :]

        if self._unbucketed and raw_strat.dtype == np.uint8:
            strat = raw_strat.astype(np.float64) / 255.0
        else:
            strat = raw_strat.astype(np.float64)

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
        from blueprint_abstraction import compute_board_cluster

        cluster_id = self._find_nearest_cluster(board)

        lines = []
        lines.append(f"Hero cards: {hero_cards}")
        lines.append(f"Board: {board}")
        lines.append(f"Cluster: {cluster_id}")
        lines.append(f"Mode: {'unbucketed' if self._unbucketed else 'bucketed'}")

        if self._unbucketed:
            cluster_idx = self._cluster_to_idx.get(cluster_id)
            if cluster_idx is not None:
                c1, c2 = int(hero_cards[0]), int(hero_cards[1])
                key = (min(c1, c2), max(c1, c2))
                hmap = self._hand_maps.get(cluster_idx, {})
                if key in hmap:
                    lines.append(f"Hand index: {hmap[key]} (exact match)")
                else:
                    lines.append("Hand index: fallback (equity-nearest)")
        else:
            equity = self._equity_engine.compute_equity(
                list(hero_cards), list(board), list(dead_cards or []))
            bucket = self._compute_bucket(hero_cards, board)
            lines.append(f"Equity: {equity:.3f}")
            lines.append(f"Bucket: {bucket}/{self.n_buckets}")

        probs = self.get_action_probabilities(hero_cards, board)
        if probs:
            lines.append("Strategy:")
            for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {prob:.3f}")
        else:
            lines.append("No strategy available")

        return "\n".join(lines)
