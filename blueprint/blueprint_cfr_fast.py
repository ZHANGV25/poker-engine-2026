"""
Numba JIT-accelerated CFR engine for blueprint strategy computation.

Converts the game tree into flat numpy arrays and runs the entire CFR
iteration loop inside @njit-compiled functions, eliminating Python overhead.
Provides 10-50x speedup over the pure-Python BlueprintCFR implementation.

The key insight is that the tree traversal is converted from recursive
Python method calls to an iterative post-order traversal using an explicit
stack, with all data stored in contiguous numpy arrays.
"""

import os
import sys
import time
import itertools

import numpy as np
import numba

_submission_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission")
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

# ---------------------------------------------------------------------------
# Constants for Numba (must be module-level for njit to see them)
# ---------------------------------------------------------------------------
_TERM_NONE = np.int8(TERM_NONE)
_TERM_FOLD_HERO = np.int8(TERM_FOLD_HERO)
_TERM_FOLD_OPP = np.int8(TERM_FOLD_OPP)
_TERM_SHOWDOWN = np.int8(TERM_SHOWDOWN)


# ---------------------------------------------------------------------------
# Numba-compiled core routines
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _regret_match(regrets, n_actions):
    """Compute strategy from regrets via regret matching (CFR+)."""
    strategy = np.empty(n_actions, dtype=np.float64)
    total = 0.0
    for a in range(n_actions):
        v = regrets[a]
        if v > 0.0:
            strategy[a] = v
            total += v
        else:
            strategy[a] = 0.0
    if total > 0.0:
        inv = 1.0 / total
        for a in range(n_actions):
            strategy[a] *= inv
    else:
        uniform = 1.0 / n_actions
        for a in range(n_actions):
            strategy[a] = uniform
    return strategy


@numba.njit(cache=True)
def _traverse_cfr(node_player, node_terminal, node_hero_pot, node_opp_pot,
                  node_num_actions, children, hero_node_map, opp_node_map,
                  equity,
                  hero_regrets_hb, hero_strategy_sum_hb,
                  opp_regrets_ob, opp_strategy_sum_ob,
                  max_actions, hero_reach_init, opp_reach_init,
                  n_nodes):
    """
    Iterative post-order CFR+ traversal for one (hero_bucket, opp_bucket) pair.

    Uses an explicit stack to avoid Python recursion limitations in Numba.
    Each stack frame stores (node_id, action_index, phase) where:
      phase 0 = first visit (compute strategy, push first child)
      phase 1 = returning from child action a (accumulate value, push next child or finalize)

    Args:
        hero_regrets_hb: 2D array (n_hero_nodes, max_actions) for this hero bucket
        hero_strategy_sum_hb: 2D array (n_hero_nodes, max_actions) for this hero bucket
        opp_regrets_ob: 2D array (n_opp_nodes, max_actions) for this opp bucket
        opp_strategy_sum_ob: 2D array (n_opp_nodes, max_actions) for this opp bucket

    Returns:
        float: hero's expected value at the root
    """
    # Stack arrays - pre-allocate for max depth
    # Each entry: (node_id, current_action, hero_reach, opp_reach)
    max_depth = n_nodes  # upper bound
    stack_node = np.empty(max_depth, dtype=np.int32)
    stack_action = np.empty(max_depth, dtype=np.int32)
    stack_hero_reach = np.empty(max_depth, dtype=np.float64)
    stack_opp_reach = np.empty(max_depth, dtype=np.float64)
    # Per-node strategy and action values stored on stack
    stack_strategy = np.zeros((max_depth, max_actions), dtype=np.float64)
    stack_action_values = np.zeros((max_depth, max_actions), dtype=np.float64)
    stack_node_value = np.zeros(max_depth, dtype=np.float64)

    # Initialize with root
    sp = 0  # stack pointer
    stack_node[0] = 0
    stack_action[0] = -1  # -1 means "first visit"
    stack_hero_reach[0] = hero_reach_init
    stack_opp_reach[0] = opp_reach_init

    root_value = 0.0

    while sp >= 0:
        node_id = stack_node[sp]
        cur_action = stack_action[sp]
        hero_reach = stack_hero_reach[sp]
        opp_reach = stack_opp_reach[sp]

        # Terminal node check
        term = node_terminal[node_id]
        if term != _TERM_NONE:
            hero_pot = node_hero_pot[node_id]
            opp_pot = node_opp_pot[node_id]
            if term == _TERM_FOLD_HERO:
                val = -hero_pot
            elif term == _TERM_FOLD_OPP:
                val = opp_pot
            else:  # TERM_SHOWDOWN
                pot_won = hero_pot if hero_pot < opp_pot else opp_pot
                val = (2.0 * equity - 1.0) * pot_won

            # Return value to parent
            sp -= 1
            if sp >= 0:
                parent_action = stack_action[sp]
                stack_action_values[sp, parent_action] = val
            else:
                root_value = val
            continue

        n_act = node_num_actions[node_id]
        player = node_player[node_id]

        if cur_action == -1:
            # First visit: compute strategy
            if player == 0:
                idx = hero_node_map[node_id]
                strategy = _regret_match(hero_regrets_hb[idx], n_act)
            else:
                idx = opp_node_map[node_id]
                strategy = _regret_match(opp_regrets_ob[idx], n_act)

            for a in range(n_act):
                stack_strategy[sp, a] = strategy[a]
            stack_node_value[sp] = 0.0

            # Start with action 0
            stack_action[sp] = 0
            child_id = children[node_id, 0]

            if player == 0:
                new_hero_reach = hero_reach * strategy[0]
                new_opp_reach = opp_reach
            else:
                new_hero_reach = hero_reach
                new_opp_reach = opp_reach * strategy[0]

            # Push child
            sp += 1
            stack_node[sp] = child_id
            stack_action[sp] = -1
            stack_hero_reach[sp] = new_hero_reach
            stack_opp_reach[sp] = new_opp_reach

        else:
            # Returning from child action cur_action
            # The child's value is already in stack_action_values[sp, cur_action]
            child_val = stack_action_values[sp, cur_action]
            stack_node_value[sp] += stack_strategy[sp, cur_action] * child_val

            next_action = cur_action + 1
            if next_action < n_act:
                # Push next child
                stack_action[sp] = next_action
                child_id = children[node_id, next_action]

                if player == 0:
                    new_hero_reach = hero_reach * stack_strategy[sp, next_action]
                    new_opp_reach = opp_reach
                else:
                    new_hero_reach = hero_reach
                    new_opp_reach = opp_reach * stack_strategy[sp, next_action]

                sp += 1
                stack_node[sp] = child_id
                stack_action[sp] = -1
                stack_hero_reach[sp] = new_hero_reach
                stack_opp_reach[sp] = new_opp_reach
            else:
                # All children processed - update regrets and strategy sum
                node_val = stack_node_value[sp]

                if player == 0:
                    idx = hero_node_map[node_id]
                    for a in range(n_act):
                        regret = opp_reach * (stack_action_values[sp, a] - node_val)
                        new_regret = hero_regrets_hb[idx, a] + regret
                        if new_regret > 0.0:
                            hero_regrets_hb[idx, a] = new_regret
                        else:
                            hero_regrets_hb[idx, a] = 0.0
                    for a in range(n_act):
                        hero_strategy_sum_hb[idx, a] += hero_reach * stack_strategy[sp, a]
                else:
                    idx = opp_node_map[node_id]
                    for a in range(n_act):
                        regret = hero_reach * (node_val - stack_action_values[sp, a])
                        new_regret = opp_regrets_ob[idx, a] + regret
                        if new_regret > 0.0:
                            opp_regrets_ob[idx, a] = new_regret
                        else:
                            opp_regrets_ob[idx, a] = 0.0
                    for a in range(n_act):
                        opp_strategy_sum_ob[idx, a] += opp_reach * stack_strategy[sp, a]

                # Return value to parent
                sp -= 1
                if sp >= 0:
                    parent_action = stack_action[sp]
                    stack_action_values[sp, parent_action] = node_val
                else:
                    root_value = node_val

    return root_value


@numba.njit(cache=True)
def _cfr_iterations(node_player, node_terminal, node_hero_pot, node_opp_pot,
                    node_num_actions, children, hero_node_map, opp_node_map,
                    bucket_equity_matrix, bucket_valid,
                    n_hero_buckets, n_opp_buckets,
                    n_iterations, n_hero_nodes, n_opp_nodes,
                    max_actions, n_nodes):
    """
    Run all CFR+ iterations entirely in compiled code.

    Args:
        node_player: int8 array (n_nodes,) - 0=hero, 1=opp
        node_terminal: int8 array (n_nodes,) - terminal type
        node_hero_pot: float64 array (n_nodes,)
        node_opp_pot: float64 array (n_nodes,)
        node_num_actions: int32 array (n_nodes,)
        children: int32 array (n_nodes, max_actions) - child node IDs, -1=unused
        hero_node_map: int32 array (n_nodes,) - node_id -> hero_node_index, -1 if not hero
        opp_node_map: int32 array (n_nodes,) - node_id -> opp_node_index, -1 if not opp
        bucket_equity_matrix: float64 (n_hero_buckets, n_opp_buckets) - avg equity
        bucket_valid: bool (n_hero_buckets, n_opp_buckets) - whether pair has valid matchups
        n_hero_buckets, n_opp_buckets: bucket counts
        n_iterations: number of CFR iterations
        n_hero_nodes, n_opp_nodes: decision node counts
        max_actions: max actions at any node
        n_nodes: total nodes in tree

    Returns:
        (hero_strategy_sum, opp_strategy_sum) - both float64 3D arrays
    """
    hero_regrets = np.zeros((n_hero_buckets, n_hero_nodes, max_actions), dtype=np.float64)
    hero_strategy_sum = np.zeros((n_hero_buckets, n_hero_nodes, max_actions), dtype=np.float64)
    opp_regrets = np.zeros((n_opp_buckets, n_opp_nodes, max_actions), dtype=np.float64)
    opp_strategy_sum = np.zeros((n_opp_buckets, n_opp_nodes, max_actions), dtype=np.float64)

    for t in range(n_iterations):
        for hb in range(n_hero_buckets):
            for ob in range(n_opp_buckets):
                if not bucket_valid[hb, ob]:
                    continue

                eq = bucket_equity_matrix[hb, ob]

                _traverse_cfr(
                    node_player, node_terminal, node_hero_pot, node_opp_pot,
                    node_num_actions, children, hero_node_map, opp_node_map,
                    eq,
                    hero_regrets[hb], hero_strategy_sum[hb],
                    opp_regrets[ob], opp_strategy_sum[ob],
                    max_actions, 1.0, 1.0,
                    n_nodes,
                )

    return hero_strategy_sum, opp_strategy_sum


@numba.njit(cache=True)
def _normalize_strategy_numba(strategy_sum, node_num_actions_list, n_buckets):
    """Normalize strategy sums to probability distributions."""
    result = strategy_sum.copy()
    n_nodes = result.shape[1]
    for b in range(n_buckets):
        for i in range(n_nodes):
            n_act = node_num_actions_list[i]
            total = 0.0
            for a in range(n_act):
                total += result[b, i, a]
            if total > 0.0:
                inv = 1.0 / total
                for a in range(n_act):
                    result[b, i, a] *= inv
                for a in range(n_act, result.shape[2]):
                    result[b, i, a] = 0.0
            else:
                uniform = 1.0 / n_act
                for a in range(n_act):
                    result[b, i, a] = uniform
                for a in range(n_act, result.shape[2]):
                    result[b, i, a] = 0.0
    return result


# ---------------------------------------------------------------------------
# Tree flattening: convert GameTree to numpy arrays for Numba
# ---------------------------------------------------------------------------

def _flatten_tree(tree):
    """
    Convert a GameTree (Python lists/dicts) into flat numpy arrays
    suitable for Numba @njit functions.

    Returns:
        dict with all numpy arrays needed by _cfr_iterations
    """
    n_nodes = tree.size
    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    # Compute max actions across all decision nodes
    max_actions = 1
    for nid in tree.hero_node_ids:
        if tree.num_actions[nid] > max_actions:
            max_actions = tree.num_actions[nid]
    for nid in tree.opp_node_ids:
        if tree.num_actions[nid] > max_actions:
            max_actions = tree.num_actions[nid]

    # Flat arrays
    node_player = np.array(tree.player, dtype=np.int8)
    node_terminal = np.array(tree.terminal, dtype=np.int8)
    node_hero_pot = np.array(tree.hero_pot, dtype=np.float64)
    node_opp_pot = np.array(tree.opp_pot, dtype=np.float64)
    node_num_actions = np.array(tree.num_actions, dtype=np.int32)

    # Children as 2D array: children[node_id, action_idx] = child_node_id
    children = np.full((n_nodes, max_actions), -1, dtype=np.int32)
    for nid in range(n_nodes):
        for a, (act_id, child_id) in enumerate(tree.children[nid]):
            children[nid, a] = child_id

    # Index maps: node_id -> decision node index (-1 if not a decision node of that type)
    hero_node_map = np.full(n_nodes, -1, dtype=np.int32)
    opp_node_map = np.full(n_nodes, -1, dtype=np.int32)
    for i, nid in enumerate(tree.hero_node_ids):
        hero_node_map[nid] = i
    for i, nid in enumerate(tree.opp_node_ids):
        opp_node_map[nid] = i

    # num_actions for hero and opp decision nodes (for normalization)
    hero_node_num_actions = np.zeros(n_hero_nodes, dtype=np.int32)
    opp_node_num_actions = np.zeros(n_opp_nodes, dtype=np.int32)
    for i, nid in enumerate(tree.hero_node_ids):
        hero_node_num_actions[i] = tree.num_actions[nid]
    for i, nid in enumerate(tree.opp_node_ids):
        opp_node_num_actions[i] = tree.num_actions[nid]

    return {
        'n_nodes': n_nodes,
        'n_hero_nodes': n_hero_nodes,
        'n_opp_nodes': n_opp_nodes,
        'max_actions': max_actions,
        'node_player': node_player,
        'node_terminal': node_terminal,
        'node_hero_pot': node_hero_pot,
        'node_opp_pot': node_opp_pot,
        'node_num_actions': node_num_actions,
        'children': children,
        'hero_node_map': hero_node_map,
        'opp_node_map': opp_node_map,
        'hero_node_num_actions': hero_node_num_actions,
        'opp_node_num_actions': opp_node_num_actions,
    }


# ---------------------------------------------------------------------------
# Public class with same interface as BlueprintCFR
# ---------------------------------------------------------------------------

class BlueprintCFRFast:
    """
    Numba JIT-accelerated full-range CFR+ solver for blueprint strategies.

    Same interface as BlueprintCFR, but the inner CFR iteration loop runs
    entirely in compiled machine code via Numba @njit.

    Typical speedups: 10-50x over the pure-Python BlueprintCFR.
    First call includes ~2-3 second JIT compilation overhead.
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
        Run full-range CFR+ using Numba JIT and return the converged strategy.

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
                                 called every checkpoint_interval iterations.
                                 NOTE: checkpointing requires running iterations in chunks,
                                 which adds some overhead vs running all iterations at once.
            checkpoint_interval: how often to call checkpoint_callback

        Returns:
            dict with:
                'hero_strategy': np.array (n_hero_buckets, n_hero_nodes, max_actions)
                    Average strategy probabilities for hero
                'opp_strategy': np.array (n_opp_buckets, n_opp_nodes, max_actions)
                    Average strategy probabilities for opponent
                'tree': GameTree instance
                'hand_matchups': None (not stored in fast version)
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

        # Flatten tree to numpy arrays
        flat = _flatten_tree(tree)

        # Enumerate all possible hands
        all_hands = list(itertools.combinations(
            [c for c in range(27) if c not in set(board) | set(dead_cards)], 2
        ))

        # Compute equity of each hand vs uniform range and assign buckets
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

        # Precompute pairwise matchup equities
        matchup_equity = self._precompute_matchup_equities(all_hands, board, dead_cards)

        # Build bucket_equity_matrix: (n_hero_buckets, n_opp_buckets)
        bucket_equity_matrix = np.zeros(
            (self.n_hero_buckets, self.n_opp_buckets), dtype=np.float64)
        bucket_valid = np.zeros(
            (self.n_hero_buckets, self.n_opp_buckets), dtype=np.bool_)

        for hb in range(self.n_hero_buckets):
            if not hero_bucket_hands[hb]:
                continue
            for ob in range(self.n_opp_buckets):
                if not opp_bucket_hands[ob]:
                    continue

                n_valid = 0
                total_equity = 0.0
                for hh in hero_bucket_hands[hb]:
                    for oh in opp_bucket_hands[ob]:
                        if set(hh) & set(oh):
                            continue
                        n_valid += 1
                        key = (hh, oh) if hh < oh else (oh, hh)
                        if hh < oh:
                            total_equity += matchup_equity.get(key, 0.5)
                        else:
                            total_equity += 1.0 - matchup_equity.get(key, 0.5)

                if n_valid > 0:
                    bucket_equity_matrix[hb, ob] = total_equity / n_valid
                    bucket_valid[hb, ob] = True

        # Run CFR iterations
        if checkpoint_callback is None:
            # Run all iterations in a single JIT call (fastest)
            hero_strategy_sum, opp_strategy_sum = _cfr_iterations(
                flat['node_player'], flat['node_terminal'],
                flat['node_hero_pot'], flat['node_opp_pot'],
                flat['node_num_actions'], flat['children'],
                flat['hero_node_map'], flat['opp_node_map'],
                bucket_equity_matrix, bucket_valid,
                self.n_hero_buckets, self.n_opp_buckets,
                n_iterations, n_hero_nodes, n_opp_nodes,
                flat['max_actions'], flat['n_nodes'],
            )
        else:
            # Run in chunks with checkpoint callbacks
            hero_strategy_sum = np.zeros(
                (self.n_hero_buckets, n_hero_nodes, flat['max_actions']),
                dtype=np.float64)
            opp_strategy_sum = np.zeros(
                (self.n_opp_buckets, n_opp_nodes, flat['max_actions']),
                dtype=np.float64)

            iters_done = 0
            while iters_done < n_iterations:
                chunk = min(checkpoint_interval, n_iterations - iters_done)

                h_sum, o_sum = _cfr_iterations(
                    flat['node_player'], flat['node_terminal'],
                    flat['node_hero_pot'], flat['node_opp_pot'],
                    flat['node_num_actions'], flat['children'],
                    flat['hero_node_map'], flat['opp_node_map'],
                    bucket_equity_matrix, bucket_valid,
                    self.n_hero_buckets, self.n_opp_buckets,
                    chunk, n_hero_nodes, n_opp_nodes,
                    flat['max_actions'], flat['n_nodes'],
                )

                hero_strategy_sum += h_sum
                opp_strategy_sum += o_sum
                iters_done += chunk

                if iters_done % checkpoint_interval == 0 or iters_done == n_iterations:
                    checkpoint_callback(iters_done, hero_strategy_sum, opp_strategy_sum)

        # Normalize strategy sums
        hero_strategy = _normalize_strategy_numba(
            hero_strategy_sum, flat['hero_node_num_actions'], self.n_hero_buckets)
        opp_strategy = _normalize_strategy_numba(
            opp_strategy_sum, flat['opp_node_num_actions'], self.n_opp_buckets)

        return {
            'hero_strategy': hero_strategy,
            'opp_strategy': opp_strategy,
            'tree': tree,
            'hand_matchups': None,
            'n_iterations': n_iterations,
        }

    def _precompute_matchup_equities(self, all_hands, board, dead_cards):
        """
        Precompute hero equity for all non-overlapping hand matchups.
        Same logic as BlueprintCFR._precompute_matchup_equities.
        """
        matchups = {}
        known_base = set(board) | set(dead_cards)
        board_needed = 5 - len(board)
        seven_lookup = self.engine._seven

        board_mask = 0
        for c in board:
            board_mask |= 1 << c

        if board_needed == 0:
            # River: direct rank comparison
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


# ---------------------------------------------------------------------------
# Warmup: trigger JIT compilation without running a real solve
# ---------------------------------------------------------------------------

def warmup_jit():
    """
    Trigger Numba JIT compilation of all @njit functions.
    Call this once at startup to avoid compilation delay on the first solve.
    """
    # Minimal arrays to trigger compilation
    node_player = np.array([0, -1, -1], dtype=np.int8)
    node_terminal = np.array([_TERM_NONE, _TERM_FOLD_OPP, _TERM_FOLD_HERO], dtype=np.int8)
    node_hero_pot = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    node_opp_pot = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    node_num_actions = np.array([2, 0, 0], dtype=np.int32)
    children = np.full((3, 2), -1, dtype=np.int32)
    children[0, 0] = 1
    children[0, 1] = 2
    hero_node_map = np.array([0, -1, -1], dtype=np.int32)
    opp_node_map = np.array([-1, -1, -1], dtype=np.int32)
    bucket_equity_matrix = np.array([[0.5]], dtype=np.float64)
    bucket_valid = np.array([[True]], dtype=np.bool_)

    _cfr_iterations(
        node_player, node_terminal, node_hero_pot, node_opp_pot,
        node_num_actions, children, hero_node_map, opp_node_map,
        bucket_equity_matrix, bucket_valid,
        1, 1, 1, 1, 0, 2, 3,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from equity import ExactEquityEngine

    print("=" * 60)
    print("BlueprintCFR vs BlueprintCFRFast Benchmark")
    print("=" * 60)
    print()

    engine = ExactEquityEngine()

    board = [0, 1, 2, 9, 18]  # 5-card river board (fastest equity computation)
    dead_cards = []
    n_buckets = 10
    n_iterations = 200

    print(f"Board: {board}")
    print(f"Buckets: {n_buckets}")
    print(f"Iterations: {n_iterations}")
    print()

    # --- Warmup JIT ---
    print("Warming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    warmup_jit()
    print(f" done ({time.time() - t0:.2f}s)")
    print()

    # --- BlueprintCFRFast ---
    print("Running BlueprintCFRFast...", end="", flush=True)
    solver_fast = BlueprintCFRFast(n_buckets, n_buckets, engine)
    t0 = time.time()
    result_fast = solver_fast.solve(
        board=board, dead_cards=dead_cards,
        hero_bet=5, opp_bet=5, hero_first=True,
        n_iterations=n_iterations,
    )
    t_fast = time.time() - t0
    print(f" {t_fast:.3f}s")

    # --- BlueprintCFR (original) ---
    from blueprint_cfr import BlueprintCFR
    print("Running BlueprintCFR (original)...", end="", flush=True)
    solver_slow = BlueprintCFR(n_buckets, n_buckets, engine)
    t0 = time.time()
    result_slow = solver_slow.solve(
        board=board, dead_cards=dead_cards,
        hero_bet=5, opp_bet=5, hero_first=True,
        n_iterations=n_iterations,
    )
    t_slow = time.time() - t0
    print(f" {t_slow:.3f}s")

    print()
    print(f"Speedup: {t_slow / t_fast:.1f}x")
    print()

    # Verify strategies are similar
    hs_fast = result_fast['hero_strategy']
    hs_slow = result_slow['hero_strategy']

    if len(hs_fast) > 0 and len(hs_slow) > 0:
        # They won't be identical (different traversal order can cause minor
        # floating-point differences), but should be structurally similar
        min_nodes = min(hs_fast.shape[1], hs_slow.shape[1])
        min_acts = min(hs_fast.shape[2], hs_slow.shape[2])
        diff = np.abs(hs_fast[:, :min_nodes, :min_acts] -
                      hs_slow[:, :min_nodes, :min_acts])
        print(f"Strategy difference (max): {diff.max():.6f}")
        print(f"Strategy difference (mean): {diff.mean():.6f}")

        # Verify both produce valid probability distributions
        for label, hs in [("fast", hs_fast), ("slow", hs_slow)]:
            valid = True
            for b in range(n_buckets):
                for n in range(hs.shape[1]):
                    s = hs[b, n, :].sum()
                    if s > 0 and abs(s - 1.0) > 0.01:
                        valid = False
                        break
            print(f"{label} strategies valid probabilities: {valid}")
    print()
    print("Done.")
