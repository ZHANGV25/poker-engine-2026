"""
Multi-street CFR solver using backward induction (river -> turn -> flop).

Instead of solving each street independently with raw equity, this solver
chains streets: the river solution's EV feeds into the turn's showdown
terminals, and the turn EV feeds into the flop's showdown terminals.

This produces strategies that account for future play -- a hand with draw
potential plays differently than a dead hand of the same current strength.

Architecture:
    1. solve_river(5 cards) -> strategy + root EV[hero][opp]
    2. solve_turn(4 cards)  -> for each showdown terminal, continuation
       value = E[river_EV] over possible river cards -> root EV[hero][opp]
    3. solve_flop(3 cards)  -> for each showdown terminal, continuation
       value = E[turn_EV] over possible turn cards -> strategy

The key implementation detail is that showdown terminals in the game tree
have varying pot sizes (check-check vs bet-call produce different pots).
We use per-terminal-node showdown values instead of a single equity value,
so the continuation value correctly reflects the pot at each specific
terminal node.
"""

import os
import sys
import time
import itertools
import logging

import numpy as np
import numba

_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)
if _submission_dir not in sys.path:
    sys.path.insert(0, _submission_dir)

from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)
from abstraction import compute_board_features

logger = logging.getLogger(__name__)

_TERM_NONE = np.int8(TERM_NONE)
_TERM_FOLD_HERO = np.int8(TERM_FOLD_HERO)
_TERM_FOLD_OPP = np.int8(TERM_FOLD_OPP)
_TERM_SHOWDOWN = np.int8(TERM_SHOWDOWN)

DEFAULT_MAX_BET = 100
DEFAULT_MIN_RAISE = 2

POT_SIZES = [
    (2, 2), (4, 4), (8, 8), (16, 16), (30, 30), (50, 50), (100, 100),
]


# ---------------------------------------------------------------------------
# Numba JIT core
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _regret_match(regrets, n_actions):
    """Regret matching for CFR+."""
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
def _traverse_cfr_pernode(
    node_player, node_terminal, node_hero_pot, node_opp_pot,
    node_num_actions, children, hero_node_map, opp_node_map,
    node_showdown_val,
    hero_regrets_h, hero_strategy_sum_h,
    opp_regrets_o, opp_strategy_sum_o,
    max_actions, hero_reach, opp_reach, n_nodes,
):
    """
    CFR+ traversal with per-node showdown values.

    At FOLD terminals, value = +/- pot (standard).
    At SHOWDOWN terminals, value = node_showdown_val[node_id] (precomputed).

    This supports backward induction: river showdown values come from hand
    ranks; earlier streets use continuation values from the next street.
    """
    max_depth = n_nodes
    stack_node = np.empty(max_depth, dtype=np.int32)
    stack_action = np.empty(max_depth, dtype=np.int32)
    stack_hero_reach = np.empty(max_depth, dtype=np.float64)
    stack_opp_reach = np.empty(max_depth, dtype=np.float64)
    stack_strategy = np.zeros((max_depth, max_actions), dtype=np.float64)
    stack_action_values = np.zeros((max_depth, max_actions), dtype=np.float64)
    stack_node_value = np.zeros(max_depth, dtype=np.float64)

    sp = 0
    stack_node[0] = 0
    stack_action[0] = -1
    stack_hero_reach[0] = hero_reach
    stack_opp_reach[0] = opp_reach
    root_value = 0.0

    while sp >= 0:
        node_id = stack_node[sp]
        cur_action = stack_action[sp]
        h_reach = stack_hero_reach[sp]
        o_reach = stack_opp_reach[sp]

        term = node_terminal[node_id]
        if term != _TERM_NONE:
            if term == _TERM_FOLD_HERO:
                val = -node_hero_pot[node_id]
            elif term == _TERM_FOLD_OPP:
                val = node_opp_pot[node_id]
            else:
                val = node_showdown_val[node_id]

            sp -= 1
            if sp >= 0:
                stack_action_values[sp, stack_action[sp]] = val
            else:
                root_value = val
            continue

        n_act = node_num_actions[node_id]
        player = node_player[node_id]

        if cur_action == -1:
            if player == 0:
                idx = hero_node_map[node_id]
                strategy = _regret_match(hero_regrets_h[idx], n_act)
            else:
                idx = opp_node_map[node_id]
                strategy = _regret_match(opp_regrets_o[idx], n_act)

            for a in range(n_act):
                stack_strategy[sp, a] = strategy[a]
            stack_node_value[sp] = 0.0
            stack_action[sp] = 0
            child_id = children[node_id, 0]

            if player == 0:
                new_h = h_reach * strategy[0]
                new_o = o_reach
            else:
                new_h = h_reach
                new_o = o_reach * strategy[0]

            sp += 1
            stack_node[sp] = child_id
            stack_action[sp] = -1
            stack_hero_reach[sp] = new_h
            stack_opp_reach[sp] = new_o
        else:
            child_val = stack_action_values[sp, cur_action]
            stack_node_value[sp] += stack_strategy[sp, cur_action] * child_val

            next_action = cur_action + 1
            if next_action < n_act:
                stack_action[sp] = next_action
                child_id = children[node_id, next_action]

                if player == 0:
                    new_h = h_reach * stack_strategy[sp, next_action]
                    new_o = o_reach
                else:
                    new_h = h_reach
                    new_o = o_reach * stack_strategy[sp, next_action]

                sp += 1
                stack_node[sp] = child_id
                stack_action[sp] = -1
                stack_hero_reach[sp] = new_h
                stack_opp_reach[sp] = new_o
            else:
                node_val = stack_node_value[sp]

                if player == 0:
                    idx = hero_node_map[node_id]
                    for a in range(n_act):
                        regret = o_reach * (stack_action_values[sp, a] - node_val)
                        new_r = hero_regrets_h[idx, a] + regret
                        hero_regrets_h[idx, a] = new_r if new_r > 0.0 else 0.0
                    for a in range(n_act):
                        hero_strategy_sum_h[idx, a] += h_reach * stack_strategy[sp, a]
                else:
                    idx = opp_node_map[node_id]
                    for a in range(n_act):
                        regret = h_reach * (node_val - stack_action_values[sp, a])
                        new_r = opp_regrets_o[idx, a] + regret
                        opp_regrets_o[idx, a] = new_r if new_r > 0.0 else 0.0
                    for a in range(n_act):
                        opp_strategy_sum_o[idx, a] += o_reach * stack_strategy[sp, a]

                sp -= 1
                if sp >= 0:
                    stack_action_values[sp, stack_action[sp]] = node_val
                else:
                    root_value = node_val

    return root_value


@numba.njit(cache=True)
def _cfr_pernode_iterations(
    node_player, node_terminal, node_hero_pot, node_opp_pot,
    node_num_actions, children_arr, hero_node_map, opp_node_map,
    showdown_values, valid_pairs,
    n_hero, n_opp,
    n_iterations, n_hero_nodes, n_opp_nodes,
    max_actions, n_nodes,
    showdown_node_ids, n_showdown_nodes,
):
    """
    Run CFR+ iterations with per-node, per-pair showdown values.

    showdown_values: (n_showdown_nodes, n_hero, n_opp) hero chip values
    showdown_node_ids: int32 array mapping showdown_idx -> tree node_id
    """
    hero_regrets = np.zeros((n_hero, n_hero_nodes, max_actions), dtype=np.float64)
    hero_strat_sum = np.zeros((n_hero, n_hero_nodes, max_actions), dtype=np.float64)
    opp_regrets = np.zeros((n_opp, n_opp_nodes, max_actions), dtype=np.float64)
    opp_strat_sum = np.zeros((n_opp, n_opp_nodes, max_actions), dtype=np.float64)
    root_ev = np.zeros((n_hero, n_opp), dtype=np.float64)

    node_sd_val = np.zeros(n_nodes, dtype=np.float64)

    for t in range(n_iterations):
        for hi in range(n_hero):
            for oi in range(n_opp):
                if not valid_pairs[hi, oi]:
                    continue

                # Fill per-node showdown values for this pair
                for si in range(n_showdown_nodes):
                    nid = showdown_node_ids[si]
                    node_sd_val[nid] = showdown_values[si, hi, oi]

                val = _traverse_cfr_pernode(
                    node_player, node_terminal, node_hero_pot, node_opp_pot,
                    node_num_actions, children_arr, hero_node_map, opp_node_map,
                    node_sd_val,
                    hero_regrets[hi], hero_strat_sum[hi],
                    opp_regrets[oi], opp_strat_sum[oi],
                    max_actions, 1.0, 1.0, n_nodes,
                )
                root_ev[hi, oi] = val

    return hero_strat_sum, opp_strat_sum, root_ev


@numba.njit(cache=True)
def _normalize_strategy_numba(strategy_sum, node_num_actions_list, n_hands):
    """Normalize strategy sums to probability distributions."""
    result = strategy_sum.copy()
    n_nodes = result.shape[1]
    for h in range(n_hands):
        for i in range(n_nodes):
            n_act = node_num_actions_list[i]
            total = 0.0
            for a in range(n_act):
                total += result[h, i, a]
            if total > 0.0:
                inv = 1.0 / total
                for a in range(n_act):
                    result[h, i, a] *= inv
                for a in range(n_act, result.shape[2]):
                    result[h, i, a] = 0.0
            else:
                uniform = 1.0 / n_act
                for a in range(n_act):
                    result[h, i, a] = uniform
                for a in range(n_act, result.shape[2]):
                    result[h, i, a] = 0.0
    return result


# ---------------------------------------------------------------------------
# Tree flattening
# ---------------------------------------------------------------------------

def _flatten_tree(tree):
    """Convert a GameTree to flat numpy arrays for Numba."""
    n_nodes = tree.size
    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    max_actions = 1
    for nid in tree.hero_node_ids + tree.opp_node_ids:
        if tree.num_actions[nid] > max_actions:
            max_actions = tree.num_actions[nid]

    node_player = np.array(tree.player, dtype=np.int8)
    node_terminal = np.array(tree.terminal, dtype=np.int8)
    node_hero_pot = np.array(tree.hero_pot, dtype=np.float64)
    node_opp_pot = np.array(tree.opp_pot, dtype=np.float64)
    node_num_actions = np.array(tree.num_actions, dtype=np.int32)

    children = np.full((n_nodes, max_actions), -1, dtype=np.int32)
    for nid in range(n_nodes):
        for a, (act_id, child_id) in enumerate(tree.children[nid]):
            children[nid, a] = child_id

    hero_node_map = np.full(n_nodes, -1, dtype=np.int32)
    opp_node_map = np.full(n_nodes, -1, dtype=np.int32)
    for i, nid in enumerate(tree.hero_node_ids):
        hero_node_map[nid] = i
    for i, nid in enumerate(tree.opp_node_ids):
        opp_node_map[nid] = i

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
# Hand utilities
# ---------------------------------------------------------------------------

def _enumerate_hands(board):
    """All valid 2-card hands not overlapping with board."""
    board_set = set(board)
    remaining = [c for c in range(27) if c not in board_set]
    return sorted(itertools.combinations(remaining, 2))


def _build_hand_index(hands):
    """Build {(c1,c2): index} for fast lookup."""
    return {h: i for i, h in enumerate(hands)}


def _build_valid_pairs(hands):
    """Boolean matrix of non-overlapping (hero, opp) hand pairs."""
    n = len(hands)
    masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hands], dtype=np.int64)
    return _build_valid_pairs_fast(masks, n)


@numba.njit(cache=True)
def _build_valid_pairs_fast(masks, n):
    """Numba-accelerated valid pair computation using bitmasks."""
    valid = np.ones((n, n), dtype=numba.boolean)
    for i in range(n):
        valid[i, i] = False
        for j in range(i + 1, n):
            if masks[i] & masks[j]:
                valid[i, j] = False
                valid[j, i] = False
    return valid


# ---------------------------------------------------------------------------
# Single-street solve with per-node showdown values
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vectorized CFR+ (matrix-valued nodes, ~100x faster)
# ---------------------------------------------------------------------------

def _regret_match_vec(regrets, n_act):
    """Vectorized regret matching for all hands at once.

    Args:
        regrets: (n_hands, max_actions)
        n_act: number of valid actions

    Returns:
        (n_hands, n_act) strategy array
    """
    pos = np.maximum(regrets[:, :n_act], 0.0)
    totals = pos.sum(axis=1, keepdims=True)
    uniform = np.full_like(pos, 1.0 / n_act)
    return np.where(totals > 0, pos / np.maximum(totals, 1e-30), uniform)


def _cfr_vectorized(flat, showdown_values, valid_pairs, showdown_ids,
                    n_hands, n_iterations):
    """
    Vectorized CFR+ that processes ALL hands simultaneously at each tree node.

    Instead of iterating n_hero x n_opp pairs and doing a tree traversal
    for each (O(n^2 * tree_size * iters)), this traverses the tree once
    per iteration with (n_hands, n_hands) matrix values at each node
    (O(tree_size * n^2 * iters) but with numpy BLAS ops on the n^2 part).

    ~100-300x faster for n_hands > 50.
    """
    n_nodes = flat['n_nodes']
    n_hero_nodes = flat['n_hero_nodes']
    n_opp_nodes = flat['n_opp_nodes']
    max_actions = flat['max_actions']

    node_player = flat['node_player']
    node_terminal = flat['node_terminal']
    node_hero_pot = flat['node_hero_pot']
    node_opp_pot = flat['node_opp_pot']
    node_num_actions = flat['node_num_actions']
    children = flat['children']
    hero_node_map = flat['hero_node_map']
    opp_node_map = flat['opp_node_map']

    valid_f = valid_pairs.astype(np.float64)

    # Regrets and strategy sums: (node_idx, n_hands, max_actions)
    hero_regrets = np.zeros((n_hero_nodes, n_hands, max_actions))
    hero_strat_sum = np.zeros((n_hero_nodes, n_hands, max_actions))
    opp_regrets = np.zeros((n_opp_nodes, n_hands, max_actions))
    opp_strat_sum = np.zeros((n_opp_nodes, n_hands, max_actions))

    # Build showdown index map: node_id -> showdown array index
    sd_idx_map = {}
    for si, nid in enumerate(showdown_ids):
        sd_idx_map[nid] = si

    def traverse(node_id, hero_reach, opp_reach):
        """
        Recursive tree traversal with matrix-valued nodes.

        Args:
            node_id: int
            hero_reach: (n_hands,) hero reach probabilities
            opp_reach: (n_hands,) opp reach probabilities

        Returns:
            (n_hands, n_hands) value matrix from hero's perspective
        """
        term = node_terminal[node_id]

        if term == _TERM_FOLD_HERO:
            return valid_f * (-node_hero_pot[node_id])
        elif term == _TERM_FOLD_OPP:
            return valid_f * node_opp_pot[node_id]
        elif term == _TERM_SHOWDOWN:
            return showdown_values[sd_idx_map[node_id]]

        n_act = int(node_num_actions[node_id])
        player = int(node_player[node_id])

        if player == 0:  # Hero
            idx = int(hero_node_map[node_id])
            strategies = _regret_match_vec(hero_regrets[idx], n_act)

            action_values = []
            node_value = np.zeros((n_hands, n_hands))

            for a in range(n_act):
                child_id = int(children[node_id, a])
                new_hero_reach = hero_reach * strategies[:, a]
                child_val = traverse(child_id, new_hero_reach, opp_reach)
                action_values.append(child_val)
                node_value += strategies[:, a:a+1] * child_val

            # Update regrets: cf_regret[h] = sum_o(opp_reach[o] * (av[h,o] - nv[h,o]))
            for a in range(n_act):
                cf_regret = (action_values[a] - node_value) @ opp_reach
                hero_regrets[idx, :, a] = np.maximum(
                    0.0, hero_regrets[idx, :, a] + cf_regret)

            # Update strategy sum
            for a in range(n_act):
                hero_strat_sum[idx, :, a] += hero_reach * strategies[:, a]

            return node_value

        else:  # Opp
            idx = int(opp_node_map[node_id])
            strategies = _regret_match_vec(opp_regrets[idx], n_act)

            action_values = []
            node_value = np.zeros((n_hands, n_hands))

            for a in range(n_act):
                child_id = int(children[node_id, a])
                new_opp_reach = opp_reach * strategies[:, a]
                child_val = traverse(child_id, hero_reach, new_opp_reach)
                action_values.append(child_val)
                node_value += child_val * strategies[:, a:a+1].T

            # Update opp regrets: cf_regret[o] = sum_h(hero_reach[h] * (nv[h,o] - av[h,o]))
            for a in range(n_act):
                cf_regret = hero_reach @ (node_value - action_values[a])
                opp_regrets[idx, :, a] = np.maximum(
                    0.0, opp_regrets[idx, :, a] + cf_regret)

            # Update strategy sum
            for a in range(n_act):
                opp_strat_sum[idx, :, a] += opp_reach * strategies[:, a]

            return node_value

    # Run iterations
    hero_reach_init = np.ones(n_hands)
    opp_reach_init = np.ones(n_hands)
    root_ev = None

    for _ in range(n_iterations):
        root_ev = traverse(0, hero_reach_init, opp_reach_init)

    # Transpose to match old interface: (n_hands, n_nodes, max_actions)
    h_strat = hero_strat_sum.transpose(1, 0, 2).copy()
    o_strat = opp_strat_sum.transpose(1, 0, 2).copy()

    if root_ev is None:
        root_ev = np.zeros((n_hands, n_hands))

    return h_strat, o_strat, root_ev


# ---------------------------------------------------------------------------
# Single-street solver
# ---------------------------------------------------------------------------

def _solve_street(tree, hands, showdown_values, showdown_ids, valid_pairs,
                  n_iterations, return_opp=False, flat=None):
    """
    Solve one street's betting with per-terminal showdown values.

    Args:
        tree: GameTree
        hands: list of (c1, c2) tuples
        showdown_values: (n_showdown, n_hands, n_hands) hero chip values
        showdown_ids: list of showdown terminal node IDs
        valid_pairs: (n_hands, n_hands) bool
        n_iterations: CFR iterations
        return_opp: if True, also return opp strategy (for position-aware play)
        flat: pre-flattened tree dict (optimization to avoid re-flattening)

    Returns:
        (hero_strategy, root_ev) or (hero_strategy, opp_strategy, root_ev)
        hero_strategy: (n_hands, n_hero_nodes, max_actions) normalized
        opp_strategy: (n_hands, n_opp_nodes, max_actions) normalized
        root_ev: (n_hands, n_hands) hero's EV at root
    """
    if flat is None:
        flat = _flatten_tree(tree)
    n_hands = len(hands)
    n_hero_nodes = flat['n_hero_nodes']
    n_opp_nodes = flat['n_opp_nodes']

    if n_hero_nodes == 0 and n_opp_nodes == 0:
        empty = np.zeros((n_hands, 1, 1), dtype=np.float64)
        zero_ev = np.zeros((n_hands, n_hands), dtype=np.float64)
        if return_opp:
            return empty, empty, zero_ev
        return empty, zero_ev

    # Use vectorized CFR (processes all hands simultaneously with matrix ops)
    hero_strat_sum, opp_strat_sum, root_ev = _cfr_vectorized(
        flat, showdown_values, valid_pairs, showdown_ids,
        n_hands, n_iterations)

    hero_strategy = _normalize_strategy_numba(
        hero_strat_sum, flat['hero_node_num_actions'], n_hands)

    if return_opp:
        opp_strategy = _normalize_strategy_numba(
            opp_strat_sum, flat['opp_node_num_actions'], n_hands)
        return hero_strategy, opp_strategy, root_ev

    return hero_strategy, root_ev


# ---------------------------------------------------------------------------
# Showdown value computation
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _map_ev_to_parent(parent_sum, parent_count, child_ev, child_valid,
                       child_to_parent, n_child):
    """Map child street EVs to parent street hand indices."""
    for hi in range(n_child):
        pi_h = child_to_parent[hi]
        if pi_h < 0:
            continue
        for oi in range(n_child):
            pi_o = child_to_parent[oi]
            if pi_o < 0:
                continue
            if not child_valid[hi, oi]:
                continue
            parent_sum[pi_h, pi_o] += child_ev[hi, oi]
            parent_count[pi_h, pi_o] += 1.0


@numba.njit(cache=True)
def _phase1_river_cfr_all(
    board_3, remaining_arr, hands_arr, n_hands,
    seven_lookup_arr, pot_won,
    node_player, node_terminal, node_hero_pot, node_opp_pot,
    node_num_actions, children_arr, hero_node_map, opp_node_map,
    sd_node_ids, n_sd, n_hero_nodes, n_opp_nodes, max_actions, n_nodes,
    river_iters,
):
    """Numba-compiled Phase 1: river CFR for ALL turn/river combinations.

    Computes turn continuation EVs by solving river sub-games with CFR.
    Everything stays in Numba — no Python overhead between the 552 sub-games.

    Returns:
        cont_ev_all: (n_remaining, n_hands, n_hands) continuation EVs
                     indexed by full hand list, averaged over river cards.
    """
    n_remaining = remaining_arr.shape[0]

    # Output: cont EVs per turn card, indexed by original hand indices
    cont_ev_sum = np.zeros((n_remaining, n_hands, n_hands), dtype=np.float64)
    cont_ev_count = np.zeros((n_remaining, n_hands, n_hands), dtype=np.float64)

    # Board bitmask for flop
    flop_mask = (1 << board_3[0]) | (1 << board_3[1]) | (1 << board_3[2])

    # Precompute hand bitmasks
    hand_masks = np.empty(n_hands, dtype=np.int64)
    for hi in range(n_hands):
        hand_masks[hi] = (1 << hands_arr[hi, 0]) | (1 << hands_arr[hi, 1])

    for ti in range(n_remaining):
        turn_card = remaining_arr[ti]
        turn_bit = np.int64(1) << turn_card

        # Identify turn-valid hands (don't contain turn card)
        turn_hand_indices = np.empty(n_hands, dtype=np.int32)
        n_turn = 0
        for hi in range(n_hands):
            if not (hand_masks[hi] & turn_bit):
                turn_hand_indices[n_turn] = hi
                n_turn += 1

        # Build turn valid pairs
        turn_valid = np.ones((n_turn, n_turn), dtype=numba.boolean)
        for i in range(n_turn):
            turn_valid[i, i] = False
            mi = hand_masks[turn_hand_indices[i]]
            for j in range(i + 1, n_turn):
                if mi & hand_masks[turn_hand_indices[j]]:
                    turn_valid[i, j] = False
                    turn_valid[j, i] = False

        for ri in range(n_remaining):
            if ri == ti:
                continue
            river_card = remaining_arr[ri]
            river_bit = np.int64(1) << river_card
            board_mask = flop_mask | turn_bit | river_bit

            # Identify river-valid hands (turn hands not containing river card)
            river_hand_indices = np.empty(n_turn, dtype=np.int32)
            n_river = 0
            for i in range(n_turn):
                hi = turn_hand_indices[i]
                if not (hand_masks[hi] & river_bit):
                    river_hand_indices[n_river] = i  # index into turn_hand_indices
                    n_river += 1

            # Build river valid pairs
            river_valid = np.ones((n_river, n_river), dtype=numba.boolean)
            for i in range(n_river):
                river_valid[i, i] = False
                ti_i = turn_hand_indices[river_hand_indices[i]]
                mi = hand_masks[ti_i]
                for j in range(i + 1, n_river):
                    tj_j = turn_hand_indices[river_hand_indices[j]]
                    if mi & hand_masks[tj_j]:
                        river_valid[i, j] = False
                        river_valid[j, i] = False

            # Compute hand ranks
            hand_ranks = np.empty(n_river, dtype=np.int64)
            for i in range(n_river):
                hi = turn_hand_indices[river_hand_indices[i]]
                mask = hand_masks[hi] | board_mask
                hand_ranks[i] = seven_lookup_arr[mask]

            # Build showdown values for CFR
            sd_vals = np.zeros((n_sd, n_river, n_river), dtype=np.float64)
            for si in range(n_sd):
                pw = pot_won  # same for all showdown terminals in this simple model
                for i in range(n_river):
                    ri_rank = hand_ranks[i]
                    for j in range(n_river):
                        if not river_valid[i, j]:
                            continue
                        rj_rank = hand_ranks[j]
                        if ri_rank < rj_rank:
                            sd_vals[si, i, j] = pw
                        elif ri_rank > rj_rank:
                            sd_vals[si, i, j] = -pw

            # Run CFR
            _, _, r_ev = _cfr_pernode_iterations(
                node_player, node_terminal, node_hero_pot, node_opp_pot,
                node_num_actions, children_arr, hero_node_map, opp_node_map,
                sd_vals, river_valid,
                n_river, n_river,
                river_iters, n_hero_nodes, n_opp_nodes,
                max_actions, n_nodes,
                sd_node_ids, n_sd,
            )

            # Map river EVs back to turn hand indices, then to full hand indices
            for i in range(n_river):
                fi_turn = river_hand_indices[i]  # index into turn hands
                fi_full = turn_hand_indices[fi_turn]  # index into full hands
                for j in range(n_river):
                    if not river_valid[i, j]:
                        continue
                    fj_turn = river_hand_indices[j]
                    fj_full = turn_hand_indices[fj_turn]
                    cont_ev_sum[ti, fi_full, fj_full] += r_ev[i, j]
                    cont_ev_count[ti, fi_full, fj_full] += 1.0

    # Average over river cards
    for ti in range(n_remaining):
        for hi in range(n_hands):
            for oi in range(n_hands):
                if cont_ev_count[ti, hi, oi] > 0:
                    cont_ev_sum[ti, hi, oi] /= cont_ev_count[ti, hi, oi]

    return cont_ev_sum


@numba.njit(cache=True)
def _vectorized_river_ev(hand_ranks, valid_pairs, pot_won, n):
    """Compute river continuation values accounting for betting.

    Models a one-round betting game where:
    - Winners bet for value and extract more from strong opponents
    - Losers face bets and fold (weak) or call (strong)
    All in Numba for speed.
    """
    bet_size = pot_won * 0.7

    ev = np.zeros((n, n), dtype=np.float64)

    # Compute strength (fraction of valid opponents this hand beats)
    strength = np.zeros(n, dtype=np.float64)
    valid_count = np.zeros(n, dtype=np.float64)
    for hi in range(n):
        w = 0.0
        vc = 0.0
        for oi in range(n):
            if not valid_pairs[hi, oi]:
                continue
            vc += 1.0
            if hand_ranks[hi] < hand_ranks[oi]:
                w += 1.0
            elif hand_ranks[hi] == hand_ranks[oi]:
                w += 0.5
        if vc > 0:
            strength[hi] = w / vc
        valid_count[hi] = vc

    # Compute adjusted EVs
    for hi in range(n):
        ri = hand_ranks[hi]
        sh = strength[hi]
        for oi in range(n):
            if not valid_pairs[hi, oi]:
                continue
            ro = hand_ranks[oi]
            so = strength[oi]

            if ri < ro:
                # Hero wins. Opp folds if weak, calls if strong.
                if so < 0.35:
                    ev[hi, oi] = pot_won
                else:
                    ev[hi, oi] = pot_won + bet_size * 0.5
            elif ri > ro:
                # Hero loses. Hero folds if weak, calls if strong.
                if sh < 0.35:
                    ev[hi, oi] = -pot_won
                else:
                    ev[hi, oi] = -(pot_won + bet_size * 0.5)

    return ev


@numba.njit(cache=True)
def _fill_analytical_ev(ev, hand_ranks, valid_pairs, pot_won, n):
    """Analytical river EV: winner gets pot_won, loser loses pot_won."""
    for hi in range(n):
        ri = hand_ranks[hi]
        for oi in range(n):
            if not valid_pairs[hi, oi]:
                continue
            ro = hand_ranks[oi]
            if ri < ro:
                ev[hi, oi] = pot_won
            elif ri > ro:
                ev[hi, oi] = -pot_won


@numba.njit(cache=True)
def _simplified_river_cfr(hand_ranks, valid_pairs, pot_won, n, n_iters):
    """Simplified river CFR with check/bet, fold/call tree.

    Tree (7 nodes):
      Node 0 (hero): CHECK→1, BET→4
      Node 1 (opp):  CHECK→2(sd), BET→3
      Node 3 (hero): FOLD→(fold_hero), CALL→(sd at bet_pot)
      Node 4 (opp):  FOLD→(fold_opp), CALL→(sd at bet_pot)

    This captures the key dynamic that analytical EVs miss:
    weak hands face bets and fold, getting lower EV than raw equity.
    """
    bet_frac = 0.7  # bet 70% of pot
    bet_size = pot_won * bet_frac  # each player's additional bet
    pot_after_bet = pot_won + bet_size  # pot won at showdown after bet-call

    # Regrets and strategy sums for hero (2 decision nodes × n hands × 2 actions)
    # Node 0: hero root (check/bet)
    # Node 1: hero facing opp bet after check (fold/call)
    hero_regrets = np.zeros((2, n, 2), dtype=np.float64)
    hero_strat_sum = np.zeros((2, n, 2), dtype=np.float64)
    opp_regrets = np.zeros((1, n, 2), dtype=np.float64)   # opp after hero check
    opp_strat_sum = np.zeros((1, n, 2), dtype=np.float64)

    root_ev = np.zeros((n, n), dtype=np.float64)

    for _ in range(n_iters):
        for hi in range(n):
            for oi in range(n):
                if not valid_pairs[hi, oi]:
                    continue

                # Showdown result
                ri = hand_ranks[hi]
                ro = hand_ranks[oi]
                if ri < ro:
                    sd_val = pot_won       # hero wins at check-check
                    sd_val_bet = pot_after_bet  # hero wins at bet-call
                elif ri > ro:
                    sd_val = -pot_won
                    sd_val_bet = -pot_after_bet
                else:
                    sd_val = 0.0
                    sd_val_bet = 0.0

                # Hero root strategy (check/bet)
                h0 = hero_regrets[0, hi]
                s0_check = max(0.0, h0[0])
                s0_bet = max(0.0, h0[1])
                t0 = s0_check + s0_bet
                if t0 > 0:
                    s0_check /= t0
                    s0_bet /= t0
                else:
                    s0_check = 0.5
                    s0_bet = 0.5

                # Opp strategy after hero check (check/bet)
                o0 = opp_regrets[0, oi]
                so_check = max(0.0, o0[0])
                so_bet = max(0.0, o0[1])
                to = so_check + so_bet
                if to > 0:
                    so_check /= to
                    so_bet /= to
                else:
                    so_check = 0.5
                    so_bet = 0.5

                # Hero strategy facing opp bet (fold/call)
                h1 = hero_regrets[1, hi]
                s1_fold = max(0.0, h1[0])
                s1_call = max(0.0, h1[1])
                t1 = s1_fold + s1_call
                if t1 > 0:
                    s1_fold /= t1
                    s1_call /= t1
                else:
                    s1_fold = 0.5
                    s1_call = 0.5

                # Compute values for each path
                # Path: hero checks → opp checks → showdown
                v_check_check = sd_val
                # Path: hero checks → opp bets → hero folds
                v_check_bet_fold = -pot_won
                # Path: hero checks → opp bets → hero calls
                v_check_bet_call = sd_val_bet
                # Path: hero bets → opp folds
                v_bet_fold = pot_won
                # Path: hero bets → opp calls
                v_bet_call = sd_val_bet

                # Hero facing opp bet: value
                v_facing_bet = s1_fold * v_check_bet_fold + s1_call * v_check_bet_call

                # After hero check: opp decides
                v_after_check = so_check * v_check_check + so_bet * v_facing_bet

                # After hero bet: opp decides (opp fold/call - use simple response)
                # Opp folds if hand is weak enough. For simplicity, use fixed frequencies
                # based on pot odds: need 41% equity to call a 70% pot bet
                # Just use the actual showdown value
                opp_call_val = -sd_val_bet  # opp's value of calling (negative of hero's)
                if opp_call_val >= -pot_after_bet * 0.5:  # rough threshold
                    v_after_bet = v_bet_call
                else:
                    v_after_bet = v_bet_fold

                # Root value
                node_val = s0_check * v_after_check + s0_bet * v_after_bet
                root_ev[hi, oi] = node_val

                # Update hero root regrets
                r_check = v_after_check - node_val
                r_bet = v_after_bet - node_val
                hero_regrets[0, hi, 0] = max(0.0, hero_regrets[0, hi, 0] + r_check)
                hero_regrets[0, hi, 1] = max(0.0, hero_regrets[0, hi, 1] + r_bet)
                hero_strat_sum[0, hi, 0] += s0_check
                hero_strat_sum[0, hi, 1] += s0_bet

                # Update hero facing-bet regrets
                r_fold = v_check_bet_fold - v_facing_bet
                r_call = v_check_bet_call - v_facing_bet
                hero_regrets[1, hi, 0] = max(0.0, hero_regrets[1, hi, 0] + r_fold)
                hero_regrets[1, hi, 1] = max(0.0, hero_regrets[1, hi, 1] + r_call)

                # Update opp regrets (after hero check)
                # Opp utility = -hero utility
                opp_r_check = -(v_check_check - v_after_check)
                opp_r_bet = -(v_facing_bet - v_after_check)
                opp_regrets[0, oi, 0] = max(0.0, opp_regrets[0, oi, 0] + opp_r_check)
                opp_regrets[0, oi, 1] = max(0.0, opp_regrets[0, oi, 1] + opp_r_bet)

    return root_ev


@numba.njit(cache=True)
def _fill_river_showdown(sd_vals, hand_ranks, valid_pairs, pot_won_arr, n, n_sd):
    """Numba-accelerated inner loop for river showdown values."""
    for si in range(n_sd):
        pot_won = pot_won_arr[si]
        for hi in range(n):
            ri = hand_ranks[hi]
            for oi in range(n):
                if not valid_pairs[hi, oi]:
                    continue
                ro = hand_ranks[oi]
                if ri < ro:
                    sd_vals[si, hi, oi] = pot_won
                elif ri > ro:
                    sd_vals[si, hi, oi] = -pot_won


def _river_showdown_values(tree, hands, board_5, valid_pairs, engine):
    """
    Compute per-terminal showdown values for a river board.
    Uses Numba for the inner loop (was pure Python, now ~50x faster).
    """
    n = len(hands)
    board_mask = 0
    for c in board_5:
        board_mask |= 1 << c
    seven_lookup = engine._seven

    hand_ranks = np.zeros(n, dtype=np.int64)
    for i, hand in enumerate(hands):
        mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
        hand_ranks[i] = seven_lookup[mask]

    sd_ids = [nid for nid in tree.terminal_node_ids
              if tree.terminal[nid] == TERM_SHOWDOWN]
    n_sd = len(sd_ids)
    sd_vals = np.zeros((n_sd, n, n), dtype=np.float64)

    pot_won_arr = np.array([min(tree.hero_pot[nid], tree.opp_pot[nid])
                            for nid in sd_ids], dtype=np.float64)

    _fill_river_showdown(sd_vals, hand_ranks, valid_pairs, pot_won_arr, n, n_sd)

    return sd_vals, sd_ids


def _continuation_showdown_values(tree, n_hands, continuation_ev, valid_pairs):
    """
    Compute per-terminal showdown values using next-street continuation EVs.

    At each showdown terminal (representing "go to next street"), the value
    is the continuation EV. Since hero_pot == opp_pot at all showdown
    terminals, and the continuation EV was computed for that specific pot
    as the starting pot of the next street, we scale the continuation EV
    by the ratio of the actual pot to the reference pot.

    continuation_ev: (n_hands, n_hands) hero's EV from next street root
    ref_pot: the pot for which continuation_ev was computed (hero_bet = opp_bet)
    """
    sd_ids = [nid for nid in tree.terminal_node_ids
              if tree.terminal[nid] == TERM_SHOWDOWN]
    n_sd = len(sd_ids)
    sd_vals = np.zeros((n_sd, n_hands, n_hands), dtype=np.float64)

    # The continuation EV was computed for the pot size we passed to the
    # next-street solver (hero_bet, opp_bet). At each showdown terminal,
    # hero_pot == opp_pot (always true after check-check or call). The
    # actual pot may differ from the reference pot.
    #
    # However, since we solve the next street for the SAME pot size as
    # the current street's starting pot, and showdown terminals can have
    # different pot sizes (e.g., check-check keeps original pot, but
    # bet-call increases it), we need to handle pot scaling.
    #
    # For now, we use the continuation EV directly. This is correct when
    # all showdown terminals have the same pot (which is NOT generally true).
    # For more accuracy, we'd solve the next street for each distinct pot.
    # As a practical approximation, we scale linearly with pot ratio.
    #
    # Reference pot = starting hero_bet + opp_bet (the pot the next street
    # was solved for). At each showdown terminal with pot P:
    #   scaled_ev = continuation_ev * (P / ref_pot)

    # Find the minimum showdown pot (corresponds to check-check = starting pot)
    if n_sd == 0:
        return sd_vals, sd_ids

    min_sd_pot = float('inf')
    for nid in sd_ids:
        pot = tree.hero_pot[nid] + tree.opp_pot[nid]
        if pot < min_sd_pot:
            min_sd_pot = pot

    ref_pot = min_sd_pot  # check-check pot = starting pot

    scales = np.array([(tree.hero_pot[nid] + tree.opp_pot[nid]) / ref_pot
                        if ref_pot > 0 else 1.0 for nid in sd_ids], dtype=np.float64)

    _fill_continuation(sd_vals, continuation_ev, valid_pairs, scales, n_hands, n_sd)

    return sd_vals, sd_ids


@numba.njit(cache=True)
def _fill_continuation(sd_vals, cont_ev, valid_pairs, scales, n_hands, n_sd):
    for si in range(n_sd):
        s = scales[si]
        for hi in range(n_hands):
            for oi in range(n_hands):
                if not valid_pairs[hi, oi]:
                    continue
                sd_vals[si, hi, oi] = cont_ev[hi, oi] * s


# ---------------------------------------------------------------------------
# Main backward induction solver
# ---------------------------------------------------------------------------

def solve_flop_board(board_3_cards, engine, n_iterations=500, pot_sizes=None,
                     turn_save_pot_idx=1, position_aware=False):
    """
    Solve a complete flop board using backward induction.

    For each pot size, chains river -> turn -> flop solutions so that
    each street's strategy accounts for optimal play on future streets.

    Args:
        board_3_cards: list of 3 ints (flop cards)
        engine: ExactEquityEngine instance
        n_iterations: CFR iterations per street
        pot_sizes: list of (hero_bet, opp_bet) tuples
        turn_save_pot_idx: which pot index to save turn strategies for
                          (default 1 = (4,4) pot). Set to -1 to save all.

    Returns:
        dict with:
            'flop_strategies': uint8 (n_pots, n_hands, n_hero_nodes, n_actions)
            'hands': list of (c1, c2) tuples
            'board': the 3 flop cards
            'pot_sizes': the pot sizes used
            'board_features': 12-float feature vector
            'action_types': int8 (n_pots, n_hero_nodes, n_actions)
    """
    if pot_sizes is None:
        pot_sizes = POT_SIZES

    board_3 = list(board_3_cards)
    board_set = set(board_3)

    # All hands for this flop (cards not on board)
    hands = _enumerate_hands(board_3)
    hand_index = _build_hand_index(hands)
    valid_pairs = _build_valid_pairs(hands)
    n_hands = len(hands)

    remaining = [c for c in range(27) if c not in board_set]

    # Iteration counts per street. River sub-games only need root EV
    # (not full strategy), so fewer iterations suffice — EV converges
    # faster than strategy. Turn and flop get more for strategy quality.
    river_iters = 20   # root EV converges fast, ~552 solves per pot
    turn_iters = max(40, n_iterations // 2)
    flop_iters = n_iterations

    t_total = time.time()
    logger.info("Multi-street solve: board=%s, %d hands, %d remaining",
                board_3, n_hands, len(remaining))

    all_strategies = []
    all_opp_strategies = []  # position-aware: P1 (acting-second) strategies
    all_trees = []
    saved_turn_data = {}  # {(pot_idx, turn_card): (hands, idx, valid, cont_ev, strategy, tree)}

    for pot_idx, (hero_bet, opp_bet) in enumerate(pot_sizes):
        t_pot = time.time()
        logger.info("  Pot %d/%d: (%d,%d)",
                    pot_idx + 1, len(pot_sizes), hero_bet, opp_bet)

        # -------------------------------------------------------
        # Phase 1: River CFR backward induction
        # -------------------------------------------------------
        # Solve actual river betting games instead of analytical EVs.
        # This correctly accounts for value betting, bluffing, and
        # folding — weak hands get lower EV because they face bets.
        turn_data = {}

        seven_lookup = engine._seven

        # Build ONE river tree per pot (reuse across all river cards)
        r_tree = GameTree(hero_bet, opp_bet, DEFAULT_MIN_RAISE,
                          DEFAULT_MAX_BET, True)
        r_flat = _flatten_tree(r_tree)  # cache flattened tree for reuse

        for turn_card in remaining:
            board_4 = board_3 + [turn_card]
            turn_hands = [(c1, c2) for c1, c2 in hands
                          if c1 != turn_card and c2 != turn_card]
            turn_idx = {h: i for i, h in enumerate(turn_hands)}
            n_turn = len(turn_hands)
            turn_valid = _build_valid_pairs(turn_hands)

            river_cards = [c for c in remaining if c != turn_card]
            ev_sum = np.zeros((n_turn, n_turn), dtype=np.float64)
            ev_count = np.zeros((n_turn, n_turn), dtype=np.float64)

            for river_card in river_cards:
                board_5 = board_4 + [river_card]

                r_hands = [(c1, c2) for c1, c2 in turn_hands
                           if c1 != river_card and c2 != river_card]
                n_river = len(r_hands)
                r_valid = _build_valid_pairs(r_hands)

                # Solve the river betting game with full CFR
                r_sd_vals, r_sd_ids = _river_showdown_values(
                    r_tree, r_hands, board_5, r_valid, engine)
                _, r_ev = _solve_street(
                    r_tree, r_hands, r_sd_vals, r_sd_ids,
                    r_valid, river_iters, flat=r_flat)

                r_to_t = np.full(n_river, -1, dtype=np.int32)
                for ri, rh in enumerate(r_hands):
                    ti = turn_idx.get(rh)
                    if ti is not None:
                        r_to_t[ri] = ti
                _map_ev_to_parent(ev_sum, ev_count, r_ev, r_valid,
                                  r_to_t, n_river)

            cont_ev = np.zeros((n_turn, n_turn), dtype=np.float64)
            mask = ev_count > 0
            cont_ev[mask] = ev_sum[mask] / ev_count[mask]

            turn_data[turn_card] = (turn_hands, turn_idx, turn_valid, cont_ev)

        logger.info("    Phase 1 (river CFR) done: %.1fs", time.time() - t_pot)

        # -------------------------------------------------------
        # Phase 2: Solve all turns, aggregate into flop EVs
        # -------------------------------------------------------
        t_phase2 = time.time()
        flop_ev_sum = np.zeros((n_hands, n_hands), dtype=np.float64)
        flop_ev_count = np.zeros((n_hands, n_hands), dtype=np.float64)

        for turn_card in remaining:
            t_hands, t_idx, t_valid, t_cont_ev = turn_data[turn_card]
            n_turn = len(t_hands)

            t_tree = GameTree(hero_bet, opp_bet, DEFAULT_MIN_RAISE,
                              DEFAULT_MAX_BET, True)
            t_sd_vals, t_sd_ids = _continuation_showdown_values(
                t_tree, n_turn, t_cont_ev, t_valid)

            if position_aware:
                turn_strategy, turn_opp_strategy, t_root_ev = _solve_street(
                    t_tree, t_hands, t_sd_vals, t_sd_ids,
                    t_valid, turn_iters, return_opp=True)
            else:
                turn_strategy, t_root_ev = _solve_street(
                    t_tree, t_hands, t_sd_vals, t_sd_ids,
                    t_valid, turn_iters)
                turn_opp_strategy = None

            # Save turn strategy for this turn card
            turn_data[turn_card] = (turn_data[turn_card][0], turn_data[turn_card][1],
                                     turn_data[turn_card][2], turn_data[turn_card][3],
                                     turn_strategy, t_tree, turn_opp_strategy)

            # Map turn EVs to flop hand indices (vectorized)
            t_to_f = np.full(n_turn, -1, dtype=np.int32)
            for ti, th in enumerate(t_hands):
                fi = hand_index.get(th)
                if fi is not None:
                    t_to_f[ti] = fi
            _map_ev_to_parent(flop_ev_sum, flop_ev_count, t_root_ev, t_valid,
                              t_to_f, n_turn)

        logger.info("    Phase 2 (turns) done: %.1fs", time.time() - t_phase2)

        # Save turn strategies for the designated pot
        if turn_save_pot_idx < 0 or pot_idx == turn_save_pot_idx:
            for turn_card in remaining:
                td = turn_data.get(turn_card)
                if td is not None and len(td) >= 6:
                    saved_turn_data[(pot_idx, turn_card)] = td

        # -------------------------------------------------------
        # Phase 3: Solve the flop
        # -------------------------------------------------------
        t_phase3 = time.time()
        flop_cont = np.zeros((n_hands, n_hands), dtype=np.float64)
        mask = flop_ev_count > 0
        flop_cont[mask] = flop_ev_sum[mask] / flop_ev_count[mask]

        f_tree = GameTree(hero_bet, opp_bet, DEFAULT_MIN_RAISE,
                          DEFAULT_MAX_BET, True)
        f_sd_vals, f_sd_ids = _continuation_showdown_values(
            f_tree, n_hands, flop_cont, valid_pairs)

        if position_aware:
            flop_strategy, flop_opp_strategy, _ = _solve_street(
                f_tree, hands, f_sd_vals, f_sd_ids,
                valid_pairs, flop_iters, return_opp=True)
            all_opp_strategies.append(flop_opp_strategy)
        else:
            flop_strategy, _ = _solve_street(
                f_tree, hands, f_sd_vals, f_sd_ids,
                valid_pairs, flop_iters)

        all_strategies.append(flop_strategy)
        all_trees.append(f_tree)

        logger.info("    Phase 3 (flop) done: %.1fs (total pot: %.1fs)",
                    time.time() - t_phase3, time.time() - t_pot)

    # -------------------------------------------------------
    # Package results
    # -------------------------------------------------------
    n_pot = len(pot_sizes)
    max_hero_nodes = max(s.shape[1] for s in all_strategies)
    max_actions = max(s.shape[2] for s in all_strategies)

    strategies = np.zeros((n_pot, n_hands, max_hero_nodes, max_actions),
                          dtype=np.uint8)
    action_types = np.full((n_pot, max_hero_nodes, max_actions), -1,
                           dtype=np.int8)

    for pi, (strat, tree) in enumerate(zip(all_strategies, all_trees)):
        nh, nn, na = strat.shape
        q = np.clip(np.round(strat * 255.0), 0, 255).astype(np.uint8)
        strategies[pi, :nh, :nn, :na] = q

        for i, nid in enumerate(tree.hero_node_ids):
            if i >= max_hero_nodes:
                break
            for a, (act_id, _) in enumerate(tree.children[nid]):
                if a >= max_actions:
                    break
                action_types[pi, i, a] = act_id

    board_features = compute_board_features(board_3)
    total_time = time.time() - t_total

    logger.info("Board %s complete: %.1fs", board_3, total_time)

    # Package opp strategies (position-aware: P1 acting second)
    opp_strategies = None
    opp_action_types = None
    if position_aware and all_opp_strategies:
        max_opp_nodes = max(s.shape[1] for s in all_opp_strategies)
        max_opp_actions = max(s.shape[2] for s in all_opp_strategies)
        opp_strategies = np.zeros((n_pot, n_hands, max_opp_nodes, max_opp_actions),
                                   dtype=np.uint8)
        opp_action_types = np.full((n_pot, max_opp_nodes, max_opp_actions), -1,
                                    dtype=np.int8)
        for pi, (strat, tree) in enumerate(zip(all_opp_strategies, all_trees)):
            nh, nn, na = strat.shape
            q = np.clip(np.round(strat * 255.0), 0, 255).astype(np.uint8)
            opp_strategies[pi, :nh, :nn, :na] = q
            for i, nid in enumerate(tree.opp_node_ids):
                if i >= max_opp_nodes:
                    break
                for a, (act_id, _) in enumerate(tree.children[nid]):
                    if a >= max_opp_actions:
                        break
                    opp_action_types[pi, i, a] = act_id

    # Package turn strategies from saved_turn_data (collected from designated pot)
    turn_strategies_out = {}
    for (pot_idx, turn_card), td in saved_turn_data.items():
        if len(td) < 6:
            continue
        t_hands = td[0]
        t_strat = td[4]
        t_tree = td[5]
        t_opp_strat = td[6] if len(td) > 6 else None
        turn_strategies_out[(pot_idx, turn_card)] = {
            'strategy': t_strat,
            'hands': t_hands,
            'tree': t_tree,
            'opp_strategy': t_opp_strat,
        }

    result = {
        'flop_strategies': strategies,
        'turn_strategies': turn_strategies_out,
        'hands': hands,
        'board': board_3,
        'pot_sizes': pot_sizes,
        'board_features': board_features,
        'action_types': action_types,
    }
    if opp_strategies is not None:
        result['flop_opp_strategies'] = opp_strategies
        result['opp_action_types'] = opp_action_types
    return result


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit():
    """Trigger Numba JIT compilation."""
    node_player = np.array([0, -1, -1], dtype=np.int8)
    node_terminal = np.array([_TERM_NONE, _TERM_FOLD_OPP, _TERM_SHOWDOWN],
                             dtype=np.int8)
    hp = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    op = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    na = np.array([2, 0, 0], dtype=np.int32)
    ch = np.full((3, 2), -1, dtype=np.int32)
    ch[0, 0] = 1
    ch[0, 1] = 2
    hm = np.array([0, -1, -1], dtype=np.int32)
    om = np.array([-1, -1, -1], dtype=np.int32)

    # Warmup _traverse_cfr_pernode
    sv = np.zeros(3, dtype=np.float64)
    hr = np.zeros((1, 2), dtype=np.float64)
    hs = np.zeros((1, 2), dtype=np.float64)
    orr = np.zeros((1, 1), dtype=np.float64)
    os_ = np.zeros((1, 1), dtype=np.float64)
    _traverse_cfr_pernode(node_player, node_terminal, hp, op, na, ch, hm, om,
                          sv, hr, hs, orr, os_, 2, 1.0, 1.0, 3)

    # Warmup _cfr_pernode_iterations
    sd_v = np.zeros((1, 1, 1), dtype=np.float64)
    vp = np.array([[True]], dtype=np.bool_)
    sd_n = np.array([2], dtype=np.int32)
    _cfr_pernode_iterations(node_player, node_terminal, hp, op, na, ch, hm, om,
                            sd_v, vp, 1, 1, 1, 1, 0, 2, 3, sd_n, 1)

    # Warmup _normalize_strategy_numba
    _normalize_strategy_numba(np.zeros((1, 1, 2), dtype=np.float64),
                              np.array([2], dtype=np.int32), 1)
    _regret_match(np.array([1.0, 2.0], dtype=np.float64), 2)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    from equity import ExactEquityEngine

    print("=" * 60)
    print("Multi-Street Solver Test")
    print("=" * 60)

    engine = ExactEquityEngine()

    print("\nWarming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    warmup_jit()
    print(f" done ({time.time() - t0:.2f}s)")

    board = [0, 1, 2]  # 2d, 3d, 4d
    print(f"\nSolving board {board} (quick test, 1 pot, 30 iters)...")

    t0 = time.time()
    result = solve_flop_board(board, engine, n_iterations=30,
                               pot_sizes=[(2, 2)])
    elapsed = time.time() - t0

    strats = result['flop_strategies']
    hands = result['hands']
    print(f"Solved in {elapsed:.1f}s")
    print(f"Strategy shape: {strats.shape}")
    print(f"Hands: {len(hands)}")

    # Validate: show strategy spread
    s_float = strats[0].astype(np.float64) / 255.0
    at = result['action_types']
    act_names = {0: 'fold', 1: 'check', 2: 'call',
                 3: 'r40', 4: 'r70', 5: 'r100', 6: 'r150'}

    print("\nSample strategies (root node):")
    step = max(1, len(hands) // 10)
    for i in range(0, len(hands), step):
        s = s_float[i, 0, :]
        acts = at[0, 0, :]
        parts = [f"{act_names.get(int(acts[a]), '?')}={s[a]:.2f}"
                 for a in range(len(s)) if acts[a] >= 0]
        print(f"  {hands[i]}: {', '.join(parts)}")

    print("\nDone.")
