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
    valid = np.ones((n, n), dtype=np.bool_)
    for i in range(n):
        hi_set = set(hands[i])
        valid[i, i] = False
        for j in range(i + 1, n):
            if hi_set & set(hands[j]):
                valid[i, j] = False
                valid[j, i] = False
    return valid


# ---------------------------------------------------------------------------
# Single-street solve with per-node showdown values
# ---------------------------------------------------------------------------

def _solve_street(tree, hands, showdown_values, showdown_ids, valid_pairs,
                  n_iterations):
    """
    Solve one street's betting with per-terminal showdown values.

    Args:
        tree: GameTree
        hands: list of (c1, c2) tuples
        showdown_values: (n_showdown, n_hands, n_hands) hero chip values
        showdown_ids: list of showdown terminal node IDs
        valid_pairs: (n_hands, n_hands) bool
        n_iterations: CFR iterations

    Returns:
        (hero_strategy, root_ev)
        hero_strategy: (n_hands, n_hero_nodes, max_actions) normalized
        root_ev: (n_hands, n_hands) hero's EV at root
    """
    flat = _flatten_tree(tree)
    n_hands = len(hands)
    n_hero_nodes = flat['n_hero_nodes']
    n_opp_nodes = flat['n_opp_nodes']

    if n_hero_nodes == 0 and n_opp_nodes == 0:
        return (np.zeros((n_hands, 1, 1), dtype=np.float64),
                np.zeros((n_hands, n_hands), dtype=np.float64))

    sd_node_ids = np.array(showdown_ids, dtype=np.int32)
    n_sd = len(showdown_ids)

    hero_strat_sum, _, root_ev = _cfr_pernode_iterations(
        flat['node_player'], flat['node_terminal'],
        flat['node_hero_pot'], flat['node_opp_pot'],
        flat['node_num_actions'], flat['children'],
        flat['hero_node_map'], flat['opp_node_map'],
        showdown_values, valid_pairs,
        n_hands, n_hands,
        n_iterations, n_hero_nodes, n_opp_nodes,
        flat['max_actions'], flat['n_nodes'],
        sd_node_ids, n_sd,
    )

    hero_strategy = _normalize_strategy_numba(
        hero_strat_sum, flat['hero_node_num_actions'], n_hands)

    return hero_strategy, root_ev


# ---------------------------------------------------------------------------
# Showdown value computation
# ---------------------------------------------------------------------------

def _river_showdown_values(tree, hands, board_5, valid_pairs, engine):
    """
    Compute per-terminal showdown values for a river board.

    On the river, hands are compared deterministically. At each showdown
    terminal with pot = min(hero_pot, opp_pot):
        hero wins:  +pot
        hero loses: -pot
        tie:        0
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

    for si, nid in enumerate(sd_ids):
        pot_won = min(tree.hero_pot[nid], tree.opp_pot[nid])
        for hi in range(n):
            for oi in range(n):
                if not valid_pairs[hi, oi]:
                    continue
                ri, ro = hand_ranks[hi], hand_ranks[oi]
                if ri < ro:
                    sd_vals[si, hi, oi] = pot_won
                elif ri > ro:
                    sd_vals[si, hi, oi] = -pot_won

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

    for si, nid in enumerate(sd_ids):
        actual_pot = tree.hero_pot[nid] + tree.opp_pot[nid]
        scale = actual_pot / ref_pot if ref_pot > 0 else 1.0

        for hi in range(n_hands):
            for oi in range(n_hands):
                if not valid_pairs[hi, oi]:
                    continue
                sd_vals[si, hi, oi] = continuation_ev[hi, oi] * scale

    return sd_vals, sd_ids


# ---------------------------------------------------------------------------
# Main backward induction solver
# ---------------------------------------------------------------------------

def solve_flop_board(board_3_cards, engine, n_iterations=500, pot_sizes=None):
    """
    Solve a complete flop board using backward induction.

    For each pot size, chains river -> turn -> flop solutions so that
    each street's strategy accounts for optimal play on future streets.

    Args:
        board_3_cards: list of 3 ints (flop cards)
        engine: ExactEquityEngine instance
        n_iterations: CFR iterations per street
        pot_sizes: list of (hero_bet, opp_bet) tuples

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

    # Iteration counts per street. River/turn sub-solves use fewer iterations
    # because they only need approximate EVs for backward induction (the
    # flop strategy, which is what we're extracting, gets full iterations).
    # With 500 total: river=50, turn=100, flop=500.
    river_iters = max(10, n_iterations // 10)
    turn_iters = max(20, n_iterations // 5)
    flop_iters = n_iterations

    t_total = time.time()
    logger.info("Multi-street solve: board=%s, %d hands, %d remaining",
                board_3, n_hands, len(remaining))

    all_strategies = []
    all_trees = []

    for pot_idx, (hero_bet, opp_bet) in enumerate(pot_sizes):
        t_pot = time.time()
        logger.info("  Pot %d/%d: (%d,%d)",
                    pot_idx + 1, len(pot_sizes), hero_bet, opp_bet)

        # -------------------------------------------------------
        # Phase 1: Solve all rivers, aggregate into turn EVs
        # -------------------------------------------------------
        # For each turn card t, solve all river boards (flop+t+r)
        # and average root EVs over river cards r.
        turn_data = {}  # turn_card -> (turn_hands, turn_idx, valid, cont_ev)

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

                r_tree = GameTree(hero_bet, opp_bet, DEFAULT_MIN_RAISE,
                                  DEFAULT_MAX_BET, True)
                r_sd_vals, r_sd_ids = _river_showdown_values(
                    r_tree, r_hands, board_5, r_valid, engine)

                _, r_root_ev = _solve_street(
                    r_tree, r_hands, r_sd_vals, r_sd_ids,
                    r_valid, river_iters)

                # Map river EVs to turn hand indices
                r_idx = {h: i for i, h in enumerate(r_hands)}
                for hi_r, hh in enumerate(r_hands):
                    hi_t = turn_idx.get(hh)
                    if hi_t is None:
                        continue
                    for oi_r, oh in enumerate(r_hands):
                        oi_t = turn_idx.get(oh)
                        if oi_t is None:
                            continue
                        if not r_valid[hi_r, oi_r]:
                            continue
                        ev_sum[hi_t, oi_t] += r_root_ev[hi_r, oi_r]
                        ev_count[hi_t, oi_t] += 1.0

            # Average to get turn continuation values
            cont_ev = np.zeros((n_turn, n_turn), dtype=np.float64)
            mask = ev_count > 0
            cont_ev[mask] = ev_sum[mask] / ev_count[mask]

            turn_data[turn_card] = (turn_hands, turn_idx, turn_valid, cont_ev)

        logger.info("    Phase 1 (rivers) done: %.1fs", time.time() - t_pot)

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

            _, t_root_ev = _solve_street(
                t_tree, t_hands, t_sd_vals, t_sd_ids,
                t_valid, turn_iters)

            # Map turn EVs to flop hand indices
            for hi_t, hh in enumerate(t_hands):
                hi_f = hand_index.get(hh)
                if hi_f is None:
                    continue
                for oi_t, oh in enumerate(t_hands):
                    oi_f = hand_index.get(oh)
                    if oi_f is None:
                        continue
                    if not t_valid[hi_t, oi_t]:
                        continue
                    flop_ev_sum[hi_f, oi_f] += t_root_ev[hi_t, oi_t]
                    flop_ev_count[hi_f, oi_f] += 1.0

        logger.info("    Phase 2 (turns) done: %.1fs", time.time() - t_phase2)

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

    return {
        'flop_strategies': strategies,
        'hands': hands,
        'board': board_3,
        'pot_sizes': pot_sizes,
        'board_features': board_features,
        'action_types': action_types,
    }


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
