"""
Unbucketed CFR solver: every individual hand gets its own strategy.

Instead of grouping hands into equity buckets (where hands in the same bucket
share a single strategy), this solver treats each possible 2-card hand as its
own information set. For a flop board with 3 community cards, there are
C(24,2)=276 possible hero hands and C(22,2)=231 possible opponent hands per
hero hand.

The Numba JIT inner loop is modeled on blueprint_cfr_fast.py but iterates
over (hero_hand, opp_hand) pairs instead of (hero_bucket, opp_bucket) pairs.

Performance characteristics:
  - Flop: 276 hero hands x ~231 avg opp hands = ~63,756 pairs per iteration
  - Turn: 253 hero hands x ~210 avg opp hands = ~53,130 pairs
  - River: 231 hero hands x ~190 avg opp hands = ~43,890 pairs (instant equity)
  - Compared to 30x30=900 bucket pairs in the bucketed version
  - But each pair traversal is identical cost, so ~70x more work per iteration
  - Compensated by needing fewer iterations for convergence (exact equity)
"""

import os
import sys
import time
import itertools
import logging

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for Numba (module-level for njit visibility)
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
                  hero_regrets_h, hero_strategy_sum_h,
                  opp_regrets_o, opp_strategy_sum_o,
                  max_actions, hero_reach_init, opp_reach_init,
                  n_nodes):
    """
    Iterative post-order CFR+ traversal for one (hero_hand, opp_hand) pair.

    Identical to blueprint_cfr_fast._traverse_cfr, but here each "bucket"
    is a single hand. The regret/strategy arrays passed in are slices for
    one specific hero hand and one specific opp hand.

    Args:
        hero_regrets_h: 2D array (n_hero_nodes, max_actions) for this hero hand
        hero_strategy_sum_h: 2D array (n_hero_nodes, max_actions) for this hero hand
        opp_regrets_o: 2D array (n_opp_nodes, max_actions) for this opp hand
        opp_strategy_sum_o: 2D array (n_opp_nodes, max_actions) for this opp hand

    Returns:
        float: hero's expected value at the root
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
    stack_hero_reach[0] = hero_reach_init
    stack_opp_reach[0] = opp_reach_init

    root_value = 0.0

    while sp >= 0:
        node_id = stack_node[sp]
        cur_action = stack_action[sp]
        hero_reach = stack_hero_reach[sp]
        opp_reach = stack_opp_reach[sp]

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
                new_hero_reach = hero_reach * strategy[0]
                new_opp_reach = opp_reach
            else:
                new_hero_reach = hero_reach
                new_opp_reach = opp_reach * strategy[0]

            sp += 1
            stack_node[sp] = child_id
            stack_action[sp] = -1
            stack_hero_reach[sp] = new_hero_reach
            stack_opp_reach[sp] = new_opp_reach
        else:
            child_val = stack_action_values[sp, cur_action]
            stack_node_value[sp] += stack_strategy[sp, cur_action] * child_val

            next_action = cur_action + 1
            if next_action < n_act:
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
                node_val = stack_node_value[sp]

                if player == 0:
                    idx = hero_node_map[node_id]
                    for a in range(n_act):
                        regret = opp_reach * (stack_action_values[sp, a] - node_val)
                        new_regret = hero_regrets_h[idx, a] + regret
                        if new_regret > 0.0:
                            hero_regrets_h[idx, a] = new_regret
                        else:
                            hero_regrets_h[idx, a] = 0.0
                    for a in range(n_act):
                        hero_strategy_sum_h[idx, a] += hero_reach * stack_strategy[sp, a]
                else:
                    idx = opp_node_map[node_id]
                    for a in range(n_act):
                        regret = hero_reach * (node_val - stack_action_values[sp, a])
                        new_regret = opp_regrets_o[idx, a] + regret
                        if new_regret > 0.0:
                            opp_regrets_o[idx, a] = new_regret
                        else:
                            opp_regrets_o[idx, a] = 0.0
                    for a in range(n_act):
                        opp_strategy_sum_o[idx, a] += opp_reach * stack_strategy[sp, a]

                sp -= 1
                if sp >= 0:
                    parent_action = stack_action[sp]
                    stack_action_values[sp, parent_action] = node_val
                else:
                    root_value = node_val

    return root_value


@numba.njit(cache=True)
def _cfr_iterations_unbucketed(
    node_player, node_terminal, node_hero_pot, node_opp_pot,
    node_num_actions, children, hero_node_map, opp_node_map,
    equity_matrix, valid_pairs,
    n_hero_hands, n_opp_hands,
    n_iterations, n_hero_nodes, n_opp_nodes,
    max_actions, n_nodes,
):
    """
    Run all CFR+ iterations for unbucketed hands entirely in compiled code.

    Args:
        equity_matrix: float64 (n_hero_hands, n_opp_hands) - pairwise equity
        valid_pairs: bool (n_hero_hands, n_opp_hands) - non-overlapping pairs
        n_hero_hands: number of distinct hero hands
        n_opp_hands: number of distinct opp hands (same set, but indexed separately)
        n_iterations: number of CFR iterations to run
        n_hero_nodes, n_opp_nodes: decision node counts
        max_actions: max actions at any node
        n_nodes: total nodes in tree

    Returns:
        (hero_strategy_sum, opp_strategy_sum) - both float64 3D arrays
        hero_strategy_sum: (n_hero_hands, n_hero_nodes, max_actions)
        opp_strategy_sum: (n_opp_hands, n_opp_nodes, max_actions)
    """
    hero_regrets = np.zeros((n_hero_hands, n_hero_nodes, max_actions), dtype=np.float64)
    hero_strategy_sum = np.zeros((n_hero_hands, n_hero_nodes, max_actions), dtype=np.float64)
    opp_regrets = np.zeros((n_opp_hands, n_opp_nodes, max_actions), dtype=np.float64)
    opp_strategy_sum = np.zeros((n_opp_hands, n_opp_nodes, max_actions), dtype=np.float64)

    for t in range(n_iterations):
        for hi in range(n_hero_hands):
            for oi in range(n_opp_hands):
                if not valid_pairs[hi, oi]:
                    continue

                eq = equity_matrix[hi, oi]

                _traverse_cfr(
                    node_player, node_terminal, node_hero_pot, node_opp_pot,
                    node_num_actions, children, hero_node_map, opp_node_map,
                    eq,
                    hero_regrets[hi], hero_strategy_sum[hi],
                    opp_regrets[oi], opp_strategy_sum[oi],
                    max_actions, 1.0, 1.0,
                    n_nodes,
                )

    return hero_strategy_sum, opp_strategy_sum


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
# Tree flattening (same as blueprint_cfr_fast)
# ---------------------------------------------------------------------------

def _flatten_tree(tree):
    """Convert a GameTree to flat numpy arrays for Numba."""
    n_nodes = tree.size
    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    max_actions = 1
    for nid in tree.hero_node_ids:
        if tree.num_actions[nid] > max_actions:
            max_actions = tree.num_actions[nid]
    for nid in tree.opp_node_ids:
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
# Equity precomputation
# ---------------------------------------------------------------------------

def precompute_equity_matrix(all_hands, board, dead_cards, equity_engine):
    """
    Precompute the full pairwise equity matrix for all hands.

    Args:
        all_hands: sorted list of (c1, c2) tuples
        board: list of board card ints
        dead_cards: list of dead card ints
        equity_engine: ExactEquityEngine instance

    Returns:
        equity_matrix: np.array (n_hands, n_hands), equity[i,j] = hero i vs opp j
        valid_pairs: np.array (n_hands, n_hands) bool, True if hands don't overlap
    """
    n = len(all_hands)
    equity_matrix = np.full((n, n), 0.5, dtype=np.float64)
    valid_pairs = np.ones((n, n), dtype=np.bool_)

    seven_lookup = equity_engine._seven
    board_mask = 0
    for c in board:
        board_mask |= 1 << c

    board_needed = 5 - len(board)
    known = set(board) | set(dead_cards)
    remaining_base = [c for c in range(27) if c not in known]

    # Mark invalid (overlapping) pairs and same-hand diagonal
    for i in range(n):
        hi_set = set(all_hands[i])
        valid_pairs[i, i] = False
        for j in range(i + 1, n):
            if hi_set & set(all_hands[j]):
                valid_pairs[i, j] = False
                valid_pairs[j, i] = False

    if board_needed == 0:
        # River: deterministic equity
        hand_ranks = np.zeros(n, dtype=np.int64)
        for i, hand in enumerate(all_hands):
            mask = (1 << hand[0]) | (1 << hand[1]) | board_mask
            hand_ranks[i] = seven_lookup[mask]

        for i in range(n):
            for j in range(i + 1, n):
                if not valid_pairs[i, j]:
                    continue
                ri = hand_ranks[i]
                rj = hand_ranks[j]
                if ri < rj:
                    equity_matrix[i, j] = 1.0
                    equity_matrix[j, i] = 0.0
                elif ri == rj:
                    equity_matrix[i, j] = 0.5
                    equity_matrix[j, i] = 0.5
                else:
                    equity_matrix[i, j] = 0.0
                    equity_matrix[j, i] = 1.0
    else:
        # Flop (2 runout cards) or Turn (1 runout card)
        for i in range(n):
            ha = all_hands[i]
            ha_set = set(ha)
            ha_mask = (1 << ha[0]) | (1 << ha[1])

            for j in range(i + 1, n):
                if not valid_pairs[i, j]:
                    continue

                hb = all_hands[j]
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

                eq = wins_a / total if total > 0 else 0.5
                equity_matrix[i, j] = eq
                equity_matrix[j, i] = 1.0 - eq

    return equity_matrix, valid_pairs


# ---------------------------------------------------------------------------
# Public solve function
# ---------------------------------------------------------------------------

def solve(board, dead_cards, hero_bet, opp_bet, hero_first,
          n_iterations, min_raise=2, max_bet=100, equity_engine=None):
    """
    Run unbucketed CFR+ where every individual hand gets its own strategy.

    Args:
        board: list of card ints (3-5 community cards)
        dead_cards: list of card ints (discards, removed from play)
        hero_bet: hero's current cumulative bet
        opp_bet: opponent's current cumulative bet
        hero_first: True if hero acts first
        n_iterations: number of CFR iterations to run
        min_raise: minimum raise increment
        max_bet: maximum total bet per player
        equity_engine: ExactEquityEngine instance. If None, creates one.

    Returns:
        dict with:
            'hero_strategy': np.array (n_hero_hands, n_hero_nodes, max_actions)
            'opp_strategy': np.array (n_opp_hands, n_opp_nodes, max_actions)
            'hero_hands': list of (c1, c2) tuples (sorted, deterministic order)
            'opp_hands': list of (c1, c2) tuples (same set as hero_hands)
            'tree': GameTree instance
            'n_iterations': number of iterations completed
    """
    if equity_engine is None:
        from equity import ExactEquityEngine
        equity_engine = ExactEquityEngine()

    # Build game tree
    tree = GameTree(hero_bet, opp_bet, min_raise, max_bet, hero_first)
    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    # Enumerate all valid hands (sorted for deterministic ordering)
    known = set(board) | set(dead_cards)
    all_hands = sorted(itertools.combinations(
        [c for c in range(27) if c not in known], 2
    ))
    n_hands = len(all_hands)

    logger.info("Unbucketed CFR: board=%s, %d hands, %d hero_nodes, %d opp_nodes, "
                "%d iterations", board, n_hands, n_hero_nodes, n_opp_nodes, n_iterations)

    if n_hero_nodes == 0 and n_opp_nodes == 0:
        return {
            'hero_strategy': np.array([]),
            'opp_strategy': np.array([]),
            'hero_hands': all_hands,
            'opp_hands': all_hands,
            'tree': tree,
            'n_iterations': 0,
        }

    # Flatten tree
    flat = _flatten_tree(tree)

    # Precompute pairwise equity matrix
    t0 = time.time()
    equity_matrix, valid_pairs = precompute_equity_matrix(
        all_hands, board, dead_cards, equity_engine)
    t_eq = time.time() - t0

    n_valid = int(valid_pairs.sum())
    logger.info("Equity matrix: %d x %d, %d valid pairs, computed in %.2fs",
                n_hands, n_hands, n_valid, t_eq)

    # Run CFR iterations (Numba JIT)
    t0 = time.time()
    hero_strategy_sum, opp_strategy_sum = _cfr_iterations_unbucketed(
        flat['node_player'], flat['node_terminal'],
        flat['node_hero_pot'], flat['node_opp_pot'],
        flat['node_num_actions'], flat['children'],
        flat['hero_node_map'], flat['opp_node_map'],
        equity_matrix, valid_pairs,
        n_hands, n_hands,
        n_iterations, n_hero_nodes, n_opp_nodes,
        flat['max_actions'], flat['n_nodes'],
    )
    t_cfr = time.time() - t0
    logger.info("CFR iterations done in %.2fs (%.4fs/iter)",
                t_cfr, t_cfr / max(n_iterations, 1))

    # Normalize strategy sums
    hero_strategy = _normalize_strategy_numba(
        hero_strategy_sum, flat['hero_node_num_actions'], n_hands)
    opp_strategy = _normalize_strategy_numba(
        opp_strategy_sum, flat['opp_node_num_actions'], n_hands)

    return {
        'hero_strategy': hero_strategy,
        'opp_strategy': opp_strategy,
        'hero_hands': all_hands,
        'opp_hands': all_hands,
        'tree': tree,
        'n_iterations': n_iterations,
    }


# ---------------------------------------------------------------------------
# Warmup: trigger JIT compilation
# ---------------------------------------------------------------------------

def warmup_jit():
    """Trigger Numba JIT compilation of all @njit functions."""
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
    equity_matrix = np.array([[0.5]], dtype=np.float64)
    valid_pairs = np.array([[True]], dtype=np.bool_)

    _cfr_iterations_unbucketed(
        node_player, node_terminal, node_hero_pot, node_opp_pot,
        node_num_actions, children, hero_node_map, opp_node_map,
        equity_matrix, valid_pairs,
        1, 1, 1, 1, 0, 2, 3,
    )

    # Also warmup normalization
    dummy = np.zeros((1, 1, 2), dtype=np.float64)
    dummy_nact = np.array([2], dtype=np.int32)
    _normalize_strategy_numba(dummy, dummy_nact, 1)


# ---------------------------------------------------------------------------
# CLI benchmark
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    from equity import ExactEquityEngine

    print("=" * 60)
    print("Unbucketed CFR Solver Benchmark")
    print("=" * 60)
    print()

    engine = ExactEquityEngine()

    # Test with a river board (fastest equity, good for validation)
    board = [0, 1, 2, 9, 18]
    dead_cards = []
    n_iterations = 200

    print(f"Board (river): {board}")
    print(f"Iterations: {n_iterations}")
    print()

    print("Warming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    warmup_jit()
    print(f" done ({time.time() - t0:.2f}s)")
    print()

    print("Running unbucketed CFR...", end="", flush=True)
    t0 = time.time()
    result = solve(
        board=board, dead_cards=dead_cards,
        hero_bet=5, opp_bet=5, hero_first=True,
        n_iterations=n_iterations,
        equity_engine=engine,
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.3f}s")

    hs = result['hero_strategy']
    print(f"Hero strategy shape: {hs.shape}")
    print(f"Hero hands: {len(result['hero_hands'])}")

    # Validate probabilities sum to ~1
    if len(hs) > 0:
        valid = True
        for h in range(min(10, hs.shape[0])):
            for n in range(hs.shape[1]):
                s = hs[h, n, :].sum()
                if s > 0 and abs(s - 1.0) > 0.01:
                    valid = False
                    print(f"  WARNING: hand {h} node {n} sum={s:.4f}")
        print(f"Strategy probabilities valid: {valid}")

    # Test with flop
    print()
    board_flop = [0, 1, 2]
    n_iterations_flop = 50
    print(f"Board (flop): {board_flop}")
    print(f"Iterations: {n_iterations_flop}")

    known = set(board_flop)
    n_possible = len(list(itertools.combinations(
        [c for c in range(27) if c not in known], 2)))
    print(f"Possible hands: {n_possible}")

    print("Running unbucketed CFR on flop...", end="", flush=True)
    t0 = time.time()
    result_flop = solve(
        board=board_flop, dead_cards=dead_cards,
        hero_bet=5, opp_bet=5, hero_first=True,
        n_iterations=n_iterations_flop,
        equity_engine=engine,
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.3f}s")
    print(f"Hero strategy shape: {result_flop['hero_strategy'].shape}")

    print()
    print("Done.")
