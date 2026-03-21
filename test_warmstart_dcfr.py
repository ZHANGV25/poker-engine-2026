#!/usr/bin/env python3
"""
Comprehensive test: Warm-started DCFR vs Cold-start DCFR on the river.

Compares 3 approaches:
  A) Cold-start: standard DCFR from zero regrets (200 iterations)
  B) Warm-start HERO: initialize hero regrets from precomputed strategy, then 100 iters
  C) Warm-start OPPONENT: lock opponent strategy to precomputed, hero solves best-response

All solve against a NARROWED opponent range (non-uniform weights).
"""

import sys
sys.path.insert(0, 'submission')

import numpy as np
import itertools
import time
import os

from river_lookup import RiverLookup
from range_solver import RangeSolver
from equity import ExactEquityEngine
from game_tree import (
    GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)
from math import pow as fpow

# ─── Setup ───────────────────────────────────────────────────────────────

extract_dir = '/tmp/river_warmstart_test'
lookup = RiverLookup(os.path.join(extract_dir, 'river'))
engine = ExactEquityEngine()
solver = RangeSolver(engine)

np.random.seed(42)

# ─── Helper: generate narrowed opponent range ────────────────────────────

def make_narrowed_range(board, dead_cards, equity_engine):
    """Create a non-uniform opponent range: remove 30% of hands randomly,
    upweight high-equity hands to simulate Bayesian narrowing."""
    known = set(board) | set(dead_cards)
    remaining = [c for c in range(27) if c not in known]
    all_hands = list(itertools.combinations(remaining, 2))

    # Compute equity for each hand
    board_mask = 0
    for c in board:
        board_mask |= 1 << c
    seven = equity_engine._seven

    equities = {}
    for h in all_hands:
        h_mask = (1 << h[0]) | (1 << h[1]) | board_mask
        rank = seven.get(h_mask, 9999)
        # Approximate equity: fraction of other hands we beat
        equities[h] = rank  # lower rank = stronger

    # Convert ranks to equities (fraction of hands beaten)
    all_ranks = list(equities.values())
    max_rank = max(all_ranks)
    min_rank = min(all_ranks)
    for h in all_hands:
        if max_rank > min_rank:
            equities[h] = 1.0 - (equities[h] - min_rank) / (max_rank - min_rank)
        else:
            equities[h] = 0.5

    # Remove 30% of hands randomly
    n_remove = int(0.3 * len(all_hands))
    remove_indices = np.random.choice(len(all_hands), n_remove, replace=False)
    remove_set = set(remove_indices)

    # Build weighted range: upweight high-equity hands
    opp_range = {}
    for i, h in enumerate(all_hands):
        if i in remove_set:
            continue
        key = (min(h[0], h[1]), max(h[0], h[1]))
        # Weight = 0.3 + 0.7 * equity (higher equity = more likely to be in range)
        w = 0.3 + 0.7 * equities[h]
        opp_range[key] = w

    # Normalize
    total = sum(opp_range.values())
    for k in opp_range:
        opp_range[k] /= total

    return opp_range

# ─── Helper: run DCFR with optional warm-start ──────────────────────────

def run_dcfr_with_warmstart(tree, opp_weights, terminal_values,
                             n_hero, n_opp, iterations,
                             hero_warmstart_regrets=None,
                             opp_locked_strategy=None):
    """Run DCFR with optional hero regret warm-start or opponent locking.

    Returns (average_strategy_at_root, hero_strat_sum) for analysis.
    Also returns intermediate checkpoints.
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

    # Apply warm-start to hero regrets if provided
    if hero_warmstart_regrets is not None:
        # hero_warmstart_regrets: (n_hero, n_act_at_root) - only for root node
        root_idx = hero_node_idx.get(0)
        if root_idx is not None:
            n_act = tree.num_actions[0]
            hero_regrets[root_idx, :, :n_act] = hero_warmstart_regrets[:, :n_act]

    hero_reach_init = np.ones(n_hero, dtype=np.float64) / n_hero

    alpha, beta, gamma = 1.5, 0.0, 2.0

    checkpoints = {}  # iteration -> root strategy

    for t in range(1, iterations + 1):
        if t > 1:
            pos_w = fpow(t - 1, alpha) / (fpow(t - 1, alpha) + 1.0)
            neg_w = fpow(t - 1, beta) / (fpow(t - 1, beta) + 1.0)
            strat_w = fpow((t - 1) / t, gamma)

            hero_regrets *= np.where(hero_regrets > 0, pos_w, neg_w)
            if opp_locked_strategy is None:
                opp_regrets *= np.where(opp_regrets > 0, pos_w, neg_w)
            hero_strat_sum *= strat_w

        solver._range_cfr_traverse(
            tree, 0, hero_reach_init.copy(), opp_weights.copy(),
            hero_regrets, hero_strat_sum, opp_regrets,
            hero_node_idx, opp_node_idx, terminal_values,
            n_hero, n_opp, max_act,
            opp_locked_strategy=opp_locked_strategy)

        # Checkpoint at specific iterations
        if t in (50, 100, 150, 200):
            root = 0
            if root in hero_node_idx:
                idx = hero_node_idx[root]
                n_act = tree.num_actions[root]
                strat_slice = hero_strat_sum[idx, :, :n_act].copy()
                totals = strat_slice.sum(axis=1, keepdims=True)
                cp_strat = np.where(totals > 0,
                                    strat_slice / np.maximum(totals, 1e-10),
                                    np.full_like(strat_slice, 1.0 / n_act))
                checkpoints[t] = cp_strat

    # Extract final average strategy at root
    root = 0
    if root not in hero_node_idx:
        return np.ones((n_hero, 1)) / 1, checkpoints

    idx = hero_node_idx[root]
    n_act = tree.num_actions[root]
    strat_slice = hero_strat_sum[idx, :, :n_act]
    totals = strat_slice.sum(axis=1, keepdims=True)
    result = np.where(totals > 0, strat_slice / np.maximum(totals, 1e-10),
                     np.full_like(strat_slice, 1.0 / n_act))
    return result, checkpoints

# ─── Helper: build warm-start regrets from precomputed strategy ──────────

def build_warmstart_regrets(lookup, board, hero_hands, tree, pot_idx=0, scale=10.0):
    """Convert precomputed strategy into regret values that reproduce it via regret matching.

    For regret matching: if regrets = [r0, r1, ...], then
    strategy[i] = max(ri, 0) / sum(max(rj, 0))

    To get target strategy [p0, p1, ...], set regrets = [p0*scale, p1*scale, ...]
    (all positive, so regret matching gives back the same proportions).
    """
    bd = lookup._lazy_load(board)
    if bd is None:
        return None

    n_act_root = tree.num_actions[0]
    root_children = tree.children[0]
    n_hero = len(hero_hands)

    warmstart = np.zeros((n_hero, n_act_root), dtype=np.float64)

    # Map precomputed action types to tree action indices
    strat_key = f'strat_{pot_idx}'
    acts_key = f'acts_{pot_idx}'
    if strat_key not in bd:
        return None

    precomp_strats = bd[strat_key].astype(np.float64) / 255.0
    precomp_acts = bd[acts_key]

    # Build mapping: precomputed action index -> tree action index
    act_map = {}
    for pai in range(len(precomp_acts)):
        pa = int(precomp_acts[pai])
        for tai, (tree_act, _) in enumerate(root_children):
            if tree_act == pa:
                act_map[pai] = tai
                break

    for hi, h in enumerate(hero_hands):
        h_sorted = (min(h[0], h[1]), max(h[0], h[1]))
        lookup_idx = bd['hand_map'].get(h_sorted)
        if lookup_idx is None:
            # Hand not in lookup (blocked by board) - uniform
            warmstart[hi] = scale / n_act_root
            continue

        strat = precomp_strats[lookup_idx]
        total = strat.sum()
        if total > 0:
            strat = strat / total
        else:
            strat = np.ones(len(strat)) / len(strat)

        for pai, tai in act_map.items():
            warmstart[hi, tai] = strat[pai] * scale

    return warmstart

# ─── Helper: build opponent locked strategy from precomputed ─────────────

def build_opp_locked_from_precomputed(lookup, board, opp_hands, tree, pot_idx=0):
    """Build opponent locked strategy from precomputed P(bet|hand).

    For opponent nodes in the tree:
    - At check/bet node: use precomputed P(bet|hand) to set check/bet probabilities
    - At facing-bet node: use equity-proportional heuristic (fold weak, call/raise strong)

    This is Noam Brown's approach: warm-start OPPONENT's strategy.
    """
    bd = lookup._lazy_load(board)
    if bd is None:
        return None

    n_opp = len(opp_hands)

    # Get P(bet|hand) from precomputed
    pbet_key = f'pbet_{pot_idx}'
    if pbet_key not in bd:
        return None

    pb_all = bd[pbet_key].astype(np.float64) / 255.0

    # Build per-hand P(bet)
    opp_pbet = np.zeros(n_opp, dtype=np.float64)
    for oi, h in enumerate(opp_hands):
        h_sorted = (min(h[0], h[1]), max(h[0], h[1]))
        lookup_idx = bd['hand_map'].get(h_sorted)
        if lookup_idx is not None:
            opp_pbet[oi] = pb_all[lookup_idx]
        else:
            opp_pbet[oi] = 0.5

    locked = {}
    for node_id in tree.opp_node_ids:
        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        strat = np.zeros((n_opp, n_act), dtype=np.float64)

        hero_pot = tree.hero_pot[node_id]
        opp_pot = tree.opp_pot[node_id]
        facing_bet = (hero_pot > opp_pot)

        if facing_bet:
            # Facing hero's bet: use P(bet) as proxy for hand strength
            # High P(bet) hands are strong -> more likely to call/raise
            fold_idx = None
            call_idx = None
            raise_indices = []
            for ai, (act_type, _) in enumerate(children):
                if act_type == ACT_FOLD:
                    fold_idx = ai
                elif act_type == ACT_CALL:
                    call_idx = ai
                else:
                    raise_indices.append(ai)

            for o in range(n_opp):
                strength = opp_pbet[o]  # proxy for hand strength
                if fold_idx is not None:
                    strat[o, fold_idx] = max(0.0, 1.0 - strength * 1.5)
                if call_idx is not None:
                    strat[o, call_idx] = min(strength * 1.2, 1.0)
                if raise_indices:
                    raise_prob = max(0.0, strength - 0.7) * 2.0
                    for ri in raise_indices:
                        strat[o, ri] = raise_prob / len(raise_indices)
        else:
            # Check/bet node: use precomputed P(bet|hand) directly
            check_idx = None
            bet_indices = []
            for ai, (act_type, _) in enumerate(children):
                if act_type == ACT_CHECK:
                    check_idx = ai
                else:
                    bet_indices.append(ai)

            for o in range(n_opp):
                pb = opp_pbet[o]
                if check_idx is not None:
                    strat[o, check_idx] = max(0.05, 1.0 - pb)
                if bet_indices:
                    per_bet = pb / len(bet_indices)
                    for bi in bet_indices:
                        strat[o, bi] = per_bet

        # Normalize
        row_sums = strat.sum(axis=1, keepdims=True)
        strat = np.where(row_sums > 0,
                         strat / np.maximum(row_sums, 1e-10),
                         np.full_like(strat, 1.0 / n_act))
        locked[node_id] = strat

    return locked

# ─── Helper: compute EV of a strategy ───────────────────────────────────

def compute_strategy_ev(strategy, equity_matrix, not_blocked, opp_weights,
                        tree, n_hero, n_opp):
    """Compute expected value for each hero hand under the given root strategy.

    Uses the tree structure: for each action, compute the expected payoff
    weighted by opponent range.

    Simpler approach: EV = sum over opp hands of
      (opp_weight * not_blocked * strategy-weighted terminal value)

    For acting-first (check/bet):
    - CHECK: EV = equity * pot (goes to showdown after both check or opp checks back)
    - BET: EV = P(fold)*pot + P(call)*equity_vs_callers*(2*bet)

    But this is complex. Instead, use the solver's own logic:
    compute the counterfactual value at the root for the given strategy.

    Actually, simplest: just compute EV = sum_over_actions(strategy[a] * action_EV[a])
    where action_EV is derived from equity.
    """
    # Use equity matrix directly for a simpler EV computation
    # EV per hero hand = sum_o(w_o * nb[h,o] * (2*eq[h,o] - 1) * pot_size) for check
    # For bet, it's more complex. Let's use a unified approach.

    # Actually, let's compute the VALUE of the strategy by doing one pass through
    # the tree with the fixed strategy. This is the "game value" approach.

    root_children = tree.children[0]
    n_act = tree.num_actions[0]

    # Normalize opponent weights with blocking
    hero_ev = np.zeros(n_hero, dtype=np.float64)

    for hi in range(n_hero):
        ev = 0.0
        for ai in range(n_act):
            act_type, child_id = root_children[ai]
            p_act = strategy[hi, ai]
            if p_act < 1e-10:
                continue

            term = tree.terminal[child_id]
            if term == TERM_SHOWDOWN:
                # Both checked or call happened - showdown
                pot_won = min(tree.hero_pot[child_id], tree.opp_pot[child_id])
                hero_invest = tree.hero_pot[child_id]

                # EV = sum_o(w_o * nb * (2*eq - 1) * pot_won)
                weights = opp_weights * not_blocked[hi]
                w_sum = weights.sum()
                if w_sum > 0:
                    eq = (equity_matrix[hi] * weights).sum() / w_sum
                    act_ev = (2 * eq - 1) * pot_won
                else:
                    act_ev = 0.0
            elif term == TERM_FOLD_OPP:
                # Opponent folds - we win their bet
                act_ev = tree.opp_pot[child_id]
            elif term == TERM_FOLD_HERO:
                # We fold - lose our bet
                act_ev = -tree.hero_pot[child_id]
            elif term == TERM_NONE:
                # Non-terminal child = opponent decision node
                # Need to recurse. For simplicity, approximate using subtree.
                act_ev = _compute_subtree_ev(
                    tree, child_id, hi, equity_matrix, not_blocked,
                    opp_weights, n_opp)
            else:
                act_ev = 0.0

            ev += p_act * act_ev
        hero_ev[hi] = ev

    return hero_ev


def _compute_subtree_ev(tree, node_id, hero_idx, equity_matrix, not_blocked,
                         opp_weights, n_opp):
    """Approximate EV at a subtree node for a single hero hand.

    For opponent nodes, assume opponent plays uniformly (or use equity-proportional).
    For terminal nodes, compute directly.
    """
    term = tree.terminal[node_id]
    if term == TERM_SHOWDOWN:
        pot_won = min(tree.hero_pot[node_id], tree.opp_pot[node_id])
        weights = opp_weights * not_blocked[hero_idx]
        w_sum = weights.sum()
        if w_sum > 0:
            eq = (equity_matrix[hero_idx] * weights).sum() / w_sum
            return (2 * eq - 1) * pot_won
        return 0.0
    elif term == TERM_FOLD_OPP:
        return tree.opp_pot[node_id]
    elif term == TERM_FOLD_HERO:
        return -tree.hero_pot[node_id]
    elif term != TERM_NONE:
        return 0.0

    # Non-terminal opponent node: approximate with uniform play
    n_act = tree.num_actions[node_id]
    children = tree.children[node_id]
    if n_act == 0:
        return 0.0

    total_ev = 0.0
    for ai in range(n_act):
        _, child_id = children[ai]
        child_ev = _compute_subtree_ev(tree, child_id, hero_idx,
                                        equity_matrix, not_blocked,
                                        opp_weights, n_opp)
        total_ev += child_ev / n_act

    return total_ev

# ─── Helper: range-weighted EV ───────────────────────────────────────────

def range_weighted_ev(hero_ev, hero_hands, board, not_blocked, opp_weights):
    """Compute range-weighted average EV (EV over all hero hands, excluding blocked)."""
    known = set(board)
    total_ev = 0.0
    count = 0
    for hi, h in enumerate(hero_hands):
        if set(h) & known:
            continue
        # Weight by number of valid opponent matchups
        valid_w = (not_blocked[hi] * opp_weights).sum()
        if valid_w > 0:
            total_ev += hero_ev[hi] * valid_w
            count += valid_w
    if count > 0:
        return total_ev / count
    return 0.0

# ─── Main test ───────────────────────────────────────────────────────────

def generate_test_boards(n=30):
    """Generate n random 5-card boards from 27-card deck."""
    all_boards = list(itertools.combinations(range(27), 5))
    indices = np.random.choice(len(all_boards), n, replace=False)
    return [all_boards[i] for i in indices]


def classify_hand(equity, n_hero):
    """Classify hand strength by equity rank."""
    if equity > 0.75:
        return "strong"
    elif equity > 0.55:
        return "medium"
    elif equity > 0.35:
        return "weak"
    elif equity > 0.15:
        return "draw"
    else:
        return "bluff"


def run_test():
    print("=" * 80)
    print("WARM-START DCFR vs COLD-START DCFR: River Strategy Comparison")
    print("=" * 80)
    print()

    boards = generate_test_boards(30)

    # Track results
    results = []
    all_check_to_bet = []  # Cases where warm-start changes CHECK -> BET
    timing = {'cold': [], 'warm_hero': [], 'warm_opp': []}
    convergence_data = {'cold': {}, 'warm_hero': {}, 'warm_opp': {}}
    agreement_count = 0
    total_hands = 0

    my_bet = 4
    opp_bet = 4
    min_raise = 2
    pot_idx = 0  # small pot (4,4)

    for bi, board in enumerate(boards):
        board_list = list(board)
        known = set(board_list)
        dead_cards = []

        print(f"\n--- Board {bi+1}/30: {board_list} ---")

        # Generate narrowed opponent range
        opp_range = make_narrowed_range(board_list, dead_cards, engine)
        if not opp_range:
            print("  Skipping: no valid opponent range")
            continue

        # Setup: hero hands, opp hands, weights
        opp_hands = []
        opp_weights_list = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights_list.append(w)

        opp_weights = np.array(opp_weights_list, dtype=np.float64)
        opp_weights /= opp_weights.sum()

        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        print(f"  Hero hands: {n_hero}, Opp hands: {n_opp}")

        # Build tree (full, hero acts first)
        tree = solver._get_tree(my_bet, opp_bet, min_raise, 100, compact=False)
        if tree.size < 2:
            print("  Skipping: tree too small")
            continue

        # Compute equity matrix
        equity_matrix, not_blocked = solver._compute_equity_and_mask(
            hero_hands, opp_hands, board_list, dead_cards, street=3)

        # Terminal values
        terminal_values = solver._compute_terminal_values(
            tree, equity_matrix, not_blocked)

        n_act = tree.num_actions[0]
        root_children = tree.children[0]
        action_names = []
        for act_type, _ in root_children:
            names = {ACT_CHECK: 'CHECK', ACT_FOLD: 'FOLD', ACT_CALL: 'CALL',
                     ACT_RAISE_HALF: 'BET_40%', ACT_RAISE_POT: 'BET_70%',
                     ACT_RAISE_ALLIN: 'BET_100%', ACT_RAISE_OVERBET: 'BET_150%'}
            action_names.append(names.get(act_type, f'ACT_{act_type}'))

        # ─── Approach A: Cold-start (200 iterations) ─────────────────────
        t0 = time.time()
        cold_strat, cold_cp = run_dcfr_with_warmstart(
            tree, opp_weights, terminal_values,
            n_hero, n_opp, 200)
        t_cold = time.time() - t0
        timing['cold'].append(t_cold)

        # ─── Approach B: Warm-start HERO (100 iterations) ────────────────
        warmstart_regrets = build_warmstart_regrets(
            lookup, board_list, hero_hands, tree, pot_idx=pot_idx, scale=10.0)

        t0 = time.time()
        if warmstart_regrets is not None:
            warm_hero_strat, warm_hero_cp = run_dcfr_with_warmstart(
                tree, opp_weights, terminal_values,
                n_hero, n_opp, 100,
                hero_warmstart_regrets=warmstart_regrets)
        else:
            warm_hero_strat = cold_strat.copy()
            warm_hero_cp = cold_cp
            print("  WARNING: No precomputed data for warm-start hero")
        t_warm_hero = time.time() - t0
        timing['warm_hero'].append(t_warm_hero)

        # ─── Approach C: Warm-start OPPONENT (200 iterations) ────────────
        opp_locked = build_opp_locked_from_precomputed(
            lookup, board_list, opp_hands, tree, pot_idx=pot_idx)

        t0 = time.time()
        if opp_locked is not None:
            warm_opp_strat, warm_opp_cp = run_dcfr_with_warmstart(
                tree, opp_weights, terminal_values,
                n_hero, n_opp, 200,
                opp_locked_strategy=opp_locked)
        else:
            warm_opp_strat = cold_strat.copy()
            warm_opp_cp = cold_cp
            print("  WARNING: No precomputed data for warm-start opp")
        t_warm_opp = time.time() - t0
        timing['warm_opp'].append(t_warm_opp)

        # ─── Compute EVs ──────────────────────────────────────────────────
        cold_ev = compute_strategy_ev(cold_strat, equity_matrix, not_blocked,
                                       opp_weights, tree, n_hero, n_opp)
        warm_hero_ev = compute_strategy_ev(warm_hero_strat, equity_matrix,
                                            not_blocked, opp_weights, tree,
                                            n_hero, n_opp)
        warm_opp_ev = compute_strategy_ev(warm_opp_strat, equity_matrix,
                                           not_blocked, opp_weights, tree,
                                           n_hero, n_opp)

        # Range-weighted EV
        cold_rwev = range_weighted_ev(cold_ev, hero_hands, board_list,
                                       not_blocked, opp_weights)
        warm_hero_rwev = range_weighted_ev(warm_hero_ev, hero_hands, board_list,
                                            not_blocked, opp_weights)
        warm_opp_rwev = range_weighted_ev(warm_opp_ev, hero_hands, board_list,
                                           not_blocked, opp_weights)

        results.append({
            'board': board_list,
            'cold_ev': cold_rwev,
            'warm_hero_ev': warm_hero_rwev,
            'warm_opp_ev': warm_opp_rwev,
            't_cold': t_cold,
            't_warm_hero': t_warm_hero,
            't_warm_opp': t_warm_opp,
        })

        print(f"  Range-weighted EV: Cold={cold_rwev:+.4f}  WarmHero={warm_hero_rwev:+.4f}  WarmOpp={warm_opp_rwev:+.4f}")
        print(f"  Time: Cold={t_cold:.3f}s  WarmHero={t_warm_hero:.3f}s  WarmOpp={t_warm_opp:.3f}s")

        # ─── Per-hand analysis: find CHECK->BET changes ──────────────────
        # Identify check index and bet indices
        check_idx = None
        bet_indices = []
        for ai, (act_type, _) in enumerate(root_children):
            if act_type == ACT_CHECK:
                check_idx = ai
            elif act_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                              ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
                bet_indices.append(ai)

        if check_idx is not None and bet_indices:
            for hi in range(n_hero):
                if set(hero_hands[hi]) & known:
                    continue

                # Compute equity of this hand vs narrowed range
                weights = opp_weights * not_blocked[hi]
                w_sum = weights.sum()
                if w_sum > 0:
                    eq = (equity_matrix[hi] * weights).sum() / w_sum
                else:
                    continue

                cold_check_prob = cold_strat[hi, check_idx]
                cold_bet_prob = cold_strat[hi, bet_indices].sum()
                warm_hero_check = warm_hero_strat[hi, check_idx]
                warm_hero_bet = warm_hero_strat[hi, bet_indices].sum()
                warm_opp_check = warm_opp_strat[hi, check_idx]
                warm_opp_bet = warm_opp_strat[hi, bet_indices].sum()

                # Cold says CHECK (>60%), high equity (>60%), warm says BET
                if cold_check_prob > 0.6 and eq > 0.60:
                    if warm_hero_bet > 0.5 or warm_opp_bet > 0.5:
                        all_check_to_bet.append({
                            'board': board_list,
                            'hand': hero_hands[hi],
                            'equity': eq,
                            'cold_check': cold_check_prob,
                            'cold_bet': cold_bet_prob,
                            'warm_hero_check': warm_hero_check,
                            'warm_hero_bet': warm_hero_bet,
                            'warm_opp_check': warm_opp_check,
                            'warm_opp_bet': warm_opp_bet,
                        })

                # Agreement tracking
                cold_best = np.argmax(cold_strat[hi])
                warm_hero_best = np.argmax(warm_hero_strat[hi])
                warm_opp_best = np.argmax(warm_opp_strat[hi])
                total_hands += 1
                if cold_best == warm_hero_best == warm_opp_best:
                    agreement_count += 1

        # ─── Sample 5 hands for detailed output ─────────────────────────
        # Pick representative hands by equity
        hand_equities = []
        for hi in range(n_hero):
            if set(hero_hands[hi]) & known:
                continue
            weights = opp_weights * not_blocked[hi]
            w_sum = weights.sum()
            if w_sum > 0:
                eq = (equity_matrix[hi] * weights).sum() / w_sum
            else:
                eq = 0.5
            hand_equities.append((hi, eq))

        hand_equities.sort(key=lambda x: -x[1])
        sample_indices = []
        if len(hand_equities) >= 5:
            # Strong, medium-strong, medium, medium-weak, weak
            positions = [0, len(hand_equities)//4, len(hand_equities)//2,
                        3*len(hand_equities)//4, len(hand_equities)-1]
            sample_indices = [hand_equities[p][0] for p in positions]

        if sample_indices:
            print(f"\n  Sample hands (actions: {action_names}):")
            for hi in sample_indices:
                h = hero_hands[hi]
                weights = opp_weights * not_blocked[hi]
                w_sum = weights.sum()
                eq = (equity_matrix[hi] * weights).sum() / w_sum if w_sum > 0 else 0.5

                cold_s = cold_strat[hi, :n_act]
                wh_s = warm_hero_strat[hi, :n_act]
                wo_s = warm_opp_strat[hi, :n_act]

                cold_best = action_names[np.argmax(cold_s)]
                wh_best = action_names[np.argmax(wh_s)]
                wo_best = action_names[np.argmax(wo_s)]

                def fmt_strat(s):
                    return '[' + ', '.join(f'{x:.2f}' for x in s) + ']'

                print(f"    Hand {h} (eq={eq:.3f}):")
                print(f"      Cold:     {fmt_strat(cold_s)} -> {cold_best}")
                print(f"      WarmHero: {fmt_strat(wh_s)} -> {wh_best}")
                print(f"      WarmOpp:  {fmt_strat(wo_s)} -> {wo_best}")

        # Convergence data
        for cp_iter in (50, 100, 150, 200):
            for label, cp_dict in [('cold', cold_cp), ('warm_hero', warm_hero_cp),
                                    ('warm_opp', warm_opp_cp)]:
                if cp_iter in cp_dict:
                    cp_strat = cp_dict[cp_iter]
                    cp_ev = compute_strategy_ev(cp_strat, equity_matrix,
                                                not_blocked, opp_weights,
                                                tree, n_hero, n_opp)
                    cp_rwev = range_weighted_ev(cp_ev, hero_hands, board_list,
                                                not_blocked, opp_weights)
                    if cp_iter not in convergence_data[label]:
                        convergence_data[label][cp_iter] = []
                    convergence_data[label][cp_iter].append(cp_rwev)

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results:
        print("No valid boards tested!")
        return

    # EV comparison
    cold_evs = [r['cold_ev'] for r in results]
    warm_hero_evs = [r['warm_hero_ev'] for r in results]
    warm_opp_evs = [r['warm_opp_ev'] for r in results]

    print(f"\n  Average Range-Weighted EV (across {len(results)} boards):")
    print(f"    Cold-start (200 iter):       {np.mean(cold_evs):+.5f} (std={np.std(cold_evs):.5f})")
    print(f"    Warm-start Hero (100 iter):  {np.mean(warm_hero_evs):+.5f} (std={np.std(warm_hero_evs):.5f})")
    print(f"    Warm-start Opp (200 iter):   {np.mean(warm_opp_evs):+.5f} (std={np.std(warm_opp_evs):.5f})")

    # Per-board wins
    hero_wins = sum(1 for i in range(len(results)) if warm_hero_evs[i] > cold_evs[i] + 0.001)
    opp_wins = sum(1 for i in range(len(results)) if warm_opp_evs[i] > cold_evs[i] + 0.001)
    cold_wins = sum(1 for i in range(len(results)) if cold_evs[i] > max(warm_hero_evs[i], warm_opp_evs[i]) + 0.001)
    ties = len(results) - hero_wins - opp_wins - cold_wins

    # Actually recount properly
    print(f"\n  Board-by-board EV comparison (warm > cold by >0.001):")
    hero_better = sum(1 for i in range(len(results)) if warm_hero_evs[i] > cold_evs[i] + 0.001)
    hero_worse = sum(1 for i in range(len(results)) if warm_hero_evs[i] < cold_evs[i] - 0.001)
    opp_better = sum(1 for i in range(len(results)) if warm_opp_evs[i] > cold_evs[i] + 0.001)
    opp_worse = sum(1 for i in range(len(results)) if warm_opp_evs[i] < cold_evs[i] - 0.001)
    print(f"    Warm-Hero beats Cold: {hero_better}/{len(results)}, loses: {hero_worse}/{len(results)}")
    print(f"    Warm-Opp beats Cold:  {opp_better}/{len(results)}, loses: {opp_worse}/{len(results)}")

    # EV deltas
    hero_deltas = [warm_hero_evs[i] - cold_evs[i] for i in range(len(results))]
    opp_deltas = [warm_opp_evs[i] - cold_evs[i] for i in range(len(results))]
    print(f"\n  EV Delta (warm - cold):")
    print(f"    Warm-Hero: mean={np.mean(hero_deltas):+.5f}, min={min(hero_deltas):+.5f}, max={max(hero_deltas):+.5f}")
    print(f"    Warm-Opp:  mean={np.mean(opp_deltas):+.5f}, min={min(opp_deltas):+.5f}, max={max(opp_deltas):+.5f}")

    # Timing
    print(f"\n  Timing (seconds per board):")
    print(f"    Cold (200 iter):      mean={np.mean(timing['cold']):.3f}s")
    print(f"    Warm-Hero (100 iter): mean={np.mean(timing['warm_hero']):.3f}s")
    print(f"    Warm-Opp (200 iter):  mean={np.mean(timing['warm_opp']):.3f}s")

    # Agreement
    if total_hands > 0:
        print(f"\n  Action Agreement (all 3 approaches agree on best action):")
        print(f"    {agreement_count}/{total_hands} = {100*agreement_count/total_hands:.1f}%")

    # CHECK -> BET cases
    print(f"\n  CHECK -> BET cases (cold CHECK>60%, equity>60%, warm BET>50%):")
    print(f"    Found: {len(all_check_to_bet)} cases")
    if all_check_to_bet:
        for i, case in enumerate(all_check_to_bet[:10]):
            print(f"    [{i+1}] Board={case['board']}, Hand={case['hand']}, Eq={case['equity']:.3f}")
            print(f"        Cold:  CHECK={case['cold_check']:.2f} BET={case['cold_bet']:.2f}")
            print(f"        WHero: CHECK={case['warm_hero_check']:.2f} BET={case['warm_hero_bet']:.2f}")
            print(f"        WOpp:  CHECK={case['warm_opp_check']:.2f} BET={case['warm_opp_bet']:.2f}")
        if len(all_check_to_bet) > 10:
            print(f"    ... and {len(all_check_to_bet) - 10} more")

    # Convergence
    print(f"\n  Convergence (average EV at iteration checkpoints):")
    print(f"    {'Iter':>6}  {'Cold':>10}  {'WarmHero':>10}  {'WarmOpp':>10}")
    for cp_iter in (50, 100, 150, 200):
        cold_val = np.mean(convergence_data['cold'].get(cp_iter, [0]))
        wh_val = np.mean(convergence_data['warm_hero'].get(cp_iter, [0]))
        wo_val = np.mean(convergence_data['warm_opp'].get(cp_iter, [0]))
        print(f"    {cp_iter:>6}  {cold_val:>+10.5f}  {wh_val:>+10.5f}  {wo_val:>+10.5f}")

    # Final recommendation
    print(f"\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    mean_hero_delta = np.mean(hero_deltas)
    mean_opp_delta = np.mean(opp_deltas)

    if mean_hero_delta > 0.005:
        print(f"  Warm-start HERO shows meaningful improvement (+{mean_hero_delta:.4f} EV).")
        print(f"  It also uses fewer iterations (100 vs 200) = ~2x speed improvement.")
        best = "WARM_HERO"
    elif mean_opp_delta > 0.005:
        print(f"  Warm-start OPPONENT shows meaningful improvement (+{mean_opp_delta:.4f} EV).")
        best = "WARM_OPP"
    elif abs(mean_hero_delta) < 0.002 and abs(mean_opp_delta) < 0.002:
        print(f"  No significant difference between approaches (deltas < 0.002).")
        print(f"  Warm-start Hero saves ~50% time for same quality - use it for speed.")
        best = "WARM_HERO_FOR_SPEED"
    else:
        print(f"  Cold-start appears slightly better or equivalent.")
        print(f"  Warm-start doesn't hurt but doesn't help significantly either.")
        best = "COLD"

    print(f"\n  Verdict: {best}")
    if best in ("WARM_HERO", "WARM_HERO_FOR_SPEED"):
        print(f"  Implementation: In solve_and_act, load precomputed regrets from")
        print(f"  river_lookup before calling _run_dcfr, and halve the iteration count.")
    elif best == "WARM_OPP":
        print(f"  Implementation: Already supported via opp_locked_strategy param.")
        print(f"  Use precomputed P(bet|hand) to set initial opponent strategy.")
    else:
        print(f"  Keep current cold-start approach. The 300-iter precompute against")
        print(f"  uniform range doesn't transfer well to narrowed ranges.")


if __name__ == '__main__':
    run_test()
