#!/usr/bin/env python3
"""Benchmark optimized range solver for river acting first.

Keeps the existing simultaneous-traversal structure (already vectorized),
adds DCFR (Brown's params) and proper card blocking in terminal values.
Measures timing, convergence, strategy stability.
"""
import sys
import os
import time
import numpy as np
import itertools
from math import pow as fpow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))
from equity import ExactEquityEngine
from game_tree import (
    GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

RANKS = "23456789A"
SUITS = "dhs"

def card_str(c):
    return RANKS[c // 3] + SUITS[c % 3]

def hand_str(h):
    return card_str(h[0]) + card_str(h[1])


class OptimizedRangeSolver:
    """Range solver with DCFR and card blocking. Simultaneous traversal."""

    def __init__(self, engine):
        self.engine = engine
        self._tree_cache = {}

    def solve(self, board, dead_cards, opp_range, my_bet, opp_bet,
              iterations=1000, compact=False):
        known = set(board) | set(dead_cards)

        opp_hands = []
        opp_weights_list = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights_list.append(w)
        if not opp_hands:
            return None

        opp_weights = np.array(opp_weights_list, dtype=np.float64)
        opp_weights /= opp_weights.sum()
        n_opp = len(opp_hands)

        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        n_hero = len(hero_hands)

        # Precompute hand strengths
        board_list = list(board)
        hero_strengths = np.array([self.engine.lookup_seven(list(h) + board_list) for h in hero_hands], dtype=np.int32)
        opp_strengths = np.array([self.engine.lookup_seven(list(h) + board_list) for h in opp_hands], dtype=np.int32)

        # Card blocking matrix: blocked[hi, oi] = True if hands share a card
        blocked = np.zeros((n_hero, n_opp), dtype=bool)
        hero_cards_sets = [set(h) for h in hero_hands]
        opp_cards_sets = [set(h) for h in opp_hands]
        for hi in range(n_hero):
            for oi in range(n_opp):
                if hero_cards_sets[hi] & opp_cards_sets[oi]:
                    blocked[hi, oi] = True

        # Equity matrix with card blocking
        # equity[hi, oi] = hero hand hi's equity vs opp hand oi (0 if blocked)
        equity = np.zeros((n_hero, n_opp), dtype=np.float64)
        for hi in range(n_hero):
            for oi in range(n_opp):
                if blocked[hi, oi]:
                    continue
                if hero_strengths[hi] < opp_strengths[oi]:
                    equity[hi, oi] = 1.0
                elif hero_strengths[hi] == opp_strengths[oi]:
                    equity[hi, oi] = 0.5

        # Build tree
        tree = self._get_tree(my_bet, opp_bet, 2, 100, compact)

        # Terminal values with card blocking
        # For fold terminals: blocked hands get 0 value (can't be in that matchup)
        not_blocked = (~blocked).astype(np.float64)  # 1 where valid, 0 where blocked
        terminal_values = {}
        for node_id in tree.terminal_node_ids:
            tt = tree.terminal[node_id]
            hp = tree.hero_pot[node_id]
            op = tree.opp_pot[node_id]
            if tt == TERM_FOLD_HERO:
                terminal_values[node_id] = -hp * not_blocked
            elif tt == TERM_FOLD_OPP:
                terminal_values[node_id] = op * not_blocked
            elif tt == TERM_SHOWDOWN:
                pot_won = min(hp, op)
                terminal_values[node_id] = (2.0 * equity - 1.0) * pot_won

        t0 = time.time()

        # CFR setup
        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)
        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}
        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1), 1)

        hero_regrets = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)
        hero_strat_sum = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float64)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        hero_reach_init = np.ones(n_hero, dtype=np.float64) / n_hero
        opp_reach_init = opp_weights.copy()

        # DCFR params
        alpha, beta, gamma = 1.5, 0.0, 2.0

        for t in range(1, iterations + 1):
            if t > 1:
                pos_w = fpow(t - 1, alpha) / (fpow(t - 1, alpha) + 1.0)
                neg_w = fpow(t - 1, beta) / (fpow(t - 1, beta) + 1.0)
                strat_w = fpow((t - 1) / t, gamma)
                hero_regrets *= np.where(hero_regrets > 0, pos_w, neg_w)
                opp_regrets *= np.where(opp_regrets > 0, pos_w, neg_w)
                hero_strat_sum *= strat_w

            self._traverse(
                tree, 0, hero_reach_init.copy(), opp_reach_init.copy(),
                hero_regrets, hero_strat_sum, opp_regrets,
                hero_idx, opp_idx, terminal_values,
                n_hero, n_opp, max_act)

        elapsed = (time.time() - t0) * 1000

        # Extract root strategy
        root = 0
        if root not in hero_idx:
            return None
        idx = hero_idx[root]
        n_act = tree.num_actions[root]
        strat = hero_strat_sum[idx, :, :n_act]
        totals = strat.sum(axis=1, keepdims=True)
        avg_strat = np.where(totals > 0, strat / np.maximum(totals, 1e-10),
                            np.full_like(strat, 1.0 / n_act))

        # Action names
        root_children = tree.children[root]
        action_names = []
        for act_type, child_id in root_children:
            if act_type == ACT_CHECK:
                action_names.append("CHECK")
            elif act_type == ACT_RAISE_HALF:
                action_names.append("BET_40%")
            elif act_type == ACT_RAISE_POT:
                action_names.append("BET_70%")
            elif act_type == ACT_RAISE_ALLIN:
                action_names.append("BET_100%")
            elif act_type == ACT_RAISE_OVERBET:
                action_names.append("BET_150%")
            else:
                action_names.append(f"ACT_{act_type}")

        # Per-hand results
        hand_strategies = {}
        hand_equities = {}
        for hi, hh in enumerate(hero_hands):
            s = {}
            for ai, name in enumerate(action_names):
                s[name] = float(avg_strat[hi, ai])
            hand_strategies[hh] = s
            # Equity with blocking
            valid_w = opp_weights * (~blocked[hi]).astype(np.float64)
            vw_sum = valid_w.sum()
            if vw_sum > 0:
                hand_equities[hh] = float(np.dot(equity[hi], valid_w / vw_sum))
            else:
                hand_equities[hh] = 0.5

        return {
            'strategies': hand_strategies,
            'equities': hand_equities,
            'elapsed_ms': elapsed,
            'tree_size': tree.size,
            'n_hero': n_hero,
            'n_opp': n_opp,
            'action_names': action_names,
        }

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet, compact):
        key = (hero_bet, opp_bet, min_raise, max_bet, True, compact)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(
                hero_bet, opp_bet, min_raise, max_bet, True, compact=compact)
        return self._tree_cache[key]

    @staticmethod
    def _regret_match(regrets, n_act):
        pos = np.maximum(regrets[:, :n_act], 0)
        totals = pos.sum(axis=1, keepdims=True)
        return np.where(totals > 0, pos / np.maximum(totals, 1e-10),
                       np.full_like(pos, 1.0 / n_act))

    def _traverse(self, tree, node_id, hero_reach, opp_reach,
                  hero_regrets, hero_strat_sum, opp_regrets,
                  hero_idx, opp_idx, terminal_values,
                  n_hero, n_opp, max_act):
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero
            idx = hero_idx[node_id]
            strategies = self._regret_match(hero_regrets[idx], n_act)

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float64)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._traverse(
                    tree, children[a][1], hero_reach * strategies[:, a], opp_reach,
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1] * action_values[a]

            diff = action_values - node_value[np.newaxis, :, :]
            cf_regrets = np.tensordot(diff, opp_reach, axes=([2], [0]))
            hero_regrets[idx, :, :n_act] += cf_regrets.T

            hero_strat_sum[idx, :, :n_act] += hero_reach[:, np.newaxis] * strategies
            return node_value

        else:  # Opponent
            idx = opp_idx[node_id]
            strategies = self._regret_match(opp_regrets[idx], n_act)

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float64)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._traverse(
                    tree, children[a][1], hero_reach, opp_reach * strategies[:, a],
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1].T * action_values[a]

            diff = node_value[np.newaxis, :, :] - action_values
            cf_regrets = np.tensordot(diff, hero_reach, axes=([1], [0]))
            opp_regrets[idx, :, :n_act] += cf_regrets.T

            return node_value


def make_range(board, dead):
    known = set(board) | set(dead)
    rng = {}
    for h in itertools.combinations(range(27), 2):
        if not (set(h) & known):
            rng[h] = 1.0
    total = sum(rng.values())
    for k in rng:
        rng[k] /= total
    return rng


def make_narrowed_range(board, dead, engine):
    """Top 60% of hands by strength."""
    known = set(board) | set(dead)
    board_list = list(board)
    hands = []
    for h in itertools.combinations(range(27), 2):
        if not (set(h) & known):
            strength = engine.lookup_seven(list(h) + board_list)
            hands.append((h, strength))
    hands.sort(key=lambda x: x[1])
    keep = int(len(hands) * 0.6)
    rng = {}
    for h, s in hands[:keep]:
        rng[h] = 1.0
    total = sum(rng.values())
    for k in rng:
        rng[k] /= total
    return rng


def main():
    print("Loading equity engine...")
    engine = ExactEquityEngine()
    solver = OptimizedRangeSolver(engine)

    boards = [
        ([0, 3, 6, 9, 12], "flush-heavy"),
        ([0, 4, 8, 15, 24], "rainbow"),
        ([0, 1, 6, 7, 18], "paired"),
    ]
    dead = [20, 21, 22, 23, 25, 26]

    # =========================================================================
    print("\n" + "=" * 80)
    print("1. TIMING & CONVERGENCE — Full tree, pot=(30,30), DCFR")
    print("=" * 80)

    board, bname = boards[0]
    opp_range = make_range(board, dead)
    remaining = [c for c in range(27) if c not in set(board) and c not in set(dead)]
    n_hero = len(list(itertools.combinations(remaining, 2)))
    n_opp = sum(1 for h, w in opp_range.items() if w > 0.001 and not (set(h) & (set(board) | set(dead))))
    print(f"Board: {' '.join(card_str(c) for c in board)} | {n_hero} hero × {n_opp} opp hands\n")

    print(f"{'Iters':>6} {'Time':>8} {'ms/it':>6} {'Bet%':>6} {'Bluff<20':>9} {'Val>80':>7} {'Stability':>10}")
    print("-" * 65)

    prev_strats = None
    for iters in [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]:
        r = solver.solve(board, dead, opp_range, 30, 30, iterations=iters, compact=False)
        if not r:
            print(f"{iters:>6} FAILED"); continue

        strats = r['strategies']
        eqs = r['equities']
        n = len(strats)
        total_bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in strats.values()) / n
        bluff_hands = [s for hh, s in strats.items() if eqs[hh] < 0.20]
        bluff_bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in bluff_hands) / max(len(bluff_hands), 1)
        value_hands = [s for hh, s in strats.items() if eqs[hh] > 0.80]
        value_bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in value_hands) / max(len(value_hands), 1)

        stability = ""
        if prev_strats:
            diffs = []
            for hh in strats:
                if hh in prev_strats:
                    for k in strats[hh]:
                        diffs.append(abs(strats[hh][k] - prev_strats[hh].get(k, 0)))
            stability = f"  Δ={max(diffs):.4f}" if diffs else ""
        prev_strats = dict(strats)

        print(f"{iters:>6} {r['elapsed_ms']:>7.0f}ms {r['elapsed_ms']/iters:>5.1f} "
              f"{total_bet:>5.1%} {bluff_bet:>8.1%} {value_bet:>6.1%}{stability}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("2. POT SIZE SWEEP — 500 iters DCFR, full tree")
    print("=" * 80)

    for board, bname in boards:
        opp_range = make_range(board, dead)
        print(f"\n  {bname}:")
        for pot_each in [4, 8, 16, 25, 30, 40, 50]:
            r = solver.solve(board, dead, opp_range, pot_each, pot_each, iterations=500, compact=False)
            if not r: continue
            strats = r['strategies']
            eqs = r['equities']
            n = len(strats)
            bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in strats.values()) / n
            bluff = sum(sum(v for k, v in s.items() if 'BET' in k)
                       for hh, s in strats.items() if eqs[hh] < 0.20) / max(sum(1 for hh in strats if eqs[hh] < 0.20), 1)
            val = sum(sum(v for k, v in s.items() if 'BET' in k)
                     for hh, s in strats.items() if eqs[hh] > 0.80) / max(sum(1 for hh in strats if eqs[hh] > 0.80), 1)
            print(f"    pot=({pot_each:>2},{pot_each:>2}): bet={bet:>5.1%}  bluff={bluff:>5.1%}  value={val:>5.1%}  {r['elapsed_ms']:.0f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print("3. UNIFORM vs NARROWED opponent range — 500 iters, pot=(30,30)")
    print("=" * 80)

    for board, bname in boards:
        uniform = make_range(board, dead)
        narrowed = make_narrowed_range(board, dead, engine)
        for rng, rlabel in [(uniform, "uniform"), (narrowed, "top60%")]:
            r = solver.solve(board, dead, rng, 30, 30, iterations=500, compact=False)
            if not r: continue
            strats = r['strategies']
            eqs = r['equities']
            n = len(strats)
            bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in strats.values()) / n
            bluff = sum(sum(v for k, v in s.items() if 'BET' in k)
                       for hh, s in strats.items() if eqs[hh] < 0.20) / max(sum(1 for hh in strats if eqs[hh] < 0.20), 1)
            print(f"  {bname:<12} {rlabel:<10}: bet={bet:>5.1%} bluff={bluff:>5.1%} "
                  f"({r['n_opp']} opp) {r['elapsed_ms']:.0f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print("4. COMPACT vs FULL TREE — 500 iters, pot=(30,30)")
    print("=" * 80)

    board, bname = boards[1]
    opp_range = make_range(board, dead)
    for compact, label in [(True, "COMPACT"), (False, "FULL")]:
        r = solver.solve(board, dead, opp_range, 30, 30, iterations=500, compact=compact)
        if not r: continue
        strats = r['strategies']
        eqs = r['equities']
        n = len(strats)
        bet = sum(sum(v for k, v in s.items() if 'BET' in k) for s in strats.values()) / n
        print(f"  {label:<10}: bet={bet:>5.1%} tree={r['tree_size']} nodes  {r['elapsed_ms']:.0f}ms")

    # =========================================================================
    print("\n" + "=" * 80)
    print("5. ARM64 BUDGET ESTIMATE")
    print("=" * 80)

    # Use a mid-complexity solve as reference
    board = boards[1][0]
    opp_range = make_range(board, dead)
    r500 = solver.solve(board, dead, opp_range, 30, 30, iterations=500, compact=False)
    mac_500 = r500['elapsed_ms']

    arm_factor = 2.5
    arm_500 = mac_500 * arm_factor

    print(f"\n  Reference solve: {r500['n_hero']} hero × {r500['n_opp']} opp, "
          f"tree={r500['tree_size']} nodes")
    print(f"  Mac: {mac_500:.0f}ms / 500 iters = {mac_500/500:.1f} ms/iter")
    print(f"  ARM64 est: {arm_500:.0f}ms / 500 iters = {arm_500/500:.1f} ms/iter")

    print(f"\n  Match budget (1500s total, ~100s used for other streets):")
    for iters in [200, 300, 500]:
        arm_ms = mac_500 * arm_factor * (iters / 500)
        for n_decisions in [100, 150, 200, 300]:
            total_s = n_decisions * arm_ms / 1000
            remaining = 1400
            pct = total_s / remaining * 100
            flag = " ✓" if pct < 70 else " ✗" if pct > 100 else " ~"
            print(f"    {iters} iters × {n_decisions} decisions = {total_s:>6.0f}s "
                  f"({pct:>4.0f}% of budget){flag}")

    print("\nDone.")


if __name__ == '__main__':
    main()
