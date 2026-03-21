#!/usr/bin/env python3
"""Test one-hand solver for river acting first with full tree (re-raises).

Measures bet frequency, action distribution, and timing across multiple
boards/hands to see if the solver gives reasonable acting-first strategies
when the opponent can re-raise (not just call/fold).

Tests both vanilla CFR+ and DCFR (discounted CFR).
"""
import sys
import os
import time
import numpy as np
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))

from equity import ExactEquityEngine as EquityEngine
from game_tree import GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL, ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET, TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN

RANKS = "23456789A"
SUITS = "dhs"

def card_str(c):
    return RANKS[c // 3] + SUITS[c % 3]

def hand_str(h):
    return card_str(h[0]) + card_str(h[1])


class TestSolver:
    """One-hand solver with both CFR+ and DCFR for acting-first testing."""

    def __init__(self, engine):
        self.engine = engine

    def solve_acting_first(self, hero_cards, board, dead_cards, opp_range,
                           my_bet, opp_bet, iterations=500, use_dcfr=False):
        """Solve river acting first with full tree (4 sizes, 2 raises max).

        Returns (strategy_dict, elapsed_ms, tree_size).
        strategy_dict maps action names to probabilities.
        """
        known = set(hero_cards) | set(board) | set(dead_cards)
        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

        if not opp_hands:
            return None, 0, 0

        opp_weights = np.array(opp_weights, dtype=np.float64)
        opp_weights /= opp_weights.sum()
        n_opp = len(opp_hands)

        # Full tree: 4 bet sizes, 2 raises max, hero acts first
        tree = GameTree(my_bet, opp_bet, 2, 100, hero_first=True, compact=False)

        # Compute terminal values
        hero_rank = self.engine.lookup_seven(list(hero_cards) + list(board))
        equity_vec = np.zeros(n_opp, dtype=np.float64)
        for i, oh in enumerate(opp_hands):
            opp_rank = self.engine.lookup_seven(list(oh) + list(board))
            if hero_rank < opp_rank:
                equity_vec[i] = 1.0
            elif hero_rank == opp_rank:
                equity_vec[i] = 0.5

        terminal_values = {}
        for node_id in tree.terminal_node_ids:
            tt = tree.terminal[node_id]
            hp = tree.hero_pot[node_id]
            op = tree.opp_pot[node_id]
            if tt == TERM_FOLD_HERO:
                terminal_values[node_id] = np.full(n_opp, -hp, dtype=np.float64)
            elif tt == TERM_FOLD_OPP:
                terminal_values[node_id] = np.full(n_opp, op, dtype=np.float64)
            elif tt == TERM_SHOWDOWN:
                pot_won = min(hp, op)
                terminal_values[node_id] = (2.0 * equity_vec - 1.0) * pot_won

        # Run CFR
        t0 = time.time()
        hero_strategy = self._run_cfr(tree, opp_weights, terminal_values,
                                       n_opp, iterations, use_dcfr)
        elapsed = (time.time() - t0) * 1000

        # Map strategy to action names
        root_children = tree.children[0]
        action_names = []
        for act_type, child_id in root_children:
            if act_type == ACT_CHECK:
                action_names.append("CHECK")
            elif act_type == ACT_FOLD:
                action_names.append("FOLD")
            elif act_type == ACT_RAISE_HALF:
                bet_amt = tree.hero_pot[child_id] - opp_bet
                action_names.append(f"BET_40%({bet_amt})")
            elif act_type == ACT_RAISE_POT:
                bet_amt = tree.hero_pot[child_id] - opp_bet
                action_names.append(f"BET_70%({bet_amt})")
            elif act_type == ACT_RAISE_ALLIN:
                bet_amt = tree.hero_pot[child_id] - opp_bet
                action_names.append(f"BET_100%({bet_amt})")
            elif act_type == ACT_RAISE_OVERBET:
                bet_amt = tree.hero_pot[child_id] - opp_bet
                action_names.append(f"BET_150%({bet_amt})")

        result = {}
        for i, name in enumerate(action_names):
            if i < len(hero_strategy):
                result[name] = hero_strategy[i]

        equity = np.dot(equity_vec, opp_weights)
        return result, elapsed, tree.size, equity

    def _run_cfr(self, tree, opp_weights, terminal_values, n_opp, iterations, use_dcfr):
        n_hero = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1), 1)

        hero_regrets = np.zeros((n_hero, max_act), dtype=np.float64)
        hero_strategy_sum = np.zeros((n_hero, max_act), dtype=np.float64)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float64)

        for t in range(iterations):
            # DCFR discount factors
            if use_dcfr and t > 0:
                # Positive regrets weighted by t^1.5 / (t^1.5 + 1)
                # Negative regrets weighted by t^0.5 / (t^0.5 + 1)
                # Strategy sum weighted by (t / (t+1))^2
                pos_w = (t ** 1.5) / (t ** 1.5 + 1)
                neg_w = (t ** 0.5) / (t ** 0.5 + 1)
                strat_w = ((t / (t + 1)) ** 2)

                hero_regrets *= np.where(hero_regrets > 0, pos_w, neg_w)
                opp_regrets *= np.where(opp_regrets > 0, pos_w, neg_w)
                hero_strategy_sum *= strat_w

            self._cfr_traverse(
                tree, 0, 1.0, opp_weights,
                hero_regrets, hero_strategy_sum, opp_regrets,
                hero_idx, opp_idx, terminal_values, n_opp, max_act, t)

        # Extract root strategy
        root = 0
        if root in hero_idx:
            idx = hero_idx[root]
            n_act = tree.num_actions[root]
            total = hero_strategy_sum[idx, :n_act].sum()
            if total > 0:
                return hero_strategy_sum[idx, :n_act] / total
            return np.ones(n_act) / n_act
        return np.array([1.0])

    def _cfr_traverse(self, tree, node_id, hero_reach, opp_reach_vec,
                       hero_regrets, hero_strategy_sum, opp_regrets,
                       hero_idx, opp_idx, terminal_values, n_opp, max_act, t):
        if tree.terminal[node_id] != TERM_NONE:
            return terminal_values[node_id]

        n_act = tree.num_actions[node_id]
        children = tree.children[node_id]
        player = tree.player[node_id]

        if player == 0:  # Hero
            idx = hero_idx[node_id]
            reg = hero_regrets[idx, :n_act]
            pos = np.maximum(reg, 0.0)
            total = pos.sum()
            strategy = pos / total if total > 0 else np.full(n_act, 1.0 / n_act)

            action_values = np.empty((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._cfr_traverse(
                    tree, children[a][1], hero_reach * strategy[a], opp_reach_vec,
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[a] * action_values[a]

            for a in range(n_act):
                cf_regret = np.dot(action_values[a] - node_value, opp_reach_vec)
                hero_regrets[idx, a] = max(0.0, hero_regrets[idx, a] + cf_regret)

            hero_strategy_sum[idx, :n_act] += hero_reach * strategy
            return node_value

        else:  # Opponent
            idx = opp_idx[node_id]
            reg = opp_regrets[idx, :, :n_act]
            pos = np.maximum(reg, 0.0)
            totals = pos.sum(axis=1, keepdims=True)
            strategy = np.where(totals > 0, pos / np.maximum(totals, 1e-10),
                               np.full_like(pos, 1.0 / n_act))

            action_values = np.empty((n_act, n_opp), dtype=np.float64)
            node_value = np.zeros(n_opp, dtype=np.float64)

            for a in range(n_act):
                action_values[a] = self._cfr_traverse(
                    tree, children[a][1], hero_reach, opp_reach_vec * strategy[:, a],
                    hero_regrets, hero_strategy_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values, n_opp, max_act, t)
                node_value += strategy[:, a] * action_values[a]

            hr = hero_reach
            for a in range(n_act):
                inst_regret = hr * (node_value - action_values[a])
                opp_regrets[idx, :, a] = np.maximum(0.0, opp_regrets[idx, :, a] + inst_regret)

            return node_value


def make_uniform_range(board, dead, hero_cards):
    """Build uniform opponent range excluding known cards."""
    known = set(board) | set(dead) | set(hero_cards)
    rng = {}
    for h in itertools.combinations(range(27), 2):
        if not (set(h) & known):
            rng[h] = 1.0
    total = sum(rng.values())
    for k in rng:
        rng[k] /= total
    return rng


def main():
    print("Loading equity engine...")
    engine = EquityEngine()
    solver = TestSolver(engine)

    # Test scenarios: various boards and hero hands at different equity levels
    # All river boards (5 community cards)
    scenarios = [
        # (name, hero_cards, board, pot_each)
        ("Nuts (straight flush)", (0, 1), [2, 3, 4, 5, 6], 10),     # 2d2h, board 2s3d3h3s4d = likely nuts
        ("Strong (trips)",       (0, 3), [1, 6, 9, 12, 15], 10),    # 2d3h on 2h4d5d6d7d
        ("Medium (top pair)",    (24, 25), [0, 3, 6, 9, 12], 10),   # AhAs on 2d3h4d5d6d
        ("Weak (low pair)",      (0, 9), [3, 6, 12, 15, 18], 10),   # 2d5d on 3h4d6d7d8d
        ("Air (no pair)",        (1, 4), [6, 9, 12, 18, 24], 10),   # 2h3h on 4d5d6d8d Ah
        ("Medium pot strong",    (24, 25), [0, 3, 6, 9, 12], 30),   # Same but pot=60
        ("Big pot air",          (1, 4), [6, 9, 12, 18, 24], 30),   # Same but pot=60
    ]

    # First, show tree structure
    print("\n=== TREE STRUCTURE ===")
    for pot_each in [10, 30, 50]:
        tree_full = GameTree(pot_each, pot_each, 2, 100, hero_first=True, compact=False)
        tree_compact = GameTree(pot_each, pot_each, 2, 100, hero_first=True, compact=True)
        print(f"  Pot ({pot_each},{pot_each}): Full tree = {tree_full.size} nodes "
              f"({len(tree_full.hero_node_ids)} hero, {len(tree_full.opp_node_ids)} opp) | "
              f"Compact = {tree_compact.size} nodes")

        # Show root actions for full tree
        root_children = tree_full.children[0]
        actions = []
        for act_type, child_id in root_children:
            if act_type == ACT_CHECK:
                actions.append("CHECK")
            else:
                bet = tree_full.hero_pot[child_id] - pot_each
                actions.append(f"BET({bet})")
        print(f"    Root actions: {', '.join(actions)}")

    # Test iterations and DCFR
    iter_counts = [100, 200, 500, 1000, 2000]

    print("\n=== CONVERGENCE TEST (Medium hand, pot=10,10) ===")
    hero_cards = (24, 25)  # Ah, As
    board = [0, 3, 6, 9, 12]
    dead = [1, 2, 4, 5, 7]  # some discards
    opp_range = make_uniform_range(board, dead, hero_cards)

    print(f"Hero: {hand_str(hero_cards)}, Board: {' '.join(card_str(c) for c in board)}")
    print(f"Dead: {' '.join(card_str(c) for c in dead)}, Opp range size: {len(opp_range)}")

    for iters in iter_counts:
        for dcfr in [False, True]:
            label = f"{'DCFR' if dcfr else 'CFR+'} {iters:4d} iters"
            result, ms, sz, eq = solver.solve_acting_first(
                hero_cards, board, dead, opp_range, 10, 10, iters, dcfr)
            if result is None:
                print(f"  {label}: FAILED")
                continue
            bet_freq = sum(v for k, v in result.items() if 'BET' in k)
            check_freq = result.get('CHECK', 0)
            parts = [f"{k}={v:.1%}" for k, v in sorted(result.items())]
            print(f"  {label}: bet={bet_freq:.1%} check={check_freq:.1%} | "
                  f"{' '.join(parts)} | eq={eq:.2f} {ms:.0f}ms")

    # Test across multiple hands and equity levels
    print("\n=== BET FREQUENCY BY HAND STRENGTH ===")
    print(f"{'Scenario':<25} {'Equity':>6} {'CFR+ 500':>10} {'DCFR 500':>10} "
          f"{'CFR+ 1000':>10} {'DCFR 1000':>10} {'ms':>6}")

    for name, hero, board, pot in scenarios:
        dead = []
        # Pick some plausible dead cards (discards)
        known = set(hero) | set(board)
        for c in range(27):
            if c not in known and len(dead) < 6:
                dead.append(c)
        opp_range = make_uniform_range(board, dead, hero)

        results = {}
        for iters in [500, 1000]:
            for dcfr in [False, True]:
                key = f"{'DCFR' if dcfr else 'CFR+'} {iters}"
                r, ms, sz, eq = solver.solve_acting_first(
                    hero, board, dead, opp_range, pot, pot, iters, dcfr)
                if r:
                    bet_freq = sum(v for k, v in r.items() if 'BET' in k)
                    results[key] = (bet_freq, ms)
                else:
                    results[key] = (0, 0)

        # Get equity from first result
        _, _, _, eq = solver.solve_acting_first(hero, board, dead, opp_range, pot, pot, 10, False)
        line = f"{name:<25} {eq:>6.2f}"
        for key in ['CFR+ 500', 'DCFR 500', 'CFR+ 1000', 'DCFR 1000']:
            bf, ms = results[key]
            line += f" {bf:>9.1%}"
        line += f" {results['DCFR 1000'][1]:>5.0f}"
        print(line)

    # Aggregate test: sample many random hands on a few boards
    print("\n=== AGGREGATE BET FREQUENCY (100 random hands per board, DCFR 1000) ===")

    np.random.seed(42)
    test_boards = [
        [0, 3, 6, 9, 12],
        [1, 4, 7, 10, 13],
        [2, 5, 8, 11, 14],
        [0, 4, 8, 12, 16],
        [3, 7, 11, 15, 19],
    ]

    total_bet = 0
    total_hands = 0
    total_ms = 0
    equity_bins = {
        '0.0-0.2': [0, 0],
        '0.2-0.4': [0, 0],
        '0.4-0.6': [0, 0],
        '0.6-0.8': [0, 0],
        '0.8-1.0': [0, 0],
    }

    for board in test_boards:
        board_set = set(board)
        all_hands = [(c1, c2) for c1, c2 in itertools.combinations(range(27), 2)
                     if c1 not in board_set and c2 not in board_set]
        np.random.shuffle(all_hands)
        hands_to_test = all_hands[:100]

        # Pick dead cards
        dead = []
        for c in range(27):
            if c not in board_set and len(dead) < 6:
                dead.append(c)

        for hero in hands_to_test:
            if set(hero) & set(dead):
                continue
            opp_range = make_uniform_range(board, dead, hero)
            r, ms, sz, eq = solver.solve_acting_first(
                hero, board, dead, opp_range, 10, 10, 1000, use_dcfr=True)
            if r is None:
                continue

            bet_freq = sum(v for k, v in r.items() if 'BET' in k)
            total_bet += bet_freq
            total_hands += 1
            total_ms += ms

            # Bin by equity
            for bname, (lo, hi) in [('0.0-0.2', (0, 0.2)), ('0.2-0.4', (0.2, 0.4)),
                                      ('0.4-0.6', (0.4, 0.6)), ('0.6-0.8', (0.6, 0.8)),
                                      ('0.8-1.0', (0.8, 1.01))]:
                if lo <= eq < hi:
                    equity_bins[bname][0] += bet_freq
                    equity_bins[bname][1] += 1
                    break

    if total_hands > 0:
        print(f"\nOverall: {total_bet/total_hands:.1%} bet frequency across {total_hands} hands")
        print(f"Avg time: {total_ms/total_hands:.0f}ms per solve")
        print(f"\nBet frequency by equity bucket:")
        for bname in sorted(equity_bins.keys()):
            bf_sum, count = equity_bins[bname]
            if count > 0:
                print(f"  Equity {bname}: {bf_sum/count:.1%} bet freq ({count} hands)")
            else:
                print(f"  Equity {bname}: no hands")


if __name__ == '__main__':
    main()
