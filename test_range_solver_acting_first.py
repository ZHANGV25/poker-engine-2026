#!/usr/bin/env python3
"""Test range solver acting first with full tree (re-raises allowed).

Measures bet frequency, strategy shape, and timing across many hands/boards.
Compares compact tree (current, no re-raise) vs full tree (re-raise allowed).
"""
import sys
import os
import time
import numpy as np
import itertools

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


class FullTreeRangeSolver:
    """Range solver that can use full or compact tree."""

    def __init__(self, engine):
        self.engine = engine
        self._tree_cache = {}

    def solve_acting_first(self, board, dead_cards, opp_range,
                           my_bet, opp_bet, iterations=1000,
                           compact=False, use_dcfr=False):
        """Solve river acting first for ALL hero hands.

        Returns:
            dict: hand -> {action_name: prob}, plus metadata
        """
        known = set(board) | set(dead_cards)
        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

        if not opp_hands:
            return None

        opp_weights = np.array(opp_weights, dtype=np.float32)
        opp_weights /= opp_weights.sum()
        n_opp = len(opp_hands)

        # Build hero hands
        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        n_hero = len(hero_hands)

        # Build tree
        tree = self._get_tree(my_bet, opp_bet, 2, 100, compact)

        # Equity matrix
        equity = np.zeros((n_hero, n_opp), dtype=np.float32)
        board_list = list(board)
        for hi, hh in enumerate(hero_hands):
            hh_set = set(hh)
            hr = self.engine.lookup_seven(list(hh) + board_list)
            for oi, oh in enumerate(opp_hands):
                if set(oh) & hh_set:
                    continue
                opr = self.engine.lookup_seven(list(oh) + board_list)
                if hr < opr:
                    equity[hi, oi] = 1.0
                elif hr == opr:
                    equity[hi, oi] = 0.5

        # Terminal values
        terminal_values = {}
        for node_id in tree.terminal_node_ids:
            tt = tree.terminal[node_id]
            hp = tree.hero_pot[node_id]
            op = tree.opp_pot[node_id]
            if tt == TERM_FOLD_HERO:
                terminal_values[node_id] = np.full((n_hero, n_opp), -hp, dtype=np.float32)
            elif tt == TERM_FOLD_OPP:
                terminal_values[node_id] = np.full((n_hero, n_opp), op, dtype=np.float32)
            elif tt == TERM_SHOWDOWN:
                pot_won = min(hp, op)
                terminal_values[node_id] = (2.0 * equity - 1.0) * pot_won

        # Run CFR
        t0 = time.time()
        hero_strategy = self._run_range_cfr(
            tree, opp_weights, terminal_values,
            n_hero, n_opp, iterations, use_dcfr)
        elapsed = (time.time() - t0) * 1000

        # Map actions
        root_children = tree.children[0]
        action_names = []
        for act_type, child_id in root_children:
            if act_type == ACT_CHECK:
                action_names.append("CHECK")
            elif act_type == ACT_RAISE_HALF:
                action_names.append(f"BET_40%")
            elif act_type == ACT_RAISE_POT:
                action_names.append(f"BET_70%")
            elif act_type == ACT_RAISE_ALLIN:
                action_names.append(f"BET_100%")
            elif act_type == ACT_RAISE_OVERBET:
                action_names.append(f"BET_150%")
            else:
                action_names.append(f"ACT_{act_type}")

        # Per-hand equities
        hand_equities = {}
        for hi, hh in enumerate(hero_hands):
            hand_equities[hh] = float(np.dot(equity[hi], opp_weights))

        # Per-hand strategies
        hand_strategies = {}
        for hi, hh in enumerate(hero_hands):
            strat = {}
            for ai, name in enumerate(action_names):
                if ai < hero_strategy.shape[1]:
                    strat[name] = float(hero_strategy[hi, ai])
            hand_strategies[hh] = strat

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

    def _run_range_cfr(self, tree, opp_weights, terminal_values,
                       n_hero, n_opp, iterations, use_dcfr):
        n_hero_nodes = len(tree.hero_node_ids)
        n_opp_nodes = len(tree.opp_node_ids)

        hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
        opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

        max_act = max(
            max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
            max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1), 1)

        hero_regrets = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float32)
        hero_strat_sum = np.zeros((n_hero_nodes, n_hero, max_act), dtype=np.float32)
        opp_regrets = np.zeros((n_opp_nodes, n_opp, max_act), dtype=np.float32)

        hero_reach_init = np.ones(n_hero, dtype=np.float32) / n_hero

        for t in range(iterations):
            if use_dcfr and t > 0:
                pos_w = (t ** 1.5) / (t ** 1.5 + 1)
                neg_w = (t ** 0.5) / (t ** 0.5 + 1)
                strat_w = ((t / (t + 1)) ** 2)
                hero_regrets *= np.where(hero_regrets > 0, pos_w, neg_w).astype(np.float32)
                opp_regrets *= np.where(opp_regrets > 0, pos_w, neg_w).astype(np.float32)
                hero_strat_sum *= strat_w

            self._traverse(
                tree, 0, hero_reach_init.copy(), opp_weights.copy(),
                hero_regrets, hero_strat_sum, opp_regrets,
                hero_idx, opp_idx, terminal_values,
                n_hero, n_opp, max_act)

        root = 0
        if root not in hero_idx:
            return np.ones((n_hero, 1), dtype=np.float32)

        idx = hero_idx[root]
        n_act = tree.num_actions[root]
        strat = hero_strat_sum[idx, :, :n_act]
        totals = strat.sum(axis=1, keepdims=True)
        return np.where(totals > 0, strat / np.maximum(totals, 1e-10),
                       np.full_like(strat, 1.0 / n_act))

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

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float32)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float32)

            for a in range(n_act):
                action_values[a] = self._traverse(
                    tree, children[a][1], hero_reach * strategies[:, a], opp_reach,
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1] * action_values[a]

            diff = action_values - node_value[np.newaxis, :, :]
            cf_regrets = np.tensordot(diff, opp_reach, axes=([2], [0]))
            hero_regrets[idx, :, :n_act] = np.maximum(
                0, hero_regrets[idx, :, :n_act] + cf_regrets.T)
            hero_strat_sum[idx, :, :n_act] += hero_reach[:, np.newaxis] * strategies

            return node_value

        else:  # Opponent
            idx = opp_idx[node_id]
            strategies = self._regret_match(opp_regrets[idx], n_act)

            action_values = np.zeros((n_act, n_hero, n_opp), dtype=np.float32)
            node_value = np.zeros((n_hero, n_opp), dtype=np.float32)

            for a in range(n_act):
                action_values[a] = self._traverse(
                    tree, children[a][1], hero_reach, opp_reach * strategies[:, a],
                    hero_regrets, hero_strat_sum, opp_regrets,
                    hero_idx, opp_idx, terminal_values,
                    n_hero, n_opp, max_act)
                node_value += strategies[:, a:a+1].T * action_values[a]

            diff = node_value[np.newaxis, :, :] - action_values
            cf_regrets = np.tensordot(diff, hero_reach, axes=([1], [0]))
            opp_regrets[idx, :, :n_act] = np.maximum(
                0, opp_regrets[idx, :, :n_act] + cf_regrets.T)

            return node_value


def make_uniform_range(board, dead, exclude=None):
    known = set(board) | set(dead)
    if exclude:
        known |= set(exclude)
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
    engine = ExactEquityEngine()
    solver = FullTreeRangeSolver(engine)

    # Test boards
    boards = [
        [0, 3, 6, 9, 12],   # 2d 3h 4d 5d 6d — flush-heavy
        [1, 4, 7, 10, 24],  # 2h 3h 4h 5h Ah — flush board
        [0, 1, 6, 7, 18],   # 2d 2h 4d 4h 8d — paired board
        [2, 5, 8, 11, 14],  # 2s 3s 4s 5s 6s — straight flush board
        [0, 4, 8, 15, 24],  # 2d 3h 4s 7d Ah — rainbow disconnected
    ]
    board_names = ["flush-heavy", "flush board", "paired board", "SF board", "rainbow"]

    dead_cards = [20, 21, 22, 23, 25, 26]  # 8s 9d 9h 9s As

    # =========================================================================
    # 1. Compact vs Full tree comparison on one board
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPACT vs FULL TREE — Board: 2d 3h 4d 5d 6d, pot=(10,10), 1000 iters")
    print("=" * 80)

    board = boards[0]
    opp_range = make_uniform_range(board, dead_cards)

    for compact, label in [(True, "COMPACT (no reraise)"), (False, "FULL (reraise)")]:
        r = solver.solve_acting_first(board, dead_cards, opp_range,
                                       10, 10, iterations=1000, compact=compact)
        if not r:
            print(f"\n{label}: FAILED")
            continue

        strats = r['strategies']
        eqs = r['equities']

        # Aggregate stats
        total_bet = 0
        bet_by_action = {}
        for name in r['action_names']:
            bet_by_action[name] = 0
        for hh, s in strats.items():
            for name, prob in s.items():
                if 'BET' in name:
                    total_bet += prob
                bet_by_action[name] = bet_by_action.get(name, 0) + prob
        n = len(strats)

        print(f"\n{label}: tree={r['tree_size']} nodes, {r['n_hero']} hero hands, "
              f"{r['n_opp']} opp hands, {r['elapsed_ms']:.0f}ms")
        print(f"  Aggregate bet freq: {total_bet/n:.1%}")
        for name in r['action_names']:
            print(f"    {name}: {bet_by_action[name]/n:.1%}")

        # Show strategy by equity bucket
        print(f"\n  Strategy by equity bucket:")
        buckets = [(0, 0.15, "0-15%"), (0.15, 0.30, "15-30%"), (0.30, 0.50, "30-50%"),
                   (0.50, 0.70, "50-70%"), (0.70, 0.85, "70-85%"), (0.85, 1.01, "85-100%")]
        for lo, hi, bname in buckets:
            hands_in = [(hh, s) for hh, s in strats.items() if lo <= eqs[hh] < hi]
            if not hands_in:
                print(f"    Equity {bname}: no hands")
                continue
            avg_bet = sum(sum(v for k, v in s.items() if 'BET' in k) for _, s in hands_in) / len(hands_in)
            avg_check = sum(s.get('CHECK', 0) for _, s in hands_in) / len(hands_in)
            # Show a few example hands
            sorted_hands = sorted(hands_in, key=lambda x: sum(v for k,v in x[1].items() if 'BET' in k), reverse=True)
            examples = []
            for hh, s in sorted_hands[:3]:
                bf = sum(v for k,v in s.items() if 'BET' in k)
                examples.append(f"{hand_str(hh)}(eq={eqs[hh]:.0%},bet={bf:.0%})")
            print(f"    Equity {bname}: avg bet={avg_bet:.1%} check={avg_check:.1%} "
                  f"({len(hands_in)} hands) | top: {', '.join(examples)}")

    # =========================================================================
    # 2. Full tree across multiple boards and pot sizes
    # =========================================================================
    print("\n" + "=" * 80)
    print("FULL TREE ACROSS BOARDS AND POT SIZES — 1000 iters")
    print("=" * 80)

    for bi, (board, bname) in enumerate(zip(boards, board_names)):
        for pot_each in [10, 30]:
            opp_range = make_uniform_range(board, dead_cards)
            r = solver.solve_acting_first(board, dead_cards, opp_range,
                                           pot_each, pot_each, iterations=1000,
                                           compact=False)
            if not r:
                continue

            strats = r['strategies']
            eqs = r['equities']
            n = len(strats)
            total_bet = sum(
                sum(v for k, v in s.items() if 'BET' in k)
                for s in strats.values()) / n

            # Bet freq by equity bucket
            bucket_bets = {}
            for lo, hi, bname2 in [(0, 0.30, "low"), (0.30, 0.60, "mid"),
                                    (0.60, 0.85, "high"), (0.85, 1.01, "nuts")]:
                hands_in = [s for hh, s in strats.items() if lo <= eqs[hh] < hi]
                if hands_in:
                    bucket_bets[bname2] = sum(
                        sum(v for k, v in s.items() if 'BET' in k)
                        for s in hands_in) / len(hands_in)
                else:
                    bucket_bets[bname2] = 0

            print(f"  {bname:<20} pot=({pot_each},{pot_each}): "
                  f"bet={total_bet:.1%} | "
                  f"low={bucket_bets['low']:.0%} mid={bucket_bets['mid']:.0%} "
                  f"high={bucket_bets['high']:.0%} nuts={bucket_bets['nuts']:.0%} | "
                  f"{r['elapsed_ms']:.0f}ms")

    # =========================================================================
    # 3. Convergence: full tree at different iteration counts
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONVERGENCE — Full tree, Board: 2d 3h 4d 5d 6d, pot=(10,10)")
    print("=" * 80)

    board = boards[0]
    opp_range = make_uniform_range(board, dead_cards)

    for iters in [200, 500, 1000, 2000, 3000]:
        for dcfr in [False, True]:
            r = solver.solve_acting_first(board, dead_cards, opp_range,
                                           10, 10, iterations=iters,
                                           compact=False, use_dcfr=dcfr)
            if not r:
                continue
            strats = r['strategies']
            eqs = r['equities']
            n = len(strats)
            total_bet = sum(
                sum(v for k, v in s.items() if 'BET' in k)
                for s in strats.values()) / n

            # Bluff check: hands with equity < 0.20 that bet
            bluff_hands = [(hh, s) for hh, s in strats.items() if eqs[hh] < 0.20]
            bluff_bet = sum(sum(v for k, v in s.items() if 'BET' in k)
                           for _, s in bluff_hands) / max(len(bluff_hands), 1)

            value_hands = [(hh, s) for hh, s in strats.items() if eqs[hh] > 0.80]
            value_bet = sum(sum(v for k, v in s.items() if 'BET' in k)
                           for _, s in value_hands) / max(len(value_hands), 1)

            label = "DCFR" if dcfr else "CFR+"
            print(f"  {label} {iters:4d}: bet={total_bet:.1%} | "
                  f"bluffs(eq<20%)={bluff_bet:.1%} value(eq>80%)={value_bet:.1%} | "
                  f"{r['elapsed_ms']:.0f}ms")

    # =========================================================================
    # 4. Detailed hand-by-hand on one board
    # =========================================================================
    print("\n" + "=" * 80)
    print("HAND-BY-HAND — Full tree, 2000 iters, Board: 2d 3h 4s 7d Ah, pot=(10,10)")
    print("=" * 80)

    board = boards[4]  # rainbow disconnected
    opp_range = make_uniform_range(board, dead_cards)
    r = solver.solve_acting_first(board, dead_cards, opp_range,
                                   10, 10, iterations=2000, compact=False)
    if r:
        strats = r['strategies']
        eqs = r['equities']

        # Sort by equity, show all
        sorted_hands = sorted(strats.items(), key=lambda x: eqs[x[0]])
        print(f"\n{'Hand':<8} {'Equity':>6}  ", end="")
        for name in r['action_names']:
            print(f"{name:>10}", end="")
        print()
        print("-" * (16 + 10 * len(r['action_names'])))

        for hh, s in sorted_hands:
            eq = eqs[hh]
            bet_freq = sum(v for k, v in s.items() if 'BET' in k)
            # Only show interesting hands (betting or high equity)
            if bet_freq > 0.05 or eq > 0.70 or eq < 0.15:
                print(f"{hand_str(hh):<8} {eq:>5.0%}  ", end="")
                for name in r['action_names']:
                    print(f"{s.get(name, 0):>10.1%}", end="")
                print()

    print("\nDone.")


if __name__ == '__main__':
    main()
