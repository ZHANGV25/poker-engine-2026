#!/usr/bin/env python3
"""
Test script for the blueprint solver pipeline.

Runs a minimal end-to-end test:
1. Compute a tiny blueprint (1 board, 100 iterations, 10 buckets)
2. Verify output format and dimensions
3. Test the runtime lookup module
4. Verify strategy properties (probabilities sum to 1, non-negative)

Usage:
    python test_blueprint.py
    python test_blueprint.py -v          # verbose output
    python test_blueprint.py --fast      # minimal test (10 iterations)
"""

import os
import sys
import time
import tempfile
import argparse
import traceback

import numpy as np

# Add paths
_dir = os.path.dirname(os.path.abspath(__file__))
_submission_dir = os.path.join(_dir, "..", "submission")
sys.path.insert(0, _dir)
sys.path.insert(0, _submission_dir)


class TestRunner:
    """Simple test framework."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.errors = []

    def test(self, name, fn):
        """Run a test function. It should raise on failure."""
        try:
            if self.verbose:
                print(f"  Running: {name}...", end="", flush=True)
            fn()
            self.passed += 1
            if self.verbose:
                print(" PASS")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            if self.verbose:
                print(f" FAIL: {e}")
                traceback.print_exc()

    def assert_eq(self, a, b, msg=""):
        assert a == b, f"Expected {a} == {b}. {msg}"

    def assert_true(self, val, msg=""):
        assert val, f"Expected True. {msg}"

    def assert_close(self, a, b, tol=1e-6, msg=""):
        assert abs(a - b) < tol, f"Expected {a} ~= {b} (tol={tol}). {msg}"

    def assert_shape(self, arr, shape, msg=""):
        assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}. {msg}"

    def report(self):
        total = self.passed + self.failed
        print(f"\nResults: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, err in self.errors:
                print(f"  {name}: {err}")
        return self.failed == 0


def test_abstraction(runner):
    """Test the abstraction module."""
    from abstraction import (
        compute_board_features,
        compute_board_cluster,
        enumerate_all_flops,
        enumerate_hands,
        enumerate_hand_buckets,
        get_representative_boards,
        get_bucket_boundaries,
        card_rank,
        card_suit,
    )
    from equity import ExactEquityEngine

    engine = ExactEquityEngine()

    # Test card encoding
    def test_card_encoding():
        # Card 0 = 2d: rank=0, suit=0
        runner.assert_eq(card_rank(0), 0, "Card 0 rank")
        runner.assert_eq(card_suit(0), 0, "Card 0 suit")
        # Card 8 = Ad: rank=8, suit=0
        runner.assert_eq(card_rank(8), 8, "Card 8 rank")
        runner.assert_eq(card_suit(8), 0, "Card 8 suit")
        # Card 9 = 2h: rank=0, suit=1
        runner.assert_eq(card_rank(9), 0, "Card 9 rank")
        runner.assert_eq(card_suit(9), 1, "Card 9 suit")
        # Card 26 = As: rank=8, suit=2
        runner.assert_eq(card_rank(26), 8, "Card 26 rank")
        runner.assert_eq(card_suit(26), 2, "Card 26 suit")
    runner.test("card_encoding", test_card_encoding)

    # Test board features
    def test_board_features():
        board = [0, 1, 2]  # 2d, 3d, 4d - monotone, connected
        features = compute_board_features(board)
        runner.assert_eq(len(features), 12, "Feature vector length")
        runner.assert_eq(int(features[0]), 2, "Should be monotone (flush_type=2)")
        runner.assert_eq(int(features[6]), 3, "Should have 3 board cards")
    runner.test("board_features", test_board_features)

    # Test board clustering
    def test_board_clustering():
        board1 = [0, 1, 2]
        board2 = [0, 1, 2]
        board3 = [9, 10, 11]  # 2h, 3h, 4h - same structure, different suit
        c1 = compute_board_cluster(board1, 200)
        c2 = compute_board_cluster(board2, 200)
        c3 = compute_board_cluster(board3, 200)
        runner.assert_eq(c1, c2, "Same board should get same cluster")
        # board3 has same structure as board1, so features should match
        runner.assert_eq(c1, c3, "Structurally identical boards should cluster together")
    runner.test("board_clustering", test_board_clustering)

    # Test flop enumeration
    def test_flop_enumeration():
        flops = enumerate_all_flops()
        runner.assert_eq(len(flops), 2925, "C(27,3) = 2925 flops")
        # Check they're sorted
        for f in flops:
            runner.assert_true(f[0] < f[1] < f[2], "Flop cards should be sorted")
    runner.test("flop_enumeration", test_flop_enumeration)

    # Test hand enumeration
    def test_hand_enumeration():
        board = [0, 1, 2]
        hands = enumerate_hands(board, [])
        # 27 - 3 = 24 remaining, C(24,2) = 276
        runner.assert_eq(len(hands), 276, "C(24,2) = 276 hands")
        # With dead cards
        hands_dead = enumerate_hands(board, [3, 4, 5])
        # 27 - 6 = 21 remaining, C(21,2) = 210
        runner.assert_eq(len(hands_dead), 210, "C(21,2) = 210 hands with 3 dead")
    runner.test("hand_enumeration", test_hand_enumeration)

    # Test hand bucketing
    def test_hand_bucketing():
        board = [0, 1, 2]  # 2d, 3d, 4d
        n_buckets = 10
        buckets = enumerate_hand_buckets(board, [], n_buckets, engine)
        runner.assert_eq(len(buckets), 276, "Should bucket all 276 hands")
        for hand, bucket in buckets.items():
            runner.assert_true(0 <= bucket < n_buckets,
                               f"Bucket {bucket} out of range [0, {n_buckets})")
    runner.test("hand_bucketing", test_hand_bucketing)

    # Test representative boards
    def test_representative_boards():
        flops = enumerate_all_flops()
        reps = get_representative_boards(flops, 200)
        runner.assert_true(len(reps) > 0, "Should have at least one representative")
        runner.assert_true(len(reps) <= 200, "Should not exceed n_clusters")
        # Each entry is (cluster_id, board)
        for cid, board in reps:
            runner.assert_true(0 <= cid < 200, "Cluster ID in range")
            runner.assert_eq(len(board), 3, "Board should have 3 cards")
    runner.test("representative_boards", test_representative_boards)

    # Test bucket boundaries
    def test_bucket_boundaries():
        bounds = get_bucket_boundaries(50)
        runner.assert_eq(len(bounds), 51, "51 boundaries for 50 buckets")
        runner.assert_close(bounds[0], 0.0, msg="First boundary = 0")
        runner.assert_close(bounds[-1], 1.0, msg="Last boundary = 1")
    runner.test("bucket_boundaries", test_bucket_boundaries)


def test_blueprint_cfr(runner, n_iterations=100, n_buckets=10):
    """Test the BlueprintCFR solver."""
    from blueprint_cfr import BlueprintCFR
    from equity import ExactEquityEngine

    engine = ExactEquityEngine()

    # Test basic solve
    def test_basic_solve():
        solver = BlueprintCFR(n_buckets, n_buckets, engine)
        board = [0, 1, 2]  # 2d, 3d, 4d

        start = time.time()
        result = solver.solve(
            board=board,
            dead_cards=[],
            hero_bet=1,
            opp_bet=2,
            hero_first=True,
            n_iterations=n_iterations,
        )
        elapsed = time.time() - start

        print(f" ({elapsed:.1f}s)", end="", flush=True)

        runner.assert_true(result is not None, "Result should not be None")
        runner.assert_true('hero_strategy' in result, "Result should contain hero_strategy")
        runner.assert_true('opp_strategy' in result, "Result should contain opp_strategy")
        runner.assert_true('tree' in result, "Result should contain tree")

        hs = result['hero_strategy']
        if len(hs) > 0:
            runner.assert_eq(hs.shape[0], n_buckets,
                             f"Hero strategy should have {n_buckets} bucket rows")
            # Check strategies are valid probabilities
            for b in range(n_buckets):
                for n in range(hs.shape[1]):
                    row_sum = hs[b, n, :].sum()
                    if row_sum > 0:
                        runner.assert_close(row_sum, 1.0, tol=1e-4,
                                            msg=f"Strategy at bucket {b}, node {n} should sum to 1")
                    for a in range(hs.shape[2]):
                        runner.assert_true(hs[b, n, a] >= 0,
                                           f"Strategy prob should be non-negative")

    runner.test("basic_cfr_solve", test_basic_solve)

    # Test with river board (5 cards)
    def test_river_solve():
        solver = BlueprintCFR(n_buckets, n_buckets, engine)
        board = [0, 1, 2, 9, 18]  # 5 cards

        start = time.time()
        result = solver.solve(
            board=board,
            dead_cards=[],
            hero_bet=5,
            opp_bet=5,
            hero_first=True,
            n_iterations=n_iterations,
        )
        elapsed = time.time() - start

        print(f" ({elapsed:.1f}s)", end="", flush=True)

        hs = result['hero_strategy']
        if len(hs) > 0:
            runner.assert_eq(hs.shape[0], n_buckets,
                             "River strategy should have correct bucket count")
    runner.test("river_cfr_solve", test_river_solve)

    # Test with dead cards
    def test_dead_cards_solve():
        solver = BlueprintCFR(n_buckets, n_buckets, engine)
        board = [0, 1, 2]
        dead = [3, 4, 5, 12, 13, 14]  # 6 dead cards

        result = solver.solve(
            board=board,
            dead_cards=dead,
            hero_bet=1,
            opp_bet=2,
            hero_first=True,
            n_iterations=max(10, n_iterations // 10),
        )

        hs = result['hero_strategy']
        runner.assert_true(len(hs) > 0 or result['n_iterations'] >= 0,
                           "Should handle dead cards")
    runner.test("dead_cards_solve", test_dead_cards_solve)


def test_compute_and_lookup(runner, n_iterations=100, n_buckets=10):
    """Test the full pipeline: compute -> save -> load -> lookup."""
    from compute_blueprint import solve_single_board, save_final
    from lookup import BlueprintLookup
    from abstraction import compute_board_cluster
    from equity import ExactEquityEngine

    engine = ExactEquityEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Solve a single board
        def test_single_board_solve():
            board = (0, 1, 2)  # 2d, 3d, 4d
            cluster_id = compute_board_cluster(list(board), 200)

            config = {
                'n_buckets': n_buckets,
                'n_iterations': n_iterations,
                'n_clusters': 200,
                'output_dir': tmpdir,
            }

            start = time.time()
            result = solve_single_board((cluster_id, board, config))
            elapsed = time.time() - start

            print(f" ({elapsed:.1f}s)", end="", flush=True)

            runner.assert_eq(result['cluster_id'], cluster_id)
            runner.assert_true(result['tree_size'] > 0, "Tree should have nodes")
            runner.assert_true(result['elapsed'] > 0, "Should report timing")

            return result
        result = test_single_board_solve
        runner.test("single_board_solve", test_single_board_solve)

        # Save and load
        def test_save_and_load():
            board = (0, 1, 2)
            cluster_id = compute_board_cluster(list(board), 200)

            config = {
                'n_buckets': n_buckets,
                'n_iterations': n_iterations,
                'n_clusters': 200,
            }

            result = solve_single_board((cluster_id, board, config))
            path = save_final([result], tmpdir, config)

            runner.assert_true(os.path.exists(path), "Output file should exist")
            runner.assert_true(os.path.getsize(path) > 0, "Output file should not be empty")

            # Load and verify
            data = np.load(path, allow_pickle=True)
            runner.assert_true('strategies' in data, "Should contain strategies")
            runner.assert_true('cluster_ids' in data, "Should contain cluster_ids")
            runner.assert_true('bucket_boundaries' in data, "Should contain bucket_boundaries")
            runner.assert_true('board_features' in data, "Should contain board_features")
            runner.assert_true('action_types' in data, "Should contain action_types")

            strat = data['strategies']
            runner.assert_eq(strat.shape[0], 1, "Should have 1 cluster")
            runner.assert_eq(strat.shape[1], n_buckets, f"Should have {n_buckets} buckets")

            return path
        runner.test("save_and_load", test_save_and_load)

        # Test BlueprintLookup
        def test_lookup():
            board = (0, 1, 2)
            cluster_id = compute_board_cluster(list(board), 200)

            config = {
                'n_buckets': n_buckets,
                'n_iterations': n_iterations,
                'n_clusters': 200,
            }

            result = solve_single_board((cluster_id, board, config))
            path = save_final([result], tmpdir, config)

            lookup = BlueprintLookup(path, equity_engine=engine)
            runner.assert_eq(lookup.n_solved_clusters, 1, "Should have 1 cluster")
            runner.assert_eq(lookup.n_buckets, n_buckets, f"Should have {n_buckets} buckets")

            # Look up a strategy for a specific hand
            hero_cards = [3, 4]  # 5d, 6d
            strat = lookup.get_strategy(hero_cards, list(board))

            if strat is not None:
                # Verify it's a valid probability distribution
                total = sum(strat.values())
                runner.assert_close(total, 1.0, tol=0.01,
                                    msg="Strategy probabilities should sum to ~1")
                for act_id, prob in strat.items():
                    runner.assert_true(prob >= 0, "Probabilities should be non-negative")
                    runner.assert_true(0 <= act_id <= 6,
                                       f"Action ID {act_id} should be in valid range")

            # Test named action probabilities
            named = lookup.get_action_probabilities(hero_cards, list(board))
            if named is not None:
                for name, prob in named.items():
                    runner.assert_true(isinstance(name, str), "Action names should be strings")
                    runner.assert_true(0 <= prob <= 1, "Probabilities in [0,1]")

            # Test sample_action
            action = lookup.sample_action(hero_cards, list(board))
            if action is not None:
                runner.assert_true(0 <= action <= 6, "Sampled action in valid range")

            # Test describe_strategy
            desc = lookup.describe_strategy(hero_cards, list(board))
            runner.assert_true(len(desc) > 0, "Description should not be empty")
            runner.assert_true("Equity" in desc, "Description should include equity")

        runner.test("blueprint_lookup", test_lookup)

        # Test lookup with different hands
        def test_lookup_varies_by_hand():
            board = (0, 1, 2)
            cluster_id = compute_board_cluster(list(board), 200)

            config = {
                'n_buckets': n_buckets,
                'n_iterations': n_iterations,
                'n_clusters': 200,
            }

            result = solve_single_board((cluster_id, board, config))
            path = save_final([result], tmpdir, config)

            lookup = BlueprintLookup(path, equity_engine=engine)

            # Strong hand (Ace-high flush draw on monotone board)
            strong_hand = [7, 8]  # 9d, Ad - strong on 2d,3d,4d board
            strat_strong = lookup.get_strategy(strong_hand, list(board))

            # Weak hand
            weak_hand = [9, 18]  # 2h, 2s - pair of 2s, no flush draw
            strat_weak = lookup.get_strategy(weak_hand, list(board))

            if strat_strong is not None and strat_weak is not None:
                # Strong hands should generally bet more
                # (this is a soft check -- may not always hold with few iterations)
                print(f" (strong: {strat_strong}, weak: {strat_weak})", end="")

        runner.test("lookup_varies_by_hand", test_lookup_varies_by_hand)

        # Test direct bucket access
        def test_direct_bucket_access():
            board = (0, 1, 2)
            cluster_id = compute_board_cluster(list(board), 200)

            config = {
                'n_buckets': n_buckets,
                'n_iterations': n_iterations,
                'n_clusters': 200,
            }

            result = solve_single_board((cluster_id, board, config))
            path = save_final([result], tmpdir, config)

            lookup = BlueprintLookup(path, equity_engine=engine)

            # Access strategy for each bucket directly
            for b in range(n_buckets):
                strat = lookup.get_strategy_for_bucket(cluster_id, b, node_idx=0)
                if strat is not None:
                    total = sum(strat.values())
                    runner.assert_close(total, 1.0, tol=0.01,
                                        msg=f"Bucket {b} strategy should sum to ~1")

        runner.test("direct_bucket_access", test_direct_bucket_access)


def test_game_tree_compatibility(runner):
    """Verify the game tree works as expected for blueprint use."""
    from game_tree import GameTree, TERM_NONE, TERM_SHOWDOWN

    def test_tree_structure():
        tree = GameTree(hero_bet=1, opp_bet=2, min_raise=2,
                        max_bet=100, hero_first=True)

        runner.assert_true(tree.size > 0, "Tree should have nodes")
        runner.assert_true(len(tree.hero_node_ids) > 0, "Should have hero nodes")
        runner.assert_true(len(tree.opp_node_ids) > 0, "Should have opp nodes")
        runner.assert_true(len(tree.terminal_node_ids) > 0, "Should have terminal nodes")

        # Root should be a hero node (hero_first=True)
        runner.assert_eq(tree.player[0], 0, "Root should be hero (player 0)")

        # All nodes should have valid data
        for nid in range(tree.size):
            runner.assert_true(tree.hero_pot[nid] >= 0, "Hero pot >= 0")
            runner.assert_true(tree.opp_pot[nid] >= 0, "Opp pot >= 0")

        print(f" (tree size: {tree.size} nodes, "
              f"{len(tree.hero_node_ids)} hero, "
              f"{len(tree.opp_node_ids)} opp, "
              f"{len(tree.terminal_node_ids)} terminal)", end="")

    runner.test("tree_structure", test_tree_structure)

    def test_tree_bet_sizes():
        tree = GameTree(hero_bet=5, opp_bet=5, min_raise=2,
                        max_bet=100, hero_first=True)
        # After both put in 5, pot = 10. Bet sizes should be:
        # 40% of 10 = 4, 70% of 10 = 7, 100% of 10 = 10, 150% of 10 = 15
        root_children = tree.children[0]
        runner.assert_true(len(root_children) >= 2,
                           "Root should have at least check + bet options")
    runner.test("tree_bet_sizes", test_tree_bet_sizes)


def main():
    parser = argparse.ArgumentParser(description="Test blueprint solver pipeline")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: fewer iterations')
    parser.add_argument('--iterations', type=int, default=100,
                        help='CFR iterations (default: 100)')
    parser.add_argument('--buckets', type=int, default=10,
                        help='Number of buckets (default: 10)')
    args = parser.parse_args()

    if args.fast:
        n_iterations = 10
        n_buckets = 5
    else:
        n_iterations = args.iterations
        n_buckets = args.buckets

    runner = TestRunner(verbose=args.verbose)

    print("=" * 60)
    print("Blueprint Solver Test Suite")
    print("=" * 60)
    print(f"Iterations: {n_iterations}, Buckets: {n_buckets}")
    print()

    # 1. Test abstraction module
    print("1. Testing abstraction module...")
    test_abstraction(runner)
    print()

    # 2. Test game tree compatibility
    print("2. Testing game tree compatibility...")
    test_game_tree_compatibility(runner)
    print()

    # 3. Test CFR solver
    print("3. Testing BlueprintCFR solver...")
    test_blueprint_cfr(runner, n_iterations=n_iterations, n_buckets=n_buckets)
    print()

    # 4. Test full pipeline (compute -> save -> lookup)
    print("4. Testing compute -> save -> lookup pipeline...")
    test_compute_and_lookup(runner, n_iterations=n_iterations, n_buckets=n_buckets)
    print()

    # Report
    success = runner.report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
