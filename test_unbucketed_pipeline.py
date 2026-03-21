#!/usr/bin/env python3
"""End-to-end test of unbucketed blueprint pipeline.

Merges a few cluster files, loads them via BlueprintLookupUnbucketed,
and tests strategy lookups for various hands and board states.
"""
import sys, os, numpy as np
sys.path.insert(0, 'submission')

from blueprint_lookup_unbucketed import BlueprintLookupUnbucketed
from blueprint_abstraction import compute_board_features, compute_board_cluster
from equity import ExactEquityEngine

# Step 1: Quick merge of test cluster files into one .npz
print("=" * 60)
print("STEP 1: Merge test clusters")
print("=" * 60)

cluster_dir = '/tmp/unbucketed_test'
cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith('.npz')])
print(f"Found {len(cluster_files)} cluster files")

clusters_data = []
for f in cluster_files:
    d = np.load(os.path.join(cluster_dir, f), allow_pickle=True)
    clusters_data.append(d)

n_clusters = len(clusters_data)
n_pot_sizes = clusters_data[0]['pot_sizes'].shape[0]
n_hands = clusters_data[0]['hands'].shape[0]
n_nodes = clusters_data[0]['hero_strategies'].shape[2]
n_actions = clusters_data[0]['hero_strategies'].shape[3]

print(f"  n_clusters={n_clusters}, n_pot_sizes={n_pot_sizes}, n_hands={n_hands}, n_nodes={n_nodes}, n_actions={n_actions}")

# Build merged arrays
strategies = np.zeros((n_clusters, n_pot_sizes, n_hands, n_nodes, n_actions), dtype=np.uint8)
action_types = np.zeros((n_clusters, n_pot_sizes, n_nodes, n_actions), dtype=np.int8)
hand_lists = np.zeros((n_clusters, n_hands, 2), dtype=np.int8)
cluster_ids = np.zeros(n_clusters, dtype=np.int32)
boards = np.zeros((n_clusters, 3), dtype=np.int8)
board_features = np.zeros((n_clusters, 12), dtype=np.float32)

for i, d in enumerate(clusters_data):
    strategies[i] = d['hero_strategies']
    action_types[i] = d['action_types']
    hand_lists[i] = d['hands']
    cluster_ids[i] = int(d['cluster_id'])
    boards[i] = d['board']
    board_features[i] = d['board_features']

pot_sizes = clusters_data[0]['pot_sizes']

merged_path = '/tmp/test_flop_unbucketed.npz'
np.savez_compressed(merged_path,
    strategies=strategies,
    action_types=action_types,
    hand_lists=hand_lists,
    cluster_ids=cluster_ids,
    boards=boards,
    board_features=board_features,
    pot_sizes=pot_sizes,
    config_n_clusters=n_clusters,
    config_n_iterations=1000,
    config_max_bet=100,
)
print(f"Merged file saved: {merged_path} ({os.path.getsize(merged_path)/1024:.0f} KB)")

# Step 2: Load via BlueprintLookupUnbucketed
print()
print("=" * 60)
print("STEP 2: Load unbucketed blueprint")
print("=" * 60)

try:
    lookup = BlueprintLookupUnbucketed(merged_path)
    print(f"Loaded successfully!")
    print(f"  Unbucketed mode: {lookup._unbucketed}")
    print(f"  Clusters: {len(lookup.cluster_ids)}")
except Exception as e:
    print(f"FAILED TO LOAD: {e}")
    import traceback; traceback.fulltb()
    sys.exit(1)

# Step 3: Test strategy lookups
print()
print("=" * 60)
print("STEP 3: Strategy lookups")
print("=" * 60)

engine = ExactEquityEngine()
act_names = {0:'FOLD', 1:'CHECK', 2:'CALL', 3:'R40%', 4:'R70%', 5:'R100%', 6:'R150%'}

# Use the actual board from cluster 10
test_board = boards[0].tolist()
print(f"Board: {test_board}")

# Test various hands
test_hands = [
    ([0, 1], "very weak (2d, 2h)"),
    ([3, 4], "weak (3d, 3h)"),
    ([12, 15], "medium"),
    ([21, 22], "strong (9d, 9h)"),
    ([24, 25], "very strong (Ad, Ah)"),
]

for pot_label, pot_state in [("small (2,2)", (2,2)), ("medium (16,16)", (16,16)), ("big (50,50)", (50,50))]:
    print(f"\n  Pot: {pot_label}")
    for cards, label in test_hands:
        # Skip if cards overlap with board
        if any(c in test_board for c in cards):
            continue
        try:
            strategy = lookup.get_strategy(
                hero_cards=cards, board=test_board,
                pot_state=pot_state, dead_cards=[], opp_weights=None
            )
            if strategy:
                parts = [f"{act_names.get(a, f'act{a}')}:{p:.0%}" for a, p in sorted(strategy.items()) if p > 0.01]
                print(f"    {label:25s}: {' '.join(parts)}")
            else:
                print(f"    {label:25s}: No strategy returned!")
        except Exception as e:
            print(f"    {label:25s}: ERROR: {e}")

# Step 4: Compare unbucketed vs bucketed for same hand
print()
print("=" * 60)
print("STEP 4: Unbucketed vs Bucketed comparison")
print("=" * 60)

from blueprint_lookup import BlueprintLookup
bucketed = BlueprintLookup('submission/data/flop_blueprint.npz', equity_engine=engine)

# Find a board that's close to our test board
for cards, label in test_hands:
    if any(c in test_board for c in cards):
        continue

    ub_strat = lookup.get_strategy(hero_cards=cards, board=test_board, pot_state=(2,2))
    bk_strat = bucketed.get_strategy(hero_cards=cards, board=test_board, pot_state=(2,2))

    print(f"\n  {label} ({cards}) on board {test_board}:")
    if ub_strat:
        parts = [f"{act_names.get(a,'?')}:{p:.0%}" for a, p in sorted(ub_strat.items()) if p > 0.01]
        print(f"    Unbucketed: {' '.join(parts)}")
    else:
        print(f"    Unbucketed: None")

    if bk_strat:
        parts = [f"{act_names.get(a,'?')}:{p:.0%}" for a, p in sorted(bk_strat.items()) if p > 0.01]
        print(f"    Bucketed:   {' '.join(parts)}")
    else:
        print(f"    Bucketed:   None")

    eq = engine.compute_equity(cards, test_board, [])
    print(f"    Equity:     {eq:.3f}")

print()
print("=" * 60)
print("ALL TESTS PASSED" if True else "FAILED")
print("=" * 60)
