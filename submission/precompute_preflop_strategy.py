"""
Offline precomputation of GTO preflop strategy via two-player CFR.

Solves the preflop betting game between SB and BB where both players
have bucketed hand potentials. The Nash equilibrium gives proper
mixed strategies (raise%, call%, fold%) for each hand strength at
each bet level.

The game tree:
  SB posts 1, BB posts 2
  SB acts: FOLD / CALL(to 2) / RAISE(to 4,8,20,100)
  BB responds: FOLD / CALL / RAISE(to next level)
  Continue until call or fold (max 4 total raise levels)

Terminal values: equity between hero_potential and opp_potential,
multiplied by the pot size. This approximates post-flop EV using
the assumption that both players play optimally after the flop.

Run:  python -m submission.precompute_preflop_strategy
Output: submission/data/preflop_strategy.npz
"""

import os
import time
import numpy as np

N_BUCKETS = 30        # hand potential buckets
CFR_ITERATIONS = 2000
MAX_BET = 100
SB_BLIND = 1
BB_BLIND = 2

# Raise levels (total bet amounts in the preflop betting tree)
RAISE_LEVELS = [4, 10, 30, 100]
# Actions: FOLD=0, CALL=1, RAISE_0=2, RAISE_1=3, RAISE_2=4, RAISE_3=5
N_ACTIONS = 2 + len(RAISE_LEVELS)  # fold, call, + each raise level
ACT_FOLD = 0
ACT_CALL = 1


def build_game_tree():
    """Build the preflop betting tree as a list of nodes.

    Each node: (player, bet_sb, bet_bb, valid_actions, children)
    player: 0=SB, 1=BB
    children: dict action_id -> child_node_id (or 'terminal_fold_SB', etc.)

    Returns list of node dicts and root node id.
    """
    nodes = []

    def add_node(player, bet_sb, bet_bb):
        nid = len(nodes)
        nodes.append({
            'player': player,
            'bet_sb': bet_sb,
            'bet_bb': bet_bb,
            'children': {},
            'terminal': None,
        })
        return nid

    def expand(nid, num_raises_so_far):
        node = nodes[nid]
        p = node['player']
        bet_sb = node['bet_sb']
        bet_bb = node['bet_bb']
        my_bet = bet_sb if p == 0 else bet_bb
        opp_bet = bet_bb if p == 0 else bet_sb

        # FOLD — terminal
        fold_id = add_node(-1, bet_sb, bet_bb)
        nodes[fold_id]['terminal'] = 'fold_by_' + ('sb' if p == 0 else 'bb')
        node['children'][ACT_FOLD] = fold_id

        # CALL — match opponent's bet
        if my_bet < opp_bet:
            call_sb = opp_bet if p == 0 else bet_sb
            call_bb = opp_bet if p == 1 else bet_bb
            call_id = add_node(-1, call_sb, call_bb)
            # After SB calls BB's blind (both at 2): BB can still act (check/raise)
            if p == 0 and opp_bet == BB_BLIND and my_bet == SB_BLIND:
                # SB limps — BB gets option
                bb_node = add_node(1, call_sb, call_bb)
                node['children'][ACT_CALL] = bb_node
                # BB can check (terminal) or raise
                check_id = add_node(-1, call_sb, call_bb)
                nodes[check_id]['terminal'] = 'showdown'
                nodes[bb_node]['children'][ACT_CALL] = check_id  # "call" = check here
                # BB can raise
                if num_raises_so_far < len(RAISE_LEVELS):
                    for ri, raise_to in enumerate(RAISE_LEVELS):
                        if raise_to > call_bb and raise_to <= MAX_BET:
                            raise_id = add_node(0, call_sb, raise_to)
                            nodes[bb_node]['children'][2 + ri] = raise_id
                            expand(raise_id, num_raises_so_far + 1)
                            break  # BB only gets one raise option after SB limps
            else:
                # Call ends the betting (SB or BB called a raise)
                nodes[call_id]['terminal'] = 'showdown'
                node['children'][ACT_CALL] = call_id
        else:
            # Bets equal, check is like call
            check_id = add_node(-1, bet_sb, bet_bb)
            nodes[check_id]['terminal'] = 'showdown'
            node['children'][ACT_CALL] = check_id

        # RAISES
        if num_raises_so_far < len(RAISE_LEVELS):
            for ri, raise_to in enumerate(RAISE_LEVELS):
                if raise_to > opp_bet and raise_to <= MAX_BET:
                    if p == 0:
                        new_sb, new_bb = raise_to, bet_bb
                    else:
                        new_sb, new_bb = bet_sb, raise_to
                    raise_nid = add_node(1 - p, new_sb, new_bb)
                    node['children'][2 + ri] = raise_nid
                    expand(raise_nid, num_raises_so_far + 1)

    # Root: SB acts first (SB=1, BB=2)
    root = add_node(0, SB_BLIND, BB_BLIND)
    expand(root, 0)

    return nodes, 0


def compute_equity_matrix(n_buckets):
    """Compute equity[i][j] = probability bucket i beats bucket j.

    Simple model: higher potential = higher equity.
    Bucket centers are evenly spaced from 0 to 1 (normalized).
    """
    equity = np.zeros((n_buckets, n_buckets), dtype=np.float64)
    for i in range(n_buckets):
        for j in range(n_buckets):
            if i > j:
                equity[i][j] = 0.65  # stronger hand wins ~65% (not 100% due to board variance)
            elif i < j:
                equity[i][j] = 0.35
            else:
                equity[i][j] = 0.50
    return equity


def solve_cfr(nodes, root, n_buckets, equity_matrix, iterations):
    """Run CFR over the game tree for all bucket pairs.

    Hero can be SB or BB. We solve for BOTH simultaneously.
    regrets[node_id][bucket][action] for each player.
    """
    n_nodes = len(nodes)

    # Identify decision nodes per player
    sb_nodes = [i for i, n in enumerate(nodes) if n['player'] == 0]
    bb_nodes = [i for i, n in enumerate(nodes) if n['player'] == 1]

    # Regrets and strategy sums
    # Indexed by [node_id, bucket, action]
    regrets = np.zeros((n_nodes, n_buckets, N_ACTIONS), dtype=np.float64)
    strat_sum = np.zeros((n_nodes, n_buckets, N_ACTIONS), dtype=np.float64)

    def get_strategy(node_id, bucket):
        r = regrets[node_id, bucket]
        valid = list(nodes[node_id]['children'].keys())
        pos = np.maximum(r, 0)
        for a in range(N_ACTIONS):
            if a not in valid:
                pos[a] = 0
        total = pos.sum()
        if total > 0:
            return pos / total
        s = np.zeros(N_ACTIONS)
        for a in valid:
            s[a] = 1.0 / len(valid)
        return s

    def cfr_walk(node_id, sb_bucket, bb_bucket, sb_reach, bb_reach):
        """Walk the tree, return utility for SB."""
        node = nodes[node_id]

        # Terminal
        if node['terminal'] is not None:
            bet_sb = node['bet_sb']
            bet_bb = node['bet_bb']
            pot_won = min(bet_sb, bet_bb)

            if node['terminal'] == 'fold_by_sb':
                return -bet_sb  # SB loses their bet
            elif node['terminal'] == 'fold_by_bb':
                return bet_bb  # SB wins BB's bet
            elif node['terminal'] == 'showdown':
                eq = equity_matrix[sb_bucket, bb_bucket]
                return eq * pot_won - (1 - eq) * pot_won
            return 0

        player = node['player']
        bucket = sb_bucket if player == 0 else bb_bucket
        strategy = get_strategy(node_id, bucket)

        valid_actions = list(node['children'].keys())
        action_utils = {}
        node_util = 0.0

        for a in valid_actions:
            child_id = node['children'][a]
            if player == 0:
                action_utils[a] = cfr_walk(child_id, sb_bucket, bb_bucket,
                                           sb_reach * strategy[a], bb_reach)
            else:
                action_utils[a] = cfr_walk(child_id, sb_bucket, bb_bucket,
                                           sb_reach, bb_reach * strategy[a])
            node_util += strategy[a] * action_utils[a]

        # Update regrets
        opp_reach = bb_reach if player == 0 else sb_reach
        sign = 1.0 if player == 0 else -1.0  # SB utility; BB is negative

        for a in valid_actions:
            regret = sign * (action_utils[a] - node_util) * opp_reach
            regrets[node_id, bucket, a] = max(0, regrets[node_id, bucket, a] + regret)

        # Update strategy sum
        my_reach = sb_reach if player == 0 else bb_reach
        strat_sum[node_id, bucket] += my_reach * strategy

        return node_util

    # Run CFR iterations
    # For each iteration, sample all bucket pairs
    # (with 30 buckets, that's 900 pairs — fast)
    for t in range(iterations):
        if t % 500 == 0:
            print(f"  Iteration {t}/{iterations}...")

        for sb_b in range(n_buckets):
            for bb_b in range(n_buckets):
                # Weight by probability of this matchup
                # (uniform for simplicity)
                cfr_walk(root, sb_b, bb_b, 1.0, 1.0)

    # Extract average strategies
    # For each node and bucket, normalize strategy_sum
    avg_strats = np.zeros_like(strat_sum)
    for nid in range(n_nodes):
        for b in range(n_buckets):
            total = strat_sum[nid, b].sum()
            if total > 0:
                avg_strats[nid, b] = strat_sum[nid, b] / total
            else:
                valid = list(nodes[nid]['children'].keys())
                if valid:
                    for a in valid:
                        avg_strats[nid, b, a] = 1.0 / len(valid)

    return avg_strats


def main():
    print("Building preflop game tree...")
    nodes, root = build_game_tree()
    print(f"Tree: {len(nodes)} nodes")

    # Count decision nodes
    sb_dec = sum(1 for n in nodes if n['player'] == 0)
    bb_dec = sum(1 for n in nodes if n['player'] == 1)
    term = sum(1 for n in nodes if n['terminal'] is not None)
    print(f"  SB decisions: {sb_dec}, BB decisions: {bb_dec}, Terminals: {term}")

    print(f"\nComputing equity matrix ({N_BUCKETS}x{N_BUCKETS})...")
    equity_matrix = compute_equity_matrix(N_BUCKETS)

    print(f"Running CFR ({CFR_ITERATIONS} iterations, {N_BUCKETS} buckets)...")
    t0 = time.time()
    avg_strats = solve_cfr(nodes, root, N_BUCKETS, equity_matrix, CFR_ITERATIONS)
    elapsed = time.time() - t0
    print(f"Solved in {elapsed:.1f}s")

    # Extract strategies for the key decision points:
    # 1. SB opening (root node, player=0, bets: SB=1, BB=2)
    # 2. BB facing SB raise (each raise level)
    # 3. SB facing BB 3-bet

    # Find key nodes
    print("\nKey strategies:")
    action_names = ['fold', 'call'] + [f'raise_{r}' for r in RAISE_LEVELS]

    def print_strategy(node_id, label):
        node = nodes[node_id]
        valid = list(node['children'].keys())
        print(f"\n  {label} (node {node_id}, player={'SB' if node['player']==0 else 'BB'}):")
        for b in range(0, N_BUCKETS, N_BUCKETS // 6):
            s = avg_strats[node_id, b]
            parts = [f"{action_names[a]}:{s[a]:.0%}" for a in valid if s[a] > 0.005]
            strength = (b + 0.5) / N_BUCKETS * 100
            print(f"    Strength {strength:4.0f}%: {' '.join(parts)}")

    # Root = SB opening
    print_strategy(root, "SB Opening (facing BB=2)")

    # Find BB response nodes
    for a, child_id in nodes[root]['children'].items():
        if nodes[child_id]['player'] == 1 and a >= 2:
            print_strategy(child_id, f"BB Facing SB raise to {RAISE_LEVELS[a-2]}")
            break
    # BB after SB limps
    if ACT_CALL in nodes[root]['children']:
        limp_child = nodes[root]['children'][ACT_CALL]
        if isinstance(limp_child, int) and nodes[limp_child]['player'] == 1:
            print_strategy(limp_child, "BB After SB Limps")

    # Save: for runtime, we need the strategy for each (position, bucket, facing_bet)
    # Simplify: extract SB opening strategy and BB response strategies

    # Load preflop potential range for bucket mapping
    data_path = os.path.join(os.path.dirname(__file__), "data", "preflop_potential.npz")
    data = np.load(data_path)
    pot_min = float(data["potentials"].min())
    pot_max = float(data["potentials"].max())

    # Save strategy for key decision nodes + bucket mapping info
    out_path = os.path.join(os.path.dirname(__file__), "data", "preflop_strategy.npz")

    # Collect all strategies indexed by node
    np.savez_compressed(
        out_path,
        strategies=avg_strats.astype(np.float32),
        pot_min=np.float32(pot_min),
        pot_max=np.float32(pot_max),
        n_buckets=np.int32(N_BUCKETS),
        raise_levels=np.array(RAISE_LEVELS, dtype=np.int32),
        # Save tree structure for runtime navigation
        node_players=np.array([n['player'] for n in nodes], dtype=np.int8),
        node_bet_sb=np.array([n['bet_sb'] for n in nodes], dtype=np.int32),
        node_bet_bb=np.array([n['bet_bb'] for n in nodes], dtype=np.int32),
        node_terminals=np.array([0 if n['terminal'] is None else 1 for n in nodes], dtype=np.int8),
        # Children as a flat mapping: node_id -> action -> child_id
        # Store as: for each node, list of (action, child) pairs
        n_nodes=np.int32(len(nodes)),
    )

    # Also save children mapping separately (variable-length, use pickle)
    import pickle
    children_path = os.path.join(os.path.dirname(__file__), "data", "preflop_tree.pkl")
    children_map = {i: dict(n['children']) for i, n in enumerate(nodes)}
    with open(children_path, 'wb') as f:
        pickle.dump(children_map, f)

    file_size = os.path.getsize(out_path)
    print(f"\nSaved to {out_path} ({file_size / 1024:.1f} KB)")
    print(f"Tree saved to {children_path}")


if __name__ == "__main__":
    main()
