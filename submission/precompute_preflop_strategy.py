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

try:
    import numba
    @numba.njit(cache=True)
    def _cfr_walk_numba(node_id, sb_b, bb_b, sb_reach, bb_reach,
                         player, terminal_type, bet_sb, bet_bb,
                         children, valid_mask, equity_matrix,
                         regrets, strat_sum, n_actions):
        """Numba-compiled CFR tree walk. Returns SB utility."""
        # Terminal
        tt = terminal_type[node_id]
        if tt > 0:
            if tt == 1:  # fold_by_sb
                return -bet_sb[node_id]
            elif tt == 2:  # fold_by_bb
                return bet_bb[node_id]
            elif tt == 3:  # showdown
                pot = min(bet_sb[node_id], bet_bb[node_id])
                eq = equity_matrix[sb_b, bb_b]
                return (2.0 * eq - 1.0) * pot
            return 0.0

        p = player[node_id]
        bucket = sb_b if p == 0 else bb_b

        # Regret matching
        strategy = np.zeros(n_actions, dtype=np.float64)
        pos_sum = 0.0
        n_valid = 0
        for a in range(n_actions):
            if valid_mask[node_id, a]:
                v = max(0.0, regrets[node_id, bucket, a])
                strategy[a] = v
                pos_sum += v
                n_valid += 1

        if pos_sum > 0:
            for a in range(n_actions):
                strategy[a] /= pos_sum
        else:
            for a in range(n_actions):
                if valid_mask[node_id, a]:
                    strategy[a] = 1.0 / n_valid

        # Traverse children
        node_util = 0.0
        action_utils = np.zeros(n_actions, dtype=np.float64)
        for a in range(n_actions):
            if not valid_mask[node_id, a]:
                continue
            child = children[node_id, a]
            if p == 0:
                action_utils[a] = _cfr_walk_numba(
                    child, sb_b, bb_b, sb_reach * strategy[a], bb_reach,
                    player, terminal_type, bet_sb, bet_bb,
                    children, valid_mask, equity_matrix,
                    regrets, strat_sum, n_actions)
            else:
                action_utils[a] = _cfr_walk_numba(
                    child, sb_b, bb_b, sb_reach, bb_reach * strategy[a],
                    player, terminal_type, bet_sb, bet_bb,
                    children, valid_mask, equity_matrix,
                    regrets, strat_sum, n_actions)
            node_util += strategy[a] * action_utils[a]

        # Update regrets
        opp_reach = bb_reach if p == 0 else sb_reach
        sign = 1.0 if p == 0 else -1.0
        for a in range(n_actions):
            if valid_mask[node_id, a]:
                r = sign * (action_utils[a] - node_util) * opp_reach
                regrets[node_id, bucket, a] = max(0.0, regrets[node_id, bucket, a] + r)

        # Update strategy sum
        my_reach = sb_reach if p == 0 else bb_reach
        for a in range(n_actions):
            strat_sum[node_id, bucket, a] += my_reach * strategy[a]

        return node_util

    @numba.njit(cache=True)
    def _run_preflop_cfr(root, n_nodes, n_buckets, n_actions, iterations,
                          player, terminal_type, bet_sb, bet_bb,
                          children, valid_mask, equity_matrix,
                          regrets, strat_sum):
        for t in range(iterations):
            for sb_b in range(n_buckets):
                for bb_b in range(n_buckets):
                    _cfr_walk_numba(root, sb_b, bb_b, 1.0, 1.0,
                                    player, terminal_type, bet_sb, bet_bb,
                                    children, valid_mask, equity_matrix,
                                    regrets, strat_sum, n_actions)

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def _run_preflop_cfr(root, n_nodes, n_buckets, n_actions, iterations,
                          player, terminal_type, bet_sb, bet_bb,
                          children, valid_mask, equity_matrix,
                          regrets, strat_sum):
        """Pure Python fallback."""
        for t in range(iterations):
            if t % 100 == 0:
                print(f"  CFR iteration {t}/{iterations}", flush=True)
            for sb_b in range(n_buckets):
                for bb_b in range(n_buckets):
                    _cfr_walk_python(root, sb_b, bb_b, 1.0, 1.0,
                                     player, terminal_type, bet_sb, bet_bb,
                                     children, valid_mask, equity_matrix,
                                     regrets, strat_sum, n_actions)

    def _cfr_walk_python(node_id, sb_b, bb_b, sb_reach, bb_reach,
                          player, terminal_type, bet_sb, bet_bb,
                          children, valid_mask, equity_matrix,
                          regrets, strat_sum, n_actions):
        tt = terminal_type[node_id]
        if tt > 0:
            if tt == 1: return -bet_sb[node_id]
            elif tt == 2: return bet_bb[node_id]
            elif tt == 3:
                pot = min(bet_sb[node_id], bet_bb[node_id])
                return (2.0 * equity_matrix[sb_b, bb_b] - 1.0) * pot
            return 0.0

        p = player[node_id]
        bucket = sb_b if p == 0 else bb_b
        strategy = np.zeros(n_actions)
        pos_sum = 0.0
        n_valid = 0
        for a in range(n_actions):
            if valid_mask[node_id, a]:
                v = max(0.0, regrets[node_id, bucket, a])
                strategy[a] = v
                pos_sum += v
                n_valid += 1
        if pos_sum > 0:
            strategy /= pos_sum
        else:
            for a in range(n_actions):
                if valid_mask[node_id, a]:
                    strategy[a] = 1.0 / n_valid

        node_util = 0.0
        action_utils = np.zeros(n_actions)
        for a in range(n_actions):
            if not valid_mask[node_id, a]: continue
            child = children[node_id, a]
            if p == 0:
                action_utils[a] = _cfr_walk_python(child, sb_b, bb_b, sb_reach*strategy[a], bb_reach, player, terminal_type, bet_sb, bet_bb, children, valid_mask, equity_matrix, regrets, strat_sum, n_actions)
            else:
                action_utils[a] = _cfr_walk_python(child, sb_b, bb_b, sb_reach, bb_reach*strategy[a], player, terminal_type, bet_sb, bet_bb, children, valid_mask, equity_matrix, regrets, strat_sum, n_actions)
            node_util += strategy[a] * action_utils[a]

        opp_reach = bb_reach if p == 0 else sb_reach
        sign = 1.0 if p == 0 else -1.0
        for a in range(n_actions):
            if valid_mask[node_id, a]:
                regrets[node_id, bucket, a] = max(0.0, regrets[node_id, bucket, a] + sign*(action_utils[a]-node_util)*opp_reach)
        my_reach = sb_reach if p == 0 else bb_reach
        for a in range(n_actions):
            strat_sum[node_id, bucket, a] += my_reach * strategy[a]
        return node_util

N_BUCKETS = 200       # hand potential buckets (200 for stability)
CFR_ITERATIONS = 5000
MAX_BET = 100
SB_BLIND = 1
BB_BLIND = 2

# Full raise levels up to all-in (100). Models the complete preflop game
# so the strategy properly handles 3-bets, 4-bets, and all-in decisions.
RAISE_LEVELS = [2, 4, 8, 16, 30, 60, 100]
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


@numba.njit(cache=True)
def _equity_matrix_inner(hands_i_arr, hands_j_arr, n_hi, n_hj,
                         flops_arr, n_flops, five_lookup_arr,
                         keep_a, keep_b, n_keeps):
    """Numba-compiled inner loop for equity matrix computation.

    hands_i_arr: (n_hi, 5) int64 — sampled hands from bucket i
    hands_j_arr: (n_hj, 5) int64 — sampled hands from bucket j
    flops_arr: (n_flops, 3) int64 — presampled flops
    five_lookup_arr: flat array indexed by bitmask -> rank
    keep_a, keep_b: (10,) int64 — keep pair indices
    """
    wins = 0.0
    total = 0.0

    for hi_idx in range(n_hi):
        hi_mask = 0
        for c in range(5):
            hi_mask |= 1 << hands_i_arr[hi_idx, c]

        for hj_idx in range(n_hj):
            # Check card overlap using bitmasks
            hj_mask = 0
            for c in range(5):
                hj_mask |= 1 << hands_j_arr[hj_idx, c]
            if hi_mask & hj_mask:
                continue

            used_mask = hi_mask | hj_mask

            for fi in range(n_flops):
                # Check flop doesn't overlap with hands
                f0 = flops_arr[fi, 0]
                f1 = flops_arr[fi, 1]
                f2 = flops_arr[fi, 2]
                flop_mask = (1 << f0) | (1 << f1) | (1 << f2)
                if flop_mask & used_mask:
                    continue

                # Best keep for player i
                best_i = 999999
                for k in range(n_keeps):
                    a = keep_a[k]
                    b = keep_b[k]
                    m = (1 << hands_i_arr[hi_idx, a]) | (1 << hands_i_arr[hi_idx, b]) | flop_mask
                    r = five_lookup_arr[m]
                    if r < best_i:
                        best_i = r

                # Best keep for player j
                best_j = 999999
                for k in range(n_keeps):
                    a = keep_a[k]
                    b = keep_b[k]
                    m = (1 << hands_j_arr[hj_idx, a]) | (1 << hands_j_arr[hj_idx, b]) | flop_mask
                    r = five_lookup_arr[m]
                    if r < best_j:
                        best_j = r

                if best_i < best_j:
                    wins += 1.0
                elif best_i == best_j:
                    wins += 0.5
                total += 1.0

    return wins, total


def compute_equity_matrix(n_buckets, preflop_table):
    """Compute equity[i][j] = probability bucket i beats bucket j.

    Numba-accelerated: ~50-100x faster than pure Python.
    Symmetric: only computes upper triangle, mirrors to lower.
    """
    import random
    from submission.equity import ExactEquityEngine
    engine = ExactEquityEngine()

    # Build flat five-card lookup array for Numba (indexed by bitmask)
    max_mask = 1 << 27
    five_lookup_arr = np.zeros(max_mask, dtype=np.int32)
    for mask_val, rank in engine._five.items():
        five_lookup_arr[mask_val] = rank

    # Keep pair indices
    keep_pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    keep_a = np.array([k[0] for k in keep_pairs], dtype=np.int64)
    keep_b = np.array([k[1] for k in keep_pairs], dtype=np.int64)
    n_keeps = len(keep_pairs)

    # Group all hands by bucket
    pot_min = min(preflop_table.values())
    pot_max = max(preflop_table.values())

    buckets = [[] for _ in range(n_buckets)]
    for mask, pot in preflop_table.items():
        frac = (pot - pot_min) / (pot_max - pot_min)
        frac = max(0.0, min(1.0 - 1e-9, frac))
        b = int(frac * n_buckets)
        cards = [c for c in range(27) if mask & (1 << c)]
        buckets[b].append(cards)

    print(f"  Bucket sizes: min={min(len(b) for b in buckets)}, max={max(len(b) for b in buckets)}, "
          f"avg={sum(len(b) for b in buckets)/n_buckets:.0f}")

    # Presample flops (reused across all bucket pairs)
    rng = random.Random(42)
    n_samples = 200
    n_flops = 200  # more flops since they're cheap with Numba
    all_cards = list(range(27))
    flops_arr = np.array([rng.sample(all_cards, 3) for _ in range(n_flops)],
                         dtype=np.int64)

    # Warmup Numba
    print("  Warming up Numba equity kernel...", end="", flush=True)
    dummy_h = np.zeros((1, 5), dtype=np.int64)
    dummy_h[0] = [0, 1, 2, 3, 4]
    dummy_h2 = np.zeros((1, 5), dtype=np.int64)
    dummy_h2[0] = [5, 6, 7, 8, 9]
    _equity_matrix_inner(dummy_h, dummy_h2, 1, 1, flops_arr[:1], 1,
                         five_lookup_arr, keep_a, keep_b, n_keeps)
    print(" done", flush=True)

    equity = np.zeros((n_buckets, n_buckets), dtype=np.float64)

    import time as _time
    _eq_start = _time.time()
    for i in range(n_buckets):
        elapsed = _time.time() - _eq_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (n_buckets - i) / rate if rate > 0 else 0
        print(f"  Equity matrix row {i}/{n_buckets} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

        if len(buckets[i]) == 0:
            equity[i, :] = 0.5
            equity[:, i] = 0.5
            continue

        hands_i = rng.sample(buckets[i], min(n_samples, len(buckets[i])))
        hi_arr = np.array(hands_i, dtype=np.int64).reshape(-1, 5)

        # Only compute upper triangle (j > i), mirror for lower
        for j in range(i + 1, n_buckets):
            if len(buckets[j]) == 0:
                equity[i][j] = 0.5
                equity[j][i] = 0.5
                continue

            hands_j = rng.sample(buckets[j], min(n_samples, len(buckets[j])))
            hj_arr = np.array(hands_j, dtype=np.int64).reshape(-1, 5)

            wins, total = _equity_matrix_inner(
                hi_arr, hj_arr, len(hands_i), len(hands_j),
                flops_arr, n_flops, five_lookup_arr,
                keep_a, keep_b, n_keeps)

            eq = wins / total if total > 0 else 0.5
            equity[i][j] = eq
            equity[j][i] = 1.0 - eq

        equity[i][i] = 0.5

    return equity


def _flatten_preflop_tree(nodes):
    """Flatten tree to numpy arrays for fast iteration."""
    n = len(nodes)
    player = np.full(n, -1, dtype=np.int8)
    terminal_type = np.zeros(n, dtype=np.int8)  # 0=none, 1=fold_sb, 2=fold_bb, 3=showdown
    bet_sb = np.zeros(n, dtype=np.float64)
    bet_bb = np.zeros(n, dtype=np.float64)
    n_actions = np.zeros(n, dtype=np.int32)
    children = np.full((n, N_ACTIONS), -1, dtype=np.int32)
    valid_mask = np.zeros((n, N_ACTIONS), dtype=np.bool_)

    for i, node in enumerate(nodes):
        player[i] = node['player']
        bet_sb[i] = node['bet_sb']
        bet_bb[i] = node['bet_bb']
        if node['terminal'] == 'fold_by_sb':
            terminal_type[i] = 1
        elif node['terminal'] == 'fold_by_bb':
            terminal_type[i] = 2
        elif node['terminal'] == 'showdown':
            terminal_type[i] = 3
        na = 0
        for a, child_id in node['children'].items():
            children[i, a] = child_id
            valid_mask[i, a] = True
            na += 1
        n_actions[i] = na

    return player, terminal_type, bet_sb, bet_bb, children, valid_mask, n_actions


def solve_cfr(nodes, root, n_buckets, equity_matrix, iterations):
    """Run CFR using flattened tree for speed."""
    n_nodes = len(nodes)
    player, terminal_type, bet_sb_arr, bet_bb_arr, children, valid_mask, n_act = \
        _flatten_preflop_tree(nodes)

    regrets = np.zeros((n_nodes, n_buckets, N_ACTIONS), dtype=np.float64)
    strat_sum = np.zeros((n_nodes, n_buckets, N_ACTIONS), dtype=np.float64)

    # Precompute terminal values for each (sb_bucket, bb_bucket) pair
    # at each terminal node
    terminal_ids = [i for i in range(n_nodes) if terminal_type[i] > 0]

    # Run iterations
    _run_preflop_cfr(root, n_nodes, n_buckets, N_ACTIONS, iterations,
                     player, terminal_type, bet_sb_arr, bet_bb_arr,
                     children, valid_mask, equity_matrix,
                     regrets, strat_sum)

    # Extract average strategies
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

    # Load preflop potential table for equity matrix computation
    data_path = os.path.join(os.path.dirname(__file__), "data", "preflop_potential.npz")
    data = np.load(data_path)
    preflop_table = dict(zip(data["bitmasks"].tolist(), data["potentials"].tolist()))

    print(f"\nComputing equity matrix ({N_BUCKETS}x{N_BUCKETS}) via simulation...")
    equity_matrix = compute_equity_matrix(N_BUCKETS, preflop_table)

    print(f"Running CFR ({CFR_ITERATIONS} iterations, {N_BUCKETS} buckets)...")
    t0 = time.time()
    avg_strats = solve_cfr(nodes, root, N_BUCKETS, equity_matrix, CFR_ITERATIONS)
    elapsed = time.time() - t0
    print(f"Solved in {elapsed:.1f}s")

    # Post-solve monotonicity check: verify that stronger buckets
    # raise more and fold less at the root. Flag anomalies.
    print("\nMonotonicity check (root node):")
    root_strats = avg_strats[root]  # (n_buckets, n_actions)
    anomalies = []
    for b in range(1, N_BUCKETS):
        fold_prev = root_strats[b - 1, 0]
        fold_curr = root_strats[b, 0]
        # Fold rate should decrease with strength (some noise OK)
        if fold_curr > fold_prev + 0.15:  # 15% jump = anomaly
            anomalies.append((b, 'fold_spike', fold_curr, fold_prev))
        # Check for shove spikes
        shove = sum(root_strats[b, a] for a in range(N_ACTIONS)
                    if a >= 2 and (a - 2) < len(RAISE_LEVELS) and RAISE_LEVELS[a - 2] >= 60)
        shove_prev = sum(root_strats[b - 1, a] for a in range(N_ACTIONS)
                         if a >= 2 and (a - 2) < len(RAISE_LEVELS) and RAISE_LEVELS[a - 2] >= 60)
        if shove > 0.5 and shove > shove_prev + 0.3:
            anomalies.append((b, 'shove_spike', shove, shove_prev))

    if anomalies:
        print(f"  WARNING: {len(anomalies)} anomalous buckets detected:")
        for b, kind, curr, prev in anomalies[:5]:
            print(f"    Bucket {b}: {kind} ({prev:.0%} -> {curr:.0%})")
        # Smooth anomalies: replace with average of neighbors
        for b, kind, curr, prev in anomalies:
            if 0 < b < N_BUCKETS - 1:
                avg_strats[root, b] = (avg_strats[root, b - 1] + avg_strats[root, b + 1]) / 2
                total = avg_strats[root, b].sum()
                if total > 0:
                    avg_strats[root, b] /= total
        print(f"  Smoothed {len(anomalies)} anomalous buckets")
    else:
        print("  All buckets monotonic ✓")

    # Extract strategies for the key decision points
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
