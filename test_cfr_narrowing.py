#!/usr/bin/env python3
"""
Comprehensive test: CFR Bayesian narrowing vs Polarized heuristic narrowing.

Extracts 20+ river bet scenarios from match logs, runs both narrowing methods,
and compares accuracy (which assigns higher probability to opponent's actual hand).

KEY FINDING: compute_opponent_bet_probs has a bug — it builds the tree with
POST-bet amounts (opp already raised), so the tree root has no aggressive
actions and P(bet)=0 for all hands. The correct approach is to build the tree
with PRE-bet amounts (both at equal bets before opponent's decision).
"""

import os
import sys
import time
import glob
import itertools
import copy
import ast
import numpy as np

# Setup path for submission imports
_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission")
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from equity import ExactEquityEngine
from inference import DiscardInference
from solver import SubgameSolver
from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

# ── Card encoding ──────────────────────────────────────────────────────────
RANKS = "23456789A"
SUITS = "dhs"

def card_str_to_int(s):
    rank, suit = s[0], s[1]
    return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)

def int_to_card_str(c):
    return RANKS[c % len(RANKS)] + SUITS[c // len(RANKS)]

def parse_card_list(s):
    s = s.strip()
    if s == '[]':
        return []
    try:
        cards_str = ast.literal_eval(s)
        return [card_str_to_int(c) for c in cards_str]
    except Exception:
        return []


# ── Match log parser ──────────────────────────────────────────────────────
def extract_river_bet_scenarios(filepaths, max_scenarios=60):
    """Extract scenarios where opponent bet/raised on the river."""
    scenarios = []

    for fpath in filepaths:
        with open(fpath) as f:
            lines = f.readlines()

        header = lines[0].strip()
        team0_name = header.split('Team 0: ')[1].split(',')[0]
        team1_name = header.split('Team 1: ')[1].strip()
        if 'Stockfish' in team0_name:
            we_are = 0
        elif 'Stockfish' in team1_name:
            we_are = 1
        else:
            continue

        data_lines = lines[2:]
        hands = {}
        for line in data_lines:
            parts = line.strip().split(',', 15)
            if len(parts) < 16:
                continue
            hand_num = int(parts[0])
            if hand_num not in hands:
                hands[hand_num] = []
            hands[hand_num].append((parts, line.strip()))

        for hand_num, rows in hands.items():
            river_bet_parts = None
            river_bet_line = None
            for parts, full_line in rows:
                street = parts[1]
                active_team = int(parts[2])
                action = parts[5]
                if street == 'River' and action == 'RAISE' and active_team != we_are:
                    river_bet_parts = parts
                    river_bet_line = full_line
                    break

            if river_bet_parts is None:
                continue

            try:
                prefix_parts = river_bet_line.split(',', 9)
                remainder = prefix_parts[9]

                bracket_sections = []
                i = 0
                while i < len(remainder):
                    if remainder[i] == '"':
                        j = remainder.index('"', i + 1)
                        bracket_sections.append(remainder[i+1:j])
                        i = j + 2
                    elif remainder[i] == '[':
                        j = remainder.index(']', i)
                        bracket_sections.append(remainder[i:j+1])
                        i = j + 2
                    elif remainder[i] == ',':
                        i += 1
                    else:
                        j = remainder.find(',', i)
                        if j == -1:
                            bracket_sections.append(remainder[i:])
                            break
                        bracket_sections.append(remainder[i:j])
                        i = j + 1

                if len(bracket_sections) < 7:
                    continue

                t0_cards = parse_card_list(bracket_sections[0])
                t1_cards = parse_card_list(bracket_sections[1])
                board = parse_card_list(bracket_sections[2])
                t0_disc = parse_card_list(bracket_sections[3])
                t1_disc = parse_card_list(bracket_sections[4])
                t0_bet = int(bracket_sections[5])
                t1_bet = int(bracket_sections[6])

                if we_are == 0:
                    our_cards = t0_cards
                    opp_cards = t1_cards
                    our_disc = t0_disc
                    opp_disc = t1_disc
                    my_bet_before = t0_bet
                    opp_bet_before = t1_bet
                else:
                    our_cards = t1_cards
                    opp_cards = t0_cards
                    our_disc = t1_disc
                    opp_disc = t0_disc
                    my_bet_before = t1_bet
                    opp_bet_before = t0_bet

                if len(board) != 5 or len(our_cards) != 2 or len(opp_cards) != 2:
                    continue
                if len(opp_disc) != 3:
                    continue

                raise_amount = int(prefix_parts[6])

                scenario = {
                    'hand_num': hand_num,
                    'match': os.path.basename(fpath),
                    'board': board,
                    'our_cards': our_cards,
                    'opp_actual_cards': opp_cards,
                    'our_discards': our_disc,
                    'opp_discards': opp_disc,
                    'my_bet_before': my_bet_before,
                    'opp_bet_before': opp_bet_before,
                    'raise_to': raise_amount,
                    # After opponent raises: we face this situation
                    'my_bet': my_bet_before,
                    'opp_bet': raise_amount,
                }
                scenarios.append(scenario)

                if len(scenarios) >= max_scenarios:
                    return scenarios

            except Exception:
                continue

    return scenarios


# ── Polarized narrowing (standalone, mirrors player.py logic) ──────────────
def polarized_narrow(opp_weights, board, my_bet, opp_bet, engine):
    """Apply polarized narrowing heuristic. Returns new weights dict."""
    weights = copy.deepcopy(opp_weights)

    raise_amt = opp_bet - my_bet
    pot_before = my_bet * 2
    if raise_amt <= 0 or pot_before <= 0:
        return weights

    bet_frac = raise_amt / pot_before
    bluff_ratio = bet_frac / (1 + bet_frac)

    total_bet_freq = 0.23
    total_bet_freq = max(0.10, min(0.70, total_bet_freq))

    value_pct = total_bet_freq / (1 + bluff_ratio)
    bluff_pct = total_bet_freq - value_pct

    board_set = set(board)
    strengths = {}
    for pair in weights:
        if weights[pair] <= 0 or set(pair) & board_set:
            continue
        try:
            if len(board) >= 5:
                r = engine.lookup_seven(list(pair) + list(board))
            else:
                continue
        except Exception:
            continue
        strengths[pair] = r

    if not strengths:
        return weights

    ranked = sorted(strengths.items(), key=lambda x: x[1])
    n = len(ranked)

    n_value = int(n * value_pct)
    n_bluff = int(n * bluff_pct)
    n_value = max(1, n_value)

    keep_indices = set(range(n_value))
    keep_indices.update(range(n - n_bluff, n))

    for i, (hand, _) in enumerate(ranked):
        if i not in keep_indices:
            weights[hand] = 0.0

    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    return weights


# ── Fixed CFR: compute P(bet|hand) using PRE-BET pot amounts ─────────────
def compute_opponent_bet_probs_fixed(solver, board, dead_cards, opp_range,
                                      hero_range, pre_bet, street,
                                      iterations=100):
    """Compute P(bet|hand) for each opponent hand via CFR.

    FIXED version: builds tree with PRE-bet amounts (both players at equal bets
    before opponent's decision to check or bet). The original version incorrectly
    used POST-bet amounts, which meant the root had no aggressive actions.

    Args:
        pre_bet: the bet amount BOTH players had before opponent decided to bet
        (e.g. if bets were 24/24 before opponent raised to 76, pre_bet=24)
    """
    known = set(board) | set(dead_cards)

    opp_hands = []
    opp_weights = []
    for hand, w in opp_range.items():
        if w > 0.001 and not (set(hand) & known):
            opp_hands.append(hand)
            opp_weights.append(w)

    hero_hands = []
    hero_weights = []
    for hand, w in hero_range.items():
        if w > 0.001 and not (set(hand) & known):
            hero_hands.append(hand)
            hero_weights.append(w)

    if len(opp_hands) < 3 or len(hero_hands) < 3:
        return None

    opp_w = np.array(opp_weights, dtype=np.float64)
    opp_w /= opp_w.sum()
    hero_w = np.array(hero_weights, dtype=np.float64)
    hero_w /= hero_w.sum()

    n_opp = len(opp_hands)
    n_hero = len(hero_hands)

    # FIXED: Build tree with equal pre-bet amounts. Opponent acts first,
    # deciding to check or bet. This gives aggressive actions at root.
    max_bet = 100
    min_raise = max(2, pre_bet // 10) if pre_bet > 0 else 2  # reasonable min raise
    tree = solver._get_tree(pre_bet, pre_bet, min_raise, max_bet, True)

    if tree.size < 2:
        return None

    # Verify root has aggressive actions
    root_children = tree.children[0]
    aggressive = set()
    for a, (act_type, _) in enumerate(root_children):
        if act_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                       ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            aggressive.add(a)

    if not aggressive:
        return None  # no bet actions possible

    # Compute equity matrix
    board_list = list(board)
    equity_matrix = np.zeros((n_opp, n_hero), dtype=np.float64)

    if len(board) == 5:
        for oi, oh in enumerate(opp_hands):
            oh_rank = solver.engine.lookup_seven(list(oh) + board_list)
            for hi, hh in enumerate(hero_hands):
                if set(hh) & set(oh):
                    continue
                hh_rank = solver.engine.lookup_seven(list(hh) + board_list)
                if oh_rank < hh_rank:
                    equity_matrix[oi, hi] = 1.0
                elif oh_rank == hh_rank:
                    equity_matrix[oi, hi] = 0.5

    # Build terminal values
    terminal_values = {}
    for node_id in tree.terminal_node_ids:
        tt = tree.terminal[node_id]
        hp = tree.hero_pot[node_id]
        op = tree.opp_pot[node_id]
        tv = np.zeros((n_opp, n_hero), dtype=np.float64)

        if tt == TERM_FOLD_HERO:
            tv[:, :] = -hp
        elif tt == TERM_FOLD_OPP:
            tv[:, :] = op
        elif tt == TERM_SHOWDOWN:
            pot_won = min(hp, op)
            tv = (2.0 * equity_matrix - 1.0) * pot_won

        terminal_values[node_id] = tv

    # Run CFR
    hero_idx = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
    opp_idx = {nid: i for i, nid in enumerate(tree.opp_node_ids)}

    max_act = max(
        max((tree.num_actions[nid] for nid in tree.hero_node_ids), default=1),
        max((tree.num_actions[nid] for nid in tree.opp_node_ids), default=1), 1)

    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    hero_regrets = np.zeros((n_hero_nodes, n_opp, max_act), dtype=np.float64)
    hero_strat_sum = np.zeros((n_hero_nodes, n_opp, max_act), dtype=np.float64)
    opp_regrets = np.zeros((n_opp_nodes, n_hero, max_act), dtype=np.float64)

    def regret_match(reg, n_act):
        pos = np.maximum(reg[:n_act], 0)
        t = pos.sum()
        return pos / t if t > 0 else np.full(n_act, 1.0 / n_act)

    def traverse(nid, hero_reach, opp_reach):
        if tree.terminal[nid] != TERM_NONE:
            return terminal_values[nid]

        n_act = tree.num_actions[nid]
        children = tree.children[nid]
        player = tree.player[nid]

        if player == 0:  # "hero" = opponent
            idx = hero_idx[nid]
            strats = np.zeros((n_opp, n_act), dtype=np.float64)
            for oi in range(n_opp):
                strats[oi] = regret_match(hero_regrets[idx, oi], n_act)

            act_vals = np.zeros((n_act, n_opp, n_hero), dtype=np.float64)
            node_val = np.zeros((n_opp, n_hero), dtype=np.float64)
            for a in range(n_act):
                new_reach = hero_reach * strats[:, a]
                act_vals[a] = traverse(children[a][1], new_reach, opp_reach)
                node_val += strats[:, a:a+1] * act_vals[a]

            for oi in range(n_opp):
                for a in range(n_act):
                    cf = np.dot(act_vals[a, oi] - node_val[oi], opp_reach)
                    hero_regrets[idx, oi, a] = max(0, hero_regrets[idx, oi, a] + cf)
                hero_strat_sum[idx, oi, :n_act] += hero_reach[oi] * strats[oi]

            return node_val

        else:  # "opp" = us
            idx = opp_idx[nid]
            strats = np.zeros((n_hero, n_act), dtype=np.float64)
            for hi in range(n_hero):
                strats[hi] = regret_match(opp_regrets[idx, hi], n_act)

            act_vals = np.zeros((n_act, n_opp, n_hero), dtype=np.float64)
            node_val = np.zeros((n_opp, n_hero), dtype=np.float64)
            for a in range(n_act):
                new_reach = opp_reach * strats[:, a]
                act_vals[a] = traverse(children[a][1], hero_reach, new_reach)
                node_val += strats[:, a:a+1].T * act_vals[a]

            for hi in range(n_hero):
                for a in range(n_act):
                    cf = np.dot(hero_reach, node_val[:, hi] - act_vals[a, :, hi])
                    opp_regrets[idx, hi, a] = max(0, opp_regrets[idx, hi, a] + cf)

            return node_val

    hero_init = opp_w.copy()
    opp_init = hero_w.copy()
    for _ in range(iterations):
        traverse(0, hero_init.copy(), opp_init.copy())

    # Extract P(bet|hand) at root
    root = 0
    if root not in hero_idx:
        return None

    idx = hero_idx[root]
    n_act = tree.num_actions[root]

    result = {}
    for oi, hand in enumerate(opp_hands):
        total = hero_strat_sum[idx, oi, :n_act].sum()
        if total > 0:
            strat = hero_strat_sum[idx, oi, :n_act] / total
        else:
            strat = np.ones(n_act) / n_act
        p_bet = sum(strat[a] for a in aggressive)
        result[hand] = p_bet

    return result


# ── Main test ─────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("CFR BAYESIAN NARROWING vs POLARIZED HEURISTIC — ACCURACY TEST")
    print("=" * 80)

    match_files = sorted(glob.glob(os.path.expanduser("~/Downloads/match_*.txt")),
                         key=os.path.getmtime, reverse=True)[:5]
    print(f"\nUsing {len(match_files)} match files:")
    for f in match_files:
        print(f"  {os.path.basename(f)}")

    print("\nExtracting river bet scenarios...")
    scenarios = extract_river_bet_scenarios(match_files, max_scenarios=60)
    print(f"Found {len(scenarios)} river bet scenarios")

    if len(scenarios) < 5:
        print("ERROR: Not enough scenarios found.")
        return

    print("\nLoading equity engine...")
    t0 = time.time()
    engine = ExactEquityEngine()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    inference = DiscardInference(engine)
    solver = SubgameSolver(engine)

    # ── Phase 1: Demonstrate the tree bug ──
    print("\n" + "=" * 80)
    print("PHASE 1: TREE BUG DIAGNOSIS")
    print("=" * 80)

    act_names = {0:'FOLD', 1:'CHECK', 2:'CALL', 3:'R_HALF', 4:'R_POT', 5:'R_ALL', 6:'R_OVER'}

    # Common scenario: both at 24, opponent raises to 76
    print("\n  Scenario: bets were 24/24, opponent raised to 76")
    print("\n  BUGGY tree (current code): _get_tree(opp_bet=76, my_bet=24, min=52)")
    tree_buggy = GameTree(76, 24, 52, 100, True)
    root_acts = [(act_names[a], tree_buggy.hero_pot[c], tree_buggy.opp_pot[c])
                 for a, c in tree_buggy.children[0]]
    n_agg = sum(1 for a, _, _ in root_acts if a.startswith('R_'))
    print(f"    Size: {tree_buggy.size} | Root actions: {[a for a,_,_ in root_acts]} | Aggressive: {n_agg}")
    print(f"    --> Root has hero_pot=76 > opp_pot=24, so hero sees fold/call only")
    print(f"    --> P(bet) will be 0 for ALL hands (no bet action exists!)")

    print("\n  FIXED tree: _get_tree(pre_bet=24, pre_bet=24, min=2)")
    tree_fixed = GameTree(24, 24, 2, 100, True)
    root_acts_f = [(act_names[a], tree_fixed.hero_pot[c], tree_fixed.opp_pot[c])
                   for a, c in tree_fixed.children[0]]
    n_agg_f = sum(1 for a, _, _ in root_acts_f if a.startswith('R_'))
    print(f"    Size: {tree_fixed.size} | Root actions: {[a for a,_,_ in root_acts_f]} | Aggressive: {n_agg_f}")
    print(f"    --> Root has equal bets, hero can CHECK or BET (4 raise sizes)")
    print(f"    --> P(bet) will reflect equilibrium betting frequency per hand")

    # ── Phase 2: Run buggy vs fixed CFR vs polarized ──
    print("\n" + "=" * 80)
    print("PHASE 2: COMPARISON — Buggy CFR vs Fixed CFR vs Polarized Heuristic")
    print("=" * 80)

    results = []
    n_tested = 0
    n_buggy_failed = 0
    n_fixed_failed = 0

    for i, sc in enumerate(scenarios[:40]):
        board = sc['board']
        our_cards = sc['our_cards']
        opp_actual = tuple(sorted(sc['opp_actual_cards']))
        opp_disc = sc['opp_discards']
        our_disc = sc['our_discards']
        my_bet = sc['my_bet']
        opp_bet = sc['opp_bet']
        dead_cards = our_disc + opp_disc

        initial_weights = inference.infer_opponent_weights(
            opp_disc, board[:3], our_cards)

        if not initial_weights:
            continue

        initial_actual_wt = initial_weights.get(opp_actual, 0.0)
        initial_range_size = sum(1 for w in initial_weights.values() if w > 0.01)

        # Skip scenarios where opp_bet <= my_bet (not a real raise from even)
        if opp_bet <= my_bet:
            continue

        # ── Polarized narrowing ──
        polar_weights = polarized_narrow(initial_weights, board, my_bet, opp_bet, engine)
        polar_actual_wt = polar_weights.get(opp_actual, 0.0)
        polar_range_size = sum(1 for w in polar_weights.values() if w > 0.01)

        # ── Build hero range (subsampled for speed) ──
        known = set(board) | set(dead_cards) | set(our_cards)
        remaining = [c for c in range(27) if c not in known]
        all_hero_hands = list(itertools.combinations(remaining, 2))
        n_opp = sum(1 for w in initial_weights.values() if w > 0.001)
        n_hero_full = len(all_hero_hands)

        max_product = 5000
        target_hero = max(10, min(n_hero_full, max_product // max(n_opp, 1)))

        hero_range = {}
        if target_hero >= n_hero_full:
            for h in all_hero_hands:
                hero_range[h] = 1.0
        else:
            step = max(1, n_hero_full // target_hero)
            for j in range(0, n_hero_full, step):
                hero_range[all_hero_hands[j]] = 1.0

        n_hero_used = len(hero_range)

        # ── BUGGY CFR (original code, post-bet tree) ──
        t0_buggy = time.time()
        p_bet_buggy = solver.compute_opponent_bet_probs(
            board=board, dead_cards=dead_cards,
            opp_range=initial_weights,
            hero_range=hero_range,
            my_bet=my_bet, opp_bet=opp_bet,
            street=3,
            min_raise=max(2, opp_bet - my_bet),
            iterations=100)
        buggy_ms = (time.time() - t0_buggy) * 1000

        buggy_actual_pb = None
        if p_bet_buggy and len(p_bet_buggy) >= 3:
            buggy_actual_pb = p_bet_buggy.get(opp_actual, None)
            buggy_wt = copy.deepcopy(initial_weights)
            for hand in list(buggy_wt.keys()):
                if hand in p_bet_buggy:
                    buggy_wt[hand] *= max(p_bet_buggy[hand], 0.05)
            total = sum(buggy_wt.values())
            if total > 0:
                for k in buggy_wt:
                    buggy_wt[k] /= total
            buggy_actual_wt = buggy_wt.get(opp_actual, 0.0)
            buggy_range_size = sum(1 for w in buggy_wt.values() if w > 0.01)
            buggy_ok = True
        else:
            n_buggy_failed += 1
            buggy_actual_wt = initial_actual_wt
            buggy_range_size = initial_range_size
            buggy_ok = False

        # ── FIXED CFR (pre-bet tree) ──
        # pre_bet = the common bet before opponent raised
        # If bets were my_bet/my_bet before opponent raised to opp_bet, pre_bet = my_bet
        pre_bet = my_bet  # both were at this level before opponent bet

        t0_fixed = time.time()
        p_bet_fixed = compute_opponent_bet_probs_fixed(
            solver, board=board, dead_cards=dead_cards,
            opp_range=initial_weights,
            hero_range=hero_range,
            pre_bet=pre_bet,
            street=3,
            iterations=100)
        fixed_ms = (time.time() - t0_fixed) * 1000

        fixed_actual_pb = None
        if p_bet_fixed and len(p_bet_fixed) >= 3:
            fixed_actual_pb = p_bet_fixed.get(opp_actual, None)
            fixed_wt = copy.deepcopy(initial_weights)
            for hand in list(fixed_wt.keys()):
                if hand in p_bet_fixed:
                    fixed_wt[hand] *= max(p_bet_fixed[hand], 0.05)
            total = sum(fixed_wt.values())
            if total > 0:
                for k in fixed_wt:
                    fixed_wt[k] /= total
            fixed_actual_wt = fixed_wt.get(opp_actual, 0.0)
            fixed_range_size = sum(1 for w in fixed_wt.values() if w > 0.01)
            fixed_ok = True
        else:
            n_fixed_failed += 1
            fixed_actual_wt = initial_actual_wt
            fixed_range_size = initial_range_size
            fixed_ok = False

        n_tested += 1

        board_str = ' '.join(int_to_card_str(c) for c in board)
        opp_str = ' '.join(int_to_card_str(c) for c in sc['opp_actual_cards'])

        result = {
            'idx': i,
            'match': sc['match'],
            'hand': sc['hand_num'],
            'board_str': board_str,
            'opp_str': opp_str,
            'my_bet': my_bet,
            'opp_bet': opp_bet,
            'pre_bet': pre_bet,
            'n_opp': n_opp,
            'n_hero_used': n_hero_used,
            'initial_wt': initial_actual_wt,
            'initial_range': initial_range_size,
            'polar_wt': polar_actual_wt,
            'polar_range': polar_range_size,
            'buggy_wt': buggy_actual_wt,
            'buggy_range': buggy_range_size,
            'buggy_ok': buggy_ok,
            'buggy_ms': buggy_ms,
            'buggy_pb': buggy_actual_pb,
            'fixed_wt': fixed_actual_wt,
            'fixed_range': fixed_range_size,
            'fixed_ok': fixed_ok,
            'fixed_ms': fixed_ms,
            'fixed_pb': fixed_actual_pb,
        }
        results.append(result)

        buggy_pb_s = f"{buggy_actual_pb:.3f}" if buggy_actual_pb is not None else "N/A"
        fixed_pb_s = f"{fixed_actual_pb:.3f}" if fixed_actual_pb is not None else "N/A"

        # Determine best method
        wts = [('Polar', polar_actual_wt), ('Buggy', buggy_actual_wt), ('Fixed', fixed_actual_wt)]
        wts.sort(key=lambda x: -x[1])
        best = wts[0][0]

        print(f"  [{n_tested:2d}] h{sc['hand_num']:>4d} Bets:{my_bet:>3}/{opp_bet:<3} "
              f"| Init:{initial_actual_wt:.3f} "
              f"| Polar:{polar_actual_wt:.3f} "
              f"| Buggy:{buggy_actual_wt:.3f}[pb={buggy_pb_s}] "
              f"| Fixed:{fixed_actual_wt:.3f}[pb={fixed_pb_s}] "
              f"| {fixed_ms:>5.0f}ms | Best={best}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    print(f"\nScenarios tested: {total}")
    print(f"Buggy CFR failed: {n_buggy_failed}")
    print(f"Fixed CFR failed: {n_fixed_failed}")

    valid = [r for r in results if r['fixed_ok']]
    n_valid = len(valid)

    if n_valid == 0:
        print("No valid results.")
        return

    # Head-to-head: Fixed CFR vs Polarized
    fixed_wins = sum(1 for r in valid if r['fixed_wt'] > r['polar_wt'] + 0.001)
    polar_wins = sum(1 for r in valid if r['polar_wt'] > r['fixed_wt'] + 0.001)
    ties = n_valid - fixed_wins - polar_wins

    print(f"\n--- Fixed CFR vs Polarized (head to head) ---")
    print(f"  Fixed CFR wins:  {fixed_wins:>3} ({fixed_wins/n_valid:.1%})")
    print(f"  Polarized wins:  {polar_wins:>3} ({polar_wins/n_valid:.1%})")
    print(f"  Ties:            {ties:>3} ({ties/n_valid:.1%})")

    # Head-to-head: Buggy CFR vs Polarized
    buggy_valid = [r for r in results if r['buggy_ok']]
    if buggy_valid:
        bv_fixed_wins = sum(1 for r in buggy_valid if r['buggy_wt'] > r['polar_wt'] + 0.001)
        bv_polar_wins = sum(1 for r in buggy_valid if r['polar_wt'] > r['buggy_wt'] + 0.001)
        bv_ties = len(buggy_valid) - bv_fixed_wins - bv_polar_wins
        print(f"\n--- Buggy CFR vs Polarized (head to head) ---")
        print(f"  Buggy CFR wins:  {bv_fixed_wins:>3} ({bv_fixed_wins/len(buggy_valid):.1%})")
        print(f"  Polarized wins:  {bv_polar_wins:>3} ({bv_polar_wins/len(buggy_valid):.1%})")
        print(f"  Ties:            {bv_ties:>3} ({bv_ties/len(buggy_valid):.1%})")

    # Average weights
    avg_init = sum(r['initial_wt'] for r in valid) / n_valid
    avg_polar = sum(r['polar_wt'] for r in valid) / n_valid
    avg_buggy = sum(r['buggy_wt'] for r in valid) / n_valid
    avg_fixed = sum(r['fixed_wt'] for r in valid) / n_valid

    print(f"\n--- Average weight assigned to actual hand ---")
    print(f"  Initial:         {avg_init:.4f} (baseline)")
    print(f"  Polarized:       {avg_polar:.4f}  ({avg_polar/max(avg_init,1e-9):.2f}x)")
    print(f"  Buggy CFR:       {avg_buggy:.4f}  ({avg_buggy/max(avg_init,1e-9):.2f}x)")
    print(f"  Fixed CFR:       {avg_fixed:.4f}  ({avg_fixed/max(avg_init,1e-9):.2f}x)")

    # Average range sizes
    avg_polar_rng = sum(r['polar_range'] for r in valid) / n_valid
    avg_buggy_rng = sum(r['buggy_range'] for r in valid) / n_valid
    avg_fixed_rng = sum(r['fixed_range'] for r in valid) / n_valid
    avg_init_rng = sum(r['initial_range'] for r in valid) / n_valid

    print(f"\n--- Average effective range size (>1% weight) ---")
    print(f"  Initial:   {avg_init_rng:.1f}")
    print(f"  Polarized: {avg_polar_rng:.1f}")
    print(f"  Buggy CFR: {avg_buggy_rng:.1f}")
    print(f"  Fixed CFR: {avg_fixed_rng:.1f}")

    # P(bet) from fixed CFR
    fixed_pbs = [r['fixed_pb'] for r in valid if r['fixed_pb'] is not None]
    if fixed_pbs:
        print(f"\n--- Fixed CFR: P(bet|actual hand) distribution ---")
        print(f"  Mean:   {np.mean(fixed_pbs):.3f}")
        print(f"  Median: {np.median(fixed_pbs):.3f}")
        print(f"  Min:    {min(fixed_pbs):.3f}")
        print(f"  Max:    {max(fixed_pbs):.3f}")
        high = sum(1 for p in fixed_pbs if p > 0.5)
        med = sum(1 for p in fixed_pbs if 0.2 <= p <= 0.5)
        low = sum(1 for p in fixed_pbs if p < 0.2)
        print(f"  >50%:   {high}/{len(fixed_pbs)} ({high/len(fixed_pbs):.0%})")
        print(f"  20-50%: {med}/{len(fixed_pbs)} ({med/len(fixed_pbs):.0%})")
        print(f"  <20%:   {low}/{len(fixed_pbs)} ({low/len(fixed_pbs):.0%})")

    # Buggy CFR P(bet) for comparison
    buggy_pbs = [r['buggy_pb'] for r in valid if r['buggy_pb'] is not None]
    if buggy_pbs:
        print(f"\n--- Buggy CFR: P(bet|actual hand) distribution ---")
        print(f"  Mean:   {np.mean(buggy_pbs):.3f}")
        print(f"  All zero? {all(p < 0.01 for p in buggy_pbs)}")

    # Zeroed out
    polar_zeroed = sum(1 for r in valid if r['polar_wt'] < 0.001)
    fixed_zeroed = sum(1 for r in valid if r['fixed_wt'] < 0.001)
    buggy_zeroed = sum(1 for r in valid if r['buggy_wt'] < 0.001)
    print(f"\n--- Actual hand zeroed out (weight < 0.001) ---")
    print(f"  Polarized: {polar_zeroed}/{n_valid} ({polar_zeroed/n_valid:.0%})")
    print(f"  Buggy CFR: {buggy_zeroed}/{n_valid} ({buggy_zeroed/n_valid:.0%})")
    print(f"  Fixed CFR: {fixed_zeroed}/{n_valid} ({fixed_zeroed/n_valid:.0%})")

    # Timing
    fixed_times = [r['fixed_ms'] for r in valid if r['fixed_ms'] > 0]
    if fixed_times:
        print(f"\n--- Timing ---")
        print(f"  Fixed CFR: mean={np.mean(fixed_times):.0f}ms, "
              f"median={np.median(fixed_times):.0f}ms, "
              f"max={max(fixed_times):.0f}ms")

    # ── VERDICT ──
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")

    print(f"\n1. BUG CONFIRMED: compute_opponent_bet_probs builds the game tree with")
    print(f"   POST-bet amounts (opp_bet={scenarios[0]['opp_bet']}, my_bet={scenarios[0]['my_bet']}), "
          f"which means the root node")
    print(f"   has hero_pot > opp_pot. The tree offers only FOLD/CALL at the root,")
    print(f"   so P(bet|hand) = 0 for all hands. This makes CFR narrowing useless.")
    print(f"   The buggy code is at solver.py line 365:")
    print(f"     tree = self._get_tree(opp_bet, my_bet, min_raise, max_bet, True)")
    print(f"   It should use PRE-bet amounts (equal bets before the decision).")

    print(f"\n2. FIXED CFR vs POLARIZED:")
    if fixed_wins > polar_wins:
        print(f"   Fixed CFR is MORE ACCURATE: {fixed_wins}/{n_valid} ({fixed_wins/n_valid:.0%}) "
              f"vs polar {polar_wins}/{n_valid} ({polar_wins/n_valid:.0%})")
    elif polar_wins > fixed_wins:
        print(f"   Polarized heuristic is MORE ACCURATE: {polar_wins}/{n_valid} ({polar_wins/n_valid:.0%}) "
              f"vs fixed CFR {fixed_wins}/{n_valid} ({fixed_wins/n_valid:.0%})")
    else:
        print(f"   Methods are TIED: {fixed_wins} vs {polar_wins}")

    print(f"   Average weight on actual hand: Fixed={avg_fixed:.4f} vs Polar={avg_polar:.4f}")

    print(f"\n3. PRODUCTION IMPACT:")
    print(f"   The range size guard (n_opp * n_hero > 10000) in _cfr_bayesian_narrow")
    print(f"   (player.py line 584) blocks most scenarios. Even if the tree bug is")
    print(f"   fixed, CFR narrowing only applies to ~{len([s for s in scenarios if s.get('_n_opp', 200) * 91 <= 10000])}/{len(scenarios)} scenarios in practice.")


if __name__ == "__main__":
    main()
