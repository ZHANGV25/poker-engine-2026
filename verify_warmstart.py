#!/usr/bin/env python3
"""Test warm-start solver vs blueprint vs cold-start on real match decisions.

For each flop/turn facing-bet decision:
1. Blueprint decision (what we actually played)
2. Cold-start solver (CTS equity, no multi-street)
3. Warm-start solver (blueprint init + narrowed range refinement)
4. Compare against actual P/L outcome
"""
import sys, os, csv, time, numpy as np, itertools
from collections import defaultdict
from math import pow as fpow

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine
from range_solver import RangeSolver
from inference import DiscardInference
from game_tree import GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL, TERM_NONE
from multi_street_lookup import MultiStreetLookup

engine = ExactEquityEngine()
solver = RangeSolver(engine)
inference = DiscardInference(engine)
ms = MultiStreetLookup('submission/data/multi_street/', engine)

RANKS = '23456789A'; SUITS = 'dhs'
def pc(s):
    s = s.strip().strip("'\" ")
    return RANKS.index(s[0]) * 3 + SUITS.index(s[1])
def pcl(s):
    s = s.strip().strip('[]')
    if not s: return []
    return [pc(c) for c in s.split(',') if c.strip().strip("'\" ")]


def warm_start_solve(hero_cards, hero_hands, opp_hands, opp_w, board, dead,
                     my_bet, opp_bet, valid_actions, street, n_refine=50, bp_weight=200):
    """Solve with warm-start from blueprint."""
    n_h = len(hero_hands)
    n_o = len(opp_hands)
    known = set(board) | set(dead)

    hero_tuple = tuple(sorted(hero_cards))
    hero_idx_in_list = None
    for i, h in enumerate(hero_hands):
        if tuple(sorted(h)) == hero_tuple:
            hero_idx_in_list = i
            break
    if hero_idx_in_list is None:
        return None

    tree = solver._get_tree(my_bet, opp_bet, 2, 100, compact=False)
    if tree.size < 2:
        return None

    eq, nb = solver._compute_equity_and_mask(hero_hands, opp_hands, board, dead, street)
    tv = solver._compute_terminal_values(tree, eq, nb)

    n_hn = len(tree.hero_node_ids)
    n_on = len(tree.opp_node_ids)
    hi = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
    oi = {nid: i for i, nid in enumerate(tree.opp_node_ids)}
    ma = max(max((tree.num_actions[n] for n in tree.hero_node_ids), default=1),
             max((tree.num_actions[n] for n in tree.opp_node_ids), default=1), 1)

    hr = np.zeros((n_hn, n_h, ma), dtype=np.float64)
    hs = np.zeros((n_hn, n_h, ma), dtype=np.float64)
    opr = np.zeros((n_on, n_o, ma), dtype=np.float64)

    # Warm-start: initialize hero_strat_sum at root with blueprint
    root = 0
    if root in hi:
        root_si = hi[root]
        root_children = tree.children[root]
        n_act = tree.num_actions[root]
        solver_act_types = [root_children[a][0] for a in range(n_act)]

        hero_position = 0 if my_bet <= opp_bet else 1
        pot_state = (my_bet, opp_bet)

        for h_i, hh in enumerate(hero_hands):
            bp_s = ms.get_strategy(list(hh), board, pot_state=pot_state,
                                   hero_position=hero_position)
            if bp_s is None:
                continue
            for sol_ai in range(n_act):
                sol_act = solver_act_types[sol_ai]
                bp_prob = bp_s.get(sol_act, 0)
                if bp_prob > 0:
                    hs[root_si, h_i, sol_ai] = bp_prob * bp_weight

    # Refine with DCFR
    h_reach = np.ones(n_h, dtype=np.float64) / n_h
    alpha, beta, gamma = 1.5, 0.0, 2.0
    for t in range(1, n_refine + 1):
        if t > 1:
            pw = fpow(t-1, alpha) / (fpow(t-1, alpha) + 1)
            nw = fpow(t-1, beta) / (fpow(t-1, beta) + 1)
            sw = fpow((t-1)/t, gamma)
            hr *= np.where(hr > 0, pw, nw)
            opr *= np.where(opr > 0, pw, nw)
            hs *= sw
        solver._range_cfr_traverse(tree, 0, h_reach.copy(), opp_w.copy(),
                                    hr, hs, opr, hi, oi, tv, n_h, n_o, ma)

    # Extract strategy
    if root in hi:
        idx = hi[root]
        n_act = tree.num_actions[root]
        s = hs[idx, hero_idx_in_list, :n_act]
        total = s.sum()
        if total > 0:
            s = s / total
        else:
            s = np.ones(n_act) / n_act
        return solver._strategy_to_action(tree, s, my_bet, opp_bet, 2,
                                           100 - max(my_bet, opp_bet), valid_actions)
    return None


def extract_decisions(filepath, street_filter):
    """Extract facing-bet decisions for a given street."""
    with open(filepath) as f:
        header = f.readline().strip()
        reader = csv.DictReader(f)
        rows = list(reader)

    us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
    them = 1 - us
    street_name = {1: 'Flop', 2: 'Turn'}[street_filter]

    hands = defaultdict(list)
    for r in rows:
        hands[int(r['hand_number'])].append(r)

    decisions = []
    prev_bank = 0
    for hnum in sorted(hands.keys()):
        rows_h = hands[hnum]
        last = rows_h[-1]
        bank = int(last[f'team_{us}_bankroll'])
        pl = bank - prev_bank
        prev_bank = bank

        street_rows = [r for r in rows_h if r['street'] == street_name
                       and r['action_type'] != 'DISCARD']
        if not street_rows:
            continue

        start_bet = int(street_rows[0][f'team_{us}_bet'])
        opp_raise = 0
        our_response = None
        for r in street_rows:
            team = int(r['active_team'])
            act = r['action_type']
            amt = int(r['action_amount'])
            if team == them and act == 'RAISE':
                opp_raise += amt
            if opp_raise > 0 and team == us and act in ('FOLD', 'CALL', 'RAISE'):
                our_response = act
                break

        if opp_raise == 0 or our_response is None:
            continue

        our_cards = pcl(last[f'team_{us}_cards'])
        board = pcl(last['board_cards'])
        our_disc = pcl(last[f'team_{us}_discarded'])
        their_disc = pcl(last[f'team_{them}_discarded'])
        if street_filter == 1 and len(board) < 3:
            continue
        if street_filter == 2 and len(board) < 4:
            continue
        if len(our_cards) < 2:
            continue

        board_for_street = board[:3] if street_filter == 1 else board[:4]

        decisions.append({
            'hand': hnum, 'pl': pl,
            'our_cards': our_cards, 'board': board_for_street,
            'our_disc': our_disc, 'their_disc': their_disc,
            'my_bet': start_bet, 'opp_bet': start_bet + opp_raise,
            'response': our_response, 'street': street_filter,
        })
    return decisions


# Main
match_files = []
for f in sorted(os.listdir(os.path.expanduser('~/Downloads'))):
    if f.startswith('match_') and f.endswith('.txt'):
        num = int(f.replace('match_', '').replace('.txt', ''))
        if num >= 69000:
            match_files.append(os.path.join(os.path.expanduser('~/Downloads'), f))

for street, street_name in [(1, 'FLOP'), (2, 'TURN')]:
    all_decisions = []
    for mf in match_files:
        decs = extract_decisions(mf, street)
        all_decisions.extend(decs)

    if len(all_decisions) > 60:
        np.random.seed(42)
        indices = np.random.choice(len(all_decisions), 60, replace=False)
        all_decisions = [all_decisions[i] for i in sorted(indices)]

    print(f'\n{"="*60}')
    print(f'{street_name} FACING BET: {len(all_decisions)} decisions')
    print(f'{"="*60}')

    results = {'blueprint': {'same': 0, 'better': 0, 'worse': 0, 'chips': 0},
               'cold': {'same': 0, 'better': 0, 'worse': 0, 'chips': 0},
               'warm': {'same': 0, 'better': 0, 'worse': 0, 'chips': 0}}

    for d in all_decisions:
        board = d['board']
        dead = d['our_disc'] + d['their_disc']
        known = set(board) | set(dead)
        pl = d['pl']
        actual = d['response']
        actual_continues = actual in ('CALL', 'RAISE')

        # Build narrowed range
        if len(d['their_disc']) == 3:
            opp_weights = inference.infer_opponent_weights(d['their_disc'],
                                                           board[:3], d['our_cards'])
        else:
            opp_weights = None
        if not opp_weights:
            continue

        rem = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(rem, 2))
        opp_hands = []
        opp_w = []
        for h, w in opp_weights.items():
            if w > 0.001 and not (set(h) & known):
                opp_hands.append(h)
                opp_w.append(w)
        if not opp_hands:
            continue
        opp_w = np.array(opp_w); opp_w /= opp_w.sum()

        va = [True, True, False, True, False]  # fold/raise/x/call/x

        # Cold-start solver
        try:
            r_cold = solver.solve_and_act(d['our_cards'], board, opp_weights, dead,
                                           d['my_bet'], d['opp_bet'], d['street'],
                                           2, 100 - max(d['my_bet'], d['opp_bet']),
                                           va, 800)
        except:
            r_cold = None

        # Warm-start solver
        try:
            r_warm = warm_start_solve(d['our_cards'], hero_hands, opp_hands, opp_w,
                                       board, dead, d['my_bet'], d['opp_bet'],
                                       va, d['street'], n_refine=50, bp_weight=200)
        except:
            r_warm = None

        for label, result in [('cold', r_cold), ('warm', r_warm)]:
            if result is None:
                continue
            act = ['FOLD', 'RAISE', 'CHECK', 'CALL'][result[0]]
            solver_continues = act in ('CALL', 'RAISE')

            if solver_continues == actual_continues:
                results[label]['same'] += 1
            elif solver_continues and not actual_continues:
                # Solver calls, blueprint folded
                if pl > 0:
                    results[label]['better'] += 1
                    results[label]['chips'] += abs(pl)
                else:
                    results[label]['worse'] += 1
                    results[label]['chips'] -= abs(pl)
            elif not solver_continues and actual_continues:
                # Solver folds, blueprint called
                if pl < -10:
                    results[label]['better'] += 1
                    results[label]['chips'] += abs(pl)
                else:
                    results[label]['worse'] += 1
                    results[label]['chips'] -= abs(pl)

    print(f'{"Approach":<12} {"Same":>6} {"Better":>7} {"Worse":>7} {"Net Chips":>10}')
    print('-' * 45)
    for label in ['cold', 'warm']:
        r = results[label]
        print(f'{label:<12} {r["same"]:>6} {r["better"]:>+7} {r["worse"]:>+7} {r["chips"]:>+10}')


if __name__ == '__main__':
    pass
