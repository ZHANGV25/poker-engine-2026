#!/usr/bin/env python3
"""Definitive test: flop warm-start solver vs blueprint on ALL match data.

Tests BOTH acting first AND facing bet (equity gate) on flop.
Uses ALL available matches, breaks down by opponent strength.
"""
import sys, os, csv, time, numpy as np, itertools
from collections import defaultdict
from math import pow as fpow

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine
from range_solver import RangeSolver
from inference import DiscardInference
from multi_street_lookup import MultiStreetLookup
from game_tree import GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL

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
                     my_bet, opp_bet, valid_actions, hero_position,
                     n_refine=50, bp_weight=200):
    """Warm-start from blueprint, refine against narrowed range."""
    n_h = len(hero_hands)
    n_o = len(opp_hands)

    hero_tuple = tuple(sorted(hero_cards))
    hero_idx = None
    for i, h in enumerate(hero_hands):
        if tuple(sorted(h)) == hero_tuple:
            hero_idx = i
            break
    if hero_idx is None:
        return None

    tree = solver._get_tree(my_bet, opp_bet, 2, 100, compact=False)
    if tree.size < 2:
        return None

    eq, nb = solver._compute_equity_and_mask(hero_hands, opp_hands, board, dead, 1)
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

    # Warm-start root with blueprint
    root = 0
    if root in hi:
        root_si = hi[root]
        root_children = tree.children[root]
        n_act = tree.num_actions[root]
        solver_act_types = [root_children[a][0] for a in range(n_act)]
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

    # Refine
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

    if root in hi:
        idx = hi[root]
        n_act = tree.num_actions[root]
        s = hs[idx, hero_idx, :n_act]
        total = s.sum()
        if total > 0:
            s = s / total
        else:
            s = np.ones(n_act) / n_act
        return solver._strategy_to_action(tree, s, my_bet, opp_bet, 2,
                                           100 - max(my_bet, opp_bet), valid_actions)
    return None


def extract_flop_decisions(filepath):
    """Extract ALL flop decisions (acting first + facing bet)."""
    with open(filepath) as f:
        header = f.readline().strip()
        reader = csv.DictReader(f)
        rows = list(reader)

    us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
    them = 1 - us
    opp = header.split('Team 0: ')[1].split(',')[0] if us == 1 else header.split('Team 1: ')[1].strip()

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

        flop_rows = [r for r in rows_h if r['street'] == 'Flop' and r['action_type'] != 'DISCARD']
        if not flop_rows:
            continue

        start_bet = int(flop_rows[0][f'team_{us}_bet'])

        # Determine: did we act first or face a bet?
        our_first_action = None
        opp_raised_first = False
        opp_raise_amt = 0

        for r in flop_rows:
            team = int(r['active_team'])
            act = r['action_type']
            amt = int(r['action_amount'])

            if team == them and act == 'RAISE' and not our_first_action:
                opp_raised_first = True
                opp_raise_amt += amt

            if team == us and act in ('FOLD', 'CALL', 'RAISE', 'CHECK'):
                our_first_action = act
                break

        if our_first_action is None:
            continue

        our_cards = pcl(last[f'team_{us}_cards'])
        board = pcl(last['board_cards'])
        our_disc = pcl(last[f'team_{us}_discarded'])
        their_disc = pcl(last[f'team_{them}_discarded'])
        if len(board) < 3 or len(our_cards) < 2:
            continue

        if opp_raised_first:
            decision_type = 'facing_bet'
            my_bet = start_bet
            opp_bet = start_bet + opp_raise_amt
        else:
            decision_type = 'acting_first'
            my_bet = start_bet
            opp_bet = start_bet

        decisions.append({
            'hand': hnum, 'pl': pl, 'opp': opp,
            'our_cards': our_cards, 'board': board[:3],
            'our_disc': our_disc, 'their_disc': their_disc,
            'my_bet': my_bet, 'opp_bet': opp_bet,
            'response': our_first_action, 'type': decision_type,
        })
    return decisions


# Collect ALL flop decisions
match_files = [os.path.join(os.path.expanduser('~/Downloads'), f)
               for f in sorted(os.listdir(os.path.expanduser('~/Downloads')))
               if f.startswith('match_') and f.endswith('.txt')
               and int(f.replace('match_', '').replace('.txt', '')) >= 69000]

all_decisions = []
for mf in match_files:
    decs = extract_flop_decisions(mf)
    all_decisions.extend(decs)

af_decisions = [d for d in all_decisions if d['type'] == 'acting_first']
fb_decisions = [d for d in all_decisions if d['type'] == 'facing_bet']

print(f'Total flop decisions: {len(all_decisions)}')
print(f'  Acting first: {len(af_decisions)}')
print(f'  Facing bet: {len(fb_decisions)}')

# Sample for speed
np.random.seed(42)
if len(af_decisions) > 100:
    indices = np.random.choice(len(af_decisions), 100, replace=False)
    af_sample = [af_decisions[i] for i in sorted(indices)]
else:
    af_sample = af_decisions

print(f'Testing {len(af_sample)} acting-first decisions with warm-start...')
print()

# For each acting-first decision: blueprint vs warm-start
bp_better = ws_better = same = 0
bp_chips = ws_chips = 0

t_start = time.time()
for i, d in enumerate(af_sample):
    board = d['board']
    dead = d['our_disc'] + d['their_disc']
    known = set(board) | set(dead)
    pl = d['pl']
    actual = d['response']
    actual_bets = actual in ('RAISE',)

    # Build narrowed range
    if len(d['their_disc']) == 3:
        opp_weights = inference.infer_opponent_weights(d['their_disc'], board, d['our_cards'])
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

    va = [True, True, True, False, False]  # fold/raise/check

    try:
        r_ws = warm_start_solve(d['our_cards'], hero_hands, opp_hands, opp_w,
                                 board, dead, d['my_bet'], d['opp_bet'],
                                 va, 0, n_refine=50, bp_weight=200)
    except:
        r_ws = None

    if r_ws is None:
        continue

    ws_act = ['FOLD', 'RAISE', 'CHECK', 'CALL'][r_ws[0]]
    ws_bets = ws_act in ('RAISE',)

    if ws_bets == actual_bets:
        same += 1
    elif ws_bets and not actual_bets:
        # WS bets, BP checked
        if pl > 0:
            ws_better += 1; ws_chips += pl
        else:
            bp_better += 1; bp_chips += abs(pl)
    elif not ws_bets and actual_bets:
        # WS checks, BP bet
        if pl < -10:
            ws_better += 1; ws_chips += abs(pl)
        else:
            bp_better += 1; bp_chips += pl

    if (i + 1) % 25 == 0:
        print(f'  Processed {i+1}/{len(af_sample)} ({time.time()-t_start:.0f}s)')

elapsed = time.time() - t_start
print(f'\n{"="*60}')
print(f'FLOP ACTING FIRST: warm-start vs blueprint ({len(af_sample)} decisions, {elapsed:.0f}s)')
print(f'{"="*60}')
print(f'Same decision:     {same}')
print(f'WS better:         {ws_better} (chips: +{ws_chips})')
print(f'BP better:         {bp_better} (chips: +{bp_chips})')
print(f'Net WS advantage:  {ws_chips - bp_chips:+d}')
if ws_better + bp_better > 0:
    print(f'Per divergent:     {(ws_chips - bp_chips) / (ws_better + bp_better):+.1f}')
