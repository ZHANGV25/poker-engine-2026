#!/usr/bin/env python3
"""Full verification: all turn facing-bet decisions across 6 matches."""
import sys, os, csv, time, numpy as np, itertools
from collections import defaultdict
from math import pow as fpow

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine
from range_solver import RangeSolver
from game_tree import GameTree, TERM_NONE
from inference import DiscardInference

RANKS = '23456789A'
SUITS = 'dhs'
def cs(c): return RANKS[c//3] + SUITS[c%3]
def pc(s):
    s = s.strip().strip("'\" ")
    return RANKS.index(s[0]) * 3 + SUITS.index(s[1])
def pcl(s):
    s = s.strip().strip('[]')
    if not s: return []
    return [pc(c) for c in s.split(',') if c.strip().strip("'\" ")]

engine = ExactEquityEngine()
solver = RangeSolver(engine)
inference = DiscardInference(engine)

match_files = [
    '/Users/victor/Downloads/match_68854.txt',
    '/Users/victor/Downloads/match_68818.txt',
    '/Users/victor/Downloads/match_68756.txt',
    '/Users/victor/Downloads/match_68640.txt',
    '/Users/victor/Downloads/match_68712.txt',
    '/Users/victor/Downloads/match_68604.txt',
]

def extract_decisions(filepath):
    with open(filepath) as f:
        header = f.readline().strip()
        reader = csv.DictReader(f)
        rows = list(reader)

    us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
    them = 1 - us

    hands = defaultdict(list)
    for r in rows:
        hands[int(r['hand_number'])].append(r)

    decisions = []
    for hnum in sorted(hands.keys()):
        rows_h = hands[hnum]
        turn_rows = [r for r in rows_h if r['street'] == 'Turn' and r['action_type'] != 'DISCARD']
        if not turn_rows:
            continue

        start_bet_us = int(turn_rows[0][f'team_{us}_bet'])
        start_bet_them = int(turn_rows[0][f'team_{them}_bet'])

        opp_raise_total = 0
        our_response = None
        for r in turn_rows:
            team = int(r['active_team'])
            act = r['action_type']
            amt = int(r['action_amount'])
            if team == them and act == 'RAISE':
                opp_raise_total += amt
            if opp_raise_total > 0 and team == us and act in ('FOLD', 'CALL', 'RAISE'):
                our_response = act
                break

        if opp_raise_total == 0 or our_response is None:
            continue

        my_bet = start_bet_us
        opp_bet = start_bet_us + opp_raise_total  # raise is on top of matching

        last = rows_h[-1]
        our_cards = pcl(last[f'team_{us}_cards'])
        their_cards = pcl(last[f'team_{them}_cards'])
        board = pcl(last['board_cards'])
        our_disc = pcl(last[f'team_{us}_discarded'])
        their_disc = pcl(last[f'team_{them}_discarded'])

        prev_bank = 0
        prev_nums = [h for h in hands.keys() if h < hnum]
        if prev_nums:
            prev_bank = int(hands[max(prev_nums)][-1][f'team_{us}_bankroll'])
        pl = int(last[f'team_{us}_bankroll']) - prev_bank

        if len(board) < 4 or len(our_cards) < 2:
            continue

        decisions.append({
            'hand': hnum, 'our_cards': our_cards, 'their_cards': their_cards,
            'board4': board[:4], 'dead': our_disc + their_disc,
            'my_bet': my_bet, 'opp_bet': opp_bet,
            'response': our_response, 'pl': pl,
            'opp_disc': their_disc,
        })
    return decisions


def solve_depth_limited(d):
    """Run depth-limited turn solver for one decision. Returns (action_name, elapsed_ms)."""
    tb = d['board4']
    dead = d['dead']
    known = set(tb) | set(dead)
    rem = [c for c in range(27) if c not in known]
    hh = list(itertools.combinations(rem, 2))

    # Narrowed range
    if len(d['opp_disc']) == 3:
        ow_dict = inference.infer_opponent_weights(d['opp_disc'], tb[:3], d['our_cards'])
        oh = []; ow_l = []
        if ow_dict:
            for h, w in ow_dict.items():
                if w > 0.001 and not (set(h) & known):
                    oh.append(h); ow_l.append(w)
        if oh:
            ow = np.array(ow_l); ow /= ow.sum()
        else:
            oh = list(hh); ow = np.ones(len(oh)) / len(oh)
    else:
        oh = list(hh); ow = np.ones(len(oh)) / len(oh)

    t0 = time.time()
    ref_pot = max(min(d['my_bet'], d['opp_bet']), 2)

    # Compute continuation values: for each river card, CTS equity on 5-card board
    # (using compact tree game value would be better but CTS is faster for bulk test)
    rcs = [c for c in range(27) if c not in known]
    n_h, n_o = len(hh), len(oh)
    hm = np.array([(1 << h[0]) | (1 << h[1]) for h in hh], dtype=np.int64)
    om = np.array([(1 << o[0]) | (1 << o[1]) for o in oh], dtype=np.int64)

    # Quick river game values using compact tree solve
    gv_sum = np.zeros((n_h, n_o))
    count = np.zeros((n_h, n_o))

    for rc in rcs:
        rm = 1 << rc
        hv = np.array([(hm[h] & rm) == 0 for h in range(n_h)])
        ov = np.array([(om[o] & rm) == 0 for o in range(n_o)])
        vp = np.outer(hv, ov).astype(float)

        rb = tb + [rc]
        tree_r = solver._get_tree(ref_pot, ref_pot, 2, 100, compact=True)
        eq, nb = solver._compute_equity_and_mask(hh, oh, rb, dead, 3)
        tv_r = solver._compute_terminal_values(tree_r, eq, nb)

        nhn = len(tree_r.hero_node_ids)
        non = len(tree_r.opp_node_ids)
        hi = {n: i for i, n in enumerate(tree_r.hero_node_ids)}
        oi = {n: i for i, n in enumerate(tree_r.opp_node_ids)}
        ma = max(max((tree_r.num_actions[n] for n in tree_r.hero_node_ids), default=1),
                 max((tree_r.num_actions[n] for n in tree_r.opp_node_ids), default=1), 1)
        hr = np.zeros((nhn, n_h, ma)); hs = np.zeros((nhn, n_h, ma))
        opr = np.zeros((non, n_o, ma))
        h_reach = np.ones(n_h) / n_h

        for t in range(1, 101):  # 100 iters
            if t > 1:
                pw = fpow(t - 1, 1.5) / (fpow(t - 1, 1.5) + 1)
                sw = fpow((t - 1) / t, 2.0)
                hr *= np.where(hr > 0, pw, 0.5)
                opr *= np.where(opr > 0, pw, 0.5)
                hs *= sw
            solver._range_cfr_traverse(tree_r, 0, h_reach.copy(), ow.copy(),
                                        hr, hs, opr, hi, oi, tv_r, n_h, n_o, ma)

        # Use equilibrium-weighted terminal values as game value approximation
        # Simpler: just use the equity-weighted value (with betting bonus from solve)
        idx0 = hi.get(0)
        if idx0 is not None:
            n_act = tree_r.num_actions[0]
            s = hs[idx0, :, :n_act]
            tot = s.sum(axis=1, keepdims=True)
            avg_s = np.where(tot > 0, s / np.maximum(tot, 1e-10), np.full_like(s, 1.0 / n_act))
            # Approximate game value: check value + bet bonus
            # Check value = CTS equity
            # Bet value > CTS for strong hands
            # For continuation values, use (2*eq-1)*pot adjusted by strategy
            check_val = (2 * eq - 1) * nb * ref_pot
            # If hero bets often (avg_s[:,0] low = less checking), hands are stronger
            bet_bonus = (1 - avg_s[:, 0:1]) * ref_pot * 0.3  # ~30% pot bonus for betting
            gv = check_val + bet_bonus * nb
        else:
            gv = (2 * eq - 1) * nb * ref_pot

        gv_sum += gv * vp
        count += vp * nb

    cv = np.where(count > 0, gv_sum / np.maximum(count, 1), 0)

    # Now solve turn tree with continuation values
    tree = GameTree(d['my_bet'], d['opp_bet'], 2, 100, True, compact=False)
    if tree.size < 2:
        return None, (time.time() - t0) * 1000

    ht = tuple(sorted(d['our_cards']))
    hidx = None
    for i, h in enumerate(hh):
        if tuple(sorted(h)) == ht:
            hidx = i; break
    if hidx is None:
        return None, (time.time() - t0) * 1000

    nb_turn = ((hm[:, None] & om[None, :]) == 0).astype(float)

    tv_turn = {}
    for nid in tree.terminal_node_ids:
        tt = tree.terminal[nid]
        hp, op = tree.hero_pot[nid], tree.opp_pot[nid]
        if tt == 1:
            tv_turn[nid] = -hp * nb_turn
        elif tt == 2:
            tv_turn[nid] = op * nb_turn
        elif tt == 3:
            scale = min(hp, op) / max(ref_pot, 1)
            tv_turn[nid] = cv * scale * nb_turn

    strat = solver._run_dcfr(tree, ow, tv_turn, n_h, n_o, 200)
    s = strat[hidx]
    s = np.maximum(s, 0)
    total = s.sum()
    if total > 0:
        s = s / total
    else:
        s = np.ones(len(s)) / len(s)

    result = solver._strategy_to_action(tree, s, d['my_bet'], d['opp_bet'], 2,
                                         100 - max(d['my_bet'], d['opp_bet']),
                                         [True, True, False, True, False])
    elapsed = (time.time() - t0) * 1000
    return result, elapsed


# Main
print("Extracting turn facing-bet decisions from 6 matches...\n")

all_decisions = []
for mf in match_files:
    if not os.path.exists(mf):
        continue
    mid = os.path.basename(mf).replace('match_', '').replace('.txt', '')
    decs = extract_decisions(mf)
    for d in decs:
        d['match'] = mid
    all_decisions.extend(decs)
    opp = open(mf).readline().strip()
    opp = opp.replace('# Team 0: ', '').replace('# Team 1: ', ' vs ')
    print(f"  {mid}: {len(decs)} decisions")

print(f"\nTotal: {len(all_decisions)} decisions")
print("Running depth-limited solver on all...\n")

# Counters
dl_saves = 0  # DL folds where blueprint called/raised and lost
dl_wins = 0   # DL calls/raises where blueprint folded and would have won
bp_saves = 0  # Blueprint folds where DL would call/raise and lose
bp_wins = 0   # Blueprint calls where DL folds and it won
same = 0
dl_chips_saved = 0
dl_chips_won = 0
bp_chips_saved = 0
errors = 0

t_start = time.time()

for i, d in enumerate(all_decisions):
    try:
        result, elapsed = solve_depth_limited(d)
    except Exception as e:
        errors += 1
        continue

    if result is None:
        errors += 1
        continue

    dl_act = ['FOLD', 'RAISE', 'CHECK', 'CALL'][result[0]]
    actual = d['response']
    pl = d['pl']

    # Categorize
    actual_continues = actual in ('CALL', 'RAISE')
    dl_continues = dl_act in ('CALL', 'RAISE')

    if actual_continues == dl_continues:
        same += 1
    elif dl_continues and not actual_continues:
        # DL would continue, blueprint folded
        if pl > 0:  # folding was wrong (we would have won)
            dl_wins += 1
            dl_chips_won += pl
        else:
            bp_saves += 1
            bp_chips_saved += abs(pl)
    elif not dl_continues and actual_continues:
        # DL would fold, blueprint continued
        if pl < -10:  # continuing was wrong (we lost)
            dl_saves += 1
            dl_chips_saved += abs(pl)
        else:
            bp_wins += 1

    if (i + 1) % 50 == 0:
        elapsed_total = time.time() - t_start
        print(f"  Processed {i+1}/{len(all_decisions)} ({elapsed_total:.0f}s)")

elapsed_total = time.time() - t_start

print(f"\n{'='*60}")
print(f"RESULTS ({len(all_decisions)} decisions, {elapsed_total:.0f}s)")
print(f"{'='*60}")
print(f"  Same decision:              {same:>4}")
print(f"  DL saves (fold vs lose):    {dl_saves:>4}  chips saved: {dl_chips_saved:>+5}")
print(f"  DL wins (call vs fold-win): {dl_wins:>4}  chips won:   {dl_chips_won:>+5}")
print(f"  BP saves (fold vs DL-lose): {bp_saves:>4}  chips saved: {bp_chips_saved:>+5}")
print(f"  BP wins (call vs DL-fold):  {bp_wins:>4}")
print(f"  Errors:                     {errors:>4}")
print(f"\n  Net DL advantage: {dl_chips_saved + dl_chips_won - bp_chips_saved:>+5} chips")
print(f"  Per decision: {(dl_chips_saved + dl_chips_won - bp_chips_saved) / max(len(all_decisions), 1):>+.1f}")
