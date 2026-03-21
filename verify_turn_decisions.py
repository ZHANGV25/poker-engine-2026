#!/usr/bin/env python3
"""Verification framework: replay real match hands and compare turn decisions.

For each hand where we faced a turn bet:
1. Reconstruct the game state (cards, board, pot, bets)
2. Run the BLUEPRINT decision
3. Run the DEPTH-LIMITED SOLVER decision
4. Compare with what actually happened and the outcome

This uses real opponent behavior from actual competition matches.
"""
import sys, os, csv, time
import numpy as np
import itertools
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, 'submission')

from equity import ExactEquityEngine
from range_solver import RangeSolver
from game_tree import GameTree, ACT_CHECK, ACT_FOLD, ACT_CALL, TERM_NONE, TERM_SHOWDOWN
from inference import DiscardInference

RANKS = "23456789A"
SUITS = "dhs"

def card_str(c):
    return RANKS[c // 3] + SUITS[c % 3]

def parse_card(s):
    """Parse '2d' -> int"""
    s = s.strip().strip("'\"")
    return RANKS.index(s[0]) * 3 + SUITS.index(s[1])

def parse_card_list(s):
    """Parse "['2d', '3h']" -> [int, int]"""
    s = s.strip().strip("[]")
    if not s:
        return []
    return [parse_card(c.strip().strip("'\"")) for c in s.split(',') if c.strip().strip("'\"")]


def compute_river_game_values(solver, hero_hands, opp_hands, opp_w,
                               turn_board, dead, river_card, pot, iters=100):
    """Compute per-(hero,opp) game values for a specific river card.

    Returns (n_hero, n_opp) matrix of game values with betting.
    Uses compact tree for speed (continuation values don't need full tree).
    """
    river_board = turn_board + [river_card]
    n_hero = len(hero_hands)
    n_opp = len(opp_hands)

    tree = GameTree(pot, pot, 2, 100, True, compact=True)
    eq, nb = solver._compute_equity_and_mask(hero_hands, opp_hands, river_board, dead, 3)
    tv = solver._compute_terminal_values(tree, eq, nb)

    # Quick DCFR solve
    from math import pow as fpow
    n_hn = len(tree.hero_node_ids)
    n_on = len(tree.opp_node_ids)
    hi = {nid: i for i, nid in enumerate(tree.hero_node_ids)}
    oi = {nid: i for i, nid in enumerate(tree.opp_node_ids)}
    ma = max(max((tree.num_actions[n] for n in tree.hero_node_ids), default=1),
             max((tree.num_actions[n] for n in tree.opp_node_ids), default=1), 1)

    hr = np.zeros((n_hn, n_hero, ma))
    hs = np.zeros((n_hn, n_hero, ma))
    opr = np.zeros((n_on, n_opp, ma))
    h_reach = np.ones(n_hero) / n_hero

    for t in range(1, iters + 1):
        if t > 1:
            pw = fpow(t-1, 1.5) / (fpow(t-1, 1.5) + 1)
            nw = 0.5  # beta=0
            sw = fpow((t-1)/t, 2.0)
            hr *= np.where(hr > 0, pw, nw)
            opr *= np.where(opr > 0, pw, nw)
            hs *= sw
        solver._range_cfr_traverse(tree, 0, h_reach.copy(), opp_w.copy(),
                                    hr, hs, opr, hi, oi, tv, n_hero, n_opp, ma)

    # Compute game value matrix by evaluating the tree with avg strategy
    def eval_node(nid, h_r, o_r):
        if tree.terminal[nid] != TERM_NONE:
            return tv[nid]
        n_a = tree.num_actions[nid]
        ch = tree.children[nid]
        player = tree.player[nid]
        if player == 0:
            idx = hi[nid]
            s = hs[idx, :, :n_a]
            tot = s.sum(axis=1, keepdims=True)
            st = np.where(tot > 0, s / np.maximum(tot, 1e-10), np.full_like(s, 1.0/n_a))
            val = np.zeros((n_hero, n_opp))
            for a in range(n_a):
                val += st[:, a:a+1] * eval_node(ch[a][1], h_r * st[:, a], o_r)
            return val
        else:
            idx = oi[nid]
            r = opr[idx, :, :n_a]
            p = np.maximum(r, 0)
            tot = p.sum(axis=1, keepdims=True)
            st = np.where(tot > 0, p / np.maximum(tot, 1e-10), np.full_like(p, 1.0/n_a))
            val = np.zeros((n_hero, n_opp))
            for a in range(n_a):
                val += st[:, a:a+1].T * eval_node(ch[a][1], h_r, o_r * st[:, a])
            return val

    gv_matrix = eval_node(0, h_reach.copy(), opp_w.copy())
    return gv_matrix


def compute_turn_continuation_values(solver, hero_hands, opp_hands, opp_w,
                                      turn_board, dead, pot, river_iters=100):
    """Compute continuation values at turn showdown terminals.

    For each possible river card, runs a quick river solve to get game values,
    then averages across river cards.

    Returns (n_hero, n_opp) matrix of continuation values.
    """
    known = set(turn_board) | set(dead)
    river_cards = [c for c in range(27) if c not in known]
    n_hero = len(hero_hands)
    n_opp = len(opp_hands)

    hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands], dtype=np.int64)
    opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands], dtype=np.int64)

    # Also compute CTS equity for comparison
    eq_cts, nb_cts = solver._compute_equity_and_mask(hero_hands, opp_hands, turn_board, dead, 2)
    cts_values = (2 * eq_cts - 1) * pot * nb_cts

    # Compute game values per river card
    gv_sum = np.zeros((n_hero, n_opp))
    count = np.zeros((n_hero, n_opp))

    for rc in river_cards:
        rc_mask = 1 << rc
        # Skip hands that contain this river card
        hero_valid = np.array([(hero_masks[h] & rc_mask) == 0 for h in range(n_hero)])
        opp_valid = np.array([(opp_masks[o] & rc_mask) == 0 for o in range(n_opp)])
        valid_pairs = np.outer(hero_valid, opp_valid).astype(float)

        gv = compute_river_game_values(solver, hero_hands, opp_hands, opp_w,
                                        turn_board, dead, rc, pot, river_iters)
        gv_sum += gv * valid_pairs
        count += valid_pairs

    cont_values = np.where(count > 0, gv_sum / count, 0)
    return cont_values, cts_values


def depth_limited_turn_solve(solver, hero_cards, hero_hands, opp_hands, opp_w,
                              turn_board, dead, my_bet, opp_bet, min_raise,
                              max_raise, valid_actions, cont_values, iters=300):
    """Solve the turn with depth-limited continuation values.

    Uses cont_values at showdown terminals instead of CTS equity.
    """
    n_hero = len(hero_hands)
    n_opp = len(opp_hands)

    hero_tuple = tuple(sorted(hero_cards))
    hero_idx = None
    for i, h in enumerate(hero_hands):
        if tuple(sorted(h)) == hero_tuple:
            hero_idx = i
            break
    if hero_idx is None:
        return None

    # Build turn tree
    tree = GameTree(my_bet, opp_bet, min_raise, 100, True, compact=False)
    if tree.size < 2:
        return None

    # Build not_blocked mask
    hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands], dtype=np.int64)
    opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands], dtype=np.int64)
    overlap = hero_masks[:, None] & opp_masks[None, :]
    nb = (overlap == 0).astype(np.float64)

    # Terminal values: fold nodes use pot, showdown nodes use continuation values
    tv = {}
    for nid in tree.terminal_node_ids:
        tt = tree.terminal[nid]
        hp = tree.hero_pot[nid]
        op = tree.opp_pot[nid]
        if tt == 1:  # FOLD_HERO
            tv[nid] = -hp * nb
        elif tt == 2:  # FOLD_OPP
            tv[nid] = op * nb
        elif tt == 3:  # SHOWDOWN → use continuation values scaled by pot
            # cont_values were computed at a reference pot. Scale to this terminal's pot.
            actual_pot = min(hp, op)
            ref_pot = (my_bet + opp_bet) / 2  # approximate reference
            if ref_pot > 0:
                scale = actual_pot / ref_pot
            else:
                scale = 1.0
            tv[nid] = cont_values * scale * nb

    # Run DCFR
    strat = solver._run_dcfr(tree, opp_w, tv, n_hero, n_opp, iters)
    our_strat = strat[hero_idx]

    return solver._strategy_to_action(tree, our_strat, my_bet, opp_bet,
                                       min_raise, max_raise, valid_actions)


def analyze_match(filepath, engine, solver):
    """Analyze turn facing-bet decisions in a match."""
    with open(filepath) as f:
        header = f.readline().strip()
        reader = csv.DictReader(f)
        rows = list(reader)

    us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
    them = 1 - us

    hands = defaultdict(list)
    for r in rows:
        hands[int(r['hand_number'])].append(r)

    results = []

    for hnum in sorted(hands.keys()):
        rows_h = hands[hnum]
        turn_rows = [r for r in rows_h if r['street'] == 'Turn' and r['action_type'] != 'DISCARD']
        if not turn_rows:
            continue

        # Find if we faced a bet on the turn
        opp_bet_on_turn = False
        our_response = None
        our_response_amt = 0
        for r in turn_rows:
            team = int(r['active_team'])
            act = r['action_type']
            if team == them and act == 'RAISE':
                opp_bet_on_turn = True
            if opp_bet_on_turn and team == us and act in ('FOLD', 'CALL', 'RAISE'):
                our_response = act
                our_response_amt = int(r['action_amount'])
                break

        if not opp_bet_on_turn or our_response is None:
            continue

        # Get hand info
        last = rows_h[-1]
        our_cards = parse_card_list(last[f'team_{us}_cards'])
        their_cards = parse_card_list(last[f'team_{them}_cards'])
        board = parse_card_list(last['board_cards'])
        our_discards = parse_card_list(last[f'team_{us}_discarded'])
        their_discards = parse_card_list(last[f'team_{them}_discarded'])

        if len(board) < 4 or len(our_cards) < 2:
            continue

        turn_board = board[:4]
        dead = our_discards + their_discards

        # Get bet state when we face the turn bet
        for r in turn_rows:
            if int(r['active_team']) == us and r['action_type'] in ('FOLD', 'CALL', 'RAISE'):
                my_bet = int(r[f'team_{us}_bet'])
                their_bet_val = int(r[f'team_{them}_bet'])
                break

        # Compute hand P/L
        prev_bank = 0
        if hnum > 0:
            prev_nums = [h for h in hands.keys() if h < hnum]
            if prev_nums:
                prev_last = hands[max(prev_nums)][-1]
                prev_bank = int(prev_last[f'team_{us}_bankroll'])
        hand_pl = int(last[f'team_{us}_bankroll']) - prev_bank

        results.append({
            'hand': hnum,
            'our_cards': our_cards,
            'their_cards': their_cards,
            'turn_board': turn_board,
            'dead': dead,
            'my_bet': my_bet,
            'opp_bet': their_bet_val,
            'our_response': our_response,
            'hand_pl': hand_pl,
        })

    return results


def main():
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

    print("Loading matches and extracting turn facing-bet decisions...")
    all_decisions = []
    for mf in match_files:
        if not os.path.exists(mf):
            continue
        match_id = os.path.basename(mf).replace('match_', '').replace('.txt', '')
        decisions = analyze_match(mf, engine, solver)
        for d in decisions:
            d['match'] = match_id
        all_decisions.extend(decisions)
        print(f"  {match_id}: {len(decisions)} turn facing-bet decisions")

    print(f"\nTotal: {len(all_decisions)} decisions to analyze")

    # Sample up to 30 for speed
    if len(all_decisions) > 30:
        np.random.seed(42)
        indices = np.random.choice(len(all_decisions), 30, replace=False)
        all_decisions = [all_decisions[i] for i in sorted(indices)]
        print(f"Sampling 30 for analysis")

    # For each decision, compute what depth-limited solver would do
    print(f"\n{'H#':>4} {'Cards':<8} {'Board':<20} {'Bets':>10} {'Actual':<6} {'PL':>5} {'DL-Action':<10} {'Time':>6}")
    print("-" * 80)

    dl_better = 0
    bp_better = 0
    same = 0

    for d in all_decisions:
        turn_board = d['turn_board']
        dead = d['dead']
        our_cards = d['our_cards']
        my_bet = d['my_bet']
        opp_bet = d['opp_bet']

        known = set(turn_board) | set(dead)
        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        opp_hands = list(hero_hands)
        n_opp = len(opp_hands)
        opp_w = np.ones(n_opp) / n_opp

        # Build narrowed range from discards
        if len(d['dead']) >= 6:
            opp_discards = d['dead'][3:]  # last 3 are opponent's
            opp_weights = inference.infer_opponent_weights(opp_discards, turn_board[:3], our_cards)
            if opp_weights:
                opp_hands_n = []
                opp_w_n = []
                for h, w in opp_weights.items():
                    if w > 0.001 and not (set(h) & known):
                        opp_hands_n.append(h)
                        opp_w_n.append(w)
                if opp_hands_n:
                    opp_hands = opp_hands_n
                    opp_w = np.array(opp_w_n)
                    opp_w /= opp_w.sum()
                    n_opp = len(opp_hands)

        # Compute continuation values
        t0 = time.time()
        ref_pot = min(my_bet, opp_bet)
        try:
            cont_values, cts_values = compute_turn_continuation_values(
                solver, hero_hands, opp_hands, opp_w,
                turn_board, dead, ref_pot, river_iters=100)

            # Run depth-limited solver
            valid = [True, True, False, True, False]  # fold/raise/x/call/x
            dl_result = depth_limited_turn_solve(
                solver, our_cards, hero_hands, opp_hands, opp_w,
                turn_board, dead, my_bet, opp_bet, 2,
                100 - max(my_bet, opp_bet), valid, cont_values, iters=200)
        except Exception as e:
            dl_result = None

        elapsed = (time.time() - t0) * 1000

        if dl_result is None:
            dl_action = "ERROR"
        else:
            dl_action = ['FOLD', 'RAISE', 'CHECK', 'CALL'][dl_result[0]]
            if dl_result[0] == 1:
                dl_action += f"({dl_result[1]})"

        actual = d['our_response']
        pl = d['hand_pl']

        # Compare: if we folded and lost less, or called and won
        if actual == dl_action.split('(')[0]:
            same += 1
            marker = ""
        elif pl < -20 and actual != 'FOLD' and dl_action == 'FOLD':
            dl_better += 1
            marker = " << DL saves"
        elif pl > 0 and actual == 'FOLD' and dl_action != 'FOLD':
            dl_better += 1
            marker = " << DL wins"
        elif pl < -20 and dl_action != 'FOLD' and actual == 'FOLD':
            bp_better += 1
            marker = " << BP saves"
        else:
            same += 1
            marker = ""

        board_str = ' '.join(card_str(c) for c in turn_board)
        cards_str = card_str(our_cards[0]) + card_str(our_cards[1])
        print(f"{d['hand']:>4} {cards_str:<8} {board_str:<20} ({my_bet},{opp_bet}) "
              f"{actual:<6} {pl:>+5} {dl_action:<10} {elapsed:>5.0f}ms{marker}")

    print(f"\n{'='*60}")
    print(f"DL better: {dl_better}  BP better: {bp_better}  Same: {same}")
    print(f"Total: {len(all_decisions)}")


if __name__ == '__main__':
    main()
