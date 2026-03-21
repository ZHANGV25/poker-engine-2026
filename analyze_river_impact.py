#!/usr/bin/env python3
"""
Comprehensive river decision analysis for v38 impact estimation.

Analyzes every river decision across all match logs to quantify:
1. River acting-first: check vs bet decisions, missed value / missed bluffs
2. River facing-bet: call/fold decisions, overcalls / correct folds
3. Equity gate overrides: when fired, was it correct
4. Chip impact quantification
5. What % of hands involve river decisions at all

Usage: python analyze_river_impact.py
"""
import csv, ast, sys, os, glob, itertools
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))
from equity import ExactEquityEngine
engine = ExactEquityEngine()

RANKS = "23456789A"
SUITS = "dhs"

def card_to_int(s):
    return SUITS.index(s[1]) * 9 + RANKS.index(s[0])

def parse_cards(s):
    """Parse card string like "['5h', '4h']" to list of ints."""
    try:
        cards = ast.literal_eval(s)
        return [card_to_int(c) for c in cards]
    except:
        return []

def hand_rank_7(cards):
    """Get hand rank for 7 cards (2 hole + 5 board). Lower = better."""
    try:
        mask = 0
        for c in cards:
            mask |= 1 << c
        return engine._seven.get(mask)
    except:
        return None

def analyze_match_rivers(filepath):
    """Deep river analysis for a single match."""
    with open(filepath) as fh:
        header = fh.readline().strip()
        team0 = header.split('Team 0: ')[1].split(',')[0].strip()
        team1 = header.split('Team 1: ')[1].strip()
        # Stockfish with various emoji/text
        we_are = None
        for marker in ['Stockfish', 'stockfish']:
            if marker in team0:
                we_are = 0
                break
            if marker in team1:
                we_are = 1
                break
        if we_are is None:
            return None
        opp = 1 - we_are
        opp_name = team1 if we_are == 0 else team0
        reader = csv.reader(fh)
        col_header = next(reader)
        rows = list(reader)

    if not rows:
        return None

    # Group by hand
    hand_rows = defaultdict(list)
    hand_bankroll = {}
    for row in rows:
        try:
            h = int(row[0])
        except:
            continue
        hand_rows[h].append(row)
        if h not in hand_bankroll:
            hand_bankroll[h] = int(row[3 + we_are])

    sorted_hands = sorted(hand_rows.keys())
    if len(sorted_hands) < 2:
        return None

    # Find auto-fold boundary
    last_active = 0
    for i in range(1, len(sorted_hands)):
        h = sorted_hands[i]
        prev_h = sorted_hands[i-1]
        delta = abs(hand_bankroll[h] - hand_bankroll[prev_h])
        if delta > 2:
            last_active = h

    # Results accumulators
    total_hands = 0
    hands_reaching_river = 0

    # River acting-first analysis
    af_check_then_showdown = []  # (hand, our_cards, opp_cards, board, delta, our_rank, opp_rank)
    af_bet_then_result = []      # (hand, action, amount, delta, our_rank, opp_rank)
    af_check_opp_checks = []     # both check on river
    af_total = 0

    # River facing-bet analysis
    fb_call_results = []   # (hand, cost, pot, delta, our_rank, opp_rank, our_equity_approx)
    fb_fold_results = []   # (hand, cost, pot, our_cards, opp_cards, board)
    fb_raise_results = []  # (hand, amount, delta)
    fb_total = 0

    # Specific categories
    overcalls = []         # called and lost
    correct_calls = []     # called and won
    missed_value = []      # we checked, opp checked, we had better hand (could have bet)
    missed_bluffs = []     # we checked, opp checked, we had worse hand (could have bluffed)
    good_checks = []       # we checked, opp checked, we won (had value but checked was ok)

    # Equity gate tracking (would need runtime data, but we can infer)
    gate_correct_folds = 0
    gate_incorrect_folds = 0

    # Chip impact tracking
    river_chips_from_calls = 0
    river_chips_from_overcalls = 0
    river_chips_from_value_bets = 0
    river_chips_from_bluffs = 0
    river_chips_from_folds = 0
    river_check_check_missed_value = 0

    for i in range(len(sorted_hands) - 1):
        h = sorted_hands[i]
        if h > last_active:
            break
        total_hands += 1

        next_h = sorted_hands[i+1]
        delta = hand_bankroll[next_h] - hand_bankroll[h]

        actions = hand_rows[h]
        river_actions = [r for r in actions if r[1] == 'River']

        if not river_actions:
            continue

        hands_reaching_river += 1

        # Parse final state cards
        last_row = actions[-1]
        try:
            our_cards = parse_cards(last_row[9] if we_are == 0 else last_row[10])
            opp_cards = parse_cards(last_row[10] if we_are == 0 else last_row[9])
            board = parse_cards(last_row[11])
        except:
            continue

        if len(board) < 5 or len(our_cards) < 2 or len(opp_cards) < 2:
            continue

        our_rank = hand_rank_7(our_cards + board)
        opp_rank = hand_rank_7(opp_cards + board)

        if our_rank is None or opp_rank is None:
            continue

        we_win = our_rank < opp_rank
        they_win = opp_rank < our_rank
        tie = our_rank == opp_rank

        # Classify river actions
        our_river = [(r[5], int(r[6])) for r in river_actions if int(r[2]) == we_are]
        opp_river = [(r[5], int(r[6])) for r in river_actions if int(r[2]) == opp]

        # Get bets at river start
        first_river = river_actions[0]
        river_start_bet_0 = int(first_river[14])
        river_start_bet_1 = int(first_river[15])
        our_start_bet = river_start_bet_0 if we_are == 0 else river_start_bet_1
        opp_start_bet = river_start_bet_1 if we_are == 0 else river_start_bet_0

        # Determine who acts first on river
        first_actor = int(river_actions[0][2])
        we_act_first = (first_actor == we_are)

        # Check if anyone folded
        any_fold = any(r[5] == 'FOLD' for r in river_actions)
        we_folded = any(r[5] == 'FOLD' for r in river_actions if int(r[2]) == we_are)
        opp_folded = any(r[5] == 'FOLD' for r in river_actions if int(r[2]) == opp)

        # Categorize by who acts first and what happens
        if we_act_first:
            af_total += 1
            our_first_action = our_river[0] if our_river else ('CHECK', 0)

            if our_first_action[0] == 'CHECK':
                # We checked first
                if not opp_river:
                    continue
                opp_first = opp_river[0] if opp_river else ('CHECK', 0)

                if opp_first[0] == 'CHECK':
                    # Check-check: showdown
                    af_check_opp_checks.append({
                        'hand': h, 'delta': delta, 'we_win': we_win,
                        'our_rank': our_rank, 'opp_rank': opp_rank,
                        'our_cards': our_cards, 'opp_cards': opp_cards,
                        'board': board, 'pot': our_start_bet + opp_start_bet
                    })

                    if we_win:
                        # We won at showdown after checking - could we have bet for value?
                        # Missed value if our hand was strong
                        pot = our_start_bet + opp_start_bet
                        missed_value.append({
                            'hand': h, 'delta': delta, 'pot': pot,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'our_cards': our_cards, 'opp_cards': opp_cards,
                            'board': board
                        })
                        # Estimate missed value: could have bet ~50% pot and gotten called some %
                        # Conservative: 30% of missed value bets get called
                        potential_bet = int(pot * 0.5)
                        river_check_check_missed_value += int(potential_bet * 0.3)

                    elif they_win:
                        # We lost at showdown - could we have bluffed them off?
                        pot = our_start_bet + opp_start_bet
                        missed_bluffs.append({
                            'hand': h, 'delta': delta, 'pot': pot,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'our_cards': our_cards, 'opp_cards': opp_cards,
                            'board': board
                        })

                elif opp_first[0] == 'RAISE':
                    # Opp bet after our check - we now face a bet
                    fb_total += 1
                    opp_bet_amt = opp_first[1]

                    # Find our response
                    if len(our_river) > 1:
                        our_response = our_river[1]
                    else:
                        our_response = ('FOLD', 0)  # implied

                    last_river = river_actions[-1]
                    final_our_bet = int(last_river[14]) if we_are == 0 else int(last_river[15])
                    final_opp_bet = int(last_river[15]) if we_are == 0 else int(last_river[14])
                    cost = final_opp_bet - our_start_bet
                    pot = our_start_bet + opp_start_bet

                    if we_folded:
                        fb_fold_results.append({
                            'hand': h, 'cost': cost, 'pot': pot,
                            'our_cards': our_cards, 'opp_cards': opp_cards,
                            'board': board, 'our_rank': our_rank, 'opp_rank': opp_rank,
                            'we_would_win': we_win
                        })
                        if we_win:
                            # We folded the best hand! Gate was wrong
                            gate_incorrect_folds += 1
                        else:
                            gate_correct_folds += 1
                    elif our_response[0] == 'CALL':
                        fb_call_results.append({
                            'hand': h, 'cost': cost, 'pot': pot, 'delta': delta,
                            'our_rank': our_rank, 'opp_rank': opp_rank, 'we_win': we_win
                        })
                        if we_win:
                            correct_calls.append({'hand': h, 'delta': delta, 'cost': cost})
                            river_chips_from_calls += delta
                        else:
                            overcalls.append({'hand': h, 'delta': delta, 'cost': cost, 'pot': pot})
                            river_chips_from_overcalls += delta  # negative
                    elif our_response[0] == 'RAISE':
                        fb_raise_results.append({
                            'hand': h, 'amount': our_response[1], 'delta': delta
                        })

            elif our_first_action[0] == 'RAISE':
                # We bet first
                bet_amt = our_first_action[1]
                pot = our_start_bet + opp_start_bet

                if opp_folded:
                    # They folded to our bet
                    river_chips_from_value_bets += delta if we_win else 0
                    river_chips_from_bluffs += delta if they_win else 0
                    af_bet_then_result.append({
                        'hand': h, 'action': 'bet', 'amount': bet_amt,
                        'delta': delta, 'opp_folded': True,
                        'our_rank': our_rank, 'opp_rank': opp_rank,
                        'we_win': we_win, 'pot': pot
                    })
                elif we_folded:
                    # They re-raised, we folded
                    af_bet_then_result.append({
                        'hand': h, 'action': 'bet_then_fold', 'amount': bet_amt,
                        'delta': delta, 'opp_folded': False,
                        'our_rank': our_rank, 'opp_rank': opp_rank,
                        'we_win': we_win, 'pot': pot
                    })
                else:
                    # Showdown after our bet
                    af_bet_then_result.append({
                        'hand': h, 'action': 'bet_showdown', 'amount': bet_amt,
                        'delta': delta, 'opp_folded': False,
                        'our_rank': our_rank, 'opp_rank': opp_rank,
                        'we_win': we_win, 'pot': pot
                    })
                    if we_win:
                        river_chips_from_value_bets += delta
                    else:
                        river_chips_from_overcalls += delta  # we bet and lost

        else:
            # Opponent acts first
            opp_first = opp_river[0] if opp_river else ('CHECK', 0)

            if opp_first[0] == 'CHECK':
                # Opp checked to us
                af_total += 1  # We're acting second but it's equivalent to acting first
                our_action = our_river[0] if our_river else ('CHECK', 0)

                if our_action[0] == 'CHECK':
                    # Check-check
                    af_check_opp_checks.append({
                        'hand': h, 'delta': delta, 'we_win': we_win,
                        'our_rank': our_rank, 'opp_rank': opp_rank,
                        'our_cards': our_cards, 'opp_cards': opp_cards,
                        'board': board, 'pot': our_start_bet + opp_start_bet
                    })

                    if we_win:
                        pot = our_start_bet + opp_start_bet
                        missed_value.append({
                            'hand': h, 'delta': delta, 'pot': pot,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'our_cards': our_cards, 'opp_cards': opp_cards,
                            'board': board
                        })
                        potential_bet = int(pot * 0.5)
                        river_check_check_missed_value += int(potential_bet * 0.3)
                    elif they_win:
                        pot = our_start_bet + opp_start_bet
                        missed_bluffs.append({
                            'hand': h, 'delta': delta, 'pot': pot,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'our_cards': our_cards, 'opp_cards': opp_cards,
                            'board': board
                        })

                elif our_action[0] == 'RAISE':
                    # We bet after opp check
                    bet_amt = our_action[1]
                    pot = our_start_bet + opp_start_bet

                    if opp_folded:
                        river_chips_from_value_bets += delta if we_win else 0
                        river_chips_from_bluffs += delta if they_win else 0
                        af_bet_then_result.append({
                            'hand': h, 'action': 'bet', 'amount': bet_amt,
                            'delta': delta, 'opp_folded': True,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'we_win': we_win, 'pot': pot
                        })
                    else:
                        af_bet_then_result.append({
                            'hand': h, 'action': 'bet_showdown', 'amount': bet_amt,
                            'delta': delta, 'opp_folded': False,
                            'our_rank': our_rank, 'opp_rank': opp_rank,
                            'we_win': we_win, 'pot': pot
                        })
                        if we_win:
                            river_chips_from_value_bets += delta
                        else:
                            river_chips_from_overcalls += delta

            elif opp_first[0] == 'RAISE':
                # Opp bet, we face it
                fb_total += 1
                opp_bet_amt = opp_first[1]
                cost = opp_bet_amt  # amount above what we already have in
                pot = our_start_bet + opp_start_bet

                our_action = our_river[0] if our_river else ('FOLD', 0)

                if we_folded:
                    fb_fold_results.append({
                        'hand': h, 'cost': cost, 'pot': pot,
                        'our_cards': our_cards, 'opp_cards': opp_cards,
                        'board': board, 'our_rank': our_rank, 'opp_rank': opp_rank,
                        'we_would_win': we_win
                    })
                    if we_win:
                        gate_incorrect_folds += 1
                    else:
                        gate_correct_folds += 1

                elif our_action[0] == 'CALL':
                    fb_call_results.append({
                        'hand': h, 'cost': cost, 'pot': pot, 'delta': delta,
                        'our_rank': our_rank, 'opp_rank': opp_rank, 'we_win': we_win
                    })
                    if we_win:
                        correct_calls.append({'hand': h, 'delta': delta, 'cost': cost})
                        river_chips_from_calls += delta
                    else:
                        overcalls.append({'hand': h, 'delta': delta, 'cost': cost, 'pot': pot})
                        river_chips_from_overcalls += delta

                elif our_action[0] == 'RAISE':
                    fb_raise_results.append({
                        'hand': h, 'amount': our_action[1], 'delta': delta
                    })

    final_bankroll = hand_bankroll.get(sorted_hands[-1], 0)

    return {
        'file': os.path.basename(filepath),
        'opp': opp_name,
        'final': final_bankroll,
        'total_hands': total_hands,
        'hands_reaching_river': hands_reaching_river,
        'river_pct': hands_reaching_river / max(total_hands, 1) * 100,
        'af_total': af_total,
        'af_check_check': len(af_check_opp_checks),
        'af_bet_results': af_bet_then_result,
        'fb_total': fb_total,
        'fb_call_results': fb_call_results,
        'fb_fold_results': fb_fold_results,
        'fb_raise_results': fb_raise_results,
        'overcalls': overcalls,
        'correct_calls': correct_calls,
        'missed_value': missed_value,
        'missed_bluffs': missed_bluffs,
        'chips_from_calls': river_chips_from_calls,
        'chips_from_overcalls': river_chips_from_overcalls,
        'chips_from_value_bets': river_chips_from_value_bets,
        'chips_from_bluffs': river_chips_from_bluffs,
        'chips_missed_value': river_check_check_missed_value,
        'gate_correct_folds': gate_correct_folds,
        'gate_incorrect_folds': gate_incorrect_folds,
        'check_check_details': af_check_opp_checks,
    }


def print_full_analysis(all_results):
    """Print comprehensive river analysis."""

    # Filter out None results
    results = [r for r in all_results if r is not None]
    if not results:
        print("No valid matches to analyze")
        return

    n_matches = len(results)

    # Aggregate totals
    total_hands = sum(r['total_hands'] for r in results)
    total_river = sum(r['hands_reaching_river'] for r in results)
    total_af = sum(r['af_total'] for r in results)
    total_fb = sum(r['fb_total'] for r in results)
    total_cc = sum(r['af_check_check'] for r in results)

    all_overcalls = []
    all_correct_calls = []
    all_missed_value = []
    all_missed_bluffs = []
    all_fb_calls = []
    all_fb_folds = []
    all_af_bets = []
    all_cc_details = []

    total_chips_calls = 0
    total_chips_overcalls = 0
    total_chips_value = 0
    total_chips_bluffs = 0
    total_chips_missed = 0
    total_gate_correct = 0
    total_gate_incorrect = 0

    for r in results:
        all_overcalls.extend(r['overcalls'])
        all_correct_calls.extend(r['correct_calls'])
        all_missed_value.extend(r['missed_value'])
        all_missed_bluffs.extend(r['missed_bluffs'])
        all_fb_calls.extend(r['fb_call_results'])
        all_fb_folds.extend(r['fb_fold_results'])
        all_af_bets.extend(r['af_bet_results'])
        all_cc_details.extend(r['check_check_details'])
        total_chips_calls += r['chips_from_calls']
        total_chips_overcalls += r['chips_from_overcalls']
        total_chips_value += r['chips_from_value_bets']
        total_chips_bluffs += r['chips_from_bluffs']
        total_chips_missed += r['chips_missed_value']
        total_gate_correct += r['gate_correct_folds']
        total_gate_incorrect += r['gate_incorrect_folds']

    wins = sum(1 for r in results if r['final'] > 0)
    losses = n_matches - wins
    total_final = sum(r['final'] for r in results)

    print("=" * 90)
    print(f"RIVER DECISION ANALYSIS FOR v38 IMPACT ESTIMATION")
    print(f"  {n_matches} matches ({wins}W {losses}L), net {total_final:+d} chips")
    print("=" * 90)

    # =========================================================================
    # SECTION 1: Overview
    # =========================================================================
    print(f"\n{'='*90}")
    print("1. RIVER REACH RATE")
    print(f"{'='*90}")
    print(f"  Total hands played (active):  {total_hands}")
    print(f"  Hands reaching river:         {total_river} ({total_river/max(total_hands,1)*100:.1f}%)")
    print(f"  River decisions (act-first):   {total_af} ({total_af/max(total_hands,1)*100:.1f}% of all hands)")
    print(f"  River decisions (facing bet):  {total_fb} ({total_fb/max(total_hands,1)*100:.1f}% of all hands)")
    print(f"  River check-check:             {total_cc} ({total_cc/max(total_river,1)*100:.1f}% of river hands)")

    # =========================================================================
    # SECTION 2: Per-match summary
    # =========================================================================
    print(f"\n{'='*90}")
    print("2. PER-MATCH RIVER BREAKDOWN")
    print(f"{'='*90}")
    print(f"  {'Opponent':<22} {'Result':>6} {'River%':>6} {'AF':>4} {'FB':>4} {'CC':>4} {'OC':>3} {'MV':>3} {'CC$':>6} {'OC$':>7} {'VB$':>6}")
    print("  " + "-" * 85)
    for r in sorted(results, key=lambda x: x['final']):
        oc_count = len(r['overcalls'])
        mv_count = len(r['missed_value'])
        print(f"  {r['opp'][:22]:<22} {r['final']:>+6} {r['river_pct']:>5.1f}% "
              f"{r['af_total']:>4} {r['fb_total']:>4} {r['af_check_check']:>4} "
              f"{oc_count:>3} {mv_count:>3} "
              f"{r['chips_missed_value']:>+6} {r['chips_from_overcalls']:>+7} {r['chips_from_value_bets']:>+6}")

    # =========================================================================
    # SECTION 3: Acting-first decisions
    # =========================================================================
    print(f"\n{'='*90}")
    print("3. ACTING-FIRST RIVER DECISIONS (v38 replaces with precomputed strategy)")
    print(f"{'='*90}")

    # 3a: Check-check analysis
    cc_we_won = sum(1 for d in all_cc_details if d['we_win'])
    cc_they_won = sum(1 for d in all_cc_details if not d['we_win'])
    cc_delta_won = sum(d['delta'] for d in all_cc_details if d['we_win'])
    cc_delta_lost = sum(d['delta'] for d in all_cc_details if not d['we_win'])

    print(f"\n  3a. Check-Check Hands: {total_cc}")
    print(f"      We won showdown:  {cc_we_won} ({cc_we_won/max(total_cc,1)*100:.1f}%) for {cc_delta_won:+d} chips")
    print(f"      They won showdown: {cc_they_won} ({cc_they_won/max(total_cc,1)*100:.1f}%) for {cc_delta_lost:+d} chips")
    print(f"      Net from CC hands: {cc_delta_won + cc_delta_lost:+d} chips")

    # Missed value analysis: when we won check-check, how much could we have extracted?
    print(f"\n  3b. MISSED VALUE BETS (we won check-check):")
    print(f"      Total missed value spots: {len(all_missed_value)}")
    if all_missed_value:
        avg_pot = sum(d['pot'] for d in all_missed_value) / len(all_missed_value)
        print(f"      Average pot at river:     {avg_pot:.0f} chips")
        # Estimate: half-pot bet, 30% call rate by opponent
        est_per_hand = avg_pot * 0.5 * 0.30
        est_total = est_per_hand * len(all_missed_value)
        print(f"      Est. value per bet:       {est_per_hand:.1f} chips (50% pot, 30% call rate)")
        print(f"      Est. TOTAL missed value:  {est_total:+.0f} chips")

        # Break down by hand rank quality
        strong_mv = [d for d in all_missed_value if d['our_rank'] < d['opp_rank'] * 0.5]
        medium_mv = [d for d in all_missed_value if d['our_rank'] >= d['opp_rank'] * 0.5]
        print(f"      Strong value (rank < 50% of opp): {len(strong_mv)} hands")
        print(f"      Marginal value:                    {len(medium_mv)} hands")

    # Missed bluff analysis
    print(f"\n  3c. MISSED BLUFF OPPORTUNITIES (they won check-check):")
    print(f"      Total missed bluff spots: {len(all_missed_bluffs)}")
    if all_missed_bluffs:
        avg_pot_bluff = sum(d['pot'] for d in all_missed_bluffs) / len(all_missed_bluffs)
        print(f"      Average pot at river:     {avg_pot_bluff:.0f} chips")
        # Estimate: half-pot bluff, opponent folds ~40% to river bets
        # But not all of these are good bluffs - only ~30% would be selected by solver
        good_bluff_pct = 0.30
        fold_rate = 0.40
        est_bluff_ev = avg_pot_bluff * fold_rate - avg_pot_bluff * 0.5 * (1 - fold_rate)
        if est_bluff_ev > 0:
            est_bluff_total = est_bluff_ev * len(all_missed_bluffs) * good_bluff_pct
            print(f"      Est. EV per good bluff:   {est_bluff_ev:.1f} chips (40% fold rate)")
            print(f"      Est. TOTAL bluff value:   {est_bluff_total:+.0f} chips (30% are good bluff candidates)")
        else:
            print(f"      Bluffing would be -EV (fold rate too low)")

    # 3d: Our bet results
    print(f"\n  3d. OUR RIVER BETS (acting first):")
    bets_won = [b for b in all_af_bets if b.get('we_win')]
    bets_lost = [b for b in all_af_bets if not b.get('we_win')]
    bets_fold = [b for b in all_af_bets if b.get('opp_folded')]
    bets_showdown = [b for b in all_af_bets if not b.get('opp_folded')]
    print(f"      Total river bets by us: {len(all_af_bets)}")
    print(f"      Opponent folded:   {len(bets_fold)} ({len(bets_fold)/max(len(all_af_bets),1)*100:.0f}%)")
    print(f"      Went to showdown:  {len(bets_showdown)}")
    if bets_showdown:
        sd_won = sum(1 for b in bets_showdown if b.get('we_win'))
        sd_lost = len(bets_showdown) - sd_won
        print(f"        Won at showdown: {sd_won}, Lost: {sd_lost} ({sd_won/len(bets_showdown)*100:.0f}% WR)")
    total_bet_delta = sum(b['delta'] for b in all_af_bets)
    print(f"      Total chips from our bets: {total_bet_delta:+d}")

    # =========================================================================
    # SECTION 4: Facing-bet decisions
    # =========================================================================
    print(f"\n{'='*90}")
    print("4. FACING-BET RIVER DECISIONS (v38 uses P(bet|hand) Bayesian narrowing)")
    print(f"{'='*90}")

    total_fb_acts = len(all_fb_calls) + len(all_fb_folds)
    print(f"  Total facing-bet situations: {total_fb_acts}")
    print(f"  Calls:  {len(all_fb_calls)} ({len(all_fb_calls)/max(total_fb_acts,1)*100:.0f}%)")
    print(f"  Folds:  {len(all_fb_folds)} ({len(all_fb_folds)/max(total_fb_acts,1)*100:.0f}%)")

    # 4a: Call outcomes
    print(f"\n  4a. CALL OUTCOMES:")
    call_wins = [c for c in all_fb_calls if c['we_win']]
    call_losses = [c for c in all_fb_calls if not c['we_win']]
    print(f"      Won:  {len(call_wins)} ({len(call_wins)/max(len(all_fb_calls),1)*100:.0f}%)")
    print(f"      Lost: {len(call_losses)} ({len(call_losses)/max(len(all_fb_calls),1)*100:.0f}%)")
    call_net = sum(c['delta'] for c in all_fb_calls)
    print(f"      Net chips from calls: {call_net:+d}")
    if call_wins:
        avg_win = sum(c['delta'] for c in call_wins) / len(call_wins)
        print(f"      Avg win per correct call: {avg_win:+.1f}")
    if call_losses:
        avg_loss = sum(c['delta'] for c in call_losses) / len(call_losses)
        print(f"      Avg loss per overcall:    {avg_loss:+.1f}")

    # 4b: Overcalls detail
    print(f"\n  4b. OVERCALLS (called and lost) - {len(all_overcalls)} total:")
    print(f"      Total chips lost to overcalls: {total_chips_overcalls:+d}")
    if all_overcalls:
        avg_oc_loss = total_chips_overcalls / len(all_overcalls)
        print(f"      Avg loss per overcall: {avg_oc_loss:.1f}")
        # How many could have been saved?
        # With better narrowing, ~50% of overcalls might become folds
        salvageable = int(len(all_overcalls) * 0.5)
        est_savings = int(abs(avg_oc_loss) * salvageable)
        print(f"      If v38 saves 50% of overcalls: +{est_savings} chips recovered")

    # 4c: Fold analysis
    print(f"\n  4c. FOLD OUTCOMES:")
    folds_correct = [f for f in all_fb_folds if not f['we_would_win']]
    folds_incorrect = [f for f in all_fb_folds if f['we_would_win']]
    print(f"      Correct folds (they had better): {len(folds_correct)} ({len(folds_correct)/max(len(all_fb_folds),1)*100:.0f}%)")
    print(f"      Incorrect folds (we had better): {len(folds_incorrect)} ({len(folds_incorrect)/max(len(all_fb_folds),1)*100:.0f}%)")
    if folds_incorrect:
        # These are hands where better narrowing might have let us call
        print(f"      Incorrect folds = equity gate over-triggered or solver over-folded")
        avg_pot = sum(f['pot'] for f in folds_incorrect) / len(folds_incorrect)
        # We lost the pot we were already in
        est_lost_from_bad_folds = sum(f['pot'] + f['cost'] for f in folds_incorrect)
        print(f"      Est. value lost from incorrect folds: ~{est_lost_from_bad_folds:.0f} chips")

    # =========================================================================
    # SECTION 5: Equity Gate Analysis
    # =========================================================================
    print(f"\n{'='*90}")
    print("5. EQUITY GATE ANALYSIS (v38 removes equity gate on river)")
    print(f"{'='*90}")
    print(f"  Equity gate correct folds:   {total_gate_correct}")
    print(f"  Equity gate incorrect folds: {total_gate_incorrect}")
    gate_total = total_gate_correct + total_gate_incorrect
    if gate_total > 0:
        print(f"  Gate accuracy: {total_gate_correct/gate_total*100:.1f}%")
        print(f"  NOTE: 'Correct' just means we would have lost at showdown.")
        print(f"        Incorrect folds cost us the entire pot we'd invested.")

    # =========================================================================
    # SECTION 6: Chip Impact Summary
    # =========================================================================
    print(f"\n{'='*90}")
    print("6. CHIP IMPACT SUMMARY")
    print(f"{'='*90}")
    print(f"  Chips from correct river calls:    {total_chips_calls:>+8d}")
    print(f"  Chips lost to river overcalls:     {total_chips_overcalls:>+8d}")
    print(f"  Chips from river value bets:       {total_chips_value:>+8d}")
    print(f"  Chips from river bluffs (fold):    {total_chips_bluffs:>+8d}")
    print(f"  Est. missed value (check-check):   {total_chips_missed:>+8d}")
    total_river_impact = (total_chips_calls + total_chips_overcalls +
                          total_chips_value + total_chips_bluffs)
    print(f"  -----------------------------------------------")
    print(f"  Net river P&L (realized):          {total_river_impact:>+8d}")
    print(f"  + Missed value opportunity:         {total_chips_missed:>+8d}")
    print(f"  = Total river significance:         {total_river_impact + total_chips_missed:>+8d}")

    # =========================================================================
    # SECTION 7: v38 Impact Estimate
    # =========================================================================
    print(f"\n{'='*90}")
    print("7. v38 ESTIMATED IMPACT")
    print(f"{'='*90}")

    # Three improvements:
    # 1. Better acting-first (precomputed 500-iter vs 50-75 runtime)
    # 2. Board-specific P(bet|hand) narrowing
    # 3. Removal of equity gate

    # Impact 1: Better acting-first
    # - More value bets where currently checking
    # - Better bluff selection
    est_value_gain = total_chips_missed  # conservative
    print(f"\n  IMPROVEMENT 1: Precomputed acting-first strategies")
    print(f"    Current check-check hands: {total_cc}")
    print(f"    Missed value estimate:     {est_value_gain:+d} chips")
    print(f"    Better convergence (500 vs ~60 iter) = more precise bet/check boundary")

    # Impact 2: Better facing-bet narrowing
    # - P(bet|hand) is board-specific: accounts for texture
    # - Heuristic polarized assumes GTO ratio, but opponents differ
    oc_recovery = int(abs(total_chips_overcalls) * 0.3)  # conservative: save 30%
    bad_fold_recovery = int(sum(f['pot'] for f in folds_incorrect) * 0.2) if folds_incorrect else 0
    print(f"\n  IMPROVEMENT 2: Board-specific P(bet|hand) Bayesian narrowing")
    print(f"    Current overcall losses:   {total_chips_overcalls:+d} chips")
    print(f"    Est. overcalls recovered:  +{oc_recovery} chips (30% recovery)")
    print(f"    Bad fold recovery:         +{bad_fold_recovery} chips")

    # Impact 3: Equity gate removal
    # With correct range info from precomputed data, gate is unnecessary
    # Currently gate catches some correct folds but also misses calls
    gate_cost = int(sum(f['pot'] for f in folds_incorrect) * 0.5) if folds_incorrect else 0
    print(f"\n  IMPROVEMENT 3: Equity gate removal (solver gets correct range)")
    print(f"    Incorrect gate folds:      {total_gate_incorrect}")
    print(f"    Est. chips recovered:      +{gate_cost} chips")

    total_est = est_value_gain + oc_recovery + bad_fold_recovery + gate_cost
    print(f"\n  -----------------------------------------------")
    print(f"  TOTAL ESTIMATED v38 IMPROVEMENT: +{total_est} chips across {n_matches} matches")
    print(f"  Per-match improvement:           +{total_est/n_matches:.0f} chips/match")
    print(f"  Per-hand improvement:            +{total_est/max(total_hands,1)*100:.2f} chips/100 hands")

    net_match_result = sum(r['final'] for r in results)
    print(f"\n  Current net result:              {net_match_result:+d} chips across {n_matches} matches")
    print(f"  With v38 improvement:            ~{net_match_result + total_est:+d} chips")

    # =========================================================================
    # SECTION 8: Detailed Hand Examples
    # =========================================================================
    print(f"\n{'='*90}")
    print("8. NOTABLE HAND EXAMPLES")
    print(f"{'='*90}")

    # Worst overcalls
    if all_overcalls:
        worst = sorted(all_overcalls, key=lambda x: x['delta'])[:5]
        print(f"\n  WORST OVERCALLS (biggest losses from calling river bets):")
        for oc in worst:
            print(f"    Hand #{oc['hand']}: lost {oc['delta']:+d} chips (cost={oc['cost']}, pot={oc['pot']})")

    # Best correct calls
    if all_correct_calls:
        best_cc = sorted(all_correct_calls, key=lambda x: x['delta'], reverse=True)[:5]
        print(f"\n  BEST CORRECT CALLS:")
        for cc in best_cc:
            print(f"    Hand #{cc['hand']}: won {cc['delta']:+d} chips")

    # Incorrect folds (folded the winner)
    if folds_incorrect:
        print(f"\n  INCORRECT FOLDS (folded best hand to river bet):")
        for fi in folds_incorrect[:10]:
            print(f"    Hand #{fi['hand']}: folded (cost={fi['cost']}, pot={fi['pot']}, "
                  f"our_rank={fi['our_rank']}, opp_rank={fi['opp_rank']})")


def main():
    # Find all match files
    match_dir = os.path.expanduser("~/Downloads/poker_matches/")
    files = sorted(glob.glob(os.path.join(match_dir, "match_*.txt")))

    if not files:
        print("No match files found in ~/Downloads/poker_matches/")
        return

    # Take most recent 20 matches (by match number)
    files_by_num = []
    for f in files:
        try:
            num = int(os.path.basename(f).replace('match_', '').replace('.txt', ''))
            files_by_num.append((num, f))
        except:
            pass

    files_by_num.sort(key=lambda x: x[0], reverse=True)
    recent = [f for _, f in files_by_num[:20]]

    print(f"Analyzing {len(recent)} most recent matches...")
    print(f"Match IDs: {', '.join(os.path.basename(f).replace('match_','').replace('.txt','') for f in recent[:10])}...")

    results = []
    for f in recent:
        try:
            r = analyze_match_rivers(f)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  Error in {os.path.basename(f)}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nSuccessfully analyzed {len(results)} matches\n")
    print_full_analysis(results)


if __name__ == '__main__':
    main()
