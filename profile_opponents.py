#!/usr/bin/env python3
"""
Mass opponent profiling: analyze every opponent's strategy from match logs.
Tracks VPIP, bet frequencies, aggression, bluff rates, showdown equity when betting,
fold rates, check-trapping, and more. Identifies how far each bot deviates from GTO.
"""
import csv, ast, sys, os, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine
engine = ExactEquityEngine()

RANKS = "23456789A"
SUITS = "dhs"
def card_to_int(s): return SUITS.index(s[1]) * 9 + RANKS.index(s[0])

# Top bots from leaderboard - map name variations
TOP_BOTS = {
    'ALL IN': '#1 ALL_IN',
    'super-quantum-frogtron-500M-params': '#2 frogtron',
    'Joker': '#3 Joker',
    'Parkway Gardens': '#4 Parkway',
    "Poker? I Barely Know Her": '#5 PIBKH',
    'Attention Is All You Need': '#6 Attention',
    'Really Bad Bot': '#7 ReallyBad',
    "Ctrl+Alt+Defeat": '#8 CtrlAlt',
    'Me & Claude <3': '#9 MeClaude',
    'pokeorpoker': '#10 pokeorpoker',
    'Team': '#11 Team',
    'GradientAscent': '#12 Gradient',
    'sheep army': '#13 sheep',
    'Peace&Love': '#14 Peace',
    'SKT T1': '#15 SKT',
    'PING': '#16 PING',
    'Jackbot': '#17 Jackbot',
    'WW': '#18 WW',
    'AWA': '#19 AWA',
    'AA998': '#20 AA998',
    'AlphaFold': '#22 AlphaFold',
    'Captain Cooked': '#23 Captain',
    'rainbet': '#24 rainbet',
    'Mind the raise': '#25 Mind',
    'wushilandb00ba': '#26 wushi',
    'if my_turn{fold()}': '#27 ifmyturn',
    'while (true) { go all in }': '#28 whiletrue',
    'D4-0LP&D2-25LP': '#29 D4',
    'SixsevenPoker': '#31 Sixseven',
    'FLEXasHoldEm': '#32 FLEX',
    'VitaminMonkeys': '#34 Vitamin',
    'Alan Keating': '#35 Alan',
    'niconiconi': '#40 niconiconi',
    'bubtubs i recall was a kind soul. bless the kind fellow when you pass him by': '#41 bubtubs',
    'hehe': '#58 hehe',
    'placeholdername': '#60 placeholder',
}

def profile_opponent(filepath):
    """Extract detailed opponent strategy profile from a match log."""
    with open(filepath) as fh:
        header = fh.readline().strip()
        team0 = header.split('Team 0: ')[1].split(',')[0].strip()
        team1 = header.split('Team 1: ')[1].strip()

        # Figure out which team is us (Stockfish)
        if 'Stockfish' in team0:
            we_are, opp_idx = 0, 1
            opp_name = team1
        elif 'Stockfish' in team1:
            we_are, opp_idx = 1, 0
            opp_name = team0
        else:
            return None  # Not our match

        reader = csv.reader(fh)
        next(reader)  # skip header
        rows = list(reader)

    # Group by hand
    hands = defaultdict(list)
    for row in rows:
        hands[int(row[0])].append(row)

    # Find active portion
    hand_bankrolls = {}
    for h, actions in hands.items():
        hand_bankrolls[h] = int(actions[0][3 + we_are])

    sorted_hands = sorted(hand_bankrolls.keys())
    last_active = sorted_hands[-1]
    prev_b = 0
    for i in range(len(sorted_hands)):
        b = hand_bankrolls[sorted_hands[i]]
        if i > 0 and abs(b - prev_b) > 2:
            last_active = sorted_hands[i]
        prev_b = b

    final_b = int(rows[-1][3 + we_are])

    # --- Opponent stats ---
    stats = {
        'name': opp_name,
        'match_id': os.path.basename(filepath),
        'result': 'WIN' if final_b > 0 else 'LOSS',
        'final_bankroll': final_b,
        'active_hands': 0,
        # VPIP: hands where opp voluntarily put money in (not just posting blind)
        'vpip_hands': 0,
        'total_hands': 0,
        # Pre-flop raise
        'pfr_count': 0,
        'pf_opportunities': 0,
        # Per-street: bets/raises, checks, calls, folds
        'street_bets': defaultdict(int),
        'street_checks': defaultdict(int),
        'street_calls': defaultdict(int),
        'street_folds': defaultdict(int),
        'street_actions': defaultdict(int),
        # Fold to bet per street
        'faced_bet': defaultdict(int),
        'folded_to_bet': defaultdict(int),
        # River specifics
        'river_bet_equities': [],      # equity when opp bets river and goes to showdown
        'river_check_equities': [],    # equity when opp checks river (showdowns only)
        'river_bluffs': 0,             # bets with <50% equity
        'river_value_bets': 0,         # bets with >=50% equity
        'river_bets_total': 0,
        # Check-trapping: checks with >70% equity
        'check_traps': defaultdict(int),
        'checks_with_equity': defaultdict(list),
        # Showdown stats
        'showdown_wr_when_betting': [],  # did they win when they bet and it went to showdown?
        'showdown_wr_when_checking': [],
        # Bet sizing
        'bet_sizes_pct': defaultdict(list),  # bet as % of pot per street
        # 3-bet frequency
        'three_bet_opportunities': 0,
        'three_bets': 0,
        # All-in frequency
        'allins': 0,
    }

    for h in sorted_hands:
        if h > last_active:
            break
        stats['active_hands'] += 1
        actions = hands[h]
        stats['total_hands'] += 1

        # Track VPIP: did opp put money in voluntarily pre-flop?
        pf_actions = [r for r in actions if r[1] == 'Pre-Flop' and int(r[2]) == opp_idx and r[5] != 'DISCARD']
        opp_raised_pf = any(r[5] == 'RAISE' for r in pf_actions)
        opp_called_pf = any(r[5] == 'CALL' for r in pf_actions)
        if opp_raised_pf or opp_called_pf:
            stats['vpip_hands'] += 1
        if len(pf_actions) > 0:
            stats['pf_opportunities'] += 1
            if opp_raised_pf:
                stats['pfr_count'] += 1

        # Get final row for equity calculation
        last_row = actions[-1]
        board_cards_raw = ast.literal_eval(last_row[11])
        opp_cards_raw = ast.literal_eval(last_row[9 + opp_idx])
        our_cards_raw = ast.literal_eval(last_row[9 + we_are])

        # Track per-street actions
        prev_street_had_bet = {}
        opp_bet_this_hand_by_street = defaultdict(bool)

        for i_row, row in enumerate(actions):
            if row[5] == 'DISCARD':
                continue
            street = row[1]
            actor = int(row[2])
            action = row[5]

            if actor == opp_idx:
                stats['street_actions'][street] += 1

                if action == 'RAISE':
                    stats['street_bets'][street] += 1
                    opp_bet_this_hand_by_street[street] = True

                    # Bet sizing
                    pot = int(row[14]) + int(row[15])
                    if pot > 0:
                        bet_amt = int(row[6]) if row[6] else 0
                        if bet_amt > 0:
                            stats['bet_sizes_pct'][street].append(bet_amt / pot * 100)

                    # Check for all-in (bet >= 100 - current contribution or very large)
                    if int(row[6]) >= 90:
                        stats['allins'] += 1

                    # 3-bet: was there already a raise before this on same street?
                    prior_raises = sum(1 for pr in actions[:i_row]
                                      if pr[1] == street and pr[5] == 'RAISE')
                    if prior_raises >= 1:
                        stats['three_bets'] += 1

                elif action == 'CHECK':
                    stats['street_checks'][street] += 1
                elif action == 'CALL':
                    stats['street_calls'][street] += 1
                elif action == 'FOLD':
                    stats['street_folds'][street] += 1

            # Track "faced a bet" for fold-to-bet
            if actor == opp_idx and action in ('CALL', 'FOLD', 'RAISE'):
                # Did they face a bet? Check if there was a prior raise on this street by us
                prior_our_raises = [pr for pr in actions[:i_row]
                                    if pr[1] == street and int(pr[2]) == we_are and pr[5] == 'RAISE']
                if prior_our_raises:
                    stats['faced_bet'][street] += 1
                    if action == 'FOLD':
                        stats['folded_to_bet'][street] += 1

            # 3-bet opportunities: opp faces a raise
            if actor == opp_idx and street == 'Pre-Flop':
                prior_raises = sum(1 for pr in actions[:i_row]
                                  if pr[1] == street and pr[5] == 'RAISE')
                if prior_raises == 1:
                    stats['three_bet_opportunities'] += 1

        # Equity-based analysis (need showdown with full board)
        ended_in_fold = any(r[5] == 'FOLD' for r in actions)
        if not ended_in_fold and len(board_cards_raw) >= 5 and len(opp_cards_raw) >= 2 and len(our_cards_raw) >= 2:
            try:
                opp_cards = [card_to_int(c) for c in opp_cards_raw[:2]]
                our_cards = [card_to_int(c) for c in our_cards_raw[:2]]
                board = [card_to_int(c) for c in board_cards_raw[:5]]

                opp_rank = engine.lookup_seven(opp_cards + board)
                our_rank = engine.lookup_seven(our_cards + board)
                opp_won = 1.0 if opp_rank < our_rank else 0.5 if opp_rank == our_rank else 0.0

                # River actions
                river_actions = [r for r in actions if r[1] == 'River']
                opp_river = [r for r in river_actions if int(r[2]) == opp_idx and r[5] != 'DISCARD']

                opp_bet_river = any(r[5] == 'RAISE' for r in opp_river)
                opp_check_river = any(r[5] == 'CHECK' for r in opp_river)

                if opp_bet_river:
                    stats['river_bet_equities'].append(opp_won)
                    stats['river_bets_total'] += 1
                    stats['showdown_wr_when_betting'].append(opp_won)
                    if opp_won >= 0.5:
                        stats['river_value_bets'] += 1
                    else:
                        stats['river_bluffs'] += 1
                elif opp_check_river:
                    stats['river_check_equities'].append(opp_won)
                    stats['showdown_wr_when_checking'].append(opp_won)

                # Check-trapping per street (check with >70% equity at showdown)
                for street_name in ['Flop', 'Turn', 'River']:
                    opp_street_actions = [r for r in actions if r[1] == street_name and int(r[2]) == opp_idx]
                    if any(r[5] == 'CHECK' for r in opp_street_actions):
                        stats['checks_with_equity'][street_name].append(opp_won)
                        if opp_won >= 0.7:
                            stats['check_traps'][street_name] += 1

            except Exception:
                pass

    return stats


def merge_profiles(profiles):
    """Merge multiple match profiles for same opponent."""
    merged = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'losses': 0,
        'total_bankroll': 0,
        'total_hands': 0,
        'vpip_hands': 0,
        'pfr_count': 0,
        'pf_opportunities': 0,
        'street_bets': defaultdict(int),
        'street_checks': defaultdict(int),
        'street_calls': defaultdict(int),
        'street_folds': defaultdict(int),
        'street_actions': defaultdict(int),
        'faced_bet': defaultdict(int),
        'folded_to_bet': defaultdict(int),
        'river_bet_equities': [],
        'river_check_equities': [],
        'river_bluffs': 0,
        'river_value_bets': 0,
        'river_bets_total': 0,
        'check_traps': defaultdict(int),
        'checks_with_equity': defaultdict(list),
        'showdown_wr_when_betting': [],
        'showdown_wr_when_checking': [],
        'bet_sizes_pct': defaultdict(list),
        'three_bet_opportunities': 0,
        'three_bets': 0,
        'allins': 0,
    })

    for p in profiles:
        name = p['name']
        m = merged[name]
        m['matches'] += 1
        m['wins'] += 1 if p['result'] == 'WIN' else 0
        m['losses'] += 1 if p['result'] == 'LOSS' else 0
        m['total_bankroll'] += p['final_bankroll']
        m['total_hands'] += p['total_hands']
        m['vpip_hands'] += p['vpip_hands']
        m['pfr_count'] += p['pfr_count']
        m['pf_opportunities'] += p['pf_opportunities']
        m['river_bluffs'] += p['river_bluffs']
        m['river_value_bets'] += p['river_value_bets']
        m['river_bets_total'] += p['river_bets_total']
        m['three_bet_opportunities'] += p['three_bet_opportunities']
        m['three_bets'] += p['three_bets']
        m['allins'] += p['allins']
        m['river_bet_equities'].extend(p['river_bet_equities'])
        m['river_check_equities'].extend(p['river_check_equities'])
        m['showdown_wr_when_betting'].extend(p['showdown_wr_when_betting'])
        m['showdown_wr_when_checking'].extend(p['showdown_wr_when_checking'])
        for s in ['Pre-Flop', 'Flop', 'Turn', 'River']:
            m['street_bets'][s] += p['street_bets'][s]
            m['street_checks'][s] += p['street_checks'][s]
            m['street_calls'][s] += p['street_calls'][s]
            m['street_folds'][s] += p['street_folds'][s]
            m['street_actions'][s] += p['street_actions'][s]
            m['faced_bet'][s] += p['faced_bet'][s]
            m['folded_to_bet'][s] += p['folded_to_bet'][s]
            m['check_traps'][s] += p['check_traps'][s]
            m['checks_with_equity'][s].extend(p['checks_with_equity'][s])
            m['bet_sizes_pct'][s].extend(p['bet_sizes_pct'][s])

    return merged


def safe_pct(num, denom):
    return num / denom * 100 if denom > 0 else None

def fmt_pct(val):
    return f"{val:.0f}%" if val is not None else "  - "

def fmt_f(val, decimals=1):
    if val is None: return "  - "
    return f"{val:.{decimals}f}"


def print_profiles(merged, all_names_order):
    """Print comprehensive opponent profiles."""

    # GTO reference ranges (approximate for this game variant)
    print(f"{'='*120}")
    print(f"OPPONENT STRATEGY PROFILES — {sum(m['matches'] for m in merged.values())} matches, {sum(m['total_hands'] for m in merged.values())} hands analyzed")
    print(f"{'='*120}")
    print()
    print("GTO REFERENCE (approx): VPIP ~50-60%, PFR ~25-35%, Bet freq: Flop 25-35%, Turn 20-30%, River 15-25%")
    print("                        Fold-to-bet: ~40-50%, River bet equity: ~65-70%, Bluff ratio: ~25-33%")
    print()

    # Sort by leaderboard rank if available, otherwise by match count
    def sort_key(name):
        for full_name, label in TOP_BOTS.items():
            if full_name in name or name in full_name:
                rank = int(label.split('#')[1].split(' ')[0])
                return (0, rank)
        return (1, -merged[name]['matches'])

    sorted_names = sorted(merged.keys(), key=sort_key)

    # === TABLE 1: Core Stats ===
    print(f"{'CORE STATS':=^120}")
    print(f"{'Opponent':<35} {'#M':>3} {'W-L':>5} {'VPIP':>5} {'PFR':>5} | {'FlopBet':>7} {'TurnBet':>7} {'RivrBet':>7} | {'AF_F':>5} {'AF_T':>5} {'AF_R':>5}")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l + " "
                break

        display = (label + name)[:35]

        vpip = safe_pct(m['vpip_hands'], m['total_hands'])
        pfr = safe_pct(m['pfr_count'], m['pf_opportunities'])

        flop_bet = safe_pct(m['street_bets']['Flop'], m['street_actions']['Flop'])
        turn_bet = safe_pct(m['street_bets']['Turn'], m['street_actions']['Turn'])
        river_bet = safe_pct(m['street_bets']['River'], m['street_actions']['River'])

        # Aggression factor: (bets+raises) / calls per street
        def af(street):
            calls = m['street_calls'][street]
            bets = m['street_bets'][street]
            return bets / calls if calls > 0 else None

        print(f"{display:<35} {m['matches']:>3} {m['wins']}-{m['losses']:>1} {fmt_pct(vpip):>5} {fmt_pct(pfr):>5} | "
              f"{fmt_pct(flop_bet):>7} {fmt_pct(turn_bet):>7} {fmt_pct(river_bet):>7} | "
              f"{fmt_f(af('Flop')):>5} {fmt_f(af('Turn')):>5} {fmt_f(af('River')):>5}")

    # === TABLE 2: Defensive Stats ===
    print()
    print(f"{'DEFENSIVE STATS':=^120}")
    print(f"{'Opponent':<35} {'#M':>3} | {'FoldFlop':>8} {'FoldTurn':>8} {'FoldRivr':>8} | {'3bet%':>5} {'Allins':>6} {'AI/Hand':>7}")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l + " "
                break

        display = (label + name)[:35]

        fold_flop = safe_pct(m['folded_to_bet']['Flop'], m['faced_bet']['Flop'])
        fold_turn = safe_pct(m['folded_to_bet']['Turn'], m['faced_bet']['Turn'])
        fold_river = safe_pct(m['folded_to_bet']['River'], m['faced_bet']['River'])

        three_bet = safe_pct(m['three_bets'], m['three_bet_opportunities'])
        ai_per_hand = m['allins'] / m['total_hands'] * 100 if m['total_hands'] > 0 else None

        print(f"{display:<35} {m['matches']:>3} | {fmt_pct(fold_flop):>8} {fmt_pct(fold_turn):>8} {fmt_pct(fold_river):>8} | "
              f"{fmt_pct(three_bet):>5} {m['allins']:>6} {fmt_f(ai_per_hand):>6}%")

    # === TABLE 3: River Deep Dive (most important) ===
    print()
    print(f"{'RIVER DEEP DIVE (showdowns only)':=^120}")
    print(f"{'Opponent':<35} {'#M':>3} | {'BetEq':>6} {'ChkEq':>6} {'Bluffs':>6} {'Value':>6} {'Bluf%':>5} | {'SD_Bet':>6} {'SD_Chk':>6} | {'Traps':>5} {'MedBet':>6}")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l + " "
                break

        display = (label + name)[:35]

        bet_eq = np.mean(m['river_bet_equities']) * 100 if m['river_bet_equities'] else None
        chk_eq = np.mean(m['river_check_equities']) * 100 if m['river_check_equities'] else None
        bluff_pct = safe_pct(m['river_bluffs'], m['river_bets_total'])
        sd_bet = np.mean(m['showdown_wr_when_betting']) * 100 if m['showdown_wr_when_betting'] else None
        sd_chk = np.mean(m['showdown_wr_when_checking']) * 100 if m['showdown_wr_when_checking'] else None

        river_sizes = m['bet_sizes_pct'].get('River', [])
        med_bet = np.median(river_sizes) if river_sizes else None

        traps = m['check_traps'].get('River', 0)

        print(f"{display:<35} {m['matches']:>3} | {fmt_f(bet_eq):>5}% {fmt_f(chk_eq):>5}% {m['river_bluffs']:>6} {m['river_value_bets']:>6} {fmt_pct(bluff_pct):>5} | "
              f"{fmt_f(sd_bet):>5}% {fmt_f(sd_chk):>5}% | {traps:>5} {fmt_f(med_bet):>5}%")

    # === TABLE 4: Bet Sizing ===
    print()
    print(f"{'BET SIZING (median % of pot)':=^120}")
    print(f"{'Opponent':<35} {'#M':>3} | {'PF_med':>7} {'PF_n':>5} | {'Flop_med':>8} {'Flop_n':>6} | {'Turn_med':>8} {'Turn_n':>6} | {'Rivr_med':>8} {'Rivr_n':>6}")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l + " "
                break
        display = (label + name)[:35]

        def med_size(street):
            sizes = m['bet_sizes_pct'].get(street, [])
            return (np.median(sizes), len(sizes)) if sizes else (None, 0)

        pf_m, pf_n = med_size('Pre-Flop')
        f_m, f_n = med_size('Flop')
        t_m, t_n = med_size('Turn')
        r_m, r_n = med_size('River')

        print(f"{display:<35} {m['matches']:>3} | {fmt_f(pf_m):>6}% {pf_n:>5} | {fmt_f(f_m):>7}% {f_n:>6} | {fmt_f(t_m):>7}% {t_n:>6} | {fmt_f(r_m):>7}% {r_n:>6}")

    # === TABLE 5: Exploit Recommendations ===
    print()
    print(f"{'EXPLOIT RECOMMENDATIONS':=^120}")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        if m['total_hands'] < 100:
            continue

        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l
                break

        exploits = []

        # VPIP analysis
        vpip = safe_pct(m['vpip_hands'], m['total_hands'])
        if vpip is not None:
            if vpip > 70: exploits.append(f"VERY LOOSE (VPIP {vpip:.0f}%) — value bet wider, don't bluff")
            elif vpip < 35: exploits.append(f"VERY TIGHT (VPIP {vpip:.0f}%) — steal blinds, respect their bets")

        # River bluffing
        bluff_pct = safe_pct(m['river_bluffs'], m['river_bets_total'])
        sd_bet = np.mean(m['showdown_wr_when_betting']) * 100 if m['showdown_wr_when_betting'] else None
        if sd_bet is not None and len(m['showdown_wr_when_betting']) >= 5:
            if sd_bet > 80: exploits.append(f"NEVER BLUFFS river (SD WR {sd_bet:.0f}% when betting) — fold to their river bets more")
            elif sd_bet < 45: exploits.append(f"BLUFFS TOO MUCH river (SD WR {sd_bet:.0f}% when betting) — call down light")

        # Fold rates
        fold_river = safe_pct(m['folded_to_bet']['River'], m['faced_bet']['River'])
        if fold_river is not None and m['faced_bet']['River'] >= 10:
            if fold_river > 60: exploits.append(f"OVERFOLDS river ({fold_river:.0f}%) — bluff more on river")
            elif fold_river < 25: exploits.append(f"CALLING STATION river ({fold_river:.0f}% fold) — never bluff river, value bet thin")

        fold_flop = safe_pct(m['folded_to_bet']['Flop'], m['faced_bet']['Flop'])
        if fold_flop is not None and m['faced_bet']['Flop'] >= 10:
            if fold_flop > 55: exploits.append(f"OVERFOLDS flop ({fold_flop:.0f}%) — c-bet aggressively")

        # Aggression
        total_bets = sum(m['street_bets'][s] for s in ['Flop', 'Turn', 'River'])
        total_checks = sum(m['street_checks'][s] for s in ['Flop', 'Turn', 'River'])
        if total_bets + total_checks > 50:
            agg = total_bets / (total_bets + total_checks) * 100
            if agg > 50: exploits.append(f"HYPER-AGGRESSIVE ({agg:.0f}% bet freq) — let them bluff into you")
            elif agg < 15: exploits.append(f"PASSIVE ({agg:.0f}% bet freq) — bet for thin value, they won't raise")

        # Check-trapping
        traps = m['check_traps'].get('River', 0)
        river_checks = len(m['checks_with_equity'].get('River', []))
        if river_checks >= 10:
            trap_rate = traps / river_checks * 100
            if trap_rate > 40: exploits.append(f"CHECK-TRAPS often ({trap_rate:.0f}% checks are >70% eq) — don't stab when they check")

        if exploits:
            display = f"{label} {name}" if label else name
            print(f"\n  {display}:")
            for e in exploits:
                print(f"    → {e}")

    # === SUMMARY: Who to worry about ===
    print()
    print(f"{'GTO DEVIATION SUMMARY':=^120}")
    print(f"{'Opponent':<35} {'Hands':>6} | {'Deviation':>10} | Key Traits")
    print("-" * 120)

    for name in sorted_names:
        m = merged[name]
        if m['total_hands'] < 50:
            continue

        label = ""
        for full_name, l in TOP_BOTS.items():
            if full_name in name or name in full_name:
                label = l + " "
                break
        display = (label + name)[:35]

        # Calculate GTO deviation score
        deviations = []
        traits = []

        vpip = safe_pct(m['vpip_hands'], m['total_hands'])
        if vpip is not None:
            dev = abs(vpip - 55)  # GTO ~55% VPIP for this variant
            deviations.append(dev)
            if vpip > 70: traits.append("loose")
            elif vpip < 40: traits.append("tight")

        river_bet = safe_pct(m['street_bets']['River'], m['street_actions']['River'])
        if river_bet is not None and m['street_actions']['River'] >= 20:
            dev = abs(river_bet - 20)  # GTO ~20% river bet
            deviations.append(dev)
            if river_bet > 35: traits.append("river-aggro")
            elif river_bet < 10: traits.append("river-passive")

        bluff_pct = safe_pct(m['river_bluffs'], m['river_bets_total'])
        if bluff_pct is not None and m['river_bets_total'] >= 5:
            dev = abs(bluff_pct - 30)  # GTO ~30% bluffs in betting range
            deviations.append(dev * 0.5)  # weight bluff deviation less
            if bluff_pct < 10: traits.append("NO bluffs")
            elif bluff_pct > 50: traits.append("over-bluffs")

        fold_river = safe_pct(m['folded_to_bet']['River'], m['faced_bet']['River'])
        if fold_river is not None and m['faced_bet']['River'] >= 10:
            dev = abs(fold_river - 45)
            deviations.append(dev)
            if fold_river > 60: traits.append("overfolds")
            elif fold_river < 25: traits.append("station")

        avg_dev = np.mean(deviations) if deviations else 0

        if avg_dev < 8: category = "  ~GTO  "
        elif avg_dev < 15: category = " SLIGHT "
        elif avg_dev < 25: category = "MODERATE"
        else: category = "  LARGE "

        trait_str = ", ".join(traits) if traits else "balanced"

        print(f"{display:<35} {m['total_hands']:>6} | {category:>10} | {trait_str}")


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.expanduser('~/Downloads/match_*.txt')))
    files += sorted(glob.glob(os.path.expanduser('~/Downloads/cmu_poker_csvs/match_*.csv')))

    if not files:
        print("No match files found")
        sys.exit(1)

    # Deduplicate by match ID
    seen = set()
    unique = []
    for f in files:
        mid = os.path.basename(f).split('_')[1].split('.')[0]
        if mid not in seen:
            seen.add(mid)
            unique.append(f)
    files = unique

    print(f"Scanning {len(files)} match files...")

    profiles = []
    errors = 0
    for f in files:
        try:
            p = profile_opponent(f)
            if p:
                profiles.append(p)
        except Exception as e:
            errors += 1

    if errors:
        print(f"({errors} files had errors)")

    merged = merge_profiles(profiles)

    # Sort by leaderboard position for display
    all_names = list(merged.keys())
    print_profiles(merged, all_names)
