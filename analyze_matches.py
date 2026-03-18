#!/usr/bin/env python3
"""
Complete match analysis tool. Analyzes every aspect of our bot's play
to find exactly where we're losing chips and why.
"""
import csv, ast, sys, os, itertools
from collections import defaultdict
import numpy as np

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine
engine = ExactEquityEngine()

RANKS = "23456789A"
SUITS = "dhs"
def card_to_int(s): return SUITS.index(s[1]) * 9 + RANKS.index(s[0])

def parse_match(filepath):
    with open(filepath) as fh:
        header = fh.readline().strip()
        team0 = header.split('Team 0: ')[1].split(',')[0].strip()
        team1 = header.split('Team 1: ')[1].strip()
        we_are = 0 if 'Stockfish' in team0 else 1
        opp = 1 - we_are
        reader = csv.reader(fh)
        next(reader)
        rows = list(reader)

    opp_name = team1 if we_are == 0 else team0
    final_b = int(rows[-1][3 + we_are])

    hand_start_b = {}
    hand_rows = defaultdict(list)
    for row in rows:
        h = int(row[0])
        if h not in hand_start_b:
            hand_start_b[h] = int(row[3 + we_are])
        hand_rows[h].append(row)

    # Find active portion (before auto-fold)
    sorted_hands = sorted(hand_start_b.keys())
    prev_b = 0
    last_active = 0
    for i in range(len(sorted_hands)):
        b = hand_start_b[sorted_hands[i]]
        if i > 0 and abs(b - prev_b) > 2:
            last_active = sorted_hands[i]
        prev_b = b

    return {
        'filepath': filepath,
        'opp_name': opp_name,
        'we_are': we_are,
        'opp': opp,
        'final_b': final_b,
        'rows': rows,
        'hand_start_b': hand_start_b,
        'hand_rows': hand_rows,
        'sorted_hands': sorted_hands,
        'last_active': last_active,
    }

def analyze_match(m):
    """Full analysis of a single match."""
    results = {}
    we_are = m['we_are']
    opp = m['opp']
    sorted_hands = m['sorted_hands']
    hand_start_b = m['hand_start_b']
    hand_rows = m['hand_rows']
    last_active = m['last_active']

    results['opp'] = m['opp_name']
    results['final'] = m['final_b']
    results['active_hands'] = last_active
    results['result'] = 'WIN' if m['final_b'] > 0 else 'LOSS'

    # Street P&L (active only)
    street_pl = defaultdict(float)
    street_count = defaultdict(int)

    # Per-action P&L
    action_pl = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'chips': 0}))

    # Pot size distribution
    pot_pl = defaultdict(lambda: {'count': 0, 'chips': 0, 'wins': 0, 'losses': 0})

    # River call analysis
    river_call_w = river_call_l = 0

    # Check-check analysis
    cc_stats = defaultdict(lambda: {'count': 0, 'chips': 0})

    # Our bet frequency vs opponent
    our_bets_by_street = defaultdict(int)
    opp_bets_by_street = defaultdict(int)
    our_actions_by_street = defaultdict(int)
    opp_actions_by_street = defaultdict(int)

    # Equity at showdown
    showdown_equities = []

    for i in range(len(sorted_hands) - 1):
        h = sorted_hands[i]
        if h > last_active:
            break
        delta = hand_start_b[sorted_hands[i+1]] - hand_start_b[h]
        actions = hand_rows[h]
        last = actions[-1]
        end_street = last[1]

        # Street P&L
        street_pl[end_street] += delta
        street_count[end_street] += 1

        # Pot size
        final_pot = int(last[14]) + int(last[15])
        if final_pot <= 4: bucket = 'tiny'
        elif final_pot <= 10: bucket = 'small'
        elif final_pot <= 30: bucket = 'medium'
        elif final_pot <= 60: bucket = 'large'
        else: bucket = 'huge'
        pot_pl[bucket]['count'] += 1
        pot_pl[bucket]['chips'] += delta
        if delta > 0: pot_pl[bucket]['wins'] += 1
        elif delta < 0: pot_pl[bucket]['losses'] += 1

        # Per-street action counts
        for row in actions:
            if row[5] == 'DISCARD': continue
            street = row[1]
            if int(row[2]) == we_are:
                our_actions_by_street[street] += 1
                if row[5] == 'RAISE': our_bets_by_street[street] += 1
            else:
                opp_actions_by_street[street] += 1
                if row[5] == 'RAISE': opp_bets_by_street[street] += 1

        # River-specific analysis
        river_actions = [r for r in actions if r[1] == 'River']
        if river_actions:
            our_a = [r[5] for r in river_actions if int(r[2]) == we_are]
            opp_a = [r[5] for r in river_actions if int(r[2]) == opp]

            # Categorize
            if 'FOLD' in our_a: cat = 'we_fold'
            elif 'FOLD' in opp_a: cat = 'opp_fold'
            elif 'CALL' in our_a and 'RAISE' in opp_a:
                cat = 'we_call'
                if delta > 0: river_call_w += 1
                else: river_call_l += 1
            elif 'CALL' in opp_a and 'RAISE' in our_a:
                cat = 'they_call_W' if delta > 0 else 'they_call_L'
            elif all(a == 'CHECK' for a in our_a) and all(a == 'CHECK' for a in opp_a):
                cat = 'cc'
            else:
                cat = 'other'

            action_pl['River'][cat]['count'] += 1
            action_pl['River'][cat]['chips'] += delta

            # Equity at showdown for river hands
            try:
                our_c = [card_to_int(c) for c in ast.literal_eval(last[9] if we_are == 0 else last[10])]
                opp_c = [card_to_int(c) for c in ast.literal_eval(last[10] if we_are == 0 else last[9])]
                board = [card_to_int(c) for c in ast.literal_eval(last[11])]
                if len(board) >= 5 and len(our_c) >= 2:
                    our_rank = engine.lookup_seven(our_c + board)
                    opp_rank = engine.lookup_seven(opp_c + board)
                    we_win = 1 if our_rank < opp_rank else 0.5 if our_rank == opp_rank else 0
                    showdown_equities.append((cat, we_win, delta))
            except:
                pass

    results['street_pl'] = dict(street_pl)
    results['street_count'] = dict(street_count)
    results['pot_pl'] = dict(pot_pl)
    results['river_call_wr'] = river_call_w / (river_call_w + river_call_l) if river_call_w + river_call_l > 0 else None
    results['river_calls'] = river_call_w + river_call_l
    results['action_pl'] = {k: dict(v) for k, v in action_pl.items()}
    results['showdown_equities'] = showdown_equities

    # Bet frequencies
    results['our_bet_freq'] = {}
    results['opp_bet_freq'] = {}
    for street in ['Flop', 'Turn', 'River']:
        if our_actions_by_street[street] > 0:
            results['our_bet_freq'][street] = our_bets_by_street[street] / our_actions_by_street[street]
        if opp_actions_by_street[street] > 0:
            results['opp_bet_freq'][street] = opp_bets_by_street[street] / opp_actions_by_street[street]

    # Showdown win rate by category
    results['showdown_wr'] = {}
    cat_wins = defaultdict(list)
    for cat, win, delta in showdown_equities:
        cat_wins[cat].append(win)
    for cat, wins in cat_wins.items():
        results['showdown_wr'][cat] = np.mean(wins)

    return results


def print_analysis(matches):
    """Print comprehensive analysis across all matches."""

    wins = sum(1 for m in matches if m['result'] == 'WIN')
    losses = len(matches) - wins

    print(f"{'='*80}")
    print(f"COMPLETE ANALYSIS: {len(matches)} matches ({wins}W {losses}L = {wins/len(matches)*100:.0f}% WR)")
    print(f"{'='*80}\n")

    # Per-match summary
    print(f"{'Opponent':<25} {'Result':>6} {'Final':>6} {'Active':>6} | {'Flop':>6} {'Turn':>6} {'River':>7} | {'R.Call':>6} {'Pot':>5}")
    print("-" * 95)
    for m in sorted(matches, key=lambda x: x['final']):
        spl = m['street_pl']
        rc = f"{m['river_call_wr']:.0%}" if m['river_call_wr'] is not None else "N/A"
        pot_huge = m['pot_pl'].get('huge', {}).get('chips', 0)
        print(f"{m['opp'][:25]:<25} {m['result']:>6} {m['final']:>+6} {m['active_hands']:>5}h | "
              f"{spl.get('Flop',0):>+6.0f} {spl.get('Turn',0):>+6.0f} {spl.get('River',0):>+7.0f} | "
              f"{rc:>6} {pot_huge:>+5.0f}")

    # Aggregate street P&L
    print(f"\n{'='*80}")
    print("AGGREGATE STREET P&L (active portions only)")
    print(f"{'='*80}")
    agg_street = defaultdict(float)
    agg_count = defaultdict(int)
    for m in matches:
        for s, v in m['street_pl'].items():
            agg_street[s] += v
            agg_count[s] += m['street_count'].get(s, 0)
    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        d = agg_street.get(street, 0)
        c = agg_count.get(street, 0)
        avg = d / c if c > 0 else 0
        bar = '█' * int(abs(avg) * 5) if avg != 0 else ''
        sign = '+' if avg > 0 else '-' if avg < 0 else ' '
        print(f"  {street:10s}: {d:>+7.0f} chips / {c:4d} hands = {avg:>+5.1f}/hand {sign}{bar}")

    # Aggregate pot size P&L
    print(f"\n{'='*80}")
    print("AGGREGATE POT SIZE P&L")
    print(f"{'='*80}")
    agg_pot = defaultdict(lambda: {'count': 0, 'chips': 0, 'wins': 0, 'losses': 0})
    for m in matches:
        for bucket, stats in m['pot_pl'].items():
            for k in stats:
                agg_pot[bucket][k] += stats[k]
    for bucket in ['tiny', 'small', 'medium', 'large', 'huge']:
        d = agg_pot[bucket]
        wr = d['wins'] / (d['wins'] + d['losses']) * 100 if d['wins'] + d['losses'] > 0 else 0
        print(f"  {bucket:8s}: {d['count']:5d} hands {d['chips']:>+7d} chips WR:{wr:.0f}%")

    # River action breakdown
    print(f"\n{'='*80}")
    print("AGGREGATE RIVER ACTION BREAKDOWN")
    print(f"{'='*80}")
    agg_river = defaultdict(lambda: {'count': 0, 'chips': 0})
    for m in matches:
        for cat, stats in m.get('action_pl', {}).get('River', {}).items():
            agg_river[cat]['count'] += stats['count']
            agg_river[cat]['chips'] += stats['chips']
    for cat in sorted(agg_river.keys(), key=lambda k: agg_river[k]['chips']):
        d = agg_river[cat]
        avg = d['chips'] / d['count'] if d['count'] > 0 else 0
        print(f"  {cat:15s}: {d['count']:5d} hands {d['chips']:>+7d} chips ({avg:>+5.1f}/hand)")

    # River call win rate trend
    print(f"\n{'='*80}")
    print("RIVER CALL WIN RATE BY MATCH")
    print(f"{'='*80}")
    total_rcw = total_rcl = 0
    for m in sorted(matches, key=lambda x: x['opp']):
        if m['river_calls'] > 0:
            total_rcw += int(m['river_call_wr'] * m['river_calls']) if m['river_call_wr'] else 0
            total_rcl += m['river_calls'] - (int(m['river_call_wr'] * m['river_calls']) if m['river_call_wr'] else 0)
            print(f"  vs {m['opp'][:25]:<25}: {m['river_call_wr']:.0%} ({m['river_calls']} calls)" if m['river_call_wr'] is not None else f"  vs {m['opp'][:25]:<25}: N/A")
    total_rc = total_rcw + total_rcl
    if total_rc > 0:
        print(f"  {'TOTAL':<25}: {total_rcw/total_rc:.0%} ({total_rc} calls)")

    # Bet frequency comparison
    print(f"\n{'='*80}")
    print("BET FREQUENCY: US vs OPPONENTS")
    print(f"{'='*80}")
    our_freq = defaultdict(list)
    opp_freq = defaultdict(list)
    for m in matches:
        for s, f in m.get('our_bet_freq', {}).items():
            our_freq[s].append(f)
        for s, f in m.get('opp_bet_freq', {}).items():
            opp_freq[s].append(f)
    for street in ['Flop', 'Turn', 'River']:
        ours = np.mean(our_freq[street]) * 100 if our_freq[street] else 0
        theirs = np.mean(opp_freq[street]) * 100 if opp_freq[street] else 0
        print(f"  {street:6s}: Us {ours:.0f}% | Them {theirs:.0f}%")

    # Showdown win rate by action
    print(f"\n{'='*80}")
    print("SHOWDOWN WIN RATE BY SITUATION")
    print(f"{'='*80}")
    all_sd = defaultdict(list)
    for m in matches:
        for cat, wr in m.get('showdown_wr', {}).items():
            all_sd[cat].append(wr)
    for cat in sorted(all_sd.keys()):
        avg_wr = np.mean(all_sd[cat]) * 100
        print(f"  {cat:15s}: {avg_wr:.0f}% showdown WR (across {len(all_sd[cat])} matches)")

    # Biggest leak identification
    print(f"\n{'='*80}")
    print("BIGGEST LEAKS (by total chips lost)")
    print(f"{'='*80}")
    leaks = []
    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        if agg_street.get(street, 0) < -100:
            leaks.append((street, agg_street[street], f"Losing {-agg_street[street]:.0f} chips on {street}"))
    for cat, stats in sorted(agg_river.items(), key=lambda x: x[1]['chips']):
        if stats['chips'] < -500:
            leaks.append((f"River:{cat}", stats['chips'], f"{stats['count']} hands, {stats['chips']/stats['count']:+.1f}/hand"))
    for bucket in ['huge', 'large']:
        if agg_pot[bucket]['chips'] < -500:
            wr = agg_pot[bucket]['wins'] / max(agg_pot[bucket]['wins'] + agg_pot[bucket]['losses'], 1) * 100
            leaks.append((f"Pot:{bucket}", agg_pot[bucket]['chips'], f"WR:{wr:.0f}%, {agg_pot[bucket]['count']} hands"))

    for name, chips, desc in sorted(leaks, key=lambda x: x[1]):
        print(f"  {name:20s}: {chips:>+7.0f} chips — {desc}")


if __name__ == '__main__':
    import glob

    # Find recent match files (after match 40838 = ~4:30 PM)
    files = sorted(glob.glob(os.path.expanduser('~/Downloads/match_*.txt')))
    recent = [f for f in files if int(os.path.basename(f).split('_')[1].split('.')[0]) >= 40838]

    if not recent:
        print("No recent match files found")
        sys.exit(1)

    print(f"Analyzing {len(recent)} matches...\n")

    matches = []
    for f in recent:
        try:
            m = parse_match(f)
            analysis = analyze_match(m)
            matches.append(analysis)
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    print_analysis(matches)
