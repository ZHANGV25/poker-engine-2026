#!/usr/bin/env python3
"""Quantify chip flow categories across all matches to estimate fix impact."""
import csv, os, sys
from collections import defaultdict

def analyze_all():
    matches = []
    dl_dir = os.path.expanduser('~/Downloads')

    for f in sorted(os.listdir(dl_dir)):
        if not f.startswith('match_') or not f.endswith('.txt'):
            continue
        path = os.path.join(dl_dir, f)

        with open(path) as fh:
            header = fh.readline().strip()
            if 'Stockfish' not in header:
                continue
            reader = csv.DictReader(fh)
            rows = list(reader)

        us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
        opp_team = 1 - us

        hands = defaultdict(list)
        for r in rows:
            hands[int(r['hand_number'])].append(r)
        hand_nums = sorted(hands.keys())

        # Detect crash
        crash = None
        for hnum in hand_nums:
            oa = [r for r in hands[hnum] if int(r['active_team']) == opp_team and r['action_type'] != 'DISCARD']
            if len(oa) == 1 and oa[0]['action_type'] == 'FOLD' and oa[0]['street'] == 'Pre-Flop':
                consec = 0
                for h2 in range(hnum, min(hnum + 10, max(hand_nums) + 1)):
                    if h2 not in hands:
                        continue
                    o2 = [r for r in hands[h2] if int(r['active_team']) == opp_team and r['action_type'] != 'DISCARD']
                    if len(o2) == 1 and o2[0]['action_type'] == 'FOLD':
                        consec += 1
                    else:
                        break
                if consec >= 10:
                    crash = hnum
                    break

        active = [h for h in hand_nums if crash is None or h < crash]
        if len(active) < 30:
            continue

        # Categorize chip flow
        steals = our_folds = sd_small = sd_big = 0
        late_fold_chips = 0
        preflop_allin_loss = 0

        for i, hnum in enumerate(active):
            hr = hands[hnum]
            last = hr[-1]
            nxt = hand_nums.index(hnum) + 1
            if nxt >= len(hand_nums):
                continue
            start = int(hr[0]['team_%d_bankroll' % us])
            end = int(hands[hand_nums[nxt]][0]['team_%d_bankroll' % us])
            delta = end - start
            pot = int(last['team_0_bet']) + int(last['team_1_bet'])
            our_bet = int(last['team_%d_bet' % us])

            # Check if went all-in preflop
            flop_rows = [r for r in hr if r['street'] == 'Flop']
            flop_pot = int(flop_rows[0]['team_0_bet']) + int(flop_rows[0]['team_1_bet']) if flop_rows else 0

            if last['action_type'] == 'FOLD':
                who = int(last['active_team'])
                if who == opp_team:
                    steals += delta
                else:
                    our_folds += delta
                    if our_bet > 20:
                        late_fold_chips += delta
            elif last['action_type'] in ('CHECK', 'CALL'):
                if flop_pot >= 150 and delta < 0:
                    preflop_allin_loss += delta
                elif pot > 50:
                    sd_big += delta
                else:
                    sd_small += delta

        n_active = len(active)
        final = int(hands[hand_nums[-1]][-1]['team_%d_bankroll' % us])

        # Get opponent name
        parts = header.split('Team 0:')[1]
        t0_name = parts.split(',')[0].strip()
        t1_name = header.split('Team 1:')[1].strip()
        opp_name = t1_name if us == 0 else t0_name

        match_num = f.replace('match_', '').replace('.txt', '')

        matches.append({
            'match': match_num, 'opp': opp_name[:20], 'n': n_active,
            'result': final, 'crashed': crash is not None,
            'steals': steals, 'our_folds': our_folds, 'sd_small': sd_small,
            'sd_big': sd_big, 'late_folds': late_fold_chips,
            'pf_allin_loss': preflop_allin_loss,
        })

    # Print
    fmt = "%6s %20s %5s %7s %7s %8s %8s %8s %7s %5s"
    print(fmt % ('Match', 'Opponent', 'Hands', 'Result', 'Steals', 'OurFold', 'LateFld', 'PF_AIL', 'SD_big', 'Crash'))
    print('-' * 100)

    totals = defaultdict(int)
    n = 0
    for m in sorted(matches, key=lambda x: -int(x['match'])):
        print(fmt % (
            m['match'], m['opp'], m['n'],
            '%+d' % m['result'], '%+d' % m['steals'],
            '%+d' % m['our_folds'], '%+d' % m['late_folds'],
            '%+d' % m['pf_allin_loss'],
            '%+d' % m['sd_big'],
            'Y' if m['crashed'] else 'N'
        ))
        for k in ['our_folds', 'late_folds', 'pf_allin_loss', 'sd_big', 'steals', 'result']:
            totals[k] += m[k]
        n += 1

    print('-' * 100)
    print(fmt % (
        'TOTAL', '%d matches' % n, '',
        '%+d' % totals['result'], '%+d' % totals['steals'],
        '%+d' % totals['our_folds'], '%+d' % totals['late_folds'],
        '%+d' % totals['pf_allin_loss'],
        '%+d' % totals['sd_big'], ''
    ))
    print(fmt % (
        'AVG', '', '',
        '%+d' % (totals['result']//n), '%+d' % (totals['steals']//n),
        '%+d' % (totals['our_folds']//n), '%+d' % (totals['late_folds']//n),
        '%+d' % (totals['pf_allin_loss']//n),
        '%+d' % (totals['sd_big']//n), ''
    ))

    print("\n" + "=" * 60)
    print("ESTIMATED IMPACT OF FIXES")
    print("=" * 60)

    avg_late = totals['late_folds'] // n
    avg_pf_ail = totals['pf_allin_loss'] // n
    avg_sd_big = totals['sd_big'] // n
    avg_folds = totals['our_folds'] // n

    print(f"\nPer-match averages:")
    print(f"  Late folds (called 20+ then folded): {avg_late:+d} chips/match")
    print(f"  Preflop all-in losses:               {avg_pf_ail:+d} chips/match")
    print(f"  Big showdown losses:                 {avg_sd_big:+d} chips/match")
    print(f"  Total from our folds:                {avg_folds:+d} chips/match")

    print(f"\nFix estimates:")
    print(f"  1. Bug fixes (already deployed):")
    print(f"     - Preflop fold guard:   saves ~30-50% of PF all-in losses = {int(avg_pf_ail * -0.4):+d}")
    print(f"     - Equity guard:         saves ~50% of late folds = {int(avg_late * -0.5):+d}")
    print(f"     - hero_first + BB wts:  improves solver quality, ~10% of SD_big = {int(avg_sd_big * -0.1):+d}")
    print(f"     TOTAL BUG FIX IMPACT:   ~{int(avg_pf_ail * -0.4) + int(avg_late * -0.5) + int(avg_sd_big * -0.1):+d} chips/match")

    print(f"\n  2. Better blueprint convergence (v2, 5000i):")
    print(f"     - Fewer unconverged fallbacks = better pot control")
    print(f"     - ~10-15% of SD_big improvement = {int(avg_sd_big * -0.125):+d} chips/match")

    print(f"\n  3. Feature-based bucketing (draws vs made hands):")
    print(f"     - Draws check/call instead of bet/call/fold pattern")
    print(f"     - Saves ~30-50% of late fold chips = {int(avg_late * -0.4):+d} chips/match")
    print(f"     - Better pot control with made hands = ~10% SD_big = {int(avg_sd_big * -0.1):+d}")
    print(f"     TOTAL BUCKETING IMPACT:  ~{int(avg_late * -0.4) + int(avg_sd_big * -0.1):+d} chips/match")

    total_impact = int(avg_pf_ail * -0.4) + int(avg_late * -0.9) + int(avg_sd_big * -0.325)
    print(f"\n  ALL COMBINED:              ~{total_impact:+d} chips/match improvement")

analyze_all()
