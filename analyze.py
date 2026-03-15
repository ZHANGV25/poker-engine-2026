#!/usr/bin/env python3
"""
Match analysis dashboard for Stockfish 🐟

Reads all match log files from ~/Downloads/ and produces:
  - Overall win/loss record
  - Per-opponent breakdown
  - Leak detection (fold patterns, big losses)
  - Action distribution analysis
  - Bankroll trajectory analysis

Usage:
    python analyze.py                    # analyze all matches
    python analyze.py match_908.txt      # analyze specific match
    python analyze.py --opponent geoz    # filter by opponent
"""

import csv
import sys
import os
import glob
from collections import defaultdict
from pathlib import Path


def parse_match(filepath):
    """Parse a match log file. Returns match dict or None on error."""
    try:
        with open(filepath) as f:
            comment = f.readline().strip()
            if not comment.startswith('#'):
                return None
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    # Parse team names
    parts = comment.replace('# ', '').split(', ')
    if len(parts) != 2:
        return None
    team0 = parts[0].replace('Team 0: ', '')
    team1 = parts[1].replace('Team 1: ', '')

    # Determine which team is us
    us_names = ['Stockfish', 'Goofy Goobers']
    if any(n in team1 for n in us_names):
        us, them = 1, 0
        opp_name = team0
    elif any(n in team0 for n in us_names):
        us, them = 0, 1
        opp_name = team1
    else:
        return None

    # Group by hand
    hands = {}
    for row in rows:
        h = int(row['hand_number'])
        if h not in hands:
            hands[h] = []
        hands[h].append(row)

    sorted_hands = sorted(hands.keys())
    n_hands = len(sorted_hands)
    last_row = hands[sorted_hands[-1]][-1]
    final = int(last_row[f'team_{us}_bankroll'])

    # Per-hand P/L
    hand_pls = []
    for i in range(1, len(sorted_hands)):
        h = sorted_hands[i]
        prev_h = sorted_hands[i - 1]
        bank = int(hands[h][0][f'team_{us}_bankroll'])
        prev = int(hands[prev_h][0][f'team_{us}_bankroll'])
        hand_pls.append((sorted_hands[i - 1], bank - prev))

    # Action counts
    our_acts = defaultdict(int)
    opp_acts = defaultdict(int)
    our_folds_street = defaultdict(int)
    opp_raise_sizes = defaultdict(list)

    for row in rows:
        a = row['action_type']
        if a == 'DISCARD':
            continue
        if int(row['active_team']) == us:
            our_acts[a] += 1
            if a == 'FOLD':
                our_folds_street[row['street']] += 1
        else:
            opp_acts[a] += 1
            if a == 'RAISE':
                opp_raise_sizes[row['street']].append(int(row['action_amount']))

    # Trajectory
    trajectory = []
    for i in range(0, n_hands, max(1, n_hands // 10)):
        h = sorted_hands[min(i, n_hands - 1)]
        bank = int(hands[h][0][f'team_{us}_bankroll'])
        trajectory.append((h, bank))

    # Big losses
    big_losses = [(h, pl) for h, pl in hand_pls if pl <= -50]
    big_losses.sort(key=lambda x: x[1])

    result = 'WIN' if final > 0 else ('TIE' if final == 0 else 'LOSS')
    filename = os.path.basename(filepath)

    return {
        'filename': filename,
        'opponent': opp_name,
        'us': us,
        'them': them,
        'final': final,
        'result': result,
        'n_hands': n_hands,
        'our_acts': dict(our_acts),
        'opp_acts': dict(opp_acts),
        'our_folds_street': dict(our_folds_street),
        'opp_raise_sizes': dict(opp_raise_sizes),
        'trajectory': trajectory,
        'big_losses': big_losses,
        'hand_pls': hand_pls,
        'hands': hands,
        'sorted_hands': sorted_hands,
    }


def print_summary(matches):
    """Print overall summary across all matches."""
    wins = sum(1 for m in matches if m['result'] == 'WIN')
    losses = sum(1 for m in matches if m['result'] == 'LOSS')
    ties = sum(1 for m in matches if m['result'] == 'TIE')
    total_chips = sum(m['final'] for m in matches)

    print("=" * 70)
    print(f"  STOCKFISH MATCH DASHBOARD — {len(matches)} matches analyzed")
    print("=" * 70)
    print(f"\n  Record: {wins}W / {losses}L / {ties}T")
    print(f"  Win rate: {wins/len(matches)*100:.1f}%")
    print(f"  Total chips: {total_chips:+d}")
    print(f"  Avg chips/match: {total_chips/len(matches):+.0f}")


def print_per_opponent(matches):
    """Print per-opponent breakdown."""
    by_opp = defaultdict(list)
    for m in matches:
        by_opp[m['opponent']].append(m)

    print(f"\n{'=' * 70}")
    print(f"  PER-OPPONENT BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"\n  {'Opponent':<25s} {'W':>3s} {'L':>3s} {'T':>3s} {'Chips':>8s} {'Avg':>7s}")
    print(f"  {'—' * 25} {'—' * 3} {'—' * 3} {'—' * 3} {'—' * 8} {'—' * 7}")

    for opp in sorted(by_opp.keys(), key=lambda o: -sum(m['final'] for m in by_opp[o])):
        ms = by_opp[opp]
        w = sum(1 for m in ms if m['result'] == 'WIN')
        l = sum(1 for m in ms if m['result'] == 'LOSS')
        t = sum(1 for m in ms if m['result'] == 'TIE')
        chips = sum(m['final'] for m in ms)
        avg = chips / len(ms)
        marker = ' ⚠' if l > 0 else ''
        print(f"  {opp:<25s} {w:>3d} {l:>3d} {t:>3d} {chips:>+8d} {avg:>+7.0f}{marker}")


def print_leak_detection(matches):
    """Identify common patterns in losses."""
    losses = [m for m in matches if m['result'] == 'LOSS']
    if not losses:
        print(f"\n  No losses to analyze!")
        return

    print(f"\n{'=' * 70}")
    print(f"  LEAK DETECTION ({len(losses)} losses)")
    print(f"{'=' * 70}")

    for m in losses:
        our_total = sum(m['our_acts'].values())
        fold_pct = m['our_acts'].get('FOLD', 0) / our_total * 100 if our_total else 0
        raise_pct = m['our_acts'].get('RAISE', 0) / our_total * 100 if our_total else 0

        opp_total = sum(m['opp_acts'].values())
        opp_raise_pct = m['opp_acts'].get('RAISE', 0) / opp_total * 100 if opp_total else 0

        n_big = len(m['big_losses'])
        big_total = sum(pl for _, pl in m['big_losses'])

        # Detect pattern
        patterns = []
        if fold_pct > 25:
            patterns.append(f"HIGH FOLD ({fold_pct:.0f}%)")
        if raise_pct > 40:
            patterns.append(f"HIGH RAISE ({raise_pct:.0f}%)")

        # Check for overbet pattern (opp raises near all-in)
        flop_sizes = m['opp_raise_sizes'].get('Flop', [])
        if flop_sizes and sum(s > 80 for s in flop_sizes) > 10:
            patterns.append("OPP OVERBETS FLOP")
        if flop_sizes and sum(s < 5 for s in flop_sizes) > 50:
            patterns.append("OPP SMALL FLOP BETS")

        # Check our flop fold rate
        flop_folds = m['our_folds_street'].get('Flop', 0)
        if flop_folds > 100:
            patterns.append(f"FLOP FOLD x{flop_folds}")

        pattern_str = ', '.join(patterns) if patterns else 'No clear pattern'

        print(f"\n  {m['filename']} vs {m['opponent']}: {m['final']:+d}")
        print(f"    Big losses: {n_big} ({big_total:+d})")
        print(f"    Us:  F:{fold_pct:.0f}% R:{raise_pct:.0f}% | Opp: R:{opp_raise_pct:.0f}%")
        print(f"    Folds by street: {dict(m['our_folds_street'])}")
        print(f"    Pattern: {pattern_str}")


def print_match_detail(m):
    """Print detailed analysis of a single match."""
    print(f"\n{'=' * 70}")
    print(f"  {m['filename']} — vs {m['opponent']} — {m['result']} ({m['final']:+d})")
    print(f"{'=' * 70}")

    # Trajectory
    print(f"\n  Trajectory:")
    for h, bank in m['trajectory']:
        bar = '█' * max(0, bank // 100) if bank > 0 else '░' * max(0, -bank // 100)
        print(f"    H{h:4d}: {bank:+6d} {bar}")
    print(f"    FINAL: {m['final']:+6d}")

    # Actions
    our_total = sum(m['our_acts'].values())
    opp_total = sum(m['opp_acts'].values())
    print(f"\n  Actions (us / opp):")
    for act in ['FOLD', 'CHECK', 'CALL', 'RAISE']:
        our_pct = m['our_acts'].get(act, 0) / our_total * 100 if our_total else 0
        opp_pct = m['opp_acts'].get(act, 0) / opp_total * 100 if opp_total else 0
        print(f"    {act:6s}: {our_pct:5.1f}% / {opp_pct:5.1f}%")

    print(f"\n  Our folds: {dict(m['our_folds_street'])}")

    # Opp raise sizing
    print(f"\n  Opp raise sizes:")
    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        sizes = m['opp_raise_sizes'].get(street, [])
        if sizes:
            avg = sum(sizes) / len(sizes)
            mx = max(sizes)
            print(f"    {street:10s}: n={len(sizes):3d} avg={avg:5.1f} max={mx}")

    # Big losses
    if m['big_losses']:
        total_big = sum(pl for _, pl in m['big_losses'])
        print(f"\n  Big losses (50+): {len(m['big_losses'])} totaling {total_big:+d}")
        for h, pl in m['big_losses'][:5]:
            rows_h = m['hands'][h]
            cards = rows_h[-1][f"team_{m['us']}_cards"]
            board = rows_h[-1]['board_cards']
            seq = []
            for row in rows_h:
                a = row['action_type']
                if a == 'DISCARD':
                    continue
                who = "US" if int(row['active_team']) == m['us'] else "OP"
                seq.append(f"{row['street'][0]}:{who}-{a[:2]}{row['action_amount']}")
            print(f"    H{h} ({pl:+d}): {cards}")
            print(f"      {' '.join(seq[-8:])}")


def main():
    # Find all match files
    downloads = Path.home() / 'Downloads'
    files = sorted(downloads.glob('match_*.txt'), key=lambda f: f.stat().st_mtime)

    if not files:
        print("No match files found in ~/Downloads/")
        return

    # Parse args
    filter_file = None
    filter_opp = None
    for arg in sys.argv[1:]:
        if arg.startswith('--opponent'):
            filter_opp = sys.argv[sys.argv.index(arg) + 1] if arg == '--opponent' else arg.split('=')[1]
        elif arg.endswith('.txt'):
            filter_file = arg
        elif not arg.startswith('-'):
            filter_opp = arg

    # Parse all matches
    matches = []
    for f in files:
        if filter_file and filter_file not in str(f):
            continue
        m = parse_match(str(f))
        if m:
            if filter_opp and filter_opp.lower() not in m['opponent'].lower():
                continue
            matches.append(m)

    if not matches:
        print("No matching Stockfish games found.")
        return

    if len(matches) == 1 or filter_file:
        # Single match detail
        for m in matches:
            print_match_detail(m)
    else:
        # Multi-match overview
        print_summary(matches)
        print_per_opponent(matches)
        print_leak_detection(matches)

        # Also show latest 5 matches
        print(f"\n{'=' * 70}")
        print(f"  LATEST MATCHES")
        print(f"{'=' * 70}")
        for m in matches[-5:]:
            big = len(m['big_losses'])
            print(f"  {m['filename']:20s} vs {m['opponent']:20s} {m['result']:4s} {m['final']:+6d} "
                  f"({m['n_hands']}h, {big} big losses)")


if __name__ == '__main__':
    main()
