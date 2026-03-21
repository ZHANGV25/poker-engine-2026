#!/usr/bin/env python3
"""Extract top bot's strategy patterns from a match log.

For each postflop decision the top bot makes, records:
- Board state, their hand, pot size, bet state
- What they did (fold/check/call/raise and amount)
- Their hand strength (equity)

This creates a benchmark to compare our multi-street blueprint against.
"""
import csv, sys, os, itertools
from collections import defaultdict

sys.path.insert(0, 'submission')
from equity import ExactEquityEngine

def analyze_top_bot(path):
    engine = ExactEquityEngine()

    with open(path) as f:
        header = f.readline().strip()
        reader = csv.DictReader(f)
        rows = list(reader)

    # Figure out which team is which
    us = 0 if 'Stockfish' in header.split('Team 0:')[1].split(',')[0] else 1
    opp = 1 - us
    opp_name = header.split(f'Team {opp}:')[1].strip().split(',')[0].strip() if opp == 1 else header.split('Team 0:')[1].split(',')[0].strip()

    print(f"Analyzing {opp_name}'s strategy")
    print()

    hands = defaultdict(list)
    for r in rows:
        hands[int(r['hand_number'])].append(r)

    RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'A': 8}
    SUIT_MAP = {'d': 0, 'h': 1, 's': 2}

    def parse_card_list(s):
        s = s.strip("[]")
        cards = []
        for c in s.split(','):
            c = c.strip().strip("'\"")
            if c and len(c) == 2:
                cards.append(RANK_MAP[c[0]] * 3 + SUIT_MAP[c[1]])
        return cards

    # Collect opponent's postflop actions with context
    actions_by_context = defaultdict(list)  # (street, facing_bet, equity_bucket) -> [action, ...]

    for hnum in sorted(hands.keys()):
        hr = hands[hnum]
        for r in hr:
            if int(r['active_team']) != opp:
                continue
            if r['action_type'] == 'DISCARD':
                continue
            if r['street'] == 'Pre-Flop':
                continue

            street = r['street']
            opp_cards = parse_card_list(r[f'team_{opp}_cards'])
            board = parse_card_list(r['board_cards'])

            if len(opp_cards) < 2 or len(board) < 3:
                continue

            my_bet = int(r[f'team_{opp}_bet'])
            other_bet = int(r[f'team_{us}_bet'])
            facing_bet = other_bet > my_bet

            # Compute opponent's equity
            try:
                if len(board) >= 5:
                    # Approximate: just use hand rank
                    rank = engine.lookup_seven(opp_cards[:2] + board[:5])
                    equity = 0.5  # can't easily compute without enumerating
                elif len(board) >= 3:
                    equity = engine.compute_equity(opp_cards[:2], board[:3], [])
                else:
                    equity = 0.5
            except:
                equity = 0.5

            eq_bucket = int(equity * 10)  # 0-10

            action = r['action_type']
            amount = int(r['action_amount']) if r['action_type'] == 'RAISE' else 0
            pot = my_bet + other_bet

            actions_by_context[(street, facing_bet, eq_bucket)].append({
                'action': action,
                'amount': amount,
                'pot': pot,
                'equity': equity,
                'hand': opp_cards[:2],
                'board': board,
            })

    # Print summary
    print(f"{'Street':>8s} {'Facing':>7s} {'Equity':>8s} {'N':>4s} {'Fold%':>6s} {'Check%':>7s} {'Call%':>6s} {'Raise%':>7s} {'AvgRaise':>9s}")
    print("-" * 75)

    for street in ['Flop', 'Turn', 'River']:
        for facing in [False, True]:
            for eq_b in range(11):
                key = (street, facing, eq_b)
                if key not in actions_by_context:
                    continue
                acts = actions_by_context[key]
                n = len(acts)
                if n < 3:
                    continue

                folds = sum(1 for a in acts if a['action'] == 'FOLD')
                checks = sum(1 for a in acts if a['action'] == 'CHECK')
                calls = sum(1 for a in acts if a['action'] == 'CALL')
                raises = sum(1 for a in acts if a['action'] == 'RAISE')
                avg_raise = sum(a['amount'] for a in acts if a['action'] == 'RAISE') / max(raises, 1)

                eq_range = f"{eq_b*10}-{(eq_b+1)*10}%"
                facing_str = "bet" if facing else "none"

                print(f"{street:>8s} {facing_str:>7s} {eq_range:>8s} {n:>4d} {folds/n:>6.0%} {checks/n:>7.0%} {calls/n:>6.0%} {raises/n:>7.0%} {avg_raise:>9.1f}")

    # Key insight: how does the top bot size bets by equity?
    print()
    print("=== BET SIZING BY EQUITY (when they raise) ===")
    for street in ['Flop', 'Turn', 'River']:
        raises_by_eq = defaultdict(list)
        for key, acts in actions_by_context.items():
            if key[0] != street:
                continue
            for a in acts:
                if a['action'] == 'RAISE':
                    pot = max(a['pot'], 1)
                    bet_to_pot = a['amount'] / pot
                    raises_by_eq[int(a['equity'] * 5)].append(bet_to_pot)

        if raises_by_eq:
            print(f"\n  {street}:")
            for eq_b in sorted(raises_by_eq.keys()):
                bets = raises_by_eq[eq_b]
                eq_range = f"{eq_b*20}-{(eq_b+1)*20}%"
                avg = sum(bets) / len(bets)
                print(f"    Equity {eq_range}: avg bet {avg:.1%} pot ({len(bets)} raises)")

    # Key insight: what do they do with draws vs made hands?
    print()
    print("=== CHECK vs BET RATE BY STREET (not facing bet) ===")
    for street in ['Flop', 'Turn', 'River']:
        total = check = bet = 0
        for key, acts in actions_by_context.items():
            if key[0] != street or key[1]:  # only when not facing bet
                continue
            for a in acts:
                total += 1
                if a['action'] == 'CHECK':
                    check += 1
                elif a['action'] == 'RAISE':
                    bet += 1
        if total:
            print(f"  {street}: check {check/total:.0%}, bet {bet/total:.0%} (n={total})")

if __name__ == "__main__":
    analyze_top_bot(sys.argv[1] if len(sys.argv) > 1 else max(
        [f for f in os.listdir(os.path.expanduser('~/Downloads')) if f.startswith('match_')],
        key=lambda f: os.path.getmtime(os.path.join(os.path.expanduser('~/Downloads'), f))
    ))
