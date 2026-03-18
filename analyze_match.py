#!/usr/bin/env python3
"""Quick match analyzer. Usage: python analyze_match.py ~/Downloads/match_*.txt"""
import sys, os, glob

def analyze(fpath):
    with open(fpath) as f:
        lines = f.readlines()
    header = lines[0].strip()
    team0 = header.split('Team 0: ')[1].split(',')[0]
    team1 = header.split('Team 1: ')[1].strip()
    we_are = 0 if 'Stockfish' in team0 else 1 if 'Stockfish' in team1 else -1
    if we_are == -1: return None
    opponent = team1 if we_are == 0 else team0
    bankrolls, hands_data, acts_count = {}, {}, {'us': {}, 'them': {}}
    for line in lines[2:]:
        p = line.strip().split(',')
        if len(p) < 14: continue
        h, street, team, action = int(p[0]), p[1], int(p[2]), p[5]
        bankrolls[h] = int(p[3])
        if h not in hands_data: hands_data[h] = []
        hands_data[h].append({'street': street, 'team': team, 'action': action, 'amount': int(p[6])})
        who = 'us' if team == we_are else 'them'
        acts_count[who][(street, action)] = acts_count[who].get((street, action), 0) + 1
    final_b0 = int(lines[-1].strip().split(',')[3])
    our_result = final_b0 if we_are == 0 else -final_b0
    net = {'Pre-Flop': 0, 'Flop': 0, 'Turn': 0, 'River': 0}
    sd_w, sd_l, sd_ev, rr_loss, rc_loss = 0, 0, 0, 0, 0
    for h in range(max(bankrolls.keys())):
        if h not in bankrolls or h+1 not in bankrolls: continue
        delta = (bankrolls[h+1] - bankrolls[h]) * (1 if we_are == 0 else -1)
        acts = hands_data.get(h, [])
        if not acts: continue
        last = acts[-1]
        if last['street'] in net: net[last['street']] += delta
        if last['action'] in ('CHECK','CALL') and last['street'] == 'River':
            if delta > 0: sd_w += 1
            elif delta < 0: sd_l += 1
            sd_ev += delta
        if last['street'] == 'River' and delta < -10:
            r_acts = [a for a in acts if a['street']=='River' and a['team']==we_are]
            if any(a['action']=='RAISE' for a in r_acts): rr_loss += delta
            elif any(a['action']=='CALL' for a in r_acts): rc_loss += delta
    fr = acts_count['us'].get(('Flop','RAISE'),0)
    fc = acts_count['us'].get(('Flop','CHECK'),0)
    return {'id': os.path.basename(fpath).replace('match_','').replace('.txt',''),
            'opp': opponent, 'result': our_result, 'net': net,
            'sd': (sd_w, sd_l, sd_ev), 'rr': rr_loss, 'rc': rc_loss,
            'fr%': fr/max(fr+fc,1), 'riv_r': acts_count['us'].get(('River','RAISE'),0),
            'pf_fold': acts_count['us'].get(('Pre-Flop','FOLD'),0)}

files = sorted(glob.glob(os.path.expanduser(sys.argv[1]))) if len(sys.argv) > 1 else []
results = [r for f in files if (r := analyze(f))]
if not results: print('No matches found'); sys.exit(1)

print(f'{"Match":>8s} {"Opponent":>18s} {"Result":>8s} {"PF":>7s} {"Flop":>7s} {"Turn":>7s} {"River":>7s} {"SD":>7s} {"F.R%":>5s} {"R.R":>4s}')
print('-'*95)
T = {'Pre-Flop':0,'Flop':0,'Turn':0,'River':0}; tr=0; tsw=0; tsl=0; tse=0
for r in results:
    n=r['net']; s=r['sd']
    print(f'{r["id"]:>8s} {r["opp"]:>18s} {r["result"]:>+8d} {n["Pre-Flop"]:>+7d} {n["Flop"]:>+7d} {n["Turn"]:>+7d} {n["River"]:>+7d} {s[0]}/{s[1]:<4d} {r["fr%"]:>4.0%} {r["riv_r"]:>4d}')
    for s2 in T: T[s2]+=n[s2]
    tr+=r['result']; tsw+=s[0]; tsl+=s[1]; tse+=s[2]
print('-'*95)
print(f'{"TOTAL":>8s} {"("+str(len(results))+" matches)":>18s} {tr:>+8d} {T["Pre-Flop"]:>+7d} {T["Flop"]:>+7d} {T["Turn"]:>+7d} {T["River"]:>+7d} {tsw}/{tsl:<4d}')
print(f'\nShowdown: {tsw}W-{tsl}L ({tse:+d}) | Win rate: {tsw/max(tsw+tsl,1):.1%}')
