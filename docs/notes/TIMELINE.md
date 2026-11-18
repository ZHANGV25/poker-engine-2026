# Session Timeline — March 20, 2026

## Starting State
- **Rank: ~#10, ELO ~1370, 55% WR** (v24)
- Architecture: flop/turn blueprints + equity thresholds on river
- Core problem: losing -2.4/hand on river from overcalling, no bluffs

---

## Changes Made This Session

### Bug Fixes (definitely good)
- **Card blocking in range solver** — blocked hand pairs contributed -pot at showdown, biasing ALL solver decisions. Nuts was checking instead of betting.
- **pot_before fix** — wrong for re-raises (3 locations)
- **Tracking variables** — river raise not updating _opp_bet_at_raise etc.
- **min_raise=2** — old value could make tree empty

### River Solver (definitely good)
- Full tree (4 sizes, 2 raises) replacing compact (2 sizes, 1 raise)
- DCFR (Noam Brown's parameters) replacing CFR+
- Card blocking fix (critical)
- Vectorized equity (69x turn speedup, 29x flop)
- River used for BOTH acting first and facing bets
- **Result: +1.2/hand over equity thresholds in per-street analysis**

### River Polarized Bet-Narrowing (tested positive)
- When facing a river bet, narrow opponent range: top 30% (value) + bottom 10% (bluffs), downweight middle
- NOT double-counting — solver tree starts after bet decision
- **Tested: +552 chips over no-narrow across 80 match decisions**

### Turn Depth-Limited Solver for Facing Bets (uncertain)
- Uses CTS equity as continuation values at showdown terminals
- Narrowed opponent range from discard inference + flop betting
- Gadget safety check (fixed: uses showdown terminal, not fold terminal)
- Part of v32 (66.7% WR) and v35 architecture

### Narrowing Improvements (uncertain)
- Floors reduced from 0.05/0.10 to 0.02 (9% phantom weight vs 20%)
- Bayesian P(call|hand) at flop→turn from blueprint opp_strategies
- compute_opp_call_probs at turn→river (runtime CFR)
- River double-narrowing removed

### Dead Code Cleanup
- Removed 484 lines of unused methods (7 methods never called)

---

## Version History

| Version | Upload | Matches | WR | Key Change |
|---------|--------|---------|-----|-----------|
| v24 | pre-session | many | 55% | Baseline: blueprints + equity thresholds |
| v27 | ~1:00 PM | 2 | - | Card blocking + DCFR + full tree river |
| v29 | ~2:00 PM | 6 | 33% | Runtime solver on ALL streets + no floors |
| v31 | ~2:19 PM | 13 | 38.5% | Transitional |
| v32 | ~3:25 PM | 15 | **66.7%** | Blueprint flop/turn AF + DL turn FB + river solver |
| v33 | ~3:52 PM | 5 | 40% | Added turn acting-first solver + broken gadget |
| v34 | ~4:38 PM | 9 | 33% | Fixed gadget, still has turn AF solver |
| v35 | ~5:30 PM | running | ? | = v32 arch + river narrowing + fixes |

*AF = acting first, FB = facing bet, DL = depth-limited*

---

## What We Know For Sure

1. **Card blocking fix is critical.** Without it, solver is systematically wrong.
2. **River range solver beats equity thresholds.** Proper value+bluff balance.
3. **River polarized narrowing tests positive.** +552 chips on real match data.
4. **Adding turn acting-first solver caused regression.** v32→v33 dropped from 66.7% to 40%. But this could be opponent-dependent.
5. **Local tests don't predict competition.** +9.5/hand vs ProbAgent, 33% in competition.

## What We Don't Know

1. **Is v32's 66.7% real or variance?** 15 matches is small sample. It was 76.9% at 13 matches, now 66.7% at 15. Trending down.
2. **Does the turn facing-bet solver actually help?** It's in v32 (66.7%) but v32's WR could be from other fixes (card blocking, river solver) not the turn solver.
3. **Do the floor changes (0.02 vs 0.05) matter?** Tested no-floor (v29) = bad. 0.05 was the original. 0.02 is current. No isolated comparison.
4. **How much is opponent-dependent?** Matches fluctuate wildly. Some opponents we crush, some crush us, regardless of version.

## Current State
- **v35 running** — v32 architecture + river narrowing + bug fixes
- **Rank: ~#45** (dropped from #10 due to v33/v34 regression)
- **~18 hours to deadline**
- Need v35 results to determine if we're recovering
