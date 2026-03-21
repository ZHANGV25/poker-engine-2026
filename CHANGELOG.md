# Changelog

## v32 (March 20, 2026) — Depth-Limited Turn Solver + River Range Solver

### What's New
- **Depth-limited turn solver** for facing bets: solves against narrowed opponent range
  with river continuation values at leaf nodes. Captures implied odds (draws improve
  and bet river) that the blueprint misses. Includes gadget for safe re-solving
  (never worse than blueprint). +3.5 chips/decision vs blueprint across 341 real match hands.
- **River range solver** for ALL decisions (acting first + facing bets): full tree
  (4 bet sizes, 2 raises), DCFR, card blocking fix. Replaces equity thresholds
  that had 0 bluffs (88% showdown WR on opponent fold).
- **Vectorized equity computation**: 69x speedup on turn, 29x on flop via numpy broadcasting.
- **2% narrowing floors** (down from 5-10%): tighter ranges, 9% phantom weight vs 20%.

### Architecture
```
Preflop:  200-bucket GTO strategy (precomputed)
Discard:  Exact equity + Bayesian inference
Flop:     Blueprint (backward induction, all decisions)
Turn:     Blueprint (acting first) / Depth-limited solver (facing bets)
River:    DCFR range solver (all decisions)
```

### Performance (local tests, 100 hands)
- vs CallingStation: +14.8/hand
- vs ProbAgent: +9.5/hand (was +0.9/hand in v30)
- vs AllInAgent: +6.3/hand

### Key Bug Fixes
- **Card blocking in range solver**: blocked hand pairs contributed -pot at showdown
  terminals, biasing strong hands' EV by -0.24. Caused the nuts to CHECK instead of bet.
- **River double-narrowing removed**: pre-narrowing + solver modeling = double-counting
  "opponent bet" information. Over-narrowed toward value, causing excess folding.
- **pot_before fix**: `my_bet * 2` wrong for re-raises → `my_bet + street_start_bet`.
- **Tracking variables on river raise**: _opp_bet_at_raise etc. now updated when
  range solver returns a raise.
- **min_raise=2** for compute_opp_bet_probs (was opp_bet - my_bet, could make tree empty).

---

## What We Tried and What Worked

### ✅ Worked
1. **Full-tree range solver for river** (+1.2/hand improvement over equity thresholds)
   - Full tree with re-raise threat gives 17-27% bet frequency (balanced value+bluffs)
   - Card blocking fix was critical — without it, nuts checked
   - DCFR converges faster than CFR+

2. **Depth-limited turn solver** (+3.5 chips/decision over blueprint)
   - River game values as continuation values (captures implied odds)
   - Gadget for safe re-solving
   - 75 river iterations + 200 turn iterations = ~2.5s ARM

3. **Vectorized equity** (69x speedup on turn)
   - numpy broadcasting replaces nested Python loops
   - Turn equity: 215ms → 3ms
   - Enables runtime solving within 5s per-action limit

4. **Bayesian P(call|hand) at flop→turn transition**
   - Uses blueprint opp_strategies (74% purity, well-converged)
   - Better than MDF: knows draws call, dominated hands fold

5. **2% floors** (vs 5-10%)
   - 9% phantom weight vs 20% — tighter ranges
   - Compounds to 0.0008% after 3 streets — phantom hands naturally die

### ❌ Didn't Work
1. **Runtime solver for flop/turn ALL decisions** (v29: turn +6.4 → 0.0/hand)
   - Single-street equity doesn't capture future betting dynamics
   - Flush draws with 35% equity look -EV but are +EV with river betting
   - Blueprint's backward induction is essential for acting-first decisions

2. **No floors** (v29: 2/6 wins)
   - Narrowing aggressively zeros out hands
   - If blueprint P(bet|hand) is wrong, we zero a hand opponent actually has
   - Solver bets confidently into the "impossible" nuts → crushed

3. **One-hand solver for river acting first** (0% bet frequency)
   - Converges to never betting because opponent adapts to known single hand
   - Range solver required (hero has a range, can't be exploited per-hand)

4. **Compact tree range solver** (55% bet frequency)
   - No re-raise threat → everyone bets → P(bet|hand) ≈ 1 for all hands
   - Full tree (4 sizes, 2 raises) needed for realistic strategies

5. **Previous depth-limited attempt** (commit 359b21d, reverted)
   - One-hand solver (not range), 75 iterations, no card blocking fix
   - ~65 chip error bound from bad terminal values
   - Current version fixes all three issues

### 🔬 Learned
- Multi-street awareness > range narrowing for early streets (flop/turn acting first)
- Range narrowing > multi-street for last street (river)
- Depth-limited solving gets both (turn facing bets)
- Card blocking is critical — biased ALL range solver decisions before fix
- Double-narrowing hurts — pre-narrowing + solver modeling = over-counting
- The competition's #1 bot (frogtron) is a pure value bettor, not GTO
- Close margins (±50) are from auto-fold lead protection, not coinflips
