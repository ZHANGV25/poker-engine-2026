# Session Log — March 20, 2026

## Starting State
- Rank ~#10, ELO ~1370, 55% WR
- Bot: Stockfish v24
- Architecture: flop/turn blueprints + equity thresholds on river
- Core problem: losing chips on river (-2.4/hand), overcalling

## What We Tried & Learned

### 1. Bug #1: pot_before = my_bet * 2 (FIXED)
- Wrong for re-raises — overstates pot, understates bet fraction
- Fix: `pot_before = my_bet + self._street_start_bet`
- Impact: minor, only affects re-raise situations

### 2. Full-tree range solver for river acting first
- Compact tree (no re-raise): 55% bet frequency — way too aggressive
- Full tree (re-raise allowed): varies by pot size
  - pot=(10,10): ~0% bet (re-raise threat too large vs pot)
  - pot=(30,30): 16-22% bet (correct GTO range)
- One-hand solver can't bet acting first (converges to 0%) because it solves for ONE hand and opponent adapts
- Range solver is needed for acting first (hero has a RANGE, opponent can't exploit specific hand)

### 3. DCFR (Noam Brown's parameters)
- alpha=1.5, beta=0, gamma=2
- Converges faster than CFR+ for average strategy
- beta=0 means negative regrets halved each iteration (like CFR+)
- gamma=2 means early iterations heavily discounted in strategy sum
- At 300-500 iterations, well-converged for our game size

### 4. CRITICAL BUG: Card blocking in range solver
- Blocked hand pairs (sharing cards) contributed -pot at showdown terminals
- This biased strong hands' EV downward by ~0.24
- Caused the solver to CHECK THE NUTS instead of value betting
- Fix: not_blocked mask on all terminal values
- Before fix: strongest hands bet 0-2%, medium hands bet 61%
- After fix: strongest hands bet 48%, medium checks — correct GTO polarization

### 5. River double-narrowing (REMOVED)
- Old code: narrow range by P(bet|hand) THEN pass narrowed range to solver
- But solver ALSO models opponent's betting behavior internally
- Double-counting "opponent bet" information → over-narrowed toward value → fold too much
- Fix: removed pre-narrowing, let solver handle it

### 6. Bayesian P(call|hand) at street transitions
- Flop→Turn: uses blueprint opp_strategies P(call|hand) + P(raise|hand)
- Better than MDF because it knows draws call, dominated hands fold
- Blueprint is well-converged (74% purity, call_std=0.335)
- Turn→River: compute_opp_call_probs (runtime CFR at facing-bet node)
- Dead cards parameter bug found and fixed

### 7. Floors on narrowing
- Original: 0.10 on Bayesian paths, 0.05 on heuristic paths
- Reduced all to 0.05, then 0.02, then 0 (no floors), then back to 0.02
- 0.02 rationale: 9% phantom weight (vs 20% at 0.05), compounds to 0.0008% after 3 streets
- No floors: v29 tested, 2/6 wins — not clearly better or worse than 0.05
- Floors are insurance against narrowing errors, not the main issue

### 8. Vectorized equity computation
- Replaced nested Python loops with numpy broadcasting
- River: 1.9ms (was ~5ms) — 2.6x speedup
- Turn: 3.1ms (was 215ms) — 69x speedup
- Flop: 28ms (was ~800ms) — 29x speedup

### 9. Flop/Turn facing-bet runtime solver (REVERTED)
- Replaced blueprint with range solver for facing bets on flop/turn
- REGRESSED: Turn went from +6.4/hand to 0.0/hand
- Cause: runtime solver uses single-street equity (check-to-showdown)
- Blueprint uses backward induction with continuation values from future streets
- The multi-street advantage massively outweighs the narrowing advantage
- Reverted back to blueprint for flop/turn

### 10. Audit findings (from fresh-eyes agent)
- River raise tracking variables not updated → fixed
- compute_opp_bet_probs min_raise too large → fixed to min_raise=2
- _reraise_narrow_range is the only fully heuristic main-path method
- Several dead code methods identified

## Key Insights

1. **Multi-street > Narrowing for early streets**: Blueprint's backward induction (accounting for future betting) is more valuable than knowing the opponent's exact range. The range narrows naturally through future betting.

2. **Narrowing > Multi-street for river**: On the river there are no future streets, so knowing the opponent's range is the only edge. Runtime solving against narrowed range is correct here.

3. **Card blocking is critical**: Without proper not_blocked mask, the solver's EV estimates are systematically biased. This caused the nuts to check.

4. **One-hand solver can't act first**: It converges to never betting because the opponent adapts to the known single hand. Only range solver (hero has a range) produces balanced value+bluff strategies.

5. **Double-narrowing hurts**: Pre-narrowing the range AND having the solver model betting internally double-counts information.

6. **Floors matter but aren't the main issue**: 2% floors are tight enough to not dilute the range significantly while preventing catastrophic zeroing.

## Current Architecture (v30)
- Preflop: 200-bucket GTO strategy
- Discard: exact equity evaluation + Bayesian inference
- Flop: backward-induction blueprint (all decisions)
- Turn: backward-induction blueprint (all decisions)
- River: full-tree DCFR range solver (both acting first and facing bets)
- Narrowing: Bayesian from blueprints (flop/turn bets and checks), Bayesian P(call|hand) at transitions, no river pre-narrowing
- Floors: 2% on all narrowing paths

## The Unsolved Problem
The flop/turn blueprint doesn't account for the narrowed opponent range.
The runtime solver accounts for narrowing but doesn't model future streets.
Depth-limited re-solving (using blueprint continuation values) was attempted
(commit 359b21d) but reverted due to noisy continuation values.
This is what Pluribus solves with depth-limited subgame re-solving.

## Performance
- v24 (pre-session): 55% WR, PF +1.4, Flop +1.8, Turn +6.4, River -2.4
- v29 (runtime solver everywhere): 33% WR, River improved +1.2 but Turn collapsed -6.4
- v30 (blueprint flop/turn, solver river): expected ~PF +1.4, Flop +1.8, Turn +6.4, River -1.2
