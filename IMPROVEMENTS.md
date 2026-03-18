# Turn & River Improvement Plan

## Current State

**Turn (street 2):** Falls back to `_equity_threshold_play` — fixed thresholds with no river lookahead. No concept of implied odds, draw value, or river bet/fold dynamics.

**River (street 3):** RangeSolver when facing a bet (~500ms, 200 iters). Equity thresholds when acting first. Acting-first strategy is exploitable — only bets above fixed equity cutoffs, so opponent knows check range is capped.

## Proposed Improvements (priority order)

### 1. Turn with Approximate River Lookahead (highest impact)

Replace `_equity_threshold_play` on the turn with a one-hand CFR solver that uses approximate river continuation values instead of raw equity at showdown terminals.

**How it works:**
1. Build turn game tree (cached, ~0ms)
2. For each showdown terminal node in the turn tree, compute continuation value:
   - For each possible river card (~14 remaining):
     - Compute exact river equity for hero vs each opponent hand
     - Estimate river EV with simplified betting model (no CFR):
       - Equity > 0.75: hero bets, ~50% fold / ~50% call. EV ≈ value bet gain
       - Equity 0.40–0.75: check-call dynamics. EV ≈ equity × pot
       - Equity < 0.40: check-fold. EV ≈ 0 (lose nothing more)
     - Average across river cards → continuation value per opponent hand
3. Run one-hand CFR on turn tree with continuation values (100 iters, ~50-100ms)
4. Return action from converged strategy

**Why this matters:** Captures that weak turn hands improving on river have positive continuation value, and strong turn hands getting counterfeited have negative continuation value. Current equity thresholds miss this entirely.

**Time cost:** ~50–150ms per turn decision. Well within budget.

### 2. One-Hand Solver for River Acting First (low effort, medium impact)

Wire in `SubgameSolver.solve_and_act()` for river when acting first, instead of equity thresholds.

```python
if street == 3:
    if time_left > 50:
        result = self.solver.solve_and_act(...)
        if result: return result
    return self._equity_threshold_play(...)
```

**Why:** Equity thresholds produce exploitable betting ranges — only betting strong hands means opponent can bluff every check. The one-hand solver constructs balanced value/bluff ranges. On the river with deterministic equity, one-hand solver is fast (~50-100ms with ~40 opp hands, 200 iters).

### 3. Replace RangeSolver with One-Hand Solver for River Facing Bets (saves time budget)

RangeSolver (facing bets) costs ~500ms because it solves for ALL possible hero hands. The one-hand solver solves only for our actual hand — ~5x faster. On the river with an already-narrowed opponent range, the theoretical range-balancing benefit of the full solver is minimal.

**Freed time:** ~400ms per river facing-bet decision, redirectable to turn lookahead.

### 4. Compact Turn Abstractions (if EC2 time available)

C(27,4) = 17,550 turn boards. Abstract by board texture (flush draws, straight draws, pairing, high cards) into ~100 buckets. Store small strategy table:

100 buckets × 90 hands × 10 nodes × 7 actions × 1 byte = ~630KB

Trivially fits in submission. Requires running solves on EC2 before deadline.

### 5. Fix Range Narrowing Bugs First

All improvements depend on accurate opponent range. The 4 bugs in BUGS.md should be fixed before implementing any of the above — bad ranges propagate through everything.
