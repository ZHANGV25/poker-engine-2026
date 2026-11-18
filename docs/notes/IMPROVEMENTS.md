# Turn & River Improvement Plan

## Current State

**Turn (street 2):** Falls back to `_equity_threshold_play` — fixed thresholds with no river lookahead. No concept of implied odds, draw value, or river bet/fold dynamics.

**River (street 3):** RangeSolver when facing a bet (~4-10s on ARM, 200 iters). Equity thresholds when acting first. Acting-first strategy is exploitable — only bets above fixed equity cutoffs, so opponent knows check range is capped.

## Critical Finding: RangeSolver Convergence

The RangeSolver at 200 iterations is **not well-converged** for this game size.

### The Math

CFR+ exploitability bound: `Exploitability ≤ Δ · √|I| / √T`

**RangeSolver (91 hero × 40 opp hands):**
- |I| ≈ 91×8 + 40×8 = ~1,048 information sets
- At 200 iters: bound ≈ 460 chips (exceeds max pot of 200)
- Needs ~2,000-5,000 iterations for practical convergence (~1-3 chips error)
- At ~4-10s per solve on ARM, 2000 iters would take 40-100s per decision — impossible

**One-hand solver (1 hero × 40 opp hands):**
- |I| ≈ 1×8 + 40×8 = ~328 information sets
- Hero strategy is only ~56 parameters (8 nodes × 7 actions)
- Converges much faster in practice — 200 iters is decent, 400-500 is solid
- Cost: ~50-100ms on ARM per solve

**Conclusion:** Drop RangeSolver entirely. Use one-hand solver everywhere on river with higher iteration count (400-500). Better convergence, less compute, practically equivalent quality since the RangeSolver wasn't converged enough to be truly range-balanced anyway.

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

### 2. One-Hand Solver for ALL River Decisions (low effort, high impact)

Replace both RangeSolver (facing bets) and equity thresholds (acting first) with the one-hand solver at 400-500 iterations.

**River acting first:** One-hand solver constructs balanced value/bluff betting ranges. Equity thresholds are exploitable — they only bet strong hands, so the check range is capped and attackable.

**River facing bets:** One-hand solver at 400 iters (~100ms) replaces RangeSolver at 200 iters (~4-10s). Better convergence AND 40-100x faster. The range-balancing benefit of RangeSolver is theoretical — at 200 iterations it wasn't converged enough to be balanced anyway.

**Time budget:** ~500 river decisions × 100ms = 50s total. Frees up ~250s previously spent on RangeSolver.

### 3. Compact Turn Abstractions (if EC2 time available)

C(27,4) = 17,550 turn boards. Abstract by board texture (flush draws, straight draws, pairing, high cards) into ~100 buckets. Store small strategy table:

100 buckets × 90 hands × 10 nodes × 7 actions × 1 byte = ~630KB

Trivially fits in submission. Requires running solves on EC2 before deadline.

### 4. Fix Range Narrowing Bugs First

All improvements depend on accurate opponent range. The 4 bugs in BUGS.md should be fixed before implementing any of the above — bad ranges propagate through everything.

## Time Budget Estimate (after improvements)

| Component | Calls/match | ms/call | Total |
|---|---|---|---|
| Preflop lookup | 1000 | 1 | 1s |
| Discard eval | 1000 | 20 | 20s |
| Flop blueprint | 500 | 5 | 2.5s |
| Turn w/ lookahead | 400 | 150 | 60s |
| River one-hand solver | 500 | 100 | 50s |
| **Total** | | | **~134s of 1500s (9%)** |
