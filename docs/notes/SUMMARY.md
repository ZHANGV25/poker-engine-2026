# Session Summary: What We Tried and What Happened

## The Core Problem
We lose chips on river (-3 to -10/hand) while winning on flop (+1.5) and turn (+8.6). Specifically: we overcall river bets. Call WR is 26-45% across opponents. The 5-card selection mechanic means both players have premium hands at river, so even after narrowing the opponent's range to value-heavy, our equity often stays above pot odds (0.25).

## Architecture
- Flop: 200-iter backward induction blueprint (4-bit, 51MB LZMA)
- Turn: per-board precomputed strategies (2925 files, 575MB, lazy-loaded)
- River acting first: equity thresholds (bet if equity > 72%/82%/92%)
- River facing bet: compact tree range solver (1500-3000 CFR+ iterations)
- Range narrowing: Bayesian P(bet|hand) on flop/turn from blueprints, polarized heuristic on river

## What We Tried

### 1. Bayesian Bluff Ratio (prior_weight=5)
- **Theory**: Adjust bluff_ratio in polarized narrowing from showdown data
- **Result**: Marginal. Takes 10-15 showdowns to calibrate. In 250-hand matches (before auto-fold), not enough data.

### 2. Bet-Frequency Prior
- **Theory**: Use opponent's river bet frequency to estimate bluff rate before showdowns (available hand 40-50)
- **Result**: 67% accurate across field. Wrong for opponents like Joker (bets rarely but includes bluffs). Untested in production.

### 3. Check Narrowing in observe()
- **Theory**: When opponent checks behind, downweight strong hands
- **BUG FOUND**: Code treated opponent CALLS as CHECKS, removing strong hands when they should be kept. Caused regression from ~50% to 46% WR.

### 4. Full Tree Runtime Solver
- **Theory**: Full tree (4 sizes, 2 raises) gives better strategies than compact (2 sizes, 1 raise)
- **Result**: 99.4% same FOLD decisions as compact tree. Tree is NOT the problem.

### 5. DCFR (Discounted CFR)
- **Theory**: Faster convergence, used by Pluribus
- **Result**: Same decisions as vanilla CFR+ at our iteration counts. Not worth implementing.

### 6. O(N) Showdown Evaluation
- **Theory**: Prefix sum trick for faster equity computation (from Noam Brown's solver)
- **Result**: Only 2% speedup. Bottleneck is CFR iterations, not equity computation.

### 7. Precomputed River P(bet|hand) — Compact Tree
- **Theory**: Replace polarized heuristic with board-specific equilibrium P(bet|hand)
- **Result**: Compact tree gives 91% bet frequency (everyone bets, no re-raise threat). P(bet|hand) ≈ 0.9 for all hands. USELESS for narrowing — 0% improvement over polarized.
- **EC2**: Launched 10× c5.9xlarge twice, killed both times (first with 2000 iters, second with 500 iters).

### 8. Precomputed River P(bet|hand) — Full Tree
- **Theory**: Full tree gives realistic 16-34% bet frequency
- **Result**: Tested offline. With UNIFORM ranges: OVER-narrows (worse than polarized). With NARROWED ranges: same as compact solve-narrow (80% accuracy in test).
- **EC2**: Not launched (full tree too slow for precompute).

### 9. Runtime Solve-Narrow (Compact Tree, Narrowed Range)
- **Theory**: Solve from opponent's perspective at runtime using their actual narrowed range. Same approach Pluribus uses.
- **Offline Test**: 62.5% accuracy vs 44.5% for polarized across 1130 hands (+18%). Better for 48/61 opponents.
- **Production**: Still overcalls. Against AWA (86% bet eq), equity drops from 0.60-0.85 to 0.38-0.82 after narrowing, but ALL hands stay above pot odds (0.25). We call correctly given the range — the range just isn't narrow enough.

### 10. Various Bug Fixes (v22)
- Fixed: duplicate _equity_threshold_play, flop pot control, _opp_faces_bet undercounting, bluff raise tracking
- **Result**: 46.4% WR on 69 matches. Unclear which fix helped/hurt. Reverted to baseline.

### 11. Agent-Written Code (river blueprint stubs, modified decision flow)
- **Result**: Added ~200 lines of untested code. Contributed to v22 regression. Reverted.

### 12. Jackbot Code Analysis
- Pure heuristic bot with board-texture adjustments, commitment bands, bet-size inference
- No solver, no CFR, no range narrowing. Just equity thresholds with ~100 tuned parameters.
- Insight: "don't lose" strategy that's robust but can't exploit anyone.

### 13. Noam Brown's Poker Solver Analysis
- Key insight: ranges are weight vectors passed INTO solver, narrowed OUTSIDE by multiplying P(action|hand)
- Key insight: use position-specific bet sizing, DCFR, alternating updates
- Key insight: our approach is fundamentally correct, the gap is narrowing quality

## Key Findings

1. **Compact tree is broken for equilibrium** — no re-raise threat → bet everything → P(bet|hand) useless
2. **Full tree converges to correct equilibrium** — but takes 7x more compute and over-narrows with uniform ranges
3. **Runtime solve-narrow with narrowed ranges works in testing** (+18% accuracy) but doesn't change production outcomes because equity stays above pot odds even after narrowing
4. **The 5-card selection mechanic** means all hands are premium. Even with perfect narrowing, our equity vs a value-heavy range is 35-50% — above the 25% pot odds threshold. Overcalling is MATHEMATICALLY CORRECT given the range estimate; the range estimate just isn't accurate enough.
5. **The #5 baseline is ~50% WR**, not 78% as initially thought (small sample bias)
6. **Top 12 opponents**: 5 bluffers, 3 weak-tight, 2 value-heavy, 1 balanced, 1 unknown

## What Actually Works
- Flop blueprint: +1.5/hand ✓
- Turn blueprint: +8.6/hand ✓
- River equity thresholds for acting first: ~20-25% bet freq ✓
- Range solver for facing bets: correct call/fold given the range ✓
- Lead protection: auto-fold when mathematically won ✓

## What Doesn't Work
- Every river narrowing approach we tried
- Adding multiple changes at once (always regresses)
- Precomputing with compact tree
- Agent-written code without manual verification

## Current Status
- v24 uploaded with runtime solve-narrow (compact tree, narrowed ranges)
- EC2 fleet killed
- ~30 hours to deadline
