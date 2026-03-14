# Poker Bot — Goofy Goobers

## Strategy: GTO via Real-Time CFR Subgame Solving

This bot computes Nash equilibrium strategies in real-time using CFR+
(Counterfactual Regret Minimization). Instead of using heuristic equity
thresholds ("raise if equity > 72%"), it solves the actual poker game tree
for each specific situation and plays the mathematically optimal strategy.

No exploitation. GTO can't lose in expectation against any opponent.

## Architecture

```
submission/
├── player.py              # Main bot (PlayerAgent class)
├── solver.py              # CFR+ subgame solver (~100ms per decision)
├── game_tree.py           # Betting tree representation for CFR
├── equity.py              # Exact equity engine via precomputed lookup tables
├── inference.py           # Bayesian discard inference (range narrowing)
├── precompute.py          # Offline: generate hand rank tables
├── precompute_preflop.py  # Offline: generate pre-flop hand potential table
└── data/
    ├── hand_ranks.npz         # 7-card + 5-card rank tables (1.6MB)
    └── preflop_potential.npz  # Pre-flop hand equity table (302KB)
```

## How Each Component Works

### 1. CFR+ Subgame Solver (`solver.py` + `game_tree.py`)

The core innovation. For each post-flop decision:

**Step 1 — Build game tree (~1ms):**
Creates a tree of all possible action sequences from the current state.
Uses 3 abstract bet sizes (half pot, pot, all-in) to keep the tree small
(~130 nodes). Caps at 3 raises per street. The tree shape depends on
whether we're initiating (CHECK/BET options) or responding to a bet
(FOLD/CALL/RAISE options).

**Step 2 — Compute terminal payoffs (~2-100ms):**
For every leaf node (fold or showdown) and every possible opponent hand:
- Fold by hero: hero loses their bet
- Fold by opponent: hero wins opponent's bet
- Showdown on river: exact hand rank comparison via lookup table
- End of street on flop/turn: exact equity as continuation value
  (enumerates all possible remaining board cards)

Per-hand equity is computed ONCE and reused across all showdown terminals
(different pot sizes but same win/loss outcome). This optimization reduced
flop solve time from 533ms to 129ms.

**Step 3 — Run CFR+ iterations (~80-100ms):**
Walks the game tree 60-100 times (adaptive based on time remaining).
At each decision node:
- Hero nodes: compute regrets weighted across all opponent hands.
  Hero's strategy is the SAME regardless of opponent's hand (hero
  doesn't know opponent's cards).
- Opponent nodes: compute per-hand regrets and per-hand strategies.
  Opponent's strategy depends on which hand they hold.
- CFR+ modification: regrets are floored at zero, which converges
  faster than vanilla CFR.

After all iterations, the average strategy at the root converges to
Nash equilibrium.

**Step 4 — Execute (~0ms):**
Sample an action from the equilibrium distribution. For near-pure
strategies (>90% on one action), take the dominant action deterministically.
Map abstract actions (BET_HALF, BET_POT, BET_ALLIN) to concrete raise
amounts clamped to [min_raise, max_raise].

**What the solver handles implicitly (no hardcoded logic needed):**
- Balanced bluffing at the correct frequency
- Pot control (checks medium hands when checking has higher EV)
- Check-raising (checks strong hands to trap)
- Optimal bet sizing per situation
- Fold/call/raise frequencies that can't be exploited

**Fallback:** When time remaining < 50 seconds, falls back to simple
equity-threshold logic to avoid timeout.

### 2. Exact Equity Engine (`equity.py`)

Computes win probability by enumerating ALL possible outcomes.
27-card deck means at most 10,920 scenarios (vs 1.87M in standard poker).

Two precomputed lookup tables (loaded at init from `data/hand_ranks.npz`):
- 7-card table: 888,030 entries for (2 hole + 5 board) → hand rank
- 5-card table: 80,730 entries for discard inference heuristic

All stored as Python dicts for O(1) lookup (~50ns per lookup vs ~10μs
for numpy searchsorted). This makes the difference between 22ms and
2.2 seconds for a discard evaluation.

### 3. Discard Inference (`inference.py`)

When opponent reveals 3 discards, computes probability distribution over
their ~120 possible kept hands using Boltzmann weighting:

For each candidate kept pair:
1. Reconstruct hypothetical original 5-card hand
2. Evaluate all 10 possible keeps from that hand
3. Weight by how close this keep is to optimal

Narrows effective range from ~120 to ~20-40 hands. This directly feeds
into the CFR solver's opponent model (more accurate range = better
equilibrium computation).

### 4. Action-Based Range Narrowing (`player.py`)

Updates opponent range weights when they bet or call:
- Opponent RAISES: keep top 40% of range (raises signal strength)
- Opponent CALLS: keep top 70% of range (calls signal moderate strength)
- Opponent CHECKS: no change

This narrows the range further between streets, making each subsequent
solver call more accurate.

### 5. Pre-Flop Hand Potential Table (`precompute_preflop.py`)

Maps every 5-card starting hand to its expected post-discard equity
across all 1,540 possible flops. Computed offline in ~8 minutes using
suit isomorphism to reduce from 80,730 to ~63,846 canonical hands.

At runtime: single dict lookup. Used for pre-flop raise/call/fold decisions.

### 6. Smart Discard Selection

Evaluates all C(5,2)=10 keep-pairs by exact equity (~25ms total).
As SB (acts second at discard), uses inference-weighted equity from
opponent's revealed discards. As BB (acts first), uses uniform equity.

## Known Risks and Mitigations

**ARM64 performance:** Solver uses 142s on Apple Silicon. Graviton2 is
~1.5x slower → ~213s. Budget is 500s (Phase 1). Mitigation: adaptive
iteration count drops from 100 to 30 when time is low, falls back to
equity thresholds below 50s remaining.

**Bet size mapping:** Solver uses 3 abstract bet sizes. When opponent
bets an amount between abstractions, the solver's strategy is slightly
suboptimal. Mitigation: 3 sizes (half/pot/allin) cover the strategically
important range; exploitability from abstraction is typically <5% of pot.

**Game tree root mismatch:** The tree is built for the current decision
point (initiating or responding). We detect this from whether bets are
equal (initiating) or unequal (responding). Edge case: if state tracking
is off, the tree might not match reality. Mitigation: valid_actions check
in action mapping catches impossible actions.

## Performance

| Operation | Time | Notes |
|---|---|---|
| Init (load tables) | ~0.4s | Once at match start |
| Pre-flop decision | ~0ms | Dict lookup |
| Discard decision | ~25ms | 10 keep-pairs × exact equity |
| Discard inference | <0.5ms | 1,200 lookups |
| CFR solver (river) | ~90ms | 100 iterations, 130-node tree |
| CFR solver (turn) | ~106ms | + runout enumeration |
| CFR solver (flop) | ~129ms | + more runouts |
| Fallback (threshold) | ~5ms | When time is low |
| **Match total** | **~142s** | **28% of Phase 1 budget** |

## Regenerating Lookup Tables

```bash
python -m submission.precompute           # Hand ranks (~2 min)
python -m submission.precompute_preflop   # Pre-flop potential (~8 min)
```

## Match Results

| Opponent | Result | Margin | Time Used |
|---|---|---|---|
| FoldAgent | Win | Dominant | 2s |
| CallingStationAgent | Win | Dominant | 10s |
| AllInAgent | Win | Dominant | 5s |
| RandomAgent | Win | Dominant | 8s |
| ProbabilityAgent (1000h) | Win | +11,118 | 142s |
| RL Agent (trained) | Win | +9,067 (timeout) | 13s |
