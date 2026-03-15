# Stockfish 🐟 — CMU x Jump Trading Poker AI Tournament 2026

**Team**: Stockfish 🐟
**Members**: vortex, prithish_shan
**Competition**: CMU Data Science Club Poker Bot Competition, March 14-21 2026
**Result**: #1 ELO (1414), 78.4% win rate, 199 matches

## Strategy

Pure GTO (Game Theory Optimal) via real-time CFR subgame solving. Zero
exploitation. The bot computes Nash equilibrium strategies for every
post-flop decision and uses precomputed equilibrium strategies for pre-flop.

GTO means: even if the opponent knows exactly how we play, they cannot
improve their results against us. We never assume anything about what the
opponent tends to do. We just compute the mathematically optimal response
to their possible hands given the evidence (their revealed discards and
their betting actions).

### Why GTO Instead of Exploitation

ELO and the round-robin finals are binary — winning by 1 chip equals winning
by 10,000 chips. Exploitation can win bigger margins against weak opponents,
but it can also LOSE against opponents who don't behave as expected (we
learned this the hard way — our exploitative v2 bot lost to trapping
opponents who just called with strong hands and let us overbet into them).

GTO maximizes win RATE. It never loses in expectation against any strategy.
Against weak opponents, GTO still wins because their mistakes give us their
blind money. Against strong opponents, GTO can't be exploited. Consistency
over dominance.

## How It Works End-to-End

### Phase 1: Pre-Flop (5 cards, no board)

We look up our hand in a precomputed strategy table. This table was built
offline by solving a simplified poker game using CFR (Counterfactual Regret
Minimization):

- All possible 5-card hands are bucketed into 50 strength levels based on
  their expected win rate after optimal discard across all possible flops
- The preflop betting game (fold/call/raise at 7 different sizes) is modeled
  as a game tree with 290 nodes
- Two simulated players played this game for 1000 iterations, adjusting
  strategies based on regret. The converged average IS the Nash equilibrium
- The equity between buckets was computed by simulating actual matchups
  (sampling hands, dealing flops, finding optimal discards, comparing),
  not by using a constant approximation

At runtime: look up our bucket, find the matching node in the game tree for
the current bet level, sample an action from the equilibrium distribution.
If the exact bet level isn't in the tree, round to the nearest node.

### Phase 2: Discard Decision (5 cards → keep 2)

The flop is dealt (3 community cards). We must choose which 2 of our 5 cards
to keep.

**If we're BB (discard first):** Evaluate all 10 possible keep-pairs by
exact equity — for each pair, enumerate every possible opponent hand and
every possible turn+river card, look up who wins. Pick the best.

**If we're SB (discard second):** We first see BB's 3 discards. Run Bayesian
inference: for each of the ~120 possible opponent hands, reconstruct their
original 5-card hand and check whether keeping that pair was rational. Weight
each hand by how rational the keep would be. The temperature adapts per-hand
based on how spread out the keep options are (if all keeps are similar, many
are plausible; if one dominates, only it is likely). Then evaluate our 10
keep-pairs using equity weighted by these probabilities.

### Phase 3: Post-Flop Betting (Flop, Turn, River)

Each time it's our turn, we solve a miniature poker game in real-time:

**Step 1 — Narrow the opponent's range (if they raised):**
When opponent raises, their range gets tighter proportional to the bet size.
A 98-chip raise into a 4-chip pot means they likely have the top 30% of their
range. A 2-chip raise keeps 85%. This is pot-odds-based math: only strong
hands justify risking large amounts relative to the pot.

**Step 2 — Build a game tree (~1ms):**
From the current state, construct all possible action sequences. Three
abstract bet sizes: half-pot, pot, and all-in (whatever's left to reach 100
total). The tree has ~130 nodes. Capped at 3 raises per street.

**Step 3 — Compute terminal payoffs (~2-100ms):**
For each leaf (fold or showdown) and each possible opponent hand:
- Fold: winner takes loser's bet
- River showdown: exact hand rank comparison via lookup table
- Flop/turn end-of-street: exact equity as continuation value (enumerate
  all remaining board cards)

Per-hand equity is computed ONCE and reused across all showdown terminals
with different pot sizes.

**Step 4 — Find Nash equilibrium via CFR+ (~80-100ms):**
Walk the game tree 60-100 times. Each iteration:
- At our nodes: try each action, compute expected value, accumulate regret
  for not taking better actions
- At opponent nodes: same process but from THEIR perspective (they minimize
  our value — zero-sum game). Each possible opponent hand has its own strategy

After all iterations, the average strategy converges to Nash equilibrium.

**Step 5 — Act:**
Sample from the equilibrium distribution: e.g., "check 15%, bet pot 60%,
all-in 25%." Near-pure strategies (>90% on one action) are taken
deterministically.

### What the Solver Handles Implicitly

These all emerge from the equilibrium computation — none are hardcoded:
- Balanced bluffing at the correct frequency
- Pot control (checks medium hands when checking has higher EV)
- Check-raising (checks strong hands to trap)
- Bet sizing selection per situation
- Fold/call/raise frequencies that can't be exploited

### Where Information Flows

```
Opponent's 3 discards (always revealed)
        │
        ▼
  Bayesian inference ──► Weighted opponent range (~30-40 hands)
                                  │
                           Opponent raises?
                                  │
                                  ▼
                          Range narrowing by bet size
                          (pot-odds-based math)
                                  │
                                  ▼
                          CFR+ solver input
                          (opponent hand distribution)
                                  │
                                  ▼
                          Nash equilibrium computation
                          (60-100 CFR iterations)
                                  │
                                  ▼
                          Mixed strategy
                          (e.g., check 15%, bet 60%, all-in 25%)
                                  │
                                  ▼
                          Sample action ──► Execute
```

## Technical Components

| Component | File | What It Does | Runtime Cost |
|---|---|---|---|
| CFR+ solver | solver.py, game_tree.py | Computes Nash equilibrium per decision | ~90-130ms |
| Exact equity | equity.py | Enumerates all outcomes (10,920 max) | ~5ms |
| Discard inference | inference.py | Bayesian P(hand \| discards) | <0.5ms |
| Hand rank tables | data/hand_ranks.npz | 888K 7-card + 80K 5-card lookups | Loaded at init |
| Preflop potential | data/preflop_potential.npz | Expected win rate for all 5-card hands | 0ms (lookup) |
| Preflop strategy | data/preflop_strategy.npz | GTO mixed strategies from CFR | 0ms (lookup) |
| Range narrowing | player.py | Filters opponent range on raises | <0.1ms |

## Known Approximations

These are places where we approximate rather than compute exactly:

1. **Post-flop bet abstraction**: 3 discrete sizes (half-pot, pot, all-in)
   instead of continuous. Planned: 5 sizes in Phase 2.

2. **Depth-limited solving**: Each street solved independently. Continuation
   value uses exact equity. Planned: multi-street solving in Phase 3.

3. **Range narrowing percentiles**: The 85/70/50/30% cutoffs by bet-to-pot
   ratio are heuristic. Planned: nested subgame solving (use the solver's
   own opponent strategy) in Phase 2.

4. **Preflop equity matrix**: Simulated from sampled matchups (50 hands ×
   20 flops per bucket pair). Accurate but not exhaustive.

## Performance

| Operation | Time | Notes |
|---|---|---|
| Init | ~0.4s | Load all tables |
| Preflop decision | ~0ms | Dict lookup |
| Discard evaluation | ~25ms | 10 keep-pairs × exact equity |
| CFR solver (river) | ~90ms | 100 iterations, 130 nodes |
| CFR solver (flop) | ~130ms | + runout enumeration |
| **Avg per hand** | **~100ms** | **~200s / 1000 hands on server** |

## Evolution

| Version | Strategy | Result |
|---|---|---|
| v1 | Exact equity + pot odds | Over-raised medium hands |
| v2 | + Exploitation | Backfired against trappers |
| v3 | GTO thresholds | Deterministic, exploitable patterns |
| v4 | CFR solver (buggy) | Folded nuts, over-raised everything |
| v5 | CFR solver (fixed) | Beat all reference bots |
| **v6 (current)** | **CFR + real preflop + range narrowing** | **#1 ELO, beats former problem opponents** |

## Regenerating Offline Data

```bash
python -m submission.precompute                  # Hand rank tables (~2 min)
python -m submission.precompute_preflop          # Hand potentials (~20 min)
python -m submission.precompute_preflop_strategy # Preflop GTO ranges (~40 min)
```

## Analysis Tool

```bash
python analyze.py                    # Dashboard across all downloaded matches
python analyze.py match_908.txt      # Detail on specific match
python analyze.py AlbertLuoLovers    # Filter by opponent
```

## What Comes Next

**Phase 2 (March 16-17, 1000s budget, 2 vCPU):**
- Nested subgame solving: use solver's own equilibrium to narrow ranges
  instead of heuristic percentiles
- 5 bet sizes instead of 3 (add 33% and 75% pot)
- 200 CFR iterations instead of 100

**Phase 3 (March 18-20, 1500s budget, 4 vCPU):**
- Multi-street solving (flop+turn as single game tree)
- 300+ CFR iterations
- Potential multi-threading for parallel CFR
