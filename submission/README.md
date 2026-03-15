# Stockfish 🐟 — CMU x Jump Trading Poker AI Tournament 2026

**Team**: Stockfish 🐟
**Members**: vortex, prithish_shan
**Competition**: CMU Data Science Club Poker Bot Competition, March 14-21 2026
**Result**: #1 ELO, 92.5%+ win rate during open season

## Strategy: GTO via Real-Time CFR Subgame Solving

This bot computes Nash equilibrium strategies in real-time for every decision.
Instead of heuristic rules ("raise if equity > 72%"), it solves the actual
poker game tree using CFR+ (Counterfactual Regret Minimization) — the same
algorithm family behind Libratus and Pluribus, the AIs that beat world champion
poker players.

**Why GTO, no exploitation:** ELO and round-robin finals are binary (win/loss).
Winning by 1 chip equals winning by 10,000 chips. GTO maximizes win rate by
never losing in expectation against any opponent. Exploitation can backfire
against trapping opponents, turning wins into losses. GTO is provably
unexploitable.

## Game Variant

Modified Texas Hold'em with a 27-card deck:
- **Deck**: 9 ranks (2-9, A) × 3 suits (♦♥♠). No face cards, no clubs.
- **Deal**: 5 hole cards each. On the flop, discard 3, keep 2.
- **Discards revealed**: Opponent sees your 3 discards (and vice versa).
- **BB discards first**: SB sees BB's discards before choosing their own.
- **Max bet**: 100 chips per player per hand. Blinds 1/2.
- **Match**: 1000 hands, positions alternate, stack resets each hand.
- **No four-of-a-kind**: Only 3 suits = max trips.

## Architecture

```
submission/
├── player.py                      # Main bot (PlayerAgent class)
├── solver.py                      # Real-time CFR+ subgame solver
├── game_tree.py                   # Betting tree representation
├── equity.py                      # Exact equity via precomputed lookups
├── inference.py                   # Bayesian discard inference
├── precompute.py                  # Offline: hand rank tables
├── precompute_preflop.py          # Offline: pre-flop hand potential
├── precompute_preflop_strategy.py # Offline: GTO preflop ranges via CFR
└── data/
    ├── hand_ranks.npz             # 7-card + 5-card rank tables (1.6MB)
    ├── preflop_potential.npz      # Expected equity for all 5-card hands (302KB)
    ├── preflop_strategy.npz       # GTO preflop action frequencies (11KB)
    └── preflop_tree.pkl           # Preflop game tree for strategy lookup
```

## Decision Flow

```
                    ┌─────────────┐
                    │  Hand Dealt │
                    │  (5 cards)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  PRE-FLOP   │ Precomputed GTO mixed strategy
                    │  Betting    │ from two-player CFR (2000 iterations)
                    └──────┬──────┘ Falls back to pot-odds + range
                           │        estimation for unusual raise sizes
                    ┌──────▼──────┐
                    │   FLOP +    │ Exact equity over all 10 keep-pairs
                    │  DISCARD    │ SB uses Bayesian inference on BB's
                    │             │ revealed discards for weighted equity
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │    POST-FLOP BETTING    │
              │  (Flop / Turn / River)  │
              │                         │
              │  Real-time CFR+ solver  │
              │  ~130 node game tree    │
              │  60-100 iterations      │
              │  ~90-130ms per decision │
              │                         │
              │  Produces Nash equilib- │
              │  rium mixed strategy:   │
              │  CHECK 12%, BET 65%,    │
              │  ALL-IN 23%             │
              └─────────────────────────┘
```

## How Each Component Works

### 1. Real-Time CFR+ Subgame Solver (`solver.py` + `game_tree.py`)

The core of the bot. For each post-flop decision, solves a miniature poker
game in real-time:

**The game being solved:** Hero has one specific hand. Opponent could hold
any of ~20-40 possible hands (narrowed by discard inference). Both players
can CHECK, BET (half pot / pot / all-in), CALL, FOLD, or RAISE. The game
tree has ~130 nodes covering all possible action sequences.

**How CFR finds the Nash equilibrium:** Simulates the game 60-100 times.
Each iteration, both "players" adjust their strategies based on accumulated
regrets — if betting would have been more profitable than checking, increase
the betting frequency. After enough iterations, the average strategy converges
to the point where neither player can improve by deviating.

**What emerges from the equilibrium (not hardcoded):**
- Balanced bluffing at the mathematically correct frequency
- Pot control (checks medium hands when checking has higher EV)
- Check-raising (checks strong hands to trap aggressive opponents)
- Optimal bet sizing per situation
- Fold/call/raise frequencies that can't be exploited

**Why mixed strategies:** If we always bet strong hands and check weak hands,
opponents learn the pattern. GTO mixes actions (e.g., bet 65%, check 12%,
all-in 23% with the same hand) so opponents can't learn anything exploitable.

**Depth-limited solving:** We solve only the current street's betting. When
the street ends without a fold, exact equity (full enumeration of all possible
remaining cards) serves as the continuation value. This is the approach used
by Libratus — solve the immediate decision exactly, approximate future play
with equity.

**Adaptive computation:** Iteration count scales with time remaining:
- >300s left: 100 iterations (tight convergence)
- >150s left: 60 iterations
- >50s left: 30 iterations
- <50s left: fall back to equity-threshold logic

### 2. Precomputed GTO Preflop Strategy (`precompute_preflop_strategy.py`)

The preflop betting game is solved offline via two-player CFR:
- 50 hand-strength buckets × 7 raise levels [2,4,8,16,30,60,100] × 2000 iterations
- Equity matrix computed from simulated matchups (not a constant approximation)
- Produces proper mixed strategies with balanced fold/call/raise frequencies

**Key solved behaviors:**
- SB opening: fold 97% of worst hands, call medium, raise big with strong
- BB facing raise: fold 99% of garbage, call medium, re-raise monsters
- BB after SB limps: check weak, raise ~98% with strong hands

**Unmatched bet sizes:** When opponent raises an amount not exactly in the
precomputed tree, we round to the nearest node by bet distance. This keeps
us in the equilibrium strategy rather than falling back to heuristics.

### 3. Exact Equity Engine (`equity.py`)

Computes win probability by enumerating ALL possible outcomes — every possible
opponent hand combined with every possible remaining board card. With a
27-card deck, the worst case is only 10,920 scenarios (vs 1.87 million in
standard 52-card poker).

**Precomputed lookup tables (loaded at init, ~0.4s):**
- 7-card table: all C(27,7) = 888,030 combinations → hand rank
- 5-card table: all C(27,5) = 80,730 combinations → hand rank

Stored as Python dicts for O(1) lookup (~50 nanoseconds each). This is 100x
faster than calling the `treys` library at runtime, which is what makes
exact enumeration feasible within the time budget.

**Why exact matters:** The reference ProbabilityAgent samples 400 random
outcomes (Monte Carlo) with ~2.5% error. A hand with true 65% equity might
read as 62% (fold) or 68% (raise). Our exact computation has zero error —
65.000% every time.

### 4. Bayesian Discard Inference (`inference.py`)

When the opponent discards 3 cards (always revealed in this variant), we
infer which 2 cards they likely kept.

For each of ~120 possible opponent hands:
1. Reconstruct their hypothetical original 5 cards (candidate pair + 3 discards)
2. Evaluate all 10 possible keep-pairs from that hand
3. Assign Boltzmann weight: `w = exp(-delta / temperature)` where delta is
   how much worse this keep is vs the optimal keep

Result: the ~120 possible hands are weighted from ~1.0 (obvious best keep)
to ~0.0 (irrational keep). This narrows the effective range to ~20-40 hands,
making all subsequent equity and solver computations more accurate.

**SB advantage:** SB sees BB's discards before choosing their own. SB uses
inference-weighted equity to pick their keep-pair, giving SB an information
edge at the discard decision.

### 5. Action-Based Range Narrowing (`player.py`)

Updates opponent range weights when they take betting actions:
- Opponent RAISES: keep top 40% of range by hand strength
- Opponent CALLS: keep top 70% of range
- Opponent CHECKS: no change (consistent with any hand)

This is Bayesian inference, not exploitation — the opponent's action IS
evidence about their hand, regardless of who they are. The narrowed range
feeds into the CFR solver for more accurate equilibrium computation.

### 6. Pre-Flop Hand Potential Table (`precompute_preflop.py`)

Maps every possible 5-card starting hand to its expected post-discard equity
across all 1,540 possible flops. Computed offline (~8 minutes) using suit
isomorphism to reduce 80,730 hands to ~63,846 canonical hands.

For each canonical hand × each possible flop:
- Evaluate all 10 keep-pairs using the 5-card rank table
- Take the best keep (optimal discard decision)
- Average the best-keep rank across all flops → hand potential

At runtime: single dict lookup per pre-flop decision. Zero compute cost.

### 7. Smart Discard Selection (`player.py`)

Evaluates all C(5,2)=10 ways to keep 2 of 5 cards by exact equity (~25ms).
For each keep-pair, the 3 discarded cards become dead, and equity is computed
against all possible opponent hands and board runouts.

As SB: uses Bayesian-weighted equity from opponent's revealed discards.
As BB: uses uniform equity (opponent hasn't discarded yet).

## Evolution of the Bot

| Version | Strategy | Key Features | Weakness |
|---|---|---|---|
| v1 | Exact equity + pot odds | Exact enumeration, discard inference | Over-raised medium hands |
| v2 | + Opponent exploitation | Rolling window model, blocker bluffing | Exploitation backfired against trappers |
| v3 | GTO thresholds | Removed exploitation, pot control, re-raise protection | Still heuristic, deterministic |
| v4 | **CFR solver** | **Real-time Nash equilibrium, precomputed preflop ranges** | **Current version** |

**Key lessons:**
- Exploitation hurt more than it helped (lost to passive trapping opponents)
- Pre-flop calling was the biggest leak (calling 80-chip raises with bad hands)
- GTO mixed strategies > deterministic thresholds
- The small deck (27 cards) makes exact computation feasible everywhere

## Performance

| Operation | Time | Notes |
|---|---|---|
| Init (load tables) | ~0.4s | Once at match start |
| Preflop decision | ~0-60ms | Lookup or range estimation |
| Discard decision | ~25ms | 10 keep-pairs × exact equity |
| Discard inference | <0.5ms | 1,200 five-card lookups |
| Range narrowing | <0.1ms | Filter + renormalize weights |
| CFR solver (river) | ~90ms | 100 iterations, 130-node tree |
| CFR solver (turn) | ~106ms | + 13 runout cards |
| CFR solver (flop) | ~129ms | + C(14,2)=91 runout combos |
| Fallback (threshold) | ~5ms | When time is critically low |
| **Average per hand** | **~100ms** | **~147s total / 1000 hands** |

## Regenerating Lookup Tables

```bash
python -m submission.precompute                  # Hand ranks (~2 min)
python -m submission.precompute_preflop          # Pre-flop potential (~8 min)
python -m submission.precompute_preflop_strategy # Pre-flop GTO ranges (~5 min)
```

## Testing

```bash
python agent_test.py         # Submission validator (4 test bots)
python test_cfr_bot.py       # Comprehensive test suite (24 tests, 6200+ hands)
python run.py                # 1000-hand match (configure in agent_config.json)
```

## Game Theory FAQ

**Q: Is Bayesian discard inference exploitative?**
No. It computes P(hand | discards) — what a rational player would keep given
the cards and board. This is math applied to revealed information. Discards
are mandatory and public; there's no "bluffing" in what you discard. The
adaptive temperature handles cases where multiple keeps are similarly strong.

**Q: Is range narrowing on raises exploitative?**
Partially. The percentile cutoffs (85/70/50/30%) are heuristic. True GTO
would derive narrowing from the equilibrium itself (nested subgame solving,
planned for Phase 2). However, the narrowing is based on pot-odds math:
a 98-chip bet into a 4-chip pot is only rational with very strong hands,
regardless of who the opponent is. This is closer to inference than exploitation.

**Q: What bet sizes does the solver use?**
Three abstract sizes per action: half-pot, pot, and all-in (whatever's left
to reach 100 total bet). All-in is NOT always 100 — if you've bet 30, all-in
is 70 more. When opponent bets a non-abstract amount, the solver's game tree
already handles it because the tree is built from the CURRENT bet state.

**Q: What happens when a preflop raise doesn't match the tree?**
We round to the nearest node by bet distance. The equilibrium strategy at
the closest node is a better approximation than falling back to heuristics.

**Q: Could we use more bet sizes?**
Yes. Going from 3 to 5 sizes (adding 33% and 75% pot) would reduce abstraction
error but increase tree size ~1.8x and solver time proportionally. Planned
for Phase 2 when compute budget doubles.

**Q: Is depth-limited solving accurate?**
The continuation value uses exact equity (full enumeration, zero error).
The approximation is that both players play "correctly" on future streets.
Multi-street solving (flop+turn as one tree) would be more accurate but
needs ~14x more nodes — not feasible until Phase 3.

**Q: What are the remaining non-GTO elements?**
1. Range narrowing percentages (heuristic, not equilibrium-derived)
2. Bet size abstraction (3 discrete sizes, not continuous)
3. Depth-limited solving (equity proxy for future streets)
All are planned to be addressed in Phase 2-3.

## Local Match Results

| Opponent | Result | Margin | Time Used |
|---|---|---|---|
| FoldAgent (1000h) | Win | +1,500 | 0s |
| CallingStationAgent (1000h) | Win | +39,997 | 192s |
| AllInAgent (1000h) | Win | +40,086 | 66s |
| RandomAgent (1000h) | Win | +13,108 | 55s |
| ProbabilityAgent (1000h) | Win | +10,921 | 147s |
| RL Agent trained (260h) | Win | +9,067 (opponent timeout) | 13s |
| Self-play CFR vs CFR (500h) | Draw | +212 (variance) | 49s each |
