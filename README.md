# Poker AI — CMU x Jump Trading Tournament 2026

A poker bot for a custom 27-card variant, built for the CMU x Jump Trading Poker AI Tournament (March 2026).

## The Game

27-card deck (9 ranks × 3 suits). Each player gets 5 hole cards, discards 3 (revealed), keeps 2. Community board: flop (3), turn (1), river (1) with betting on each street. 1000 hands per match, ELO ranking.

Key properties:
- No four-of-a-kind (only 3 suits)
- Trips is mediocre — 52% of showdown hands beat trips
- Best-2-of-5 selection means everyone has strong hands (0% high card at showdown)
- Flush draws are extremely common (9 cards per suit) and complete ~51% of the time
- The game tree is small enough (~80K river boards) to solve entirely at runtime with a fast solver

## What the Optimal Approach Looks Like

1. **Fast solver (C/C++ with SIMD)** — the game is small enough to solve every decision at runtime. No precomputation needed. A 25x-optimized C solver could solve flop-to-river in ~2 seconds.

2. **Exact range tracking for BOTH players** — track hero range through the hand (narrowed by our own actions) and opponent range (narrowed by their actions). Pass both to the solver. This is how Libratus works — ranges are derived from equilibrium, not inferred from observations.

3. **Opponent modeling** — detect opponent type (aggressive, passive, trapping) within 50 hands. Solve against the detected model, not generic GTO.

4. **No precomputation** — with a fast enough solver, blueprints and EC2 fleets are unnecessary for a 27-card game.

## What We Tried

- **EC2 precomputation** — 20 × c5.9xlarge instances computing 80,730 river boards at 300 DCFR iterations. Produced 146MB of strategy data. In hindsight, overkill for a game this small.
- **Multi-street blueprints** — backward induction across all 2,925 flop boards with turn/river continuation values. 632MB of precomputed data.
- **Heuristic overrides** — equity gate margins, floor overrides, bluff injection, selective caller suppression, overbet sizing hacks. Each one patched a symptom rather than fixing the root cause.
- **Node-locked solver** — attempted to model non-GTO opponents by locking their strategy to equity-proportional play. Produced reversed behavior (bluffed with weak hands, checked strong).
- **Check-behind narrowing** — narrowing opponent range when they check behind. Proven to hurt because opponents trap (check strong hands).

## What We Ended Up With

### Architecture
```
Preflop:  Precomputed GTO table (200 buckets)
Discard:  Exact equity + draw-aware Bayesian inference
Flop:     Depth-limited solver (C DCFR) with turn+river game values
Turn:     Depth-limited solver (C DCFR) with river game values
River:    Full-tree DCFR solver (C, 4 bet sizes) + floor override + equity gate
```

### Key Components
- **C DCFR solver** (`dcfr_core.c`) — auto-compiles at startup, 4.8x speedup over Python, returns root game values for continuation value computation
- **Draw-aware discard inference** — Boltzmann-weighted with flush draw (+3000) and straight draw (+1500) bonuses. Fixed a 47 percentage point equity error.
- **Runtime P(bet|hand) narrowing** — solves from opponent's perspective on every street to compute betting probabilities. Replaces static blueprint narrowing.
- **Full subgame solving** — river game values feed into turn solver, turn game values feed into flop solver. Every decision accounts for future street betting dynamics.
- **Equity gate (+0.12 margin)** on river calls — 94.9% accuracy across 2,625 tested calls.

### The Gap We Identified Too Late

The single biggest remaining issue: **hero range tracking**. We track the opponent's range but use a UNIFORM hero range when computing P(bet|hand). Since opponents bet differently against our actual (capped) range vs uniform, this causes systematic overcalling on the river.

We implemented hero range tracking but it crashed on an edge case. We reverted instead of debugging — a 20-minute fix that could have changed the outcome.

### Lessons Learned

1. **For a 27-card game, invest in solver speed, not precompute infrastructure.** A fast C solver eliminates the need for EC2 fleets and blueprint databases.
2. **Foundation before features.** Range tracking is the foundation. Everything else (DL solvers, runtime narrowing, subgame solving) operates on the ranges. Wrong ranges → wrong everything.
3. **Debug, don't revert.** A feature that works in 19/20 tests is worth debugging, not reverting.
4. **One change at a time.** We shipped versions with 6 changes and couldn't tell which helped.
5. **Verify execution, not just compilation.** Our "runtime narrowing" was dead code for hours due to a missing import — it compiled fine but silently fell back to blueprint narrowing.
6. **Agents describe the right architecture but implement shortcuts.** Always trace actual code execution on a specific hand, not the architecture description.

## Project Structure

```
submission/
  player.py              # Main decision engine
  inference.py            # Bayesian discard inference with draw bonus
  range_solver.py         # DCFR solver + C bridge integration
  depth_limited_solver.py # Flop/turn DL solver with subgame solving
  dcfr_core.c             # C DCFR solver (auto-compiles)
  dcfr_bridge.py          # Python-C bridge with root game value extraction
  equity.py               # Exact enumeration equity engine
  game_tree.py            # Betting tree (full/lean/compact modes)
  river_lookup.py         # Lazy-loaded precomputed river strategies
  data/                   # Precomputed blueprints + river strategies
docs/
  OPTIMAL_SOLVER.md       # Detailed roadmap for the ideal architecture
```
