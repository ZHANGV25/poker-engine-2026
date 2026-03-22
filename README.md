# Stockfish -- CMU x Jump Trading Poker AI Tournament 2026

**Team**: Stockfish
**Members**: vortex, prithish_shan
**Competition**: CMU Data Science Club Poker Bot Competition, March 14-22 2026

---

## The Game

27-card deck (9 ranks x 3 suits -- no face cards, no clubs). Each player gets 5 hole cards. After preflop betting, both players discard 3 cards (keeping 2), with discards revealed to the opponent. Community cards follow: flop (3), turn (1), river (1) with betting on each street. Max bet 100 per player per hand, 1000 hands per match, binary win/loss ELO ranking.

Key properties of 27-card variant:
- Four-of-a-kind is impossible (only 3 suits)
- Trips is mediocre (52% of showdowns beat trips)
- 5-card selection means both players reach river with premium hands
- Ace is both high and low for straights (A2345 and 6789A both valid)

---

## Architecture

### Per-Street Strategy (v5.0 / Final)

```
Preflop:    200-bucket GTO strategy (precomputed, 5000 CFR iterations)
Discard:    Exact equity for all 10 keep-pairs + draw-aware Bayesian inference
Flop AF:    Backward-induction blueprint (200 iter, 4-bit, 51MB LZMA)
Flop FB:    Depth-limited solver (C DCFR) against narrowed range, blueprint fallback
Turn AF:    Backward-induction blueprint (per-board lazy-loaded, 575MB)
Turn FB:    Depth-limited solver with river continuation values + gadget safety
River AF:   Full tree DCFR range solver (4 sizes, 2 raises) + floor override
River FB:   P(bet|hand) Bayesian narrowing + runtime DCFR solver + equity gate
```

*AF = acting first, FB = facing bet*

### Full Subgame Solving Chain

The bot performs backward induction at runtime for facing-bet decisions:

1. **River equity**: exact enumeration for all hand matchups on the river board
2. **River solver**: DCFR against narrowed opponent range, produces river game values
3. **River game values** become continuation values for turn solver leaf nodes
4. **Turn solver**: depth-limited DCFR with gadget game (safe re-solving)
5. **Turn game values** feed into flop solver when available
6. **Flop solver**: depth-limited DCFR against narrowed range

For acting-first decisions on flop and turn, the precomputed blueprint (which already incorporates backward induction from river through flop) is used instead of runtime solving. This is because acting-first decisions depend heavily on multi-street dynamics (implied odds, reverse implied odds) that single-street solvers miss.

### C DCFR Solver

`dcfr_core.c` is compiled automatically at startup. Provides 4.8x speedup over the Python solver. Falls back to Python if compilation fails.

- Compile flags: `-O3 -march=native -ffast-math`
- Optional BLAS support (macOS Accelerate or Linux OpenBLAS)
- DCFR parameters: alpha=1.5, beta=0, gamma=2 (Noam Brown's Pluribus parameters)
- Extracts root game values for use as continuation values in upstream solvers

### Range Narrowing Pipeline

Every opponent action narrows their estimated range via Bayesian updates:

| Street   | On Bet                                     | On Check                        | Street Transition               |
|----------|--------------------------------------------|---------------------------------|---------------------------------|
| Discard  | --                                         | --                              | Draw-aware Boltzmann inference  |
| Flop     | Bayesian P(bet\|hand) from blueprint       | Bayesian P(check\|hand)         | P(call\|hand) from blueprint    |
| Turn     | Bayesian P(bet\|hand) from turn data       | Bayesian P(check\|hand)         | compute_opp_call_probs (runtime)|
| River    | Runtime solve P(bet\|hand) via range DCFR  | --                              | --                              |

All narrowing uses 0.5% weight floors (never zero out any hand). After 3 streets of narrowing, phantom hands compound down to ~0.0008% -- they die naturally without needing to be zeroed.

---

## Key Components

### submission/inference.py -- Discard Inference
Boltzmann-weighted Bayesian inference over opponent's kept cards. For each possible 2-card keep from the opponent's original 5, evaluates whether it was the rational choice. Draw bonuses (flush draw +3000, OESD +1500) correct the inference for hands that rank poorly by made-hand strength but have high equity through draws. This fixed a 47 percentage-point equity error on drawing hands.

### submission/range_solver.py -- DCFR Range Solver
Full-range DCFR solver with C bridge integration. Solves over all hero hands simultaneously against the opponent's weighted range. Produces strategy (action probabilities per hand) and root game values (EV per hand for continuation values). Runtime P(bet|hand) computation for Bayesian narrowing on all streets. Card blocking via `not_blocked` mask prevents impossible hand pairs from biasing EVs.

### submission/depth_limited_solver.py -- Depth-Limited Solver
Subgame solver for flop and turn facing-bet decisions. Uses river/turn continuation values at leaf nodes (captures implied odds that single-street equity misses). Includes a gadget game for safe re-solving: gives the opponent the option to "cash out" at blueprint value, guaranteeing the re-solve can only improve over the blueprint.

### submission/dcfr_core.c / dcfr_bridge.py -- C Solver Bridge
C implementation of the DCFR inner loop. `dcfr_bridge.py` serializes the Python game tree into flat arrays, calls the C solver via ctypes, and deserializes results. Auto-compiles from source at import time if the `.so` is missing. Includes root game value extraction for feeding upstream solvers.

### submission/player.py -- Main Decision Engine
~1800 lines. Orchestrates all narrowing, blueprint lookups, solver calls, equity gates, floor overrides, lead protection, and adaptive exploit tracking. Contains per-street decision routing and all override logic.

### submission/equity.py -- Exact Equity Engine
Dict-based O(1) hand rank lookups from precomputed `hand_ranks.npz`. Vectorized numpy equity computation: 3ms on turn (was 215ms, 69x speedup), 28ms on flop (was 800ms, 29x speedup).

### submission/multi_street_lookup.py -- Blueprint Lookup
Runtime lookup into precomputed flop strategies (51MB LZMA, decompressed ~2.1GB) and lazy-loaded per-board turn strategies (2925 files, 575MB on disk, ~50MB in memory via LRU cache). Background thread decompresses LZMA data after server starts.

### submission/game_tree.py -- Betting Tree
Builds the game tree for solver decisions. 4 bet sizes (40%, 70%, 100%, 150% pot), max 2 raises. Compact mode available for resource-constrained solves.

### submission/river_lookup.py -- River Precomputed Data
Lazy-loads per-board precomputed river strategies from `river.tar.lzma` (146MB, 80,730 boards, 300 DCFR iterations, 3 pot sizes). Extracted to `/tmp` at startup via background thread.

### submission/solver.py -- One-Hand CFR Solver
Fallback single-hand CFR+ solver. Used when range solver is unavailable or for quick evaluations.

---

## Precomputed Data

| Data | Size | Description |
|------|------|-------------|
| `hand_ranks.npz` | 1.6 MB | 7-card and 5-card hand rank lookup tables |
| `preflop_potential.npz` | 326 KB | Expected win rates for all 80,730 hands |
| `preflop_strategy.npz` + `preflop_tree.pkl` | ~65 KB | 200-bucket preflop GTO |
| `blueprint.pkl.lzma` | 51 MB | Flop strategies (2925 boards, 5 pots, P0+P1) |
| `turn_boards/` | 575 MB | Per-board turn strategies (2925 files) |
| `river.tar.lzma` | 146 MB | River strategies (80,730 boards, 3 pots, 300 iter) |

Total submission: ~775 MB (under 1 GB limit).

---

## Compute Infrastructure

### Blueprint (Flop + Turn)
- 10 x c5.9xlarge (36 vCPU each), 8 workers/instance, ~1.7 hours
- Backward induction: river -> turn -> flop, 200 CFR iterations per street
- S3: `s3://poker-blueprint-2026/`

### River Precompute
- 20 x c5.9xlarge, 80,730 boards, 300 DCFR iterations, 3 pot sizes
- Stored as per-board `.npz` files, compressed to `river.tar.lzma`
- All instances stopped. Data on S3.

---

## Project Structure

```
submission/                          # Competition code (uploaded to server)
  player.py                         # Main bot: PlayerAgent class
  solver.py                         # One-hand CFR+ solver (fallback)
  range_solver.py                   # DCFR range solver (full tree, card blocking)
  depth_limited_solver.py           # DL solver (flop/turn, continuation values, gadget)
  dcfr_core.c                       # C DCFR solver core
  dcfr_core.so                      # Compiled C solver (auto-compiled if missing)
  dcfr_bridge.py                    # Python-C bridge via ctypes
  game_tree.py                      # Betting tree (4 sizes, max 2 raises)
  equity.py                         # Exact equity engine (vectorized O(1) lookups)
  inference.py                      # Bayesian discard inference (draw-aware)
  multi_street_lookup.py            # Flop+turn blueprint lookup (lazy-loaded)
  river_lookup.py                   # River precomputed strategy lookup
  blueprint_lookup.py               # Single-street blueprint lookup
  blueprint_lookup_unbucketed.py    # Unbucketed blueprint lookup
  data/                             # Precomputed data (~775MB)
    hand_ranks.npz
    preflop_potential.npz
    preflop_strategy.npz
    preflop_tree.pkl
    multi_street/
      blueprint.pkl.lzma            # Flop strategies (51MB compressed)
      turn_boards/                  # Per-board turn data (2925 files)
    river.tar.lzma                  # River strategies (146MB compressed)

blueprint/                          # Offline computation (NOT uploaded)
  multi_street_solver.py            # Backward induction solver
  compute_multi_street.py           # Distributed EC2 computation
  compute_river_strategies.py       # River precompute script
  compute_river.py                  # River board computation
  compute_turn_values.py            # Turn game value precomputation
  ec2_launch_fleet.sh               # Launch EC2 fleet
  ec2_launch_river_fleet.sh         # Launch river precompute fleet
  ec2_merge_and_deploy.sh           # Download and deploy results
  merge_river.py                    # Merge river results from fleet
  merge_turn_*.py                   # Merge turn data from fleet

agents/                             # Baseline agents for local testing
  agent.py                          # Base agent class
  prob_agent.py                     # Probability-based agent
  rl_agent.py                       # RL agent (trained weights)

docs/                               # Competition documentation
  rules.md                          # Game rules
  game-engine.md                    # Engine API
  terminology.md                    # Poker terminology
  submission.md                     # Submission format
  libraries.md                      # Allowed libraries

analyze_*.py                        # Match analysis scripts
test_*.py                           # Test suites
verify_*.py                         # Verification scripts
validate_v38.py                     # Pre-upload validation
local_match.py                      # Local head-to-head testing
self_play_test.py                   # Self-play evaluation
```

---

## What Works (Proven Improvements)

- **Draw-aware inference**: flush/OESD draw bonuses fixed 47pp equity error in discard inference
- **Runtime P(bet|hand) narrowing on all streets**: Bayesian updates from blueprint (flop/turn) and runtime solving (river)
- **DL solving on flop/turn facing bets**: captures implied odds from continuation values that blueprint misses
- **River equity gate (+0.12 margin)**: 94.9% accuracy across 2,625 calls. Catches overcalls the solver misses.
- **Floor override for river AF value betting**: P(bet) >= 0.25 threshold for thin value when solver is too passive
- **C solver with game value extraction**: 4.8x speedup, enables full subgame solving chain within 5s per-action limit
- **Blueprint for flop/turn acting first**: multi-street backward induction captures dynamics that runtime solvers miss
- **Card blocking fix**: critical for all range solver decisions (without it, the nuts checks)
- **Lead protection**: when bankroll > remaining_hands x 1.5 + 10, check/fold everything. Binary ELO means winning by 1 chip = winning by 1000.
- **Weight floor 0.005**: sharper narrowing (13% phantom weight vs 37% at higher floors)
- **Selective caller suppression**: tracks bet-called WR, suppresses thin value when <40%
- **Bluff injection**: half-GTO frequency when opponent folds >50%

---

## Known Limitations and Future Work

### Hero Range Tracking (Biggest Remaining Improvement)
We track the opponent's range but use a UNIFORM hero range for P(bet|hand) computation. Opponents bet differently against our actual (capped) range vs a uniform range. This causes systematic overcalling on river (-42/hand on calls). Implementation was attempted but crashed on edge cases -- needs approximately 20 minutes of debugging to complete.

### CC Showdown Leak
Structural leak in check-check pots where no betting narrows ranges. Both players hold premium hands (5-card selection), so equity is close to 50/50 but pot size asymmetry causes chip loss. Floor override partially addresses this.

### Fundamental Approximation
True GTO derives ranges from equilibrium; we infer ranges from observations. The bot's range estimates are Bayesian approximations of the opponent's actual range, not equilibrium-derived. This is the fundamental gap between our approach and a full Nash equilibrium solver.

### Node-Locked Solver
Attempted for river AF to model non-GTO opponents. Produced reversed behavior (bluffed weak hands, checked strong hands). Needs a different opponent model formulation.

### C Solver Optimizations (Not Implemented)
- float32 arithmetic: 2-2.5x speedup (biggest easy win)
- Reach-based pruning: 1.3-1.5x
- Early termination: 1.3-2x
- Regret-based pruning: 1.3-1.5x
- Combined potential: 25-60x over Python

---

## What Doesn't Work (Exhaustively Tested)

These approaches were tested and confirmed to regress or produce no improvement:

- **Any equity gate margin on flop/turn**: equity too diffuse on early streets
- **Never-call strategy**: catastrophic regression (0/4 wins)
- **Warm-starting DCFR**: cold-start consistently wins
- **Check-behind narrowing**: opponents trap by checking behind with strong hands, so narrowing on check removes hands that are actually in their range
- **Floor override at 0.15**: thin value bleeds against good callers (0.25 threshold works)
- **Node-locked solver for river AF**: reversed strong/weak behavior
- **Overbet sizing hack**: conflicts with full tree solver structure
- **Runtime solver for flop/turn acting first**: CTS equity misses multi-street dynamics (regressed 3 separate times)
- **Turn acting-first solver**: v32 (66.7% WR) -> v33 (40%) after adding it
- **No floors on narrowing**: zeroed-out hands cause catastrophic solver errors
- **Compact tree range solver**: no re-raise threat -> 91% bet frequency -> P(bet|hand) useless for narrowing
- **Heuristic betting bonus on continuation values**: corrupted strategies, inverted strong/weak
- **Multiple changes at once**: always regresses, impossible to isolate which change helped/hurt
- **Agent-written code without manual verification**: contributed to v22 regression

---

## Performance

Competition results as of final submission:
- **v36 open competition: 100% WR (9W 0L as of upload)**
- **v35+v36 combined: ~85% WR**
- **Overall rank: ~#25-30**
- **Scrimmage vs top 5: 2W 3L (40%)**

### Local Benchmarks (100 hands)
- vs CallingStation: +14.8/hand
- vs ProbAgent: +9.5/hand
- vs AllInAgent: +6.3/hand

---

## Runtime Budget

Server constraints: 4 vCPU, 8GB RAM, 1500s total, ARM64 Graviton2, 5s per-action timeout.

| Component | Calls/match | Per call | Total | Budget % |
|-----------|------------|----------|-------|----------|
| Flop blueprint | 500 | 0ms | 0s | 0% |
| Turn blueprint (lazy) | 400 | 2ms | 1s | 0.1% |
| DL turn solver | ~60 | ~2.5s | ~150s | 10% |
| River range solver | ~120 | ~0.7s | ~84s | 5.6% |
| Equity computations | ~1000 | 1ms | 1s | 0.1% |
| **Total** | | | **~236s** | **~16%** |

### Memory

| Component | Compressed | In Memory |
|-----------|-----------|-----------|
| Flop blueprint | 51MB | ~2.1GB |
| Turn boards (LRU cache) | 575MB on disk | ~50MB |
| River strategies | 146MB on disk | ~50MB |
| Preflop data | 2MB | ~50MB |
| Python + engine | -- | ~500MB |
| **Total** | ~775MB | **~2.75GB** |

---

## Version History (Key Versions)

| Version | Key Change | WR |
|---------|-----------|-----|
| v24 | Baseline: blueprints + equity thresholds | 55% |
| v27 | Card blocking + DCFR + full tree river | -- |
| v29 | Runtime solver on ALL streets (regressed) | 33% |
| v32 | Blueprint AF + DL turn FB + river solver | 66.7% |
| v35 | v32 arch + river narrowing + bug fixes | ~70% |
| v36 | C solver + flop DL + equity gate tuning | 100% (9 matches) |
| v37 | River equity gate + adaptive narrowing | ~68% |
| v38 | River precompute + final tuning | Final |

---

## Key Lessons Learned

1. **Multi-street awareness beats range narrowing for early streets.** Blueprint backward induction captures implied odds and reverse implied odds that no single-street solver can match.
2. **Range narrowing beats multi-street awareness for the last street.** On the river, there are no future streets, so accurate opponent range estimation is what matters.
3. **Depth-limited solving gets both.** For facing-bet decisions on middle streets, DL solving with continuation values combines range narrowing with multi-street awareness.
4. **Card blocking is critical.** Without the `not_blocked` mask, impossible hand pairs bias EV by -0.24, causing the nuts to check instead of bet.
5. **Never zero out hands in narrowing.** If the blueprint P(bet|hand) is wrong for even one hand, zeroing it means the solver bets confidently into the "impossible" nuts.
6. **Local tests don't predict competition results.** +9.5/hand vs ProbAgent locally, 33% in competition (v29).
7. **Backtests can't model second-order effects.** Never-call and aggressive margins both regressed despite positive backtests, because opponents adapt their strategy.
8. **The #1 bot is a pure value bettor (0 bluffs), not GTO.** The metagame rewards exploitation-resistant value betting, not game-theoretic balance.
9. **Close margins (+/-50 chips) come from auto-fold lead protection, not coinflips.** Treat as decisive losses.
10. **Never lower thresholds when behind.** Causes a death spiral of increasingly loose play.
