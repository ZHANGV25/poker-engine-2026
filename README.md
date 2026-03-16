# Stockfish -- CMU x Jump Trading Poker AI Tournament 2026

**Team**: Stockfish
**Members**: vortex, prithish_shan
**Competition**: CMU Data Science Club Poker Bot Competition, March 14-22 2026

## The Game

27-card deck (9 ranks x 3 suits). Each player gets 5 hole cards. After preflop betting, both players discard 3 cards (keeping 2), discards revealed to opponent. Then flop (3 cards), turn (1 card), river (1 card) with betting on each street. Max bet 100 per player, 1000 hands per match, binary win/loss ELO.

## Why This Game Is Solvable

Standard Hold'em has 22,100 possible flops and 1,326 possible hands -- too many to solve exactly. This game has **2,925 flops** and **276 hands per board**. Small enough to solve every board individually with no bucketing, no clustering, no approximation of which hands are "similar." Every hand gets its own strategy.

The 27-card deck also means exact equity computation is feasible. After the flop, there are only ~16 unknown cards. Enumerating all possible opponent hands and runouts takes ~22,000 lookups (~2ms). Standard poker needs Monte Carlo sampling with inherent error. We have zero error.

## Strategy Overview

### Preflop (200-bucket GTO tree, 5000 CFR iterations)

Preflop is the one street where we can't solve per-hand -- there's no board yet, so we can't compute exact equity. Instead we bucket the C(27,5) = 80,730 possible 5-card holdings by "potential" (probability of making a strong hand across random flops). 200 buckets, ~400 hands each.

We build a full betting tree with raise levels [2, 4, 8, 16, 30, 60, 100] and solve the 2-player game via CFR with 5000 iterations. The result is a mixed strategy per bucket: fold X%, call Y%, raise-to-Z W%.

Preflop pots are small (2-8 chips typically), so approximation here costs the least EV. A shove cap (potential > 0.88 required for raises > 20) prevents noisy buckets from randomly shoving.

**Fallback**: Pot-odds calculation with margin.

### Discard (exact equity + Bayesian inference)

When the flop is dealt, both players discard 3 and keep 2. Discards are revealed.

For each of the C(5,2) = 10 possible keep-pairs:
- Treat the 3 discards as dead cards
- Enumerate all C(16,2) = 120 possible opponent hands
- For each, enumerate all possible board runouts
- Compute exact win probability via precomputed 7-card hand rank lookup tables

This takes ~22ms total. We pick the keep-pair with the highest equity.

**Bayesian inference**: When the opponent discards 3 cards, we reconstruct what their original 5-card hand must have been for each of the 120 possible kept pairs. We evaluate all 10 of their possible keep-pairs and assign a Boltzmann weight based on how close their actual keep was to optimal. This narrows the effective opponent range from 120 hands to ~20-40 realistic hands. Every subsequent decision is more accurate.

**SB advantage**: SB sees BB's discards before choosing their own keep-pair, enabling weighted equity against the inferred opponent range.

### Flop (multi-street backward induction, position-aware)

This is the core of the strategy. A hand's value on the flop depends on what happens on the turn and river. A flush draw with 45% current equity isn't worth 45% of the pot -- it depends on whether the draw completes, how betting goes on future streets, and whether weak hands can profitably fold when facing bets.

**Backward induction** solves this:

1. **River**: For each possible 5-card board, solve the full betting game (4 bet sizes: 40/70/100/150% pot, max 2 raises). Hand ranks are deterministic. 50 CFR iterations. This produces each hand's EV *accounting for the betting game* -- weak hands that face bets and fold get lower EV than raw equity suggests.

2. **Turn**: For each possible 4-card board, the "showdown" terminals are transitions to the river. The value at each terminal is the average of river EVs over all possible river cards. 100 CFR iterations. Each hand's turn EV now accounts for both turn AND river play.

3. **Flop**: Same chain -- showdown terminals use average turn EVs over all possible turn cards. 500 CFR iterations. The flop strategy reflects the complete game: a flush draw gets high value because the solver "knows" it completes ~35% of the time and plays profitably on future streets.

Solved for:
- ALL 2,925 flop boards individually (no clustering)
- ALL 276 hands per board (no bucketing)
- 7 pot sizes: (2,2), (4,4), (8,8), (16,16), (30,30), (50,50), (100,100)
- Both P0 (first-to-act) and P1 (second-to-act) strategies (position-aware)

**At runtime**: pure table lookup, ~5ms. No computation.

### Turn (backward induction, 1 pot size, position-aware)

Turn strategies are a byproduct of the flop solve -- computed during Phase 2 of backward induction. Saved for 1 pot size (4,4), all other pots rounded to it. Every turn card covered (24 per board), every hand individual (253 per turn board). Both P0 and P1 strategies saved.

**At runtime**: pure lookup, ~5ms. The strategy knows about the river because it was solved with river CFR continuation values.

### River (range solver, real-time)

The river is solved at runtime instead of using a precomputed blueprint. After a hand of betting and discard inference, we've narrowed the opponent's range from 120 hands to ~20-40. A blueprint computed against the full range is suboptimal against a narrowed range.

The range solver solves for ALL possible hero hands simultaneously against the narrowed opponent range. This produces **range-balanced** strategies -- betting frequencies across all hands are consistent, so the opponent can't exploit our river play.

100 iterations, ~555ms on ARM64. River equity is deterministic (board complete), so convergence is fast and the solution is near-optimal.

### Lead Protection

When `bankroll > remaining_hands * 1.5 + 10`, check/fold everything. Binary ELO means winning by 1 chip = winning by 1000 chips. If we can survive on blinds alone, there's no reason to risk any pot.

### Decision Cascade

At each postflop decision:
1. Multi-street flop blueprint (~5ms, best quality)
2. Multi-street turn blueprint (~5ms, backward induction)
3. River range solver (~555ms, adapts to narrowed range)
4. Single-street blueprint fallback
5. One-hand solver fallback
6. Check/fold emergency

Layers 4-6 should never fire with complete v7.1 data.

## Precomputed Data

| Data | Description | Size |
|------|-------------|------|
| `hand_ranks.npz` | All C(27,7) seven-card + C(27,5) five-card hand ranks | 1.6 MB |
| `preflop_potential.npz` | Expected win rate for all 80,730 five-card hands | 326 KB |
| `preflop_strategy.npz` + `preflop_tree.pkl` | 200-bucket preflop GTO (290 nodes, 7 raise levels) | ~65 KB |
| `multi_street/board_*.npz` | 2,925 per-board files: flop strategies (7 pots, P0+P1), turn strategies (1 pot, P0+P1), action types | ~3 GB |
| `hand_ranks.npz` lookup tables | Precomputed 7-card and 5-card hand rankings | 1.6 MB |

Total in RAM at runtime: ~3 GB (fits Phase 2's 4 GB, Phase 3's 8 GB).

## Compute Infrastructure

All blueprint strategies computed on AWS EC2:
- **Multi-street v7.1**: 10 x c5.9xlarge (36 vCPU each), 8 workers per instance, ~1.7 hours
- **Preflop v6**: 1 x c5.4xlarge (16 vCPU), ~2 hours
- S3 bucket: `s3://poker-blueprint-2026/`

## Project Structure

```
submission/                          # Uploaded to competition server
+-- player.py                       # Main bot: PlayerAgent class
+-- solver.py                       # Real-time one-hand CFR+ solver (fallback)
+-- range_solver.py                 # Range-based river solver (all hero hands)
+-- game_tree.py                    # Betting tree (4 sizes, max 2 raises/street)
+-- equity.py                       # Exact equity engine (precomputed lookups)
+-- inference.py                    # Bayesian discard inference
+-- multi_street_lookup.py          # Runtime flop+turn blueprint lookup
+-- blueprint_lookup_unbucketed.py  # Single-street blueprint lookup (fallback)
+-- precompute_preflop_strategy.py  # Offline: preflop GTO via CFR
+-- data/                           # Precomputed lookup tables + blueprints

blueprint/                          # Offline computation (NOT uploaded)
+-- multi_street_solver.py          # Backward induction solver (river->turn->flop)
+-- compute_multi_street.py         # Distributed multi-street computation
+-- abstraction.py                  # Board features for matching
+-- package_code.sh                 # Package and upload code to S3
+-- ec2_launch_fleet.sh             # Launch EC2 compute fleet
+-- ec2_monitor.sh                  # Monitor fleet progress
+-- ec2_merge_and_deploy.sh         # Download and deploy results

test_path_coverage.py               # Exhaustive verification suite (12 tests)
```

## Running

```bash
python agent_test.py                 # Quick validation
python run.py                        # Full match vs configured opponent
python test_path_coverage.py         # Exhaustive strategy verification
python test_path_coverage.py --quick # Quick verification (~2 min)
```

### Regenerating Data

```bash
# Local
python -m submission.precompute                     # Hand rank tables (~2 min)
python -m submission.precompute_preflop             # Hand potentials (~20 min)
python -m submission.precompute_preflop_strategy    # Preflop GTO (~2 hrs)

# EC2 (multi-street, requires Numba)
bash blueprint/package_code.sh                      # Upload code to S3
bash blueprint/ec2_launch_fleet.sh                  # Launch 10x c5.9xlarge
bash blueprint/ec2_monitor.sh                       # Monitor progress
bash blueprint/ec2_merge_and_deploy.sh              # Download + deploy
```

## Known Approximations (after v7.1)

1. **Turn pot rounding**: 1 pot (4,4) for all turn situations. Wrong SPR at large pots.
2. **500 CFR iterations**: ~3-5% exploitability per decision point.
3. **4 bet sizes, max 2 raises**: Missing small bet (25%), can't handle 3-bet+ sequences.
4. **Preflop bucketing**: 200 buckets = ~2 hands per bucket share strategies.
5. **Range narrowing heuristic**: Top-N% by hand strength, not polarized ranges.
6. **Linear continuation value scaling**: Approximate at bet-call terminal pots.
