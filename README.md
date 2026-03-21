# Stockfish — CMU x Jump Trading Poker AI Tournament 2026

**Team**: Stockfish
**Members**: vortex, prithish_shan
**Competition**: CMU Data Science Club Poker Bot Competition, March 14-22 2026

## The Game

27-card deck (9 ranks × 3 suits). Each player gets 5 hole cards. After preflop betting, both players discard 3 cards (keeping 2), discards revealed to opponent. Then flop (3 cards), turn (1 card), river (1 card) with betting on each street. Max bet 100 per player, 1000 hands per match, binary win/loss ELO.

## Architecture (v36)

```
Preflop:   200-bucket GTO strategy (precomputed, 5000 CFR iterations)
Discard:   Exact equity evaluation (all 10 keep-pairs) + Bayesian inference
Flop:      Backward-induction blueprint (all decisions) + equity gate on calls
Turn AF:   Backward-induction blueprint (acting first)
Turn FB:   Depth-limited solver (facing bet: CTS continuation values + narrowed range)
River:     DCFR range solver (all decisions, full tree, card blocking, polarized narrowing)
```

### Decision Flow

**Flop** — Blueprint for all decisions. When the blueprint says CALL, an equity gate checks our equity vs the narrowed opponent range. If equity < pot odds, override to FOLD. This catches overcalls against value-heavy opponents that the blueprint (solved against uniform ranges) can't detect. Tested: +1097 chips across 1201 match decisions.

**Turn acting first** — Blueprint lookup. The blueprint's backward induction accounts for river betting dynamics (implied odds, reverse implied odds). Runtime solving with CTS equity doesn't capture this and regressed from +6.4 to 0.0/hand in testing.

**Turn facing bet** — Depth-limited solver. When opponent bets, their narrowed range (from discard inference + flop betting) matters more than multi-street dynamics. CTS continuation values at showdown terminals + DCFR against narrowed range. Gadget safety check prevents worse-than-blueprint decisions.

**River** — DCFR range solver for ALL decisions. Full tree (4 bet sizes, 2 raises max) includes re-raise threat. When facing a bet, polarized narrowing applied first: opponent's betting range = top 30% value + bottom 10% bluffs, middle downweighted. This is not double-counting — the solver tree starts after the bet decision. Tested: +552 chips over no-narrow across 80 match decisions.

### Key Components

**Card blocking** — The range solver's terminal values use a `not_blocked` mask. Without it, impossible hand pairs (sharing cards) contribute -pot at showdown, biasing strong hands' EV by -0.24 and causing the nuts to check instead of bet.

**DCFR** — Noam Brown's parameters (alpha=1.5, beta=0, gamma=2). Converges faster than CFR+ for average strategy by discounting noisy early iterations.

**Vectorized equity** — numpy broadcasting replaces nested Python loops. Turn equity: 3ms (was 215ms, 69× speedup). Flop equity: 28ms (was 800ms, 29× speedup).

**Narrowing** — Bayesian P(bet|hand) from blueprint on flop/turn. Bayesian P(call|hand) from blueprint at flop→turn transition. Runtime compute_opp_call_probs at turn→river. 2% floors on all narrowing paths.

**Lead protection** — When bankroll > remaining_hands × 1.5 + 10, check/fold everything. Binary ELO means winning by 1 chip = winning by 1000.

## Precomputed Data

| Data | Size |
|------|------|
| `hand_ranks.npz` — 7-card and 5-card hand rank lookup tables | 1.6 MB |
| `preflop_potential.npz` — expected win rates for all 80,730 hands | 326 KB |
| `preflop_strategy.npz` + `preflop_tree.pkl` — 200-bucket preflop GTO | ~65 KB |
| `multi_street/blueprint.pkl.lzma` — flop strategies (2925 boards, 5 pots, P0+P1) | 51 MB |
| `multi_street/turn_boards/` — per-board turn strategies (2925 files) | 575 MB |

Total submission: ~628 MB (under 1 GB limit).

## Compute Infrastructure

Blueprint strategies computed on AWS EC2:
- **Multi-street**: 10 × c5.9xlarge (36 vCPU each), 8 workers/instance, ~1.7 hours
- **Preflop**: 1 × c5.4xlarge (16 vCPU), ~2 hours
- S3: `s3://poker-blueprint-2026/`

## Project Structure

```
submission/                          # Uploaded to competition server
├── player.py                       # Main bot: PlayerAgent class
├── solver.py                       # One-hand CFR+ solver (fallback)
├── range_solver.py                 # DCFR range solver (river, full tree, card blocking)
├── depth_limited_solver.py         # Turn facing-bet solver (continuation values + gadget)
├── game_tree.py                    # Betting tree (4 sizes, max 2 raises)
├── equity.py                       # Exact equity engine (vectorized lookups)
├── inference.py                    # Bayesian discard inference
├── multi_street_lookup.py          # Runtime flop+turn blueprint lookup
└── data/                           # Precomputed data

blueprint/                          # Offline computation (NOT uploaded)
├── multi_street_solver.py          # Backward induction solver
├── compute_multi_street.py         # Distributed computation
├── compute_turn_values.py          # Turn game value precomputation
├── ec2_launch_fleet.sh             # Launch EC2 fleet
└── ec2_merge_and_deploy.sh         # Download and deploy results

verify_*.py                         # Verification scripts for testing changes
analyze_match.py                    # Match analysis from CSV logs
test_local_matches.py               # Local head-to-head testing
```

## Performance

**v36 open competition: 100% WR (9W 0L as of upload)**
**v35+v36 combined: ~85% WR**
**Scrimmage vs top 5: 2W 3L (40%)**

## What Works and What Doesn't

### Works
- Blueprint for flop/turn acting first (multi-street dynamics > range narrowing)
- Range solver for river (narrowed range > multi-street, no future streets)
- Depth-limited turn solver for facing bets (narrowed range helps for reactive decisions)
- Equity gate on flop/turn calls (+1097 chips on 1201 decisions)
- Polarized river bet-narrowing (+552 chips on 80 decisions)
- Card blocking fix (critical for all solver decisions)

### Doesn't Work
- Runtime solver replacing blueprint for acting first (CTS equity loses multi-street dynamics, regressed 3 times)
- Warm-start solver for flop acting first (tested -41 to -243 chips, blueprint is better)
- No floors on narrowing (opponents deviate from GTO, zeroed-out hands cause catastrophic errors)
- Opponent modeling based on showdown WR (10 samples too noisy to classify)
- Heuristic betting bonus on continuation values (corrupted strategies, inverted strong/weak)
