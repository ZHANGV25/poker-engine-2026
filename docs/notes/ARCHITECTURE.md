# Stockfish Poker Bot — Architecture

## Overview

CFR-based poker bot for CMU AI Poker Competition 2026 (27-card variant).
Precomputed blueprints for flop/turn + real-time solving on river.

## Data Pipeline

### Precomputed (offline, EC2)
- **Flop blueprint** (51MB, 4-bit LZMA): 2925 boards × 5 pots × 276 hands × 21 nodes × 6 actions. 200 iterations backward induction (river→turn→flop). Hero + opponent strategies.
- **Turn data** (575MB, per-board lazy-load): 2925 boards × 5 pots × ~24 turn cards. Hero strategies (4-bit, 16 nodes) + opponent P(bet|hand) for Bayesian narrowing. Loaded on demand (~2ms per board).

### Real-time (runtime)
- **River range solver**: Vectorized CFR, compact tree (2 sizes, 1 raise), float32. 1000-3000 iterations in ~0.7s ARM.
- **River narrowing solve**: Solves from opponent's perspective to get exact GTO P(bet|hand). Equity-weighted hero range + iterative refinement. ~1.6s ARM.

## Decision Flow

```
Preflop → Precomputed GTO tree (200 buckets, 5000 iters)
Flop    → Blueprint lookup (0ms)
Turn    → Blueprint lookup from lazy-loaded per-board files (2ms)
River   → Equity thresholds acting first (bets 20-25%, includes bluffs)
        → Range solver facing bets (1000-3000 iters, coordinated call/fold)
```

## Range Narrowing Pipeline

Every opponent action narrows their estimated range:

| Street | Bet | Check | Street Transition |
|--------|-----|-------|-------------------|
| Flop | Bayesian from blueprint P(bet\|hand) with trust blending | Bayesian P(check\|hand) from blueprint | MDF on previous board |
| Turn | Bayesian from turn opp data P(bet\|hand) with trust blending | Bayesian P(check\|hand) from turn data | MDF on previous board |
| River | **Solved GTO P(bet\|hand)** from range CFR (equity-weighted hero range, iterative refinement) | — (last street) | MDF on previous board |

### Trust Blending
Against non-GTO opponents, blueprint trust is reduced:
```
α = 1 - |f_tracked - f_blueprint| / max(f_tracked, f_blueprint)
P(bet|hand) = α × P_blueprint + (1-α) × f_tracked
```

### Floors
All narrowing uses soft floors (weight *= 0.05 minimum, never zeroed). Consistent with GTO Bayesian inference.

## Adaptive Exploit Layer

On top of GTO base strategy:
- **Bluff suppression**: tracks opponent call rate. Fold <35% → never bluff. 35-50% → bluff rarely.
- **Exploit bluffing**: tracks opponent fold rate. Fold >50% → bluff weak hands at scaled frequency.
- Both respect pot control (max 2 streets raising).

## Compute Budget (1500s ARM)

| Component | Calls/match | Per call | Total | Budget % |
|-----------|------------|----------|-------|----------|
| Flop blueprint | 500 | 0ms | 0s | 0% |
| Turn blueprint (lazy) | 400 | 2ms | 1s | 0.1% |
| River range solver | 60 | 0.7s | 42s | 2.8% |
| River narrowing solve | 60 | 1.6s | 96s | 6.4% |
| Equity computations | ~1000 | 1ms | 1s | 0.1% |
| **Total** | | | **~140s** | **9.3%** |

## Memory (8GB ARM limit)

| Component | Compressed | In Memory |
|-----------|-----------|-----------|
| Flop blueprint | 51MB | ~2.1GB |
| Turn boards (cached) | 575MB on disk | ~50MB (50 board cache) |
| Preflop data | 2MB | ~50MB |
| Python + engine | — | ~500MB |
| **Total** | 628MB | **~2.7GB** |

## Key Files

- `submission/player.py` — Main bot logic, narrowing, decision flow
- `submission/solver.py` — One-hand CFR solver
- `submission/range_solver.py` — Range-balanced CFR solver (vectorized, float32, compact tree)
- `submission/multi_street_lookup.py` — Blueprint lookup + lazy turn loading
- `submission/equity.py` — Exact equity engine (precomputed hand rank tables)
- `submission/inference.py` — Boltzmann discard inference
- `submission/game_tree.py` — Betting tree builder (compact mode for range solver)
- `blueprint/compute_multi_street.py` — EC2 offline computation
- `blueprint/merge_turn_minimal.py` — Turn opp data extraction
- `blueprint/merge_turn_hero.py` — Turn hero data extraction
