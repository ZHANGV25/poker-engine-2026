# Poker Bot — Submission

## Architecture

```
submission/
├── player.py          # Main bot (PlayerAgent class)
├── equity.py          # Exact equity engine (full enumeration, zero MC error)
├── inference.py       # Bayesian discard inference (opponent range narrowing)
├── opponent.py        # Within-match opponent modeling (1000 hands)
├── precompute.py      # Offline script to generate lookup tables
└── data/
    └── hand_ranks.npz # Precomputed 7-card and 5-card hand rank tables (1.6MB)
```

## How It Works

### Core Edge: Exact Enumeration

The 27-card deck is small enough to compute equity exactly. After discards on the
flop, there are only C(16,2)×C(14,2) = 10,920 possible (opponent hand, runout)
scenarios. We evaluate all of them via precomputed lookup tables (~50ns per lookup).

Result: perfect equity in ~5ms vs the reference bot's Monte Carlo with ~2.5% error.

### Decision Flow

1. **Pre-flop**: Heuristic hand strength (pairs, aces, suited, connected). Almost
   always calls from SB. Raises strong hands.

2. **Discard**: Evaluates all 10 keep-pairs by exact equity (~25ms). As SB, uses
   Bayesian inference on opponent's revealed discards to weight equity calculation.

3. **Post-flop**: Exact equity vs pot odds. Tiered raise sizing (50-100% pot).
   Bluffs at ~12% frequency against opponents who fold enough.

### Opponent Modeling

Tracks fold/raise/call rates, aggression, VPIP across all 1000 hands. After 50
hands, adjusts thresholds to exploit detected patterns. Lead-aware play tightens
when ahead and loosens when behind in the last 300 hands.

## Regenerating Lookup Tables

If the game rules change, regenerate the hand rank tables:

```bash
python -m submission.precompute
```

This takes ~2 minutes and produces `submission/data/hand_ranks.npz`.

## Performance

- **Init**: ~0.4s (load lookup tables)
- **Discard decision**: ~25ms (10 keep-pairs × exact equity)
- **Flop equity**: ~5ms
- **Turn equity**: ~0.3ms
- **River equity**: ~0.02ms
- **Average per hand**: ~50ms (10% of Phase 1 budget)
