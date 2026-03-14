# Poker Bot — Goofy Goobers

## Strategy: GTO-First, No Exploitation

This bot plays solid, unexploitable poker. It does not try to read or exploit
opponent tendencies. Instead, it computes exact equity, makes mathematically
correct decisions, and lets opponents beat themselves by making mistakes.

Why no exploitation: ELO and round-robin scoring are binary (win/loss).
Winning by 1 chip = winning by 10,000 chips. A strategy that wins 75% of
matches by small margins beats a strategy that wins 60% by large margins.
Exploitation can backfire (e.g., raising aggressively against a trapper),
turning wins into losses. GTO never loses in expectation.

## Architecture

```
submission/
├── player.py              # Main bot (PlayerAgent class)
├── equity.py              # Exact equity engine via precomputed lookup tables
├── inference.py           # Bayesian discard inference (range narrowing)
├── opponent.py            # Opponent stat tracking (data collection only)
├── precompute.py          # Offline: generate 7-card and 5-card hand rank tables
├── precompute_preflop.py  # Offline: generate pre-flop hand potential table
└── data/
    ├── hand_ranks.npz     # 7-card (888K entries) + 5-card (80K) rank tables (1.6MB)
    └── preflop_potential.npz  # Expected post-discard equity for all 5-card hands (302KB)
```

## Every Feature Explained

### 1. Exact Equity Engine (`equity.py`)

**What it does:** Computes the exact probability of winning a hand by
enumerating every possible outcome — every possible opponent hand combined
with every possible remaining board card.

**Why it matters:** The reference ProbabilityAgent samples 400 random
outcomes (Monte Carlo) and estimates equity with ~2.5% error. We enumerate
ALL outcomes with zero error. When equity is 65%, we know it's 65.000% — not
62% or 68%. Every bet/call/fold decision is based on perfect information.

**How it works:**
1. At init, loads two precomputed lookup tables from `data/hand_ranks.npz`:
   - 7-card table: all C(27,7) = 888,030 combinations of (2 hole + 5 board)
     mapped to their hand rank (lower = better). Stored as a Python dict for
     O(1) lookup (~50ns each).
   - 5-card table: all C(27,5) = 80,730 combinations, used for discard
     inference heuristics.

2. To compute equity after discards on the flop (worst case):
   - Known cards: 11 (our 2 kept + 3 discarded + opponent's 3 discarded + 3 board)
   - Unknown: 16 cards
   - Opponent could hold any C(16,2) = 120 two-card hands
   - For each opponent hand, the turn+river could be any C(14,2) = 91 combo
   - Total: 120 × 91 = 10,920 scenarios
   - For each: look up our 7-card hand rank and opponent's 7-card hand rank
   - Count wins, ties, losses → exact equity

3. Performance:
   - Flop equity: ~5ms (10,920 × 2 lookups)
   - Turn equity: ~0.3ms (1,365 × 2 lookups)
   - River equity: ~0.02ms (91 × 2 lookups)

**Why precompute instead of calling treys at runtime:** The `treys` library
evaluator takes ~5-10μs per call. With 218K lookups for a discard decision,
that's ~2 seconds — too slow. Dict lookups take ~50ns each, making it 100x
faster.

### 2. Discard Inference (`inference.py`)

**What it does:** When the opponent discards 3 cards (which are always
revealed), computes a probability distribution over their possible kept hands.

**Why it matters:** Without inference, we treat all 120 possible opponent
hands as equally likely. With inference, we weight each hand by how rational
it would be to keep it, narrowing the effective range to ~20-40 hands. This
makes every subsequent equity calculation more accurate.

**How it works:**
1. For each of the ~120 possible 2-card hands the opponent could hold:
   - Reconstruct their hypothetical original 5-card hand:
     {candidate_pair + their 3 discards}
   - Evaluate all 10 possible keep-pairs from that 5-card hand using the
     5-card rank table (how strong is each keep on the current board?)
   - Check: was the candidate pair the best or near-best keep?

2. Assign Boltzmann weights: `weight = exp(-delta / temperature)` where
   delta = how much worse this keep is compared to the optimal keep.
   - If candidate pair was the obvious best keep → weight ≈ 1.0
   - If candidate pair was a terrible keep → weight ≈ 0.0
   - Temperature controls how "rational" we assume the opponent is.
     Default 5.0 allows for some imperfection.

3. Normalize weights to sum to 1.0. Pass to equity engine.

**Performance:** 120 candidates × 10 keep-pairs × 1 lookup = 1,200 lookups.
Cost: <0.5ms. Negligible.

**When it fires:**
- As SB on the flop: we see BB's discards before choosing our own discard.
  Inference runs on their discards, and we use weighted equity to pick our
  best keep-pair. This gives SB an information edge at the discard.
- After both players discard: inference weights are used for all subsequent
  equity computations in the hand.

### 3. Action-Based Range Narrowing (`player.py: _narrow_range_by_action`)

**What it does:** Updates the opponent's weighted range each time they take
a betting action. When they call or raise, filter out hands from their range
that wouldn't take that action.

**Why it matters:** This was the biggest missing feature in earlier versions.
Without it, we'd compute 70% equity against the opponent's full range, raise
aggressively, and lose 100 chips when they had us beat. With range narrowing,
after an opponent calls our raise, their range is filtered to the top 70% of
hands (the ones strong enough to call). Our equity against this narrowed range
might be only 50% — so we check instead of raising again.

**How it works:**
1. When we see `opp_last_action == "RAISE"` or `"CALL"` in the observation:
2. For each hand in opponent's weighted range, compute its strength (5-card
   rank on the current board)
3. Sort hands by strength
4. Zero out the weakest portion:
   - After a RAISE: keep only top 40% (raises signal strength)
   - After a CALL: keep only top 70% (calls signal moderate strength)
   - After a CHECK: no change (checks are consistent with any hand)
5. Renormalize weights
6. All subsequent equity computations use the narrowed range

**This is not exploitation — it's Bayesian inference.** The opponent's action
is evidence about their hand. Incorporating that evidence is mathematically
correct regardless of who the opponent is.

### 4. Pre-Flop Hand Potential Table (`precompute_preflop.py` + `data/preflop_potential.npz`)

**What it does:** Maps every possible 5-card starting hand to its expected
post-discard equity across all possible flops.

**Why it matters:** Pre-flop, we have 5 cards but no board. We can't compute
exact equity without knowing the flop. The potential table answers: "across
all 1,540 possible flops, if I play optimally (keep the best pair each time),
what's my average equity?" This replaces a naive heuristic (count pairs and
aces) with computed values.

**How it works (offline, ~8 minutes):**
1. Enumerate all C(27,5) = 80,730 five-card hands
2. Use suit isomorphism to reduce to ~63,846 canonical hands (3 suits are
   interchangeable pre-flop)
3. For each canonical hand:
   - Enumerate all C(22,3) = 1,540 possible flops
   - For each flop, evaluate all 10 keep-pairs using the 5-card rank table
   - Take the best keep-pair's rank (optimal discard)
   - Average across all flops → hand potential score
4. Map all 80,730 hands to their canonical potential, save as lookup table

**At runtime:** Single dict lookup per pre-flop decision. 0ms cost.

**Potential range:** 0.37 (worst hand) to 0.65 (best hand). Normalized to
0-10 scale for decision thresholds.

### 5. Pot Control (`player.py: _streets_raised tracking`)

**What it does:** Limits the number of streets where we raise to prevent
building huge pots with medium-strength hands.

**Why it matters:** In our first loss (-346 chips), 58 hands lost us 50-100
chips each. The pattern: we raised on flop, turn, AND river with two pair or
trips, building the pot to 100 chips each. Opponent just called with a better
hand every time. Those 58 hands cost us 4,900 chips.

**How it works:**
- Track `_streets_raised`: how many different streets we've raised on this hand
- Maximum 2 streets of raising unless equity > 92% (near-nuts)
- On the 3rd street, we check or call instead of raising
- This caps our maximum loss on medium hands to ~40-60 chips instead of 100

**Example:**
- We have trips. Equity vs full range: 75%.
- Flop: we raise (street 1 of raising). Good.
- Turn: we raise (street 2 of raising). Good.
- River: pot control kicks in. We CHECK instead of raising a 3rd time.
- If opponent bets, we call (equity still justifies it).
- If opponent has us beat, we lose ~50 chips instead of 100.

### 6. Re-Raise Protection (`player.py: opp re-raised check`)

**What it does:** When opponent re-raises after we already raised on a street,
require near-nuts equity (>92%) to continue raising. Otherwise just call.

**Why it matters:** A re-raise is one of the strongest signals in poker. When
an opponent raises our raise, they're representing a very strong hand. Our
equity against their full range might be 70%, but against hands that would
re-raise, it's probably 30%.

**How it works:**
- If `opp_last_action == "RAISE"` and we already raised this street:
  - Equity > 92%: raise back (we likely have the nuts)
  - Equity >= pot odds: just call (decent hand but not nuts)
  - Otherwise: fold

### 7. Check-Raising (`player.py: check-raise logic`)

**What it does:** With a strong hand as first to act, sometimes check instead
of betting, hoping the opponent bets so we can raise.

**Why it matters:** Always betting strong hands is predictable. If we bet,
opponent folds. If we check, opponent might bet (thinking we're weak), and
then we raise — trapping them for more chips.

**How it works:**
- When we're first to act (no bet to respond to)
- AND equity > 82% (strong hand)
- AND random < 30% (do it 30% of the time for balance)
- → CHECK instead of raising. If opponent bets, we'll raise on our next action.

The 30% frequency prevents us from being exploitable. If we always check-raised
strong hands, opponents would learn to check behind us.

### 8. GTO-Balanced Bluffing (`player.py: bluff section`)

**What it does:** Bluffs at a mathematically determined frequency based on
bet sizing, not opponent behavior.

**Why it matters:** In game theory, the correct bluff frequency makes the
opponent indifferent between calling and folding. If we bluff too much, they
profit by always calling. If we never bluff, they profit by always folding
to our bets. The GTO frequency is the one they can't exploit.

**How it works:**
- GTO bluff frequency = bet_size / (bet_size + pot_size)
- For a 60% pot bet: bluff_freq = 0.6P / 1.6P = 37.5%
- Scale down by street: 20% of theoretical on flop, 40% on turn, 100% on river
- Further scaled by 0.5 for conservatism
- Only bluff with equity < 25% (clear bluff candidates)
- Subject to pot control (won't bluff if already raised 2 streets)

### 9. Smart Discard Selection (`player.py: _handle_discard`)

**What it does:** Evaluates all 10 possible keep-pairs (C(5,2)=10 ways to
keep 2 of 5 cards) by exact equity and picks the best one.

**Why it matters:** The discard is the most impactful decision in this game
variant. Keeping the right 2 cards vs the wrong 2 cards can swing equity by
20-30%. Most bots either keep the first two cards or use Monte Carlo.

**How it works:**
- For each of 10 keep-pairs:
  - The 3 cards not kept become dead cards
  - Compute exact equity with the 2 kept cards against all possible opponent
    hands and board runouts
- Pick the keep-pair with the highest equity
- As SB: uses discard-inference-weighted equity (accounts for what opponent
  likely kept based on their revealed discards)
- As BB: uses uniform equity (opponent hasn't discarded yet)

**Performance:** 10 × ~10,920 lookups = ~25ms total.

### 10. Pre-Flop Strategy (`player.py: _handle_preflop`)

**What it does:** Makes pre-flop raise/call/fold decisions using the
precomputed hand potential table.

**How it works:**
- Look up hand potential (expected post-discard equity across all flops)
- Normalize to 0-10 strength scale
- Decisions:
  - Strength ≥ 7: raise (top ~20% of hands)
  - SB facing 1 chip: always call (pot odds too good to fold)
  - Facing a raise, strength ≥ 3: call
  - Big raise, strength < 2: fold
  - BB with no raise, strength ≥ 8: raise
  - Otherwise: check or call

**Why we almost always call from SB:** SB needs 1 chip to see a pot of 3 =
33% pot odds. Even the worst 5-card hand has ~37% potential after optimal
discard. Folding pre-flop from SB is almost never mathematically correct.

## Regenerating Lookup Tables

If the game rules change, regenerate the tables:

```bash
# Hand rank tables (~2 minutes)
python -m submission.precompute

# Pre-flop hand potential (~8 minutes)
python -m submission.precompute_preflop
```

## Performance

| Operation | Time | Notes |
|---|---|---|
| Init (load tables) | ~0.4s | Once at match start |
| Pre-flop decision | ~0ms | Dict lookup |
| Discard decision | ~25ms | 10 keep-pairs × exact equity |
| Discard inference | <0.5ms | 1,200 lookups |
| Range narrowing | <0.1ms | Filter + renormalize |
| Flop equity | ~5ms | 10,920 scenarios |
| Turn equity | ~0.3ms | 1,365 scenarios |
| River equity | ~0.02ms | 91 scenarios |
| **Average per hand** | **~50ms** | **10% of Phase 1 budget** |

## Testing

```bash
# Run submission validator (4 test bots)
python agent_test.py

# Run full 1000-hand match vs ProbabilityAgent
python run.py

# Run against a different bot (edit agent_config.json)
```

## Match Results

| Opponent | Result | Margin |
|---|---|---|
| FoldAgent | Win | +8 per hand |
| CallingStationAgent | Win | +24 per hand |
| AllInAgent | Win | +124 per hand |
| RandomAgent | Win | +162 per hand |
| ProbabilityAgent | Win | +7,363 over 1000 hands |
| RL Agent (trained) | Win (timeout) | +9,067 at timeout |
