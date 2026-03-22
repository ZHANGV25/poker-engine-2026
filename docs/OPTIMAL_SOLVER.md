# Optimal Solver Design for 27-Card Poker

## The Goal
A solver that plays true Nash equilibrium in real-time against any opponent, with exact range tracking and full multi-street planning.

---

## Current Architecture vs Optimal

### Range Tracking (THE critical gap)

**Current:** Infer opponent range from observations (discards, bets). Use UNIFORM hero range for P(bet|hand) computation. Both are approximations.

**Optimal:** Track BOTH players' ranges through the entire hand, derived from equilibrium strategies. At every decision node:
- Hero range = set of hands hero would reach this node with, given their strategy
- Opponent range = set of hands opponent would reach this node with, given their strategy
- Both ranges are mathematical consequences of the equilibrium, not inferences

**Why it matters:** When we compute P(bet|hand) with uniform hero range, we underestimate how wide opponents bet against our actual (capped) range. This causes systematic overcalling on the river (-42/hand on calls). Every downstream decision inherits this error.

**Implementation:**
- After each of our actions, narrow our own range based on what we just did (if we checked, remove hands we'd bet with; if we bet, remove hands we'd check with)
- Pass our actual range to `compute_opp_bet_probs` instead of uniform
- This requires tracking P(action|hand) for our own strategy, which the solver already computes

---

### Solving Approach

**Current:** Subgame solving — solve each street independently against inferred ranges. Continuation values from child subgames feed into parent.

**Optimal (Libratus approach):**
1. **Offline:** Solve the full game tree with abstraction → blueprint
2. **Online:** At each decision point, re-solve the subgame with:
   - Exact ranges derived from blueprint play up to this point
   - Gadget game ensuring re-solve ≥ blueprint value
   - Action translation for bet sizes not in the abstract tree

**Key differences:**
- Blueprint ranges are EXACT (derived from equilibrium strategy)
- Re-solving only IMPROVES over blueprint (gadget guarantee)
- Action translation handles continuous bet sizing
- Both players' strategies are consistent across streets

---

### Continuation Values

**Current:**
```
River: equity → river solver → game value ✓
Turn:  river game value → turn solver → game value ✓
Flop:  turn game value → flop solver → strategy ✓
```

This is correct in structure but the game values come from `_last_root_value` which is the LAST iteration's root value. DCFR converges in average, not point-wise — last-iteration values can be noisy.

**Optimal:**
- Use AVERAGE game values (averaged over all iterations, weighted by DCFR discounting)
- Or: use the average strategy to compute game values via one final forward pass
- The game value should be the expected payoff under the AVERAGE strategy, not the last-iteration strategy

---

### Tree Abstraction

**Current:** 4 bet sizes (40/70/100/150% pot), 2 raises max per street. Full tree = 123 nodes, lean = 39 nodes.

**Optimal:**
- Continuous bet sizing via action translation
- When opponent bets a size not in our tree (e.g., 55% pot), translate to the closest abstract actions and interpolate the response
- Finer abstraction where it matters most (river AF has the most impact)
- Board-texture-dependent tree: more sizes on connected/suited boards, fewer on dry boards

---

### Iteration Count & Convergence

**Current:** 300 iterations, DCFR with α=1.5, β=0, γ=2. Strategy converges quickly but exploitability gap is nonzero.

**Optimal:**
- Early termination: check exploitability every 50 iterations, stop when converged
- Regret-based pruning (RBP): after iteration 100, skip actions with very negative regret. Saves 30-50% compute.
- Reach-based pruning: skip hands with near-zero reach probability in deep nodes
- More iterations on high-stakes decisions (big pots), fewer on small pots

---

### Opponent Modeling

**Current:** None beyond range narrowing. We play the same strategy against every opponent.

**Optimal (Pluribus approach):**
- Maintain multiple opponent models (aggressive, passive, balanced, trapping)
- Detect opponent type within first 50-100 hands
- Solve against the detected model, not generic GTO
- Continuously update model as match progresses

**What this enables:**
- Against calling stations: bet only value (no bluffs they'll call)
- Against aggressors: trap more, call down lighter
- Against passive players: bet wider (they fold too much)
- Against GTO players: play GTO (can't exploit, can't be exploited)

---

### Performance Target

**Current:** C solver at 4.8x speedup over Python. Flop DL solve = 870ms.

**Optimal (from research agent analysis):**

| Optimization | Speedup | Cumulative |
|---|---|---|
| float32 | 2.5x | 2.5x |
| Reach pruning | 1.5x | 3.8x |
| Early termination | 1.5x | 5.7x |
| Regret-based pruning | 1.5x | 8.5x |
| Custom NEON matvec | 1.5x | 12x |
| OpenMP (4 cores) | 2.5x | 25x |

At 25x over current C (125x over Python):
- Flop DL solve: 870ms → 35ms
- Turn DL solve: 723ms → 29ms
- River solve: 193ms → 8ms
- **Could solve ALL streets multiple times per decision**
- Enables full opponent modeling (solve against multiple models, pick best)

---

## Implementation Roadmap (Priority Order)

### Week 1: Foundation
1. **Debug hero range tracking** (the 20-minute fix we missed)
   - Fix edge case crash in `infer_opponent_weights` called with reversed perspective
   - Pass hero range to all `compute_opp_bet_probs` calls
   - Test against CallingStation specifically

2. **C solver: float32 conversion**
   - Change all doubles to floats in dcfr_core.c
   - Update bridge numpy dtypes
   - Verify convergence quality (should be identical at 300 iterations)
   - 2.5x speedup for free

3. **Early termination + regret-based pruning**
   - Check max regret every 50 iterations, stop if < 0.5
   - Skip actions with regret < -1e6 after iteration 100
   - Combined 2-3x speedup

### Week 2: Multi-Street Solving
4. **Full game tree blueprint with range derivation**
   - Compute blueprint ranges at every node
   - Store as compressed lookup (range = f(action_sequence))
   - Use derived ranges instead of inferred ranges for re-solving

5. **Action translation**
   - When opponent bets non-abstract size, interpolate response
   - Map continuous sizing to abstract actions probabilistically

6. **Average game values for continuation**
   - Modify `_run_dcfr` to track average game value, not last-iteration
   - Use for continuation values in DL solver

### Week 3: Opponent Modeling
7. **Opponent type detection**
   - Track per-street bet frequency, call frequency, raise frequency
   - Classify after 50 hands: aggressive/passive/balanced/trapping
   - Select pre-computed counter-strategy

8. **Adaptive re-solving**
   - Solve against detected opponent model, not GTO
   - Use Bayesian updates to refine model each hand
   - Multiple concurrent solves → pick strategy with highest EV against model

### Week 4: Polish & Precompute
9. **Full EC2 precompute with C solver**
   - Recompute all 80K river boards with 1000+ iterations (vs 300)
   - Compute turn continuation values for all boards
   - Precompute blueprint ranges for all action sequences

10. **Tournament testing**
    - Run against pool of test opponents (aggressive, passive, GTO, random)
    - Verify no regressions from each change
    - Stress test time budget on ARM Graviton2

---

## The Key Insight

The entire architecture (DL solvers, runtime narrowing, subgame solving, C solver) is the RIGHT structure. The issue is the INPUT to the structure — the ranges. With exact ranges (hero + opponent), every component works correctly. With approximate ranges, every component inherits the error.

**Hero range tracking is the foundation.** Everything else is the house. We built the house first, then discovered the foundation was wrong. Next time: foundation first.

---

## Game-Specific Observations (27-Card Deck)

- 0% high card at showdown — everyone has at least a pair
- 67% of showdowns are trips+ vs trips+ (strong-vs-strong is the norm)
- Trips is mediocre (52% of hands beat it)
- Flush draws are extremely common (9 cards per suit, 3 suits)
- Straights are frequent (only 9 ranks, many connected possibilities)
- The discard mechanic (keep best 2 of 5) inflates hand quality massively
- Board texture matters more than in standard poker (fewer ranks = more connected boards)
- Position advantage is ~3 chips/hand (acting second on river)
