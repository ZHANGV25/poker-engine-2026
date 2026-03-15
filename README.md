# Poker AI Bot

My submission for the CMU x Jump Trading Poker AI Tournament 2026.

## How the Bot Works

Stockfish plays **Game Theory Optimal (GTO) poker** using real-time
**Counterfactual Regret Minimization (CFR+)** — the same algorithm family
behind Libratus and Pluribus, the AIs that beat world champion poker players.

### The Core Idea

Instead of heuristic rules ("raise if hand is strong"), the bot **solves
the actual poker game** for each specific situation. It builds a small game
tree (~130 nodes), simulates thousands of strategy adjustments, and converges
on the **Nash equilibrium** — the strategy that no opponent can exploit,
no matter what they do.

### Decision Flow

1. **Pre-flop**: Looks up precomputed GTO mixed strategies (solved offline
   via two-player CFR with 50 hand-strength buckets and 7 raise levels).
   Samples from equilibrium distribution: e.g., "call 65%, raise 30%, fold 5%."

2. **Discard**: Evaluates all 10 ways to keep 2 of 5 cards using exact equity
   (exhaustive enumeration of all possible outcomes — feasible because the
   27-card deck produces only ~10,000 scenarios). As SB, uses Bayesian
   inference on opponent's revealed discards to weight the evaluation.

3. **Post-flop betting**: For each decision, builds a game tree and runs
   CFR+ for 60-100 iterations (~90-130ms). The solver computes balanced
   bluffing, pot control, check-raising, and bet sizing — all emerging from
   the equilibrium, not hardcoded. When opponent raises, narrows their range
   proportional to bet-to-pot ratio (bigger bet = stronger range).

### Key Technical Edges

- **Exact equity**: Full enumeration with zero approximation error, via
  precomputed lookup tables (888K seven-card + 80K five-card hand ranks)
- **Bayesian discard inference**: Narrows opponent's ~120 possible hands to
  ~30-40 based on what they discarded and what's rational to keep
- **No exploitation**: Purely GTO — never assumes opponent tendencies. Wins
  by letting opponents make mistakes, not by trying to predict their behavior
- **Real preflop strategy**: Solved from simulated matchup equities, not
  constant approximations

### Why GTO

ELO and round-robin finals are binary (win/loss). GTO maximizes win rate by
never losing in expectation. Exploitation can backfire against opponents who
don't behave as expected — our exploitative v2 bot lost to trapping opponents.
GTO is provably unexploitable: consistency over dominance.

See `submission/README.md` for the complete technical deep-dive.

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

1. Create and activate a virtual environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Regenerating Precomputed Data (optional)

The precomputed lookup tables are included in `submission/data/`. To
regenerate them (e.g., if game rules change):

```bash
python -m submission.precompute                  # Hand rank tables (~2 min)
python -m submission.precompute_preflop          # Hand potentials (~20 min)
python -m submission.precompute_preflop_strategy # Preflop GTO ranges (~40 min)
```

## Running

### Quick Test (5 hands vs 4 test bots)

```bash
python agent_test.py
```

### Full Match (1000 hands vs configured opponent)

```bash
python run.py
```

Configure which bots play by editing `agent_config.json`.

### Comprehensive Test Suite

```bash
python test_cfr_bot.py    # 24 tests, 6200+ hands, solver verification
```

### Match Analysis

```bash
python analyze.py                    # Dashboard across all downloaded match logs
python analyze.py match_908.txt      # Detailed analysis of specific match
python analyze.py AlbertLuoLovers    # Filter by opponent name
```

Download match CSVs from the competition dashboard to `~/Downloads/` and
the analysis tool will automatically find and parse them.

## Project Structure

```
submission/                          # Bot code (uploaded to competition)
├── player.py                        # Main bot entry point (PlayerAgent class)
├── solver.py                        # Real-time CFR+ subgame solver
├── game_tree.py                     # Betting tree for CFR
├── equity.py                        # Exact equity via precomputed lookups
├── inference.py                     # Bayesian discard inference
├── precompute.py                    # Offline: hand rank tables
├── precompute_preflop.py            # Offline: hand potential table
├── precompute_preflop_strategy.py   # Offline: preflop GTO strategy via CFR
└── data/                            # Precomputed lookup tables (~2MB total)

agents/                              # Reference bots (provided by competition)
├── prob_agent.py                    # Monte Carlo equity bot (our benchmark)
├── rl_agent.py                      # Neural network bot
└── test_agents.py                   # Fold/Call/AllIn/Random bots

gym_env.py                           # Game engine
match.py                             # Match runner
run.py                               # Run a match locally
agent_test.py                        # Submission validator
test_cfr_bot.py                      # Our comprehensive test suite
analyze.py                           # Match log analysis dashboard
```
