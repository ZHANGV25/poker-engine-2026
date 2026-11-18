# Poker bot for a 27-card hold'em variant

A heads-up poker bot built by a team of two for the CMU × Jump Trading Poker AI Tournament in March 2026. Bots played each other in 1,000-hand matches and were ranked by Elo.

## The game

The deck has 27 cards: nine ranks in three suits. Each player is dealt five hole cards, discards three face up and keeps two. A flop, turn and river follow, with betting on each street as in hold'em.

Three suits change the hand values. There are no four of a kind, flush draws are common, and keeping the best two of five means most hands that reach showdown are strong. The full rules are in [docs/rules.md](docs/rules.md).

## How it plays

- **Preflop.** A precomputed strategy table over 200 hand buckets.
- **Discards.** The opponent's three face-up discards narrow what they kept. The bot weighs each holding by exact equity, with extra weight on hands that keep a flush or straight draw.
- **Flop and turn.** Strategies for all 2,925 flops were precomputed offline by backward induction from the river. When facing a bet, a depth-limited solver re-solves the street, with the later streets' values at its leaves.
- **River.** A range-against-range CFR solver written in C ([dcfr_core.c](submission/dcfr_core.c)). It compiles at startup and runs 4.8× faster than the Python version.
- **Ranges.** Every opponent action narrows the estimate of their range before the next solve.

## What we learned

The bot's biggest leak was calling too often on the river. It tracked the opponent's range through each hand but solved as if its own range were uniform, so it misjudged how the opponent would bet against it. Tracking both ranges fixes that. Our version crashed on an edge case, and we reverted it instead of debugging it.

The game is small enough that a faster runtime solver would likely have beaten the precomputed flop and turn data, which took a fleet of EC2 machines to compute and more than 600 MB to store.

## Layout

| Path | What it holds |
|---|---|
| `submission/` | The bot. [player.py](submission/player.py) is the entry point; the solvers, equity engine and C core sit beside it |
| `submission/data/` | Precomputed strategies (about 640 MB) |
| `blueprint/` | The offline computation of the flop and turn strategies |
| `agents/` | Simple opponents for testing |
| `docs/` | The tournament's rules and engine docs |
| `docs/notes/` | Our working notes from the tournament |
| `match.py`, `run.py` | The tournament's match runner |

## Running it

Python 3.13 with [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run python run.py
```

`run.py` plays the two bots named in [agent_config.json](agent_config.json): this bot against a simple probability-based opponent. The engine tests run with `uv run --with pytest --with pytest-asyncio pytest engine_test.py api_test.py`.
