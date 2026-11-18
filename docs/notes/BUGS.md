# Confirmed Bugs — Range Narrowing & Opponent Modeling

## 1. `pot_before = my_bet * 2` is wrong for re-raises

**Files:** `player.py` lines 559, 801, 488
**Functions:** `_polarized_narrow_range`, `_soft_narrow_range`, `_bayesian_range_update`

Assumes pot before opponent's bet was `2 × my_bet` (both bets equal). Only true for opening bets. When opponent re-raises after we raised, `my_bet` is our raised amount and the real pot-before was `my_bet + opp_bet_before_reraise`, not `2 × my_bet`.

**Impact:** Wrong `bet_frac` → wrong bluff ratio in polarized narrowing → range is narrowed too aggressively or too loosely depending on bet history.

## 2. MDF narrowing at street transition uses new board

**File:** `player.py` lines 923–929

When transitioning flop→turn, `board` already has 4 cards (turn card dealt). But MDF narrowing is meant to represent which hands opponent would have *called our flop bet with* — should rank by flop strength (3 cards), not turn strength (4 cards). The turn card can drastically change hand rankings (flush completing, pair on board, etc.).

**Impact:** Hands that got lucky on the turn card survive when they should have been folded out, and vice versa. Corrupts the range for all subsequent decisions.

## 3. Check-check narrows bottom 15% — wrong direction

**File:** `player.py` line 932
```python
self._mdf_narrow_range(board, dead, 0.85)
```

On check-check, both players showed weakness. Opponent's range should stay wide (or widen — they didn't bet their strong hands either). Removing the bottom 15% makes their range look artificially strong.

**Impact:** We fold too often on subsequent streets because we overestimate opponent's range strength after a check-check line.

## 4. Showdown tracking counts folds as showdowns

**File:** `player.py` lines 282–285
```python
if self._hand_reward != 0 and self._opp_bet_this_hand:
    self._opp_bet_showdown_total += 1
    if self._hand_reward < 0:
        self._opp_bet_showdown_wins += 1
```

`_hand_reward != 0` fires for both showdowns AND folds (when we fold to their bet, reward is negative). So `_opp_bet_showdown_total` conflates "opponent bet and we folded" with "opponent bet and went to showdown." The showdown win-rate (`_opp_bet_showdown_wins / _opp_bet_showdown_total`) is diluted by fold outcomes.

**Impact:** The adaptive opponent modeling in `_polarized_narrow_range` (line 581) uses a corrupted win-rate, making the value-heavy/bluff-heavy classification unreliable. When we fold a lot to their bets (common), the win-rate is artificially low → we think they're bluff-heavy → we widen their range → we overcall.
