# CMU AI Poker - Game Rules Reference

## Deck
- 27 cards: 3 suits (d=diamonds, h=hearts, s=spades), 9 ranks (2,3,4,5,6,7,8,9,A)
- No face cards (10,J,Q,K), no clubs
- Card int encoding: suit_index * 9 + rank_index (0-26)
- Four-of-a-kind impossible (only 3 suits)

## Hand Rankings (strongest to weakest)
1. **Straight Flush** — 5 consecutive same suit (e.g., 5d6d7d8d9d)
2. **Full House** — Three of a kind + pair
3. **Flush** — 5 same suit
4. **Straight** — 5 consecutive any suit
5. **Three of a Kind**
6. **Two Pair**
7. **One Pair**
8. **High Card**

**Ace Rule**: Ace is BOTH high and low for straights/straight flushes.
- A2345 is a straight, 6789A is a straight
- Lower evaluate() rank = better hand

## Hand Flow
1. **Pre-flop (Street 0)**: Deal 5 cards each. SB posts 1, BB posts 2. Betting starts with SB.
2. **Flop (Street 1)**: 3 community cards dealt. **Discard round**: BB discards first, then SB. Each keeps 2, discards 3. Discards revealed to opponent. Then flop betting starts with BB.
3. **Turn (Street 2)**: 4th community card. Betting.
4. **River (Street 3)**: 5th community card. Betting. Showdown if no fold.

## Betting Rules
- Stack: 100 chips per hand (resets each hand)
- Blinds: SB=1, BB=2
- Min raise = previous raise amount
- Max bet = 100 (MAX_PLAYER_BET)
- Raises are cumulative (total bet, not increment)

## Action Space
```
(action_type, raise_amount, keep_card_1, keep_card_2)
FOLD=0, RAISE=1, CHECK=2, CALL=3, DISCARD=4
```
- keep_card_1/2: indices 0-4 of hole cards to keep (only for DISCARD)
- Invalid actions treated as folds

## Observation
```python
{
    "street": int,           # 0-3
    "acting_agent": int,     # 0 or 1
    "my_cards": Tuple[int],  # 5 slots, -1 if empty
    "community_cards": Tuple[int],  # 5 slots, -1 if not dealt
    "my_bet": int,
    "opp_bet": int,
    "my_discarded_cards": Tuple[int],   # 3 slots
    "opp_discarded_cards": Tuple[int],  # 3 slots, revealed after discard
    "min_raise": int,
    "max_raise": int,
    "valid_actions": List[bool]  # 5 bools
}
```

## Match/Tournament
- 1000 hands per match, binary win/loss ELO
- Phase 3 (current): 4 vCPU, 8GB RAM, 1500s time bank, ARM64 Graviton2
- 5-second per-action timeout, 5 retries per action, 3 total failures = forfeit
- Budget: ~1.5s avg per hand
- Deadline: March 21, 11:59 PM EST
- Finals: March 22, top 10 ELO, round-robin
- Match requests (up to 5/day) don't affect rankings
- Python 3.12, ARM64
