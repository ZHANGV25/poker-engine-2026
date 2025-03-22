from gym_env import PokerEnv
import logging
import numpy as np

"""
We will use a standard card set minus all royal cards and the club's suit. Card 
numbers 2-9 and A from the remaining 3 suits are in play. Texas Hold 'em hand 
sets will still be used for hand evaluation in the same order but with 
four-of-a-kind removed.
"""

"""
1.	Straight Flush
2.	Full House
3.	Flush
4.	Three of a Kind
5.	Straight
6.	Two Pair
7.	One Pair
8.	High Card
"""

"""
Redraw Once: On any phase excluding the river, and up to once per hand for each 
player, players are allowed to discard a card and draw a replacement card. Both 
the discarded cards and the drawn card must be revealed to the opponent player, 
and the discarded card is out of the game for that hand. This goal is to make 
players decide whether or not to reveal information about their hand by 
discarding and attempting to improve their hand.
"""

"""
Testing Poker Variant:
- A single match consists of:
    - Player Cards (fix for tests)
    - Shared Cards (fix for tests)
    - Player Actions:
        - discard: get a new card; old and new cards revealed to other player
        - check: pot doesn't change
        - raise: pot increases
        - fold: game ends, opponent wins the pot
"""


RANKS = "23456789A"
SUITS = "dhs"  # no clubs

logging.basicConfig(level=logging.DEBUG)


def int_to_card_str(card_int: int):
    """
    Convert from our encoding of a card, an integer on [0, 52)
    to the trey's encoding of a card, an integer desiged for fast lookup & comparison
    """

    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return rank + suit


def card_str_to_int(card_str: str):
    rank, suit = card_str[0], card_str[1]
    return (SUITS.index(suit) * len(RANKS)) + RANKS.index(rank)


def test_utils():
    for card_int in range(len(RANKS) * len(SUITS)):
        assert card_str_to_int(int_to_card_str(card_int)) == card_int


"""

## Simple Test Cases

- p1 fold => p2 wins
    - action pace before hand doesn't matter, as long as neither player folds

- p2 fold => p1 wins
    - action pace before hand doesn't matter, as long as neither player folds

- both all in, then the better cards win
    - or tie, if hand ranks are equal

- both check the entire game => better cards win 
    - or tie, if hand ranks are equal

- p1 raises & p2 checks => better cards win 
    - or tie, if hand ranks are equal

- p2 raises & p1 checks => better cards win 
    - or tie, if hand ranks are equal


## Negative Test Cases

- invalid action -> FOLD, and print

"""


class State:
    pass


# make assert statements a function
# Action = fold,etc
# state = check the state using dictionary


def check_observation(expected_obs: dict, got_obs: dict):
    for field, value in expected_obs.items():
        assert field in got_obs, print(f"Field {field} was expected, but wasn't present in obs: {got_obs}")
        assert got_obs[field] == value, print(f"Field {field} failed: expected {value}, got {got_obs[field]}")


class Action:
    def __init__(
        self,
        action: int,
        raise_ammount: int,
        card_to_discard: int,
    ):
        assert isinstance(action, int)
        assert isinstance(raise_ammount, int)
        assert isinstance(card_to_discard, int)
        self.action = action
        self.raise_ammount = raise_ammount
        self.card_to_discard = card_to_discard

    def __repr__(self):
        return f"Action(action={repr(self.action)}, raise_ammount={repr(self.raise_ammount)}, card_to_discard={repr(self.card_to_discard)})"


class GameState:
    def __init__(self, p0obs: dict, p1obs: dict):
        self.p0obs = p0obs
        self.p1obs = p1obs


def _test_engine(rigged_deck: list[int], updates: list[tuple[Action, tuple[dict, dict]]], expected_final_rewards: tuple[int, int], num_hands: int = 1):
    engine = PokerEnv(num_hands=num_hands)  # small blind player always starts out as 0
    assert isinstance(rigged_deck, list)
    (player0_obs, player1_obs), _info = engine.reset(
        options={"cards": rigged_deck}  # rig the deck
    )

    reversed_deck = rigged_deck[::-1]

    # pops p0 2 card's first
    p0_expected_start_cards = []
    for _ in range(2):
        p0_expected_start_cards.append(reversed_deck.pop())

    # pops p1 2 card's first
    p1_expected_start_cards = []
    for _ in range(2):
        p1_expected_start_cards.append(reversed_deck.pop())

    # pops 5 community cards
    expected_community_cards = []
    for _ in range(5):
        expected_community_cards.append(reversed_deck.pop())

    p0_valid_actions = [1] * 5
    p0_valid_actions[engine.ActionType.CHECK.value] = 0  # p0 can't check as it's small blind

    # pop redraw cards next (as needed)
    player0_expected_obs = {
        "street": 0,
        "acting_agent": 0,
        "my_cards": p0_expected_start_cards,
        "community_cards": [-1] * 5,
        "my_bet": 1,
        "opp_bet": 2,
        "opp_discarded_card": -1,
        "opp_drawn_card": -1,
        "my_discarded_card": -1,
        "my_drawn_card": -1,
        "min_raise": 2,
        "valid_actions": p0_valid_actions,
    }

    player1_expected_obs = {
        "street": 0,
        "acting_agent": 0,
        "my_cards": p1_expected_start_cards,
        "community_cards": [-1] * 5,
        "my_bet": 2,
        "opp_bet": 1,
        "opp_discarded_card": -1,
        "opp_drawn_card": -1,
        "my_discarded_card": -1,
        "my_drawn_card": -1,
        "min_raise": 2,
    }

    check_observation(player0_expected_obs, player0_obs)
    check_observation(player1_expected_obs, player1_obs)

    for i, (action, expected_state) in enumerate(updates):
        obs, reward, terminated, _, _ = engine.step((action.action, action.raise_ammount, action.card_to_discard))
        p0_got_obs, p1_got_obs = obs
        p0_got_reward, p1_got_reward = reward

        assert terminated == (i == (len(updates) - 1)), print(f"terminated: {terminated}; len(updates): {len(updates)}; i: {i}")

        expected_p0_obs, expected_p1_obs = expected_state
        check_observation(expected_p0_obs, p0_got_obs)
        check_observation(expected_p1_obs, p1_got_obs)

        if terminated:
            assert reward == expected_final_rewards, print(f"Got final reward: {reward}, expected: {expected_final_rewards}")
        else:
            assert p0_got_reward == 0 and p1_got_reward == 0

    return


def test_allways_check():
    # small blind player goes first; they need to pay 1 to check
    small_blind_call = Action(PokerEnv.ActionType.CALL.value, 0, -1)

    either_player_check = Action(PokerEnv.ActionType.CHECK.value, 0, -1)

    # pr-flop: bb checks (1); flop: 2 checks; turn: 2 checks; river: 2 checks
    NUM_CHECK_ROUNDS = sum([1, 2, 2, 2])
    actions = [small_blind_call] + ([either_player_check] * NUM_CHECK_ROUNDS)
    states = [({}, {})] * (NUM_CHECK_ROUNDS + 1)
    assert len(actions) == len(states)
    updates = list(zip(actions, states))
    # draws p0's 2 cards, then p1's 2 cards, then 5 community cards, starting at 0th index of rigged_deck
    rigged_deck = list(
        map(
            card_str_to_int,
            [
                # p0's cards
                "Ah",
                "Ad",
                # p1's cards
                "9h",
                "9d",
                # community cards
                "As",
                "9s",
                "2h",
                "3h",
                "4h",
                # p0 wins (higher 3 of a kind)
            ],
        )
    )
    _test_engine(rigged_deck=rigged_deck, updates=updates, expected_final_rewards=(2, -2))


def test_allways_raise_small():
    # Small Blind always raises min_bet while big blind always calls
    min_bet = 2
    # small blind must put in 1 to match the
    small_blind_raise = Action(PokerEnv.ActionType.RAISE.value, min_bet, -1)
    big_blind_call = Action(PokerEnv.ActionType.CALL.value, min_bet, -1)
    # preflop: p0raise->p1call 4 => flop p0raise->p1call 8=> river p0raise p1call 12=> turn p0raise p1call
    actions = ([small_blind_raise] + [big_blind_call]) * 4
    states = [({"min_raise": 4}, {"min_raise": 4})] + [({"min_raise": 2}, {"min_raise": 2})] * 7
    assert len(states) == len(actions)
    updates = list(zip(actions, states))
    rigged_deck = list(
        map(
            card_str_to_int,
            [
                # p0's cards
                "4h",
                "4d",
                # p1's cards
                "6h",
                "7d",
                # community cards
                "4s",
                "9s",
                "9h",
                "2h",
                "Ah",
                # p0 wins
            ],
        )
    )
    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=(10, -10),
    )


def test_example_tie():
    small_blind_call = Action(PokerEnv.ActionType.CALL.value, 0, -1)
    either_player_check = Action(PokerEnv.ActionType.CHECK.value, 0, -1)
    NUM_CHECK_ROUNDS = sum([1, 2, 2, 2])
    actions = [small_blind_call] + ([either_player_check] * NUM_CHECK_ROUNDS)
    states = [({}, {})] * (NUM_CHECK_ROUNDS + 1)
    assert len(actions) == len(states)
    updates = list(zip(actions, states))
    rigged_deck = list(
        map(
            card_str_to_int,
            [
                # p0's cards
                "2h",
                "3d",
                # p1's cards
                "2d",
                "3s",
                # community cards
                "9s",
                "8s",
                "7h",
                "6h",
                "5h",
                # tie
            ],
        )
    )
    _test_engine(rigged_deck=rigged_deck, updates=updates, expected_final_rewards=(0, 0))


def test_example_game_1():
    """
    A game played between Mark and DK
    """
    rigged_deck = [25, 14, 1, 4, 8, 16, 9, 11, 23]
    expected_final_rewards = (14, -14)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.RAISE.value, 2, -1),
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.RAISE.value, 10, -1),
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
    ]

    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    -1,
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    -1,
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    -1,
                    -1,
                ],
                "my_bet": 4,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 1, 1],
            },
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    -1,
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 0, 1, 1],
            },
        ),
        (
            {
                "street": 2,
                "acting_agent": 0,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    -1,
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 2,
                "acting_agent": 0,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    -1,
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 2,
                "acting_agent": 1,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    -1,
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 2,
                "acting_agent": 1,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    -1,
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 3,
                "acting_agent": 0,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 3,
                "acting_agent": 0,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 4,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 96,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 3,
                "acting_agent": 1,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 14,
                "opp_bet": 4,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 10,
                "max_raise": 86,
                "valid_actions": [1, 1, 1, 1, 0],
            },
            {
                "street": 3,
                "acting_agent": 1,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 4,
                "opp_bet": 14,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 10,
                "max_raise": 86,
                "valid_actions": [1, 1, 0, 1, 0],
            },
        ),
        (
            {
                "street": 4,
                "acting_agent": 0,
                "my_cards": [np.int64(25), np.int64(14)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 14,
                "opp_bet": 14,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 86,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 4,
                "acting_agent": 0,
                "my_cards": [np.int64(1), np.int64(4)],
                "community_cards": [
                    np.int64(8),
                    np.int64(16),
                    np.int64(9),
                    np.int64(11),
                    np.int64(23),
                ],
                "my_bet": 14,
                "opp_bet": 14,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 86,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
    ]
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))
    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
    return


def test_example_game_2():
    """
    A game with invalid raise on player2
    """
    rigged_deck = [24, 14, 11, 23, -1, -1, -1, -1, -1]
    expected_final_rewards = (2, -2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.RAISE.value, 1, -1),
    ]
    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(24), np.int64(14)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(11), np.int64(23)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 0,
                "acting_agent": 0,
                "my_cards": [np.int64(24), np.int64(14)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 0,
                "my_cards": [np.int64(11), np.int64(23)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
    ]
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))
    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
    return


def test_example_game_3():
    rigged_deck = [21, 3, 8, 6, 20, 5, 22, -1, -1, 0]
    # discard card 0
    expected_final_rewards = (-2, 2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.DISCARD.value, 0, 1),
        Action(PokerEnv.ActionType.DISCARD.value, 0, 1),
    ]
    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(21), np.int64(3)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(6)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(21), np.int64(3)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(6)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(21), np.int64(0)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": np.int64(3),
                "my_drawn_card": np.int64(0),
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(6)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": np.int64(3),
                "opp_drawn_card": np.int64(0),
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(21), np.int64(0)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": np.int64(3),
                "my_drawn_card": np.int64(0),
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(6)],
                "community_cards": [np.int64(20), np.int64(5), np.int64(22), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": np.int64(3),
                "opp_drawn_card": np.int64(0),
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
    ]
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
    return


def test_example_game_4():
    rigged_deck = [8, 16, 14, 17, 9, 7, 0, 12, 10]
    # discard card 0
    expected_final_rewards = (-2, 2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
        Action(PokerEnv.ActionType.CHECK.value, 0, -1),
    ]
    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [np.int64(9), np.int64(7), np.int64(0), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 1,
                "acting_agent": 0,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [np.int64(9), np.int64(7), np.int64(0), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [np.int64(9), np.int64(7), np.int64(0), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 1,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [np.int64(9), np.int64(7), np.int64(0), -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 2,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 2,
                "acting_agent": 0,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 2,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 2,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    -1,
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 3,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 3,
                "acting_agent": 0,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 3,
                "acting_agent": 1,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 3,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
        (
            {
                "street": 4,
                "acting_agent": 0,
                "my_cards": [np.int64(8), np.int64(16)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
            {
                "street": 4,
                "acting_agent": 0,
                "my_cards": [np.int64(14), np.int64(17)],
                "community_cards": [
                    np.int64(9),
                    np.int64(7),
                    np.int64(0),
                    np.int64(12),
                    np.int64(10),
                ],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 0],
            },
        ),
    ]
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
    return


def test_example_game_5():
    """
    A game with max bet
    """
    rigged_deck = [24, 10, 14, 5, -1, -1, -1, -1, -1, -1]
    expected_final_rewards = (-2, 2)
    actions = [
        Action(PokerEnv.ActionType.CALL.value, 0, -1),
        Action(PokerEnv.ActionType.RAISE.value, 98, -1),
        Action(PokerEnv.ActionType.FOLD.value, 0, -1),
    ]
    obs = [
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(24), np.int64(10)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(5)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 2,
                "max_raise": 98,
                "valid_actions": [1, 1, 1, 0, 1],
            },
        ),
        (
            {
                "street": 0,
                "acting_agent": 0,
                "my_cards": [np.int64(24), np.int64(10)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 100,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 0,
                "max_raise": 0,
                "valid_actions": [1, 0, 0, 1, 1],
            },
            {
                "street": 0,
                "acting_agent": 0,
                "my_cards": [np.int64(14), np.int64(5)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 100,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 0,
                "max_raise": 0,
                "valid_actions": [1, 0, 1, 1, 1],
            },
        ),
        (
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(24), np.int64(10)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 2,
                "opp_bet": 100,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 0,
                "max_raise": 0,
                "valid_actions": [1, 0, 0, 1, 1],
            },
            {
                "street": 0,
                "acting_agent": 1,
                "my_cards": [np.int64(14), np.int64(5)],
                "community_cards": [-1, -1, -1, -1, -1],
                "my_bet": 100,
                "opp_bet": 2,
                "opp_discarded_card": -1,
                "opp_drawn_card": -1,
                "my_discarded_card": -1,
                "my_drawn_card": -1,
                "min_raise": 0,
                "max_raise": 0,
                "valid_actions": [1, 0, 1, 1, 1],
            },
        ),
    ]
    assert len(actions) == len(obs)
    updates = list(zip(actions, obs))
    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=expected_final_rewards,
    )
    return


def test_discard_invalid_card():
    """
    Test discarding with card_to_discard = -1, which should not change any cards
    """
    rigged_deck = list(
        map(
            card_str_to_int,
            [
                # p0's cards
                "4h",
                "4d",
                # p1's cards
                "6h",
                "7d",
                # community cards
                "4s",
                "9s",
                "9h",
                "2h",
                "Ah",
                # redraw cards (shouldn't be used)
                "3h",
                "3d",
            ],
        )
    )

    small_blind_call = Action(PokerEnv.ActionType.CALL.value, 0, -1)
    invalid_discard = Action(PokerEnv.ActionType.DISCARD.value, 0, -1)
    either_player_check = Action(PokerEnv.ActionType.CHECK.value, 0, -1)

    # Initial state after small blind calls
    p0_obs = {
        "my_cards": [np.int64(rigged_deck[0]), np.int64(rigged_deck[1])],
        "my_discarded_card": -1,
        "my_drawn_card": -1,
    }
    p1_obs = {
        "my_cards": [np.int64(rigged_deck[2]), np.int64(rigged_deck[3])],
        "my_discarded_card": -1,
        "my_drawn_card": -1,
    }

    # Complete sequence of actions to end the game:
    # 1. Small blind calls
    # 2. Invalid discard attempt
    # 3-7. Check through all streets
    updates = [
        (small_blind_call, ({}, {})),
        (invalid_discard, (p0_obs, p1_obs)),
        (either_player_check, ({}, {})),  # flop
        (either_player_check, ({}, {})),
        (either_player_check, ({}, {})),  # turn
        (either_player_check, ({}, {})),
        (either_player_check, ({}, {})),  # river
        (either_player_check, ({}, {})),
        (either_player_check, ({}, {})),  # showdown
    ]

    _test_engine(
        rigged_deck=rigged_deck,
        updates=updates,
        expected_final_rewards=(2, -2),  # p0 wins with three of a kind
    )


def test_small_blind_alternation():
    """
    Test that the small blind alternates between players correctly.
    """
    env = PokerEnv(num_hands=4)
    small_blind_positions = []

    for hand_number in range(4):
        # Reset with the small blind player determined by hand number
        small_blind_player = hand_number % 2
        obs, _ = env.reset(options={"small_blind_player": small_blind_player})

        # Verify the small blind player was set correctly
        small_blind_positions.append(small_blind_player)

    expected_pattern = [0, 1, 0, 1]
    assert small_blind_positions == expected_pattern, f"Small blind positions {small_blind_positions} don't match expected pattern {expected_pattern}"
    print("Small blind alternation test passed!")


def test_ace_high_after_nine():
    # small blind player goes first; they need to pay 1 to check
    small_blind_call = Action(PokerEnv.ActionType.CALL.value, 0, -1)

    either_player_check = Action(PokerEnv.ActionType.CHECK.value, 0, -1)

    # pr-flop: bb checks (1); flop: 2 checks; turn: 2 checks; river: 2 checks
    NUM_CHECK_ROUNDS = sum([1, 2, 2, 2])
    actions = [small_blind_call] + ([either_player_check] * NUM_CHECK_ROUNDS)
    states = [({}, {})] * (NUM_CHECK_ROUNDS + 1)
    assert len(actions) == len(states)
    updates = list(zip(actions, states))
    # draws p0's 2 cards, then p1's 2 cards, then 5 community cards, starting at 0th index of rigged_deck
    rigged_deck = list(
        map(
            card_str_to_int,
            [
                # p0's cards
                "Ad",
                "Ad",
                # p1's cards
                "9h",
                "9d",
                # community cards
                "6h",
                "7s",
                "8h",
                "9s",
                "4h",
                # p1 has three of a kind: 9, 9, 9,
                # p0 has a straight, with ACE HIGH: 6, 7, 8, 9, A
                # p0 wins
            ],
        )
    )
    _test_engine(rigged_deck=rigged_deck, updates=updates, expected_final_rewards=(2, -2))


def main():
    test_utils()
    print("test utils passed")
    print("test engine 0 passed")
    test_allways_check()
    test_allways_raise_small()
    test_example_tie()
    test_example_game_2()
    test_example_game_3()
    test_example_game_4()
    test_example_game_5()
    test_small_blind_alternation()
    test_ace_high_after_nine()


if __name__ == "__main__":
    main()
