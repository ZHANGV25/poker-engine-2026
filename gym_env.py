"""
CMU Poker Bot Competition Game Engine 2025

This module implements a custom Gym environment for a poker game.
It supports a two-player game with betting, discarding, and multiple streets.

References:
- https://gymnasium.farama.org
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Note: This implementation uses a Tuple space to represent the observation and
action spaces for each agent, as Gym doesn't inherently support multi-agent environments.
"""

from enum import Enum
from typing import List, Optional

import gym
import numpy as np
from gym import spaces
from pydantic import BaseModel, Field
from treys import Card, Evaluator


class ActionType(Enum):
    """Enumeration of possible action types in the poker game."""

    FOLD = 1
    RAISE = 2
    CHECK = 3
    CALL = 4
    INVALID = 5


class PlayerAction(BaseModel):
    """Represents a player's action in the game."""

    bet_amount: int = Field(..., ge=-1, le=101, description="The total bet amount after this action")
    card_to_discard: int = Field(..., ge=-1, le=3, description="Index of the card to discard, or -1 for no discard")


class PlayerState(BaseModel):
    """Represents the state of a player in the game."""

    cards: List[int] = Field(..., min_items=3, max_items=3, description="Player's hand, represented as integers")
    bet: int = Field(0, ge=0, description="Current bet amount")
    bankroll: float = Field(0.0, description="Player's total bankroll")
    shown_cards: List[int] = Field(default_factory=lambda: [-1, -1, -1], description="Cards shown during discard phase")


class GameState(BaseModel):
    """Represents the current state of the poker game."""

    street: int = Field(0, ge=0, le=3, description="Current street of the game (0-3)")
    turn: int = Field(0, ge=0, le=1, description="Index of the player whose turn it is")
    community_cards: List[int] = Field(..., min_items=5, max_items=5, description="Community cards, represented as integers")
    players: List[PlayerState] = Field(..., min_items=2, max_items=2, description="States of both players")
    small_blind_player: int = Field(0, ge=0, le=1, description="Index of the player who is the small blind")
    game_num: int = Field(1, ge=1, description="Current game number")
    min_raise: int = Field(2, ge=2, description="Minimum raise amount")


class PokerEnv(gym.Env):
    """
    A custom Gym environment for a two-player poker game.

    This environment simulates a poker game with betting, discarding, and multiple streets.
    It uses the Gym interface for reinforcement learning compatibility.
    """

    SMALL_BLIND = 1
    BIG_BLIND = 2
    NO_DISCARD = -1
    DISCARD_CARD_POS = 0

    def __init__(self, num_games: int) -> None:
        """
        Initialize the PokerEnv.

        Args:
            num_games (int): The number of games to play in a full episode.
        """
        super().__init__()
        self.num_games = num_games
        self.evaluator = Evaluator()
        self.action_space = spaces.Tuple([spaces.Discrete(102, start=-1), spaces.Discrete(4, start=-1)])
        self.observation_space = self._create_observation_space()
        self.state: GameState = self._initialize_game_state()

    @staticmethod
    def int_to_card(card_int: int) -> int:
        """
        Convert from our integer encoding of a card to the treys library encoding.

        Args:
            card_int (int): Our integer representation of a card (0-51).

        Returns:
            int: The treys library integer representation of the card.
        """
        RANKS = "23456789TJQKA"
        SUITS = "cdhs"  # clubs diamonds hearts spades
        rank = RANKS[card_int % 13]
        suit = SUITS[card_int // 13]
        return Card.new(rank + suit)

    def _create_observation_space(self):
        """
        Create the observation space for the environment.

        Returns:
            spaces.Tuple: A tuple of two identical Dict spaces, one for each player.
        """
        cards_space = spaces.Discrete(53, start=-1)
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),
                "turn": spaces.Discrete(2),
                "my_cards": spaces.Tuple([cards_space for _ in range(3)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(100, start=0),
                "opp_bet": spaces.Discrete(100, start=0),
                "my_bankroll": spaces.Box(low=-1, high=1, shape=(1,)),
                "opp_shown_cards": spaces.Tuple([cards_space for _ in range(3)]),
                "game_num": spaces.Discrete(self.num_games, start=1),
                "min_raise": spaces.Discrete(100, start=2),
            }
        )
        return spaces.Tuple([observation_space_one_player for _ in range(2)])

    def _initialize_game_state(self) -> GameState:
        """Initialize and return a new GameState object."""
        return GameState(
            street=0,
            turn=1,
            community_cards=[-1] * 5,
            players=[
                PlayerState(cards=[-1] * 3, bet=self.SMALL_BLIND, bankroll=0),
                PlayerState(cards=[-1] * 3, bet=self.BIG_BLIND, bankroll=0),
            ],
            small_blind_player=1,
            game_num=0,
            min_raise=self.BIG_BLIND,
        )

    def _get_single_player_obs(self, player_num: int):
        """
        Get the observation for a single player.

        Args:
            player_num (int): The index of the player (0 or 1).

        Returns:
            dict: The observation dictionary for the specified player.
        """
        num_cards_to_reveal = min(self.state.street + 2, 5) if self.state.street > 0 else 0
        return {
            "street": self.state.street,
            "turn": self.state.turn,
            "my_cards": self.state.players[player_num].cards,
            "community_cards": self.state.community_cards[:num_cards_to_reveal],
            "my_bet": self.state.players[player_num].bet,
            "opp_bet": self.state.players[1 - player_num].bet,
            "my_bankroll": np.array([self.state.players[player_num].bankroll]),
            "opp_shown_cards": self.state.players[1 - player_num].shown_cards,
            "game_num": self.state.game_num,
            "min_raise": self.state.min_raise,
        }

    def _get_obs(self, winner: Optional[int] = None):
        """
        Get the full observation, including rewards and other information.

        Args:
            winner (Optional[int]): The index of the winning player, if any.

        Returns:
            tuple: A tuple containing observations, rewards, termination flag, truncation flag, and info.
        """
        observation = (self._get_single_player_obs(0), self._get_single_player_obs(1))
        if winner is not None:
            if winner == 0:
                reward = (
                    min(self.state.players[0].bet, self.state.players[1].bet),
                    -min(self.state.players[0].bet, self.state.players[1].bet),
                )
            elif winner == 1:
                reward = (
                    -min(self.state.players[0].bet, self.state.players[1].bet),
                    min(self.state.players[0].bet, self.state.players[1].bet),
                )
            else:
                reward = (0, 0)
        else:
            reward = (0, 0)
        terminated = self.state.game_num > self.num_games
        truncated = False
        info = None
        return observation, reward, terminated, truncated, info

    def _update_bankrolls(self, winner: int):
        """
        Update player bankrolls based on the winner of the hand.

        Args:
            winner (int): The index of the winning player, or -1 for a tie.
        """
        if winner >= 0:
            pot = min(self.state.players[0].bet, self.state.players[1].bet)
            self.state.players[winner].bankroll += pot
            self.state.players[1 - winner].bankroll -= pot

    def _start_new_game(self):
        """Initialize the state for a new game."""
        self.state.small_blind_player = 1 - self.state.small_blind_player
        self.state.turn = self.state.small_blind_player
        cards = np.random.choice(52, 11, replace=False)
        self.state.players[0].cards = cards[:3].tolist()
        self.state.players[1].cards = cards[3:6].tolist()
        self.state.community_cards = cards[6:].tolist()
        self.state.players[self.state.small_blind_player].bet = self.SMALL_BLIND
        self.state.players[1 - self.state.small_blind_player].bet = self.BIG_BLIND
        self.state.players[0].shown_cards = [-1, -1, -1]
        self.state.players[1].shown_cards = [-1, -1, -1]
        self.state.min_raise = self.BIG_BLIND
        self.state.street = 0
        self.state.game_num += 1

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Args:
            seed (Optional[int]): Seed for the random number generator.
            options (Optional[dict]): Additional options for reset (not used).

        Returns:
            Tuple containing the initial observations for both players.
        """
        super().reset(seed=seed)
        self.state = self._initialize_game_state()
        self.state.players[0].bankroll = 0
        self.state.players[1].bankroll = 0
        self._start_new_game()
        return self._get_obs()[0]

    def _get_action_type(self, action: PlayerAction) -> ActionType:
        """
        Determine the type of action based on the player's input.

        Args:
            action (PlayerAction): The action taken by the player.

        Returns:
            ActionType: The classified type of the action.
        """
        current_player = self.state.players[self.state.turn]
        opponent = self.state.players[1 - self.state.turn]

        if action.bet_amount == -1:
            return ActionType.FOLD
        elif action.bet_amount == current_player.bet:
            return ActionType.CHECK if current_player.bet == opponent.bet else ActionType.CALL
        elif action.bet_amount > current_player.bet:
            return ActionType.RAISE if action.bet_amount - opponent.bet >= self.state.min_raise else ActionType.INVALID
        else:
            return ActionType.INVALID

    def _next_street(self):
        """Advance the game to the next street."""
        self.state.street += 1
        self.state.min_raise = self.BIG_BLIND
        self.state.turn = self.state.small_blind_player

    def _get_winner(self) -> int:
        """
        Determine the winner of the current hand.

        Returns:
            int: The index of the winning player (0 or 1), or -1 for a tie.
        """
        board_cards = [self.int_to_card(card) for card in self.state.community_cards if card != -1]
        player_1_cards = [self.int_to_card(card) for card in self.state.players[0].cards if card != -1]
        player_2_cards = [self.int_to_card(card) for card in self.state.players[1].cards if card != -1]

        assert len(player_1_cards) == 2 and len(player_2_cards) == 2 and len(board_cards) == 5

        player_1_hand_score = self.evaluator.evaluate(board_cards, player_1_cards)
        player_2_hand_score = self.evaluator.evaluate(board_cards, player_2_cards)

        if player_1_hand_score == player_2_hand_score:
            return -1  # tie
        elif player_1_hand_score < player_2_hand_score:
            return 0
        else:
            return 1

    def step(self, action: PlayerAction):
        """
        Take a step in the environment based on the given action.

        Args:
            action (PlayerAction): The action to be taken by the current player.

        Returns:
            tuple: A tuple containing the new observation, reward, termination flag, truncation flag, and info.
        """
        action_type = self._get_action_type(action)
        new_game = False
        new_street = False
        winner = None

        # Discard phase
        if self.state.street == 1 and self.state.players[self.state.turn].shown_cards[self.DISCARD_CARD_POS] == -1:
            self.state.players[self.state.turn].shown_cards[self.DISCARD_CARD_POS] = self.state.players[
                self.state.turn
            ].cards[action.card_to_discard]
            self.state.players[self.state.turn].cards[action.card_to_discard] = -1

        # We consider invalid actions as folding
        if action_type == ActionType.INVALID:
            action_type = ActionType.FOLD

        if action_type == ActionType.FOLD:
            winner = 1 - self.state.turn
            self._update_bankrolls(winner)
            new_game = True
        elif action_type == ActionType.CALL:
            self.state.players[self.state.turn].bet = self.state.players[1 - self.state.turn].bet
            new_street = True
        elif action_type == ActionType.CHECK:
            if self.state.turn == 1 - self.state.small_blind_player:
                new_street = True  # big blind checks mean next street
        elif action_type == ActionType.RAISE:
            self.state.players[self.state.turn].bet = action.bet_amount
            self.state.min_raise = max(
                self.state.min_raise, action.bet_amount - self.state.players[1 - self.state.turn].bet
            )
        else:
            assert False

        if new_street:
            self._next_street()
            if self.state.street > 3 and not new_game:
                winner = self._get_winner()
                self._update_bankrolls(winner)
                new_game = True

        if not new_game and not new_street:
            self.state.turn = 1 - self.state.turn

        obs, reward, terminated, truncated, info = self._get_obs(winner)
        if terminated:
            print("Game is terminated. Final bankrolls:", [player.bankroll for player in self.state.players])

        if new_game:
            self._start_new_game()

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = PokerEnv(100)
