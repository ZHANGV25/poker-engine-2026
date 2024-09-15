"""
CMU Poker Bot Competition Game Engine 2025

People working on this code, please refer to:
https://gymnasium.farama.org
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Keep in mind gym doesn't inherently support multi-agent environments.
We will have to use the Tuple space to represent the observation space and 
action space for each agent.

"""

import os
from enum import Enum
import numpy as np
import gym
from gym import spaces
from treys import Evaluator, Card


class PokerEnv(gym.Env):

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer desiged for fast lookup & comparison
        """
        RANKS = "23456789TJQKA"
        SUITS = "cdhs"  # clubs diamonds hearts spade
        rank = RANKS[card_int % 13]
        suit = SUITS[card_int // 13]
        return Card.new(rank + suit)

    class ActionType(Enum):
        FOLD = 1
        RAISE = 2
        CHECK = 3
        CALL = 4
        INVALID = 5

    SMALL_BIND = 1
    BIG_BLIND = 2

    DISCARD_CARD_POS = 0

    NO_DISCARD = -1

    def __init__(self, num_games) -> None:
        super().__init__()
        self.num_games = num_games

        self.evaluator = Evaluator()

        # Action space is a Tuple (total_bet, card_to_discard)
        # where action is a Discrete(4) and amount is a Discrete(400)
        # If bet_amount =< opp_bet, it is considered as folding
        # If bet_amount > opp_bet, it is considered as raising (must be a legal raise)
        # If bet_amount == opp_bet, it is considered as calling
        # Card to discard and show is only relevant in the discard game.
        # Discrete(4) cuz we have 3 cards.
        # Keep in Mind: THIS IS Cumulative betting
        self.action_space = spaces.Tuple(
            [spaces.Discrete(102, start=-1), spaces.Discrete(4, start=-1)]
        )

        # Card space is a Discrete(53), -1 means the card is not shown
        cards_space = spaces.Discrete(53, start=-1)

        # Single observation space is a Dict.
        # Since we have two players, turn is a Discrete(2)
        # Make sure to check (turn == agent_num) before taking an action
        # opp_shown_card is "0" if the opp's card is not shown
        # Two players, so the observation space is a Tuple of two single_observation_spaces
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),
                "turn": spaces.Discrete(2),
                "my_cards": spaces.Tuple([cards_space for _ in range(3)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(100, start=0),
                "opp_bet": spaces.Discrete(100, start=0),
                "my_bankroll": spaces.Box(
                    low=-1, high=1, shape=(1,)
                ),  # Normalized bankroll div by 1e4 or something idk
                "opp_shown_cards": spaces.Tuple([cards_space for _ in range(2)]),
                "game_num": spaces.Discrete(self.num_games, start=1),
                "min_raise": spaces.Discrete(100, start=2),
            }
        )

        # Since we have two players, the observation space is a tuple of
        # (observation_space_one_player, observation_space_one_player)
        self.observation_space = spaces.Tuple(
            [observation_space_one_player for _ in range(2)]
        )

        self.player_cards = (None, None)
        self.community_cards = []
        self.bets = [None, None]
        self.shown_cards = (None, None)

        # New episode
        self.reset(seed=int.from_bytes(os.urandom(32)))

    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for the player_num player.
        """
        num_cards_to_reveal = -1
        if self.street == 0:
            num_cards_to_reveal = 0
        else:
            num_cards_to_reveal = self.street + 2

        return {
            "street": self.street,
            "turn": self.turn,
            "my_cards": self.player_cards[player_num],
            "community_cards": self.community_cards[:num_cards_to_reveal],
            "my_bet": self.bets[player_num],
            "opp_bet": self.bets[1 - player_num],
            "my_bankroll": self.bankrolls[player_num],
            "opp_shown_cards": self.shown_cards[1 - player_num],
            "game_num": self.game_num,
            "min_raise": self.min_raise,
        }

    def _get_obs(self, winner):
        """
        Returns the observation for both players.
        """
        observation = (self._get_single_player_obs(0), self._get_single_player_obs(1))
        if winner == 0:
            reward = (min(self.bets), -min(self.bets))
        elif winner == 1:
            reward = (-min(self.bets), min(self.bets))
        else:
            reward = (0, 0)
        terminated = self.game_num > self.num_games
        truncated = False
        info = None
        return observation, reward, terminated, truncated, info

    def _update_bankrolls(self, winner):
        """
        End the game, update the bankrolls
        winner == -1 means a tie
        """
        assert -1 <= winner <= 1
        if winner >= 0:
            self.bankrolls[winner] += min(self.bets)
            self.bankrolls[1 - winner] -= min(self.bets)

    def _start_new_game(self):
        # Rotate the small blind
        self.small_blind_player = 1 - self.small_blind_player

        # Small blind starts
        self.turn = self.small_blind_player

        # Deal the cards
        cards = np.random.choice(52, 3 + 3 + 5, replace=False)
        self.player_cards = [cards[:3], cards[3:6]]
        self.community_cards = cards[6:]

        # Reset the bets, and no cards are shown yet
        self.bets[self.small_blind_player] = self.SMALL_BIND
        self.bets[1 - self.small_blind_player] = self.BIG_BLIND
        self.shown_cards = [np.array([-1, -1, -1]), np.array([-1, -1, -1])]
        self.min_raise = self.BIG_BLIND

        self.street = 0

        self.game_num += 1

    def reset(self, *, seed=None, options=None):
        """
        Resets the entire game.
        """
        # match initialization

        super().reset(seed=seed)

        # cumulative winnings across all games
        self.bankrolls = [0, 0]
        self.game_num = 0  # start_game will put to 1
        self.small_blind_player = 1  # start_game will put to 0

        # game initialization
        self._start_new_game()

        obs = (self._get_single_player_obs(0), self._get_single_player_obs(1))
        return obs, None

    def _get_action_type(self, action) -> ActionType:
        """
        Validates the action taken by the player.
        """
        new_bet, card_to_discard = action

        other_player = 1 - self.turn

        # detect if bet is raise, fold, or check
        other_player_old_bet = self.bets[other_player]

        # amount curr player is raising
        if new_bet < other_player_old_bet:
            return self.ActionType.FOLD
        elif new_bet == other_player_old_bet:
            if new_bet == self.bets[self.turn]:
                return self.ActionType.CHECK
            else:
                return self.ActionType.CALL
        # raise
        raised_by = new_bet - other_player_old_bet
        if raised_by < self.min_raise:
            print("Raise must be at least", self.min_raise, "but was", raised_by)
            return self.ActionType.INVALID

        # Discard has to be done in the flop and not any other streets
        if self.street == 1:
            # on the flop, must be valid discard
            # TODO: What if a player decides to fold? Do they still have to show their cards?
            if card_to_discard < 0 and self.ACTION_TYPE != self.ActionType.FOLD:
                print("Did not discard a card during the flop")
                return self.ActionType.INVALID
        else:
            if card_to_discard != self.NO_DISCARD:
                print("Discarded a card when it wasn't the flop")
                return self.ActionType.INVALID

        return self.ActionType.RAISE

    def _next_street(self):
        """
        Update to the next street of the game.
        """
        self.street += 1
        self.min_raise = self.BIG_BLIND
        self.turn = self.small_blind_player

    def _get_winner(self):
        board_cards = list(map(self.int_to_card, self.community_cards))
        player_1_cards = list(
            map(self.int_to_card, [c for c in self.player_cards[0] if c != -1])
        )
        player_2_cards = list(
            map(self.int_to_card, [c for c in self.player_cards[1] if c != -1])
        )
        assert (
            len(player_1_cards) == 2
            and len(player_2_cards) == 2
            and len(board_cards) == 5
        )
        player_1_hand_score = self.evaluator.evaluate(
            player_1_cards,
            board_cards,
        )
        player_2_hand_score = self.evaluator.evaluate(
            player_2_cards,
            board_cards,
        )
        # showdown
        if player_1_hand_score == player_2_hand_score:
            winner = -1  # tie
        elif player_1_hand_score < player_2_hand_score:
            winner = 1
        else:
            winner = 0
        return winner

    def step(self, action):
        """
        Takes a step in the game, given the action taken by the active player.
        """
        bet_amount, card_to_discard = action
        action_type = self._get_action_type(action)
        print("Action type:", action_type)
        new_game = False
        new_street = False
        winner = None

        # Discard phase
        if (
            self.street == 1
            and self.shown_cards[self.turn][self.DISCARD_CARD_POS] == -1
        ):
            self.shown_cards[self.turn][self.DISCARD_CARD_POS] = self.player_cards[
                self.turn
            ][card_to_discard]
            self.player_cards[self.turn][card_to_discard] = -1

        # We consider invalid actions as folding
        if action_type == self.ActionType.INVALID:
            action_type = self.ActionType.FOLD

        if action_type == self.ActionType.FOLD:
            winner = 1 - self.turn
            self._update_bankrolls(winner)  # by folding, we must lose
            new_game = True
        elif action_type == self.ActionType.CALL:
            self.bets[self.turn] = self.bets[1 - self.turn]
            new_street = True
        elif action_type == self.ActionType.CHECK:
            if self.turn == 1 - self.small_blind_player:
                new_street = True # big blind checks mean next street
        elif action_type == self.ActionType.RAISE:
            self.bets[self.turn] = bet_amount
            self.min_raise = max(self.min_raise, bet_amount - self.bets[1 - self.turn])
        else:
            assert False

        if new_street:
            self._next_street()
            if self.street > 3 and not new_game:
                winner = self._get_winner()
                self._update_bankrolls(winner)
                new_game = True
                
        if not new_game and not new_street:
            self.turn = 1 - self.turn

        obs, reward, terminated, truncated, info = self._get_obs(winner)
        if terminated:
            print("Game is terminated. Final bankrolls:", self.bankrolls)

        if new_game:
            self._start_new_game()

        return obs, reward, terminated, truncated, info
        


if __name__ == "__main__":
    env = PokerEnv(100)
