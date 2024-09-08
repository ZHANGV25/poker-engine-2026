"""
CMU Poker Bot Competition Game Engine 2025

People working on this code, please refer to:
https://gymnasium.farama.org
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Keep in mind gym doesn't inherently support multi-agent environments.
We will have to use the Tuple space to represent the observation space and action space for each agent.

"""

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


class PokerEnv(gym.Env):

    class ActionType(Enum):
        FOLD = 1
        RAISE = 2
        CHECK = 3
        CALL = 4
        INVALID = 5

    SMALL_BIND  = 1
    BIG_BLIND = 2

    NO_DISCARD = -1

    def __init__(self, num_games) -> None:
        super().__init__()
        self.num_games = num_games

        # Action space is a Tuple (total_bet, card_to_discard) 
        # where action is a Discrete(4) and amount is a Discrete(400)
        # If bet_amount =< opp_bet, it is considered as folding
        # If bet_amount > opp_bet, it is considered as raising (must be a legal raise)
        # If bet_amount == opp_bet, it is considered as calling 
        # Card to discard and show is only relevant in the discard game. Discrete(4) cuz we have 3 cards
        # Keep ing Mind: THIS IS Cumulative betting
        self.action_space = spaces.Tuple([spaces.Discrete(99, start=2), spaces.Discrete(4, start=-1)])

        # Card space is a Discrete(53), -1 means the card is not shown
        cards_space = spaces.Discrete(53, start=-1)

        # Single observation space is a Dict. 
        # Since we have two players, turn is a Discrete(2)
        # Make sure to check (turn == agent_num) before taking an action
        # opp_shown_card is "0" if the opp's card is not shown
        # Two players, so the observation space is a Tuple of two single_observation_spaces
        observation_space_one_player = spaces.Dict({
            "turn": spaces.Discrete(2),
            "my_cards": spaces.Tuple([cards_space for _ in range(3)]),
            "board_cards": spaces.Tuple([cards_space for _ in range(5)]),
            "my_bet": spaces.Discrete(100, start=0), 
            "opp_bet": spaces.Discrete(100, start=0), 
            "my_bankroll": spaces.Box(low=-1, high=1, shape=(1,)), # Normalized bankroll div by 1e4 or something idk
            "opp_shown_cards": spaces.Tuple([cards_space for _ in range(2)]),
            "game_num": spaces.Discrete(self.num_games, start=1),
            "min_raise": spaces.Discrete(100, start=2)
        })

        # Since we have two players, the observation space is a tuple of (observation_space_one_player, observation_space_one_player)
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(2)])

        # New episode
        self.reset(
            seed=int.from_bytes(os.urandom(32))
        )

    def _get_obs(self, player_num: int):
        """
        Returns the observation for the player_num player.
        """ 
        num_cards_to_reveal = -1
        if self.street == 0:
            num_cards_to_reveal = 0
        else:
            num_cards_to_reveal = self.street + 2

        return {
            "turn": self.turn,
            "my_cards": self.player_cards[player_num],
            "board_cards": self.board_cards[:num_cards_to_reveal],
            "my_bet": self.bets[player_num],
            "opp_bet": self.bets[1 - player_num],
            "my_bankroll": self.bankrolls[player_num],
            "opp_shown_cards": self.shown_cards[1 - player_num],
            "game_num": self.game_num,
            "min_raise": self.min_raise
        }
    
    def _end_game(self, winner):
        # End the game, update the bankrolls
        self.bankrolls[winner] += sum(self.bets)

    def _start_new_game(self):
        # Rotate the small blind
        self.small_blind_player = 1 - self.small_blind_player

        # Small blind starts
        self.turn = self.small_blind_player

        # Deal the cards
        cards = np.random.choice(53, 3+3+5, replace=False)
        self.player_cards = [cards[:3], cards[3:6]]
        self.board_cards = cards[6:]

        # Reset the bets, and no cards are shown yet
        self.bets = [0, 0]
        self.shown_cards = [np.array([-1, -1, -1]), np.array([-1, -1, -1])]
        self.min_raise = self.BIG_BLIND

        self.game_num += 1

    def reset(self, seed=None, options=None):
        """
        Resets the entire game.
        """
        super().reset(seed=seed)

        self.small_blind_player = 0
        self.turn = self.small_blind_player
        self.bankrolls = [0, 0]
        self.game_num = 1

        # There are 4 streets: Preflop, Flop, Turn, River
        self.street = 0

        self.min_raise = self.BIG_BLIND

        obs1, obs2 = self._get_obs(0), self._get_obs(1)
        return (obs1, obs2), None

    def _get_action_type(self, action) -> ActionType:
        """
        Validates the action taken by the player. 
        """
        assert self.curr_game_num <= self.num_games
        new_bet, card_to_discard = action

        other_player = 1-self.turn

        # detect if bet is raise, fold, or check
        other_player_old_bet = self.bets[other_player]

        # Discard has to be done in the flop and not any other streets
        if self.street == 1:
            # on the flop, must be valid discard
            if card_to_discard < 0:
                print("Did not discard a card during the flop")
                return self.ActionType.INVALID
        else:
            if card_to_discard != self.NO_DISCARD:
                print("Discarded a card when it wasn't the flop")
                return self.ActionType.INVALID

        # amount curr player is raising
        if new_bet < other_player_old_bet:
            return self.ActionType.FOLD
        elif new_bet == other_player_old_bet:
                
            return self.ActionType.CHECK
        # raise
        raised_by = new_bet - other_player_old_bet
        if raised_by < self.min_raise:
            print("Raise must be at least", self.min_raise, "but was", raised_by)
            return self.ActionType.INVALID
        return self.ActionType.RAISE
    
    def _next_street(self):
        """
        Update to the next street of the game.
        """
        self.street += 1
        self.min_raise = self.BIG_BLIND
        if self.street == 1:
            # Flop
            
        elif self.street == 2:
            # Turn
        elif self.street == 3:
            # River
        else:
            # end round
    
    def step(self, action):
        """
        Takes a step in the game, given the action taken by the active player.
        """
        assert self.curr_game_num <= self.num_games
        bet_amount, card_to_discard = action
        action_type = self._get_action_type(action)

        # We consider invalid actions as folding
        if action_type == self.ActionType.INVALID:
            action_type = self.ActionType.FOLD
        
        if action_type == self.ActionType.FOLD:
            # end game
            self._end_game(1-self.turn)
        if action_type == self.ActionType.CALL:
            # end game
            self._end_game(1-self.turn)
        elif action == self.ActionType.CHECK:
            if self.turn == 1-self.small_blind_player:
                # big blind checked
                # next street
                self.street += 1
            else:
                # other player's action on same street
                pass

        elif action == self.ActionType.RAISE:
            pass
        else:
            # unkown action type
            pass

        
        return observation, reward, terminated, False, info