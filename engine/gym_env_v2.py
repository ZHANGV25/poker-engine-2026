"""
CMU Poker Bot Competition Game Engine 2026

People working on this code, please refer to:
https://gymnasium.farama.org
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Keep in mind gym doesn't inherently support multi-agent environments.
We will have to use the Tuple space to represent the observation space and
action space for each agent.
"""

import logging
import os
from enum import Enum
from itertools import combinations

import gym
import numpy as np
from gym import spaces
from treys import Card, Evaluator

class PokerEnv(gym.Env):
    # Class constants (not player-dependent)
    ANTE_AMOUNT = 1
    MAX_PLAYER_BET = 100
    HAND_SIZE = 5  # Each player gets 5 cards initially

    RANKS = "23456789TJQKA"  # 13 ranks
    SUITS = "cdhs"  # 4 suits = 52 card deck

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer designed for fast lookup & comparison
        """
        if card_int == -1:
            raise ValueError("Cannot convert -1 to card")
        return Card.new(PokerEnv.int_card_to_str(card_int))

    @staticmethod
    def int_card_to_str(card_int: int):
        """Convert card int (0-51) to string like '2c', 'Ah', etc."""
        if card_int == -1:
            return "??"
        RANKS, SUITS = PokerEnv.RANKS, PokerEnv.SUITS
        rank = RANKS[card_int % len(RANKS)]
        suit = SUITS[card_int // len(RANKS)]
        return rank + suit

    @staticmethod
    def card_to_obs_encoding(card_int: int):
        """
        Convert card int to observation space encoding.
        -1 (hidden) -> 0
        0-51 (cards) -> 1-52
        """
        return card_int + 1

    @staticmethod
    def obs_encoding_to_card(obs_val: int):
        """
        Convert observation space encoding to card int.
        0 -> -1 (hidden)
        1-52 -> 0-51 (cards)
        """
        return obs_val - 1

    class ActionType(Enum):
        FOLD = 0
        RAISE = 1
        CHECK = 2
        CALL = 3
        SELECT_CARDS = 4  # Select 2 cards from 5 to keep
        INVALID = 5

    def __init__(self, num_players=6, logger=None, ante_amount=1, num_hands=1):
        """
        Represents a single hand of Bomb Pot PLO with variable player count.

        Args:
            num_players: Number of players (2-6). Default is 6.
            logger: Optional logger instance
            ante_amount: Ante amount per player
            num_hands: Number of hands (for match tracking)
        """
        super().__init__()

        # Validate player count
        assert 2 <= num_players <= 6, f"Must have 2-6 players, got {num_players}"
        self.NUM_PLAYERS = num_players

        self.logger = logger or logging.getLogger(__name__)

        self.ante_amount = ante_amount
        self.min_raise = ante_amount
        self.acting_agent = 0
        self.last_street_bet = None
        self.evaluator = Evaluator()
        self.dealer_button = 0  # Tracks dealer position

        # Action space: (action_type, raise_amount, card_idx_1, card_idx_2)
        # - action_type: FOLD, RAISE, CHECK, CALL, SELECT_CARDS
        # - raise_amount: amount to raise (only used for RAISE action)
        # - card_idx_1, card_idx_2: indices of 2 cards to keep (only used for SELECT_CARDS)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(len(self.ActionType) - 1),  # user can pass any besides INVALID
                spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET (raise amounts 1 to MAX_PLAYER_BET)
                spaces.Discrete(self.HAND_SIZE),  # card index 1 (0-4)
                spaces.Discrete(self.HAND_SIZE),  # card index 2 (0-4)
            ]
        )

        # Card space: Discrete(53) represents -1 to 51
        # We'll map: 0 -> -1 (hidden), 1 -> 0 (first card), ..., 52 -> 51 (last card)
        cards_space = spaces.Discrete(53)  # 0 represents -1, 1-52 represent cards 0-51

        # Observation space for one player (dynamic based on num_players)
        # Each player sees their own cards, community cards, and aggregate info about opponents
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),  # 0=preflop(unused), 1=flop, 2=turn, 3=river
                "acting_agent": spaces.Discrete(self.NUM_PLAYERS),
                "seat": spaces.Discrete(self.NUM_PLAYERS),  # This player's seat number
                "button": spaces.Discrete(self.NUM_PLAYERS),

                "my_cards": spaces.Tuple([cards_space for _ in range(self.HAND_SIZE)]),
                "my_hand":  spaces.Tuple([cards_space for _ in range(2)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "bets": spaces.Tuple(
                    [spaces.Discrete(self.MAX_PLAYER_BET + 1) for _ in range(self.NUM_PLAYERS)]
                ),
                "min_raise": spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET (1 to MAX_PLAYER_BET are valid)
                "max_raise": spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET

                "players_active": spaces.Tuple([spaces.Discrete(2) for _ in range(self.NUM_PLAYERS)]),
                "valid_actions": spaces.MultiBinary(len(self.ActionType) - 1),
            }
        )

        # Observation space is a tuple of N player observations (where N = num_players)
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(self.NUM_PLAYERS)])

        # New episode
        import random
        self.reset(seed=random.randint(0, 2**32 - 1))

    def _get_valid_actions(self, player_num: int):
        """
        Returns a list of valid actions for the given player.
        [FOLD, RAISE, CHECK, CALL, SELECT_CARDS]
        """
        # Card selection phase - only SELECT_CARDS is valid
        if self.selection_phase:
            valid_actions = [0] * (len(self.ActionType) - 1)
            if player_num == self.acting_agent and not self.has_fixed[player_num]:
                valid_actions[self.ActionType.SELECT_CARDS.value] = 1
            return valid_actions

        # Not active anymore (folded), no valid actions
        if not self.players_active[player_num]:
            return [0] * (len(self.ActionType) - 1)

        # Betting phase
        valid_actions = [1, 1, 1, 1, 0]  # FOLD, RAISE, CHECK, CALL always considered, SELECT_CARDS disabled

        max_bet = max(self.bets)
        my_bet = self.bets[player_num]

        # Can't CHECK if someone has a larger bet than you
        if my_bet < max_bet:
            valid_actions[self.ActionType.CHECK.value] = 0

        # Can't CALL if you already have the max bet
        if my_bet == max_bet:
            valid_actions[self.ActionType.CALL.value] = 0

        # Can't RAISE if already at max bet limit
        if max_bet >= self.MAX_PLAYER_BET:
            valid_actions[self.ActionType.RAISE.value] = 0

        return valid_actions
    
    def _handle_hand_selection(self, action: tuple[int, int, int, int]):
        """
        Handle the card fixing phase where players select 2 cards from 5 to keep.
        This happens on the flop (street 1) before betting begins.
        """
        action_type, _, card_idx_1, card_idx_2 = action
        p = self.acting_agent

        # Validate action type
        if action_type != self.ActionType.SELECT_CARDS.value:
            self.logger.error(f"Player {p} must select cards, chose action {action_type}; defaulting to (0,1)")
            card_idx_1, card_idx_2 = 0, 1

        # Validate card indices
        if not (0 <= card_idx_1 < self.HAND_SIZE and 0 <= card_idx_2 < self.HAND_SIZE and card_idx_1 != card_idx_2):
            self.logger.error(f"Player {p} invalid selection {card_idx_1},{card_idx_2}; defaulting to (0,1)")
            card_idx_1, card_idx_2 = 0, 1

        # Record chosen pair
        self.chosen_pairs[p] = (card_idx_1, card_idx_2)
        self.has_fixed[p] = True

        # Mask the 3 unchosen cards to -1 (keep only the 2 selected)
        keep = {card_idx_1, card_idx_2}
        for k in range(self.HAND_SIZE):
            if k not in keep:
                self.player_cards[p][k] = -1

        # Check if all players have fixed their cards
        if all(self.has_fixed):
            self.selection_phase = False
            # Start betting from left of dealer button
            self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS
            # Skip folded players (shouldn't happen during selection, but just in case)
            while not self.players_active[self.acting_agent]:
                self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS
        else:
            # Move to next player who hasn't fixed yet
            self.acting_agent = (p + 1) % self.NUM_PLAYERS
            while self.has_fixed[self.acting_agent]:
                self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS

        # Return observation (no winner yet)
        return self._get_obs(winner=None, invalid_action=False)


    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for a specific player.

        Args:
            player_num: Player index (0-5)

        Returns:
            Tuple of (observation dict, info dict)
        """
        # Determine community cards visibility based on street
        if self.street == 1:
            visible_count = 3  # Flop
        elif self.street == 2:
            visible_count = 4  # Turn
        else:  # street == 3 or higher (showdown)
            visible_count = 5  # River / Showdown

        # Build community_cards tuple with proper visibility
        community_visible = self.community_cards[:visible_count]
        community_hidden = [-1] * (5 - visible_count)
        community_cards_obs = tuple(
            self.card_to_obs_encoding(c) for c in (community_visible + community_hidden)
        )

        # Build my_cards tuple (encoded)
        my_cards_obs = tuple(
            self.card_to_obs_encoding(c) for c in self.player_cards[player_num]
        )

        # Build my_hand tuple (the 2 selected cards)
        if self.has_fixed[player_num]:
            my_hand_cards = [c for c in self.player_cards[player_num] if c != -1]
            assert len(my_hand_cards) == 2, f"Player {player_num} should have 2 cards after fixing, has {len(my_hand_cards)}"
            my_hand_obs = tuple(self.card_to_obs_encoding(c) for c in my_hand_cards)
        else:
            my_hand_obs = (-1, -1)  # Not yet fixed, show as hidden

        # Build observation dict
        obs = {
            "street": self.street,
            "acting_agent": self.acting_agent,
            "seat": player_num,
            "button": self.dealer_button,
            "my_cards": my_cards_obs,
            "my_hand": my_hand_obs,
            "community_cards": community_cards_obs,
            "bets": tuple(self.bets),
            "min_raise": self.min_raise,
            "max_raise": self.MAX_PLAYER_BET - max(self.bets),
            "valid_actions": self._get_valid_actions(player_num),
            "players_active": tuple(self.players_active),
        }

        # Handle all-in situation
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]

        # Build info dict for logging (human-readable)
        info = {
            "player_cards": [self.int_card_to_str(c) for c in self.player_cards[player_num] if c != -1],
            "community_cards": [self.int_card_to_str(c) for c in self.community_cards[:visible_count]],
        }

        return obs, info

    def _get_obs(self, winner, invalid_action=False):
        """
        Returns observations for all players, rewards, terminated status, and info.

        winner can be:
        - None: game continues
        - int (0-5): single winner
        - list of ints: multiple winners (tie)
        """
        # Get observations for all players
        obs_list = []
        info_list = []
        for p in range(self.NUM_PLAYERS):
            obs, info = self._get_single_player_obs(p)
            obs_list.append(obs)
            info_list.append(info)

        # Calculate rewards
        pot = sum(self.bets)
        rewards = [0] * self.NUM_PLAYERS

        if winner is not None:
            # Handle ties (winner is a list)
            if isinstance(winner, list):
                # Split pot among winners
                pot_share = pot / len(winner)
                for p in range(self.NUM_PLAYERS):
                    if p in winner:
                        rewards[p] = pot_share - self.bets[p]
                    else:
                        rewards[p] = -self.bets[p]
            else:
                # Single winner
                for p in range(self.NUM_PLAYERS):
                    if p == winner:
                        rewards[p] = pot - self.bets[p]
                    else:
                        rewards[p] = -self.bets[p]

        terminated = winner is not None
        truncated = False

        # Info dict for showdown
        is_showdown = terminated and self.street >= 1
        info = {}
        if is_showdown:
            for p in range(self.NUM_PLAYERS):
                info[f"player_{p}_cards"] = info_list[p]["player_cards"]
            info["community_cards"] = info_list[0]["community_cards"]
            info["pot"] = pot
            info["winner"] = winner
            info["invalid_action"] = invalid_action

        return tuple(obs_list), tuple(rewards), terminated, truncated, info

    def _draw_card(self):
        return self.cards.pop()

    def reset(self, *, seed=None, options=None):
        """
        Resets the game for a new hand of Bomb Pot PLO with variable players.

        Bomb Pot: All players post equal antes, no pre-flop betting.
        Game starts at street 1 (flop).
        Players then select 2 cards from their 5 hole cards to keep.
        Community cards revealed progressively: 3 at flop, +1 at turn, +1 at river.

        options is a dict with the following keys:
        - cards: a list of 52 cards to be used in the game (optional)
        - dealer_button: which player is the dealer (optional, default=0)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        DECK_SIZE = len(self.RANKS) * len(self.SUITS)  # 52 cards

        # Initialize game state variables
        self.street = 1  # Start at flop (skip pre-flop in bomb pot)
        self.selection_phase = True  # Players must fix cards before betting
        self.chosen_pairs = [None] * self.NUM_PLAYERS
        self.has_fixed = [False] * self.NUM_PLAYERS
        self.players_active = [True] * self.NUM_PLAYERS
        self.bets = [self.ANTE_AMOUNT] * self.NUM_PLAYERS  # Everyone posts ante
        self.min_raise = self.ANTE_AMOUNT
        self.last_street_bet = self.ANTE_AMOUNT

        # Initialize deck
        self.cards = list(np.arange(DECK_SIZE))
        np.random.shuffle(self.cards)

        # Override with any provided options
        if options is not None:
            if "cards" in options:
                self.cards = list(options["cards"])
            if "dealer_button" in options:
                self.dealer_button = options["dealer_button"]

        # Deal 5 cards to each player
        self.player_cards = [[self._draw_card() for _ in range(self.HAND_SIZE)] for _ in range(self.NUM_PLAYERS)]

        # Deal 5 community cards (revealed progressively: 3 at flop, +1 at turn, +1 at river)
        self.community_cards = [self._draw_card() for _ in range(5)]

        # First player to act (left of dealer button)
        self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS

        # Generate observations for all players
        obs_list_raw = [self._get_single_player_obs(p) for p in range(self.NUM_PLAYERS)]
        obs_list_filtered = tuple(obs_item for (obs_item, _) in obs_list_raw)
        info = {}

        return obs_list_filtered, info

    def _next_street(self):
        """
        Advance to the next street of the game.
        Streets: 1 (flop) -> 2 (turn) -> 3 (river) -> showdown
        """
        self.street += 1
        self.min_raise = self.ANTE_AMOUNT

        # Verify all active players have equal bets
        active_bets = [self.bets[p] for p in range(self.NUM_PLAYERS) if self.players_active[p]]
        if active_bets:
            max_bet = max(active_bets)
            for bet in active_bets:
                assert bet == max_bet, f"Bet amounts not equal when advancing street: {self.bets}"

        self.last_street_bet = max(self.bets)

        # First player to act is left of dealer
        self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS
        # Skip inactive players
        while not self.players_active[self.acting_agent]:
            self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS

    def _evaluate_plo_hand(self, hole_cards: list[int], board_cards: list[int]) -> int:
        """
        Evaluate PLO hand: must use exactly 2 from hole_cards and exactly 3 from board_cards.
        Returns the best (lowest) hand rank.

        Args:
            hole_cards: List of 2 card integers
            board_cards: List of 5 card integers

        Returns:
            Best hand rank (lower is better)
        """
        assert len(hole_cards) == 2, f"Expected 2 hole cards, got {len(hole_cards)}"
        assert len(board_cards) == 5, f"Expected 5 board cards, got {len(board_cards)}"

        # Convert to treys encoding
        hole_treys = [self.int_to_card(c) for c in hole_cards]
        board_treys = [self.int_to_card(c) for c in board_cards]

        # Since we must use exactly 2 hole cards, there's only 1 combination
        # We need to try all combinations of 3 board cards (C(5,3) = 10)
        best_rank = float('inf')

        for board_indices in combinations(range(5), 3):
            # Select 3 cards from board
            board_subset = [board_treys[i] for i in board_indices]

            # Evaluate: hole_treys (2 cards) + board_subset (3 cards)
            rank = self.evaluator.evaluate(hole_treys, board_subset)
            best_rank = min(best_rank, rank)

        return int(best_rank)

    def _get_winner(self):
        """
        Returns the winner(s) of the game using PLO evaluation.

        Returns:
            - int (0-5): single winner
            - list of ints: multiple winners (tie)
        """
        board_cards = self.community_cards[:5]  # Use first 5 community cards

        # Evaluate each active player's hand
        player_ranks = {}
        for p in range(self.NUM_PLAYERS):
            if self.players_active[p]:
                # Get the 2 cards this player kept (non -1 cards)
                player_cards = [c for c in self.player_cards[p] if c != -1]
                assert len(player_cards) == 2, f"Player {p} should have exactly 2 cards, has {len(player_cards)}"

                # Evaluate PLO hand
                rank = self._evaluate_plo_hand(player_cards, board_cards)
                player_ranks[p] = rank

                self.logger.debug(f"Player {p} cards: {[self.int_card_to_str(c) for c in player_cards]}, rank: {rank}")

        # Find winner(s) - lowest rank wins
        if not player_ranks:
            return None

        best_rank = min(player_ranks.values())
        winners = [p for p, rank in player_ranks.items() if rank == best_rank]

        # Return single winner or list of winners for tie
        if len(winners) == 1:
            return winners[0]
        else:
            return winners

    def step(self, action: tuple[int, int, int, int]):
        """
        Takes a step in the game, given the action taken by the active player.

        `action`: (action_type, raise_amount, card_idx_1, card_idx_2)
            - `action_type`: int, index of the action type [FOLD, RAISE, CHECK, CALL, SELECT_CARDS]
            - `raise_amount`: int, how much to raise (only for RAISE action)
            - `card_idx_1`: int, first card index to keep (only for SELECT_CARDS)
            - `card_idx_2`: int, second card index to keep (only for SELECT_CARDS)
        """
        # Phase 1: Handle card selection phase
        if self.selection_phase:
            return self._handle_hand_selection(action)

        # Phase 2: Extract and validate action
        action_type, raise_amount, card_idx_1, card_idx_2 = action
        valid_actions = self._get_valid_actions(self.acting_agent)

        # Log for debugging
        self.logger.debug(
            f"Player {self.acting_agent} action: {self.ActionType(action_type).name}, "
            f"raise={raise_amount}, street={self.street}, bets={self.bets}"
        )

        # Validate action type
        if not valid_actions[action_type]:
            action_name = self.ActionType(action_type).name
            valid_names = [self.ActionType(i).name for i, v in enumerate(valid_actions) if v]
            self.logger.error(
                f"Player {self.acting_agent} invalid action {action_name}. "
                f"Valid: {valid_names}. Treating as FOLD."
            )
            action_type = self.ActionType.INVALID.value

        # Validate raise amount if raising
        if action_type == self.ActionType.RAISE.value:
            max_bet = max(self.bets)
            max_raise = self.MAX_PLAYER_BET - max_bet
            if not (self.min_raise <= raise_amount <= max_raise):
                self.logger.error(
                    f"Player {self.acting_agent} invalid raise {raise_amount}. "
                    f"Must be in [{self.min_raise}, {max_raise}]. Treating as FOLD."
                )
                action_type = self.ActionType.INVALID.value

        # Phase 3: Track state for street completion detection
        # Initialize tracking variables on first action of street
        if not hasattr(self, 'actions_this_street'):
            self.actions_this_street = 0

        # Phase 4: Process action
        winner = None
        street_complete = False

        if action_type in (self.ActionType.FOLD.value, self.ActionType.INVALID.value):
            # Player folds
            self.logger.debug(f"Player {self.acting_agent} folded")
            self.players_active[self.acting_agent] = False

            # Check if only one player remains
            active_players = [p for p in range(self.NUM_PLAYERS) if self.players_active[p]]
            if len(active_players) == 1:
                winner = active_players[0]
                self.logger.debug(f"Player {winner} wins by elimination")

        elif action_type == self.ActionType.CALL.value:
            # Match the current max bet
            max_bet = max(self.bets)
            self.bets[self.acting_agent] = max_bet
            self.logger.debug(f"Player {self.acting_agent} calls to {max_bet}")
            self.actions_this_street += 1

        elif action_type == self.ActionType.CHECK.value:
            # No change to bets
            self.logger.debug(f"Player {self.acting_agent} checks")
            self.actions_this_street += 1

        elif action_type == self.ActionType.RAISE.value:
            # Increase bet
            max_bet = max(self.bets)
            new_bet = max_bet + raise_amount

            # Update min_raise for next raise (must be at least this raise amount)
            raise_so_far = max_bet - self.last_street_bet
            self.min_raise = raise_so_far + raise_amount

            self.bets[self.acting_agent] = new_bet
            self.logger.debug(f"Player {self.acting_agent} raises to {new_bet}")

            # Reset action counter - everyone must act again
            self.actions_this_street = 1  # This player has acted

        # Phase 5: Check for street completion (only if no winner yet)
        if winner is None:
            # Check if all active players have equal bets
            active_bets = [self.bets[p] for p in range(self.NUM_PLAYERS) if self.players_active[p]]
            all_bets_equal = len(set(active_bets)) == 1

            # Check if everyone has acted
            num_active = sum(self.players_active)
            everyone_acted = self.actions_this_street >= num_active

            if all_bets_equal and everyone_acted:
                street_complete = True
                self.logger.debug(f"Street {self.street} complete")

        # Phase 6: Advance to next street or determine winner
        if street_complete:
            self._next_street()
            # Reset tracking for new street
            self.actions_this_street = 0

            # Check if game is over (after river)
            if self.street > 3:
                winner = self._get_winner()
                self.logger.debug(f"Showdown - Winner: {winner}")

        # Phase 7: Advance to next active player (if game not over and street not complete)
        if winner is None and not street_complete:
            # Move to next active player
            self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS

            # Skip inactive players
            iterations = 0
            while not self.players_active[self.acting_agent]:
                self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS
                iterations += 1
                if iterations >= self.NUM_PLAYERS:
                    # Safety check - should never happen
                    self.logger.error("No active players found!")
                    break

        # Phase 8: Cap min_raise at max possible
        self.min_raise = min(self.min_raise, self.MAX_PLAYER_BET - max(self.bets))

        # Phase 9: Build and return observation
        invalid_action = (action_type == self.ActionType.INVALID.value)
        return self._get_obs(winner=winner, invalid_action=invalid_action)

