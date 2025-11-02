"""
CMU Poker Bot Competition Game Engine 2025

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

import gym
import numpy as np
from gym import spaces
from treys import Card, Evaluator

class WrappedEval(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, hand: list[int], board: list[int]) -> int:
        """
        This is the function that the user calls to get a hand rank.

        No input validation because that's cycles!
        """

        def ace_to_ten(treys_card: int):
            """Convert trey's representation of an Ace to trey's representation of a Ten"""
            as_str = Card.int_to_str(treys_card)
            alt = as_str.replace("A", "T")  # treys uses "T" for ten
            alt_as_treys = Card.new(alt)
            return alt_as_treys

        # check for the edge case of Ace used as high card after a 9
        alt_hand = list(map(ace_to_ten, hand))
        alt_board = list(map(ace_to_ten, board))

        reg_score = super().evaluate(hand, board)  # regular score
        alt_score = super().evaluate(
            alt_hand, alt_board
        )  # score if aces were tens

        if alt_score < reg_score:
            # explicit branch for pytorch coverage
            return alt_score

        return reg_score

class PokerEnv(gym.Env):
    SMALL_BLIND_PLAYER = 0
    BIG_BLIND_PLAYER = 1
    MAX_PLAYER_BET = 100

    RANKS = "23456789A"
    SUITS = "dhs"  # diamonds hearts spade

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer desiged for fast lookup & comparison
        """
        return Card.new(PokerEnv.int_card_to_str(card_int))

    @staticmethod
    def int_card_to_str(card_int: int):
        RANKS, SUITS = PokerEnv.RANKS, PokerEnv.SUITS
        rank = RANKS[card_int % len(RANKS)]
        suit = SUITS[card_int // len(RANKS)]
        return rank + suit

    class ActionType(Enum):
        FOLD = 0
        RAISE = 1
        CHECK = 2
        CALL = 3
        DISCARD = 4
        INVALID = 5

    def __init__(self, logger=None, small_blind_amount=1, num_hands=1, num_players=6):
        """
        Represents a single hand of poker.

        Args:
            logger: Optional logger instance
            small_blind_amount: Amount for small blind (deprecated for bomb-pot)
            num_hands: Number of hands in the match
            num_players: Number of players (1-6, default 6)
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.num_players = min(max(1, num_players), 6)  # Clamp between 1 and 6
        self.num_hands = num_hands
        self.current_hand = 0
        self._reset_count = 0  # Track number of resets

        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = small_blind_amount * 2
        self.min_raise = self.big_blind_amount
        self.acting_agent = PokerEnv.SMALL_BLIND_PLAYER
        self.last_street_bet = None
        self.evaluator = WrappedEval()

        # Action space is a Tuple (action_type, raise_amount, card_to_discard)
        # where action is a Discrete(4), raise_amount is a Discrete(100), and card_to_discard is a Discrete(3) (-1 means no card is discarded)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(len(self.ActionType) - 1),  # user can pass any besides INVALID
                spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                spaces.Discrete(3, start=-1),
            ]
        )

        # Card space is a Discrete(28), -1 means the card is not shown
        cards_space = spaces.Discrete((len(self.SUITS) * len(self.RANKS)) + 1, start=-1)

        # Single observation space is a Dict.
        # Since we have two players, acting_agent is a Discrete(2)
        # Make sure to check (acting_agent == agent_num) before taking an action
        # opp_shown_card is "0" if the opp's card is not shown
        # Two players, so the observation space is a Tuple of two single_observation_spaces
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),
                "acting_agent": spaces.Discrete(2),
                "my_cards": spaces.Tuple([cards_space for _ in range(2)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_discarded_card": cards_space,
                "opp_drawn_card": cards_space,
                "min_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "max_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "valid_actions": spaces.MultiBinary(len(self.ActionType) - 1),
            }
        )

        # Since we have two players, the observation space is a tuple of
        # (observation_space_one_player, observation_space_one_player)
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(2)])

        # New episode
        self.reset(seed=int.from_bytes(os.urandom(32)))

    def _get_valid_actions(self, player_num: int):
        """
        Returns valid action mask for 6-player bomb-pot rules.

        Street 0 (card selection): No betting actions, only card selection (handled separately)
        Street 1 (betting): CHECK, CALL, RAISE, FOLD
        Street 2+ (showdown): No actions

        Returns:
            List of 5 binary values indicating valid actions:
            [FOLD, RAISE, CHECK, CALL, DISCARD]
        """
        valid_actions = [1, 1, 1, 1, 1]

        # Street 0 (card selection): disable betting actions
        if self.street == 0:
            valid_actions[self.ActionType.FOLD.value] = 0
            valid_actions[self.ActionType.RAISE.value] = 0
            valid_actions[self.ActionType.CHECK.value] = 0
            valid_actions[self.ActionType.CALL.value] = 0
            # Card selection is handled separately, not via DISCARD action
            valid_actions[self.ActionType.DISCARD.value] = 0
            return valid_actions

        # Street 2+ (showdown): no actions
        if self.street >= 2:
            return [0, 0, 0, 0, 0]

        # Street 1 (betting round) - disable DISCARD
        valid_actions[self.ActionType.DISCARD.value] = 0

        # Get max bet across all players
        max_bet = max(self.bets)
        my_bet = self.bets[player_num]

        # Can't CHECK if facing a bet
        if my_bet < max_bet:
            valid_actions[self.ActionType.CHECK.value] = 0

        # Can't CALL if bets are equal
        if my_bet == max_bet:
            valid_actions[self.ActionType.CALL.value] = 0

        # Can't RAISE if at bet cap
        if max_bet >= self.MAX_PLAYER_BET:
            valid_actions[self.ActionType.RAISE.value] = 0

        return valid_actions

    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for a single player matching PlayerObservation type.

        Args:
            player_num: Seat index (0-5)

        Returns:
            PlayerObservation dict with all required fields
        """
        from poker_types import NUM_SEATS, NUM_BOARDS, BET_CAP, BOARD_CARDS_PER_BOARD

        # Community cards visibility based on street
        # Per user clarification: cards are revealed during street 0 for "informed selection"
        # Street 0 (card selection): cards VISIBLE (players see boards before choosing)
        # Street 1+ (betting/showdown): cards visible
        # All streets: reveal all 15 community cards across 3 boards
        community_cards = [
            self.community_cards[i*BOARD_CARDS_PER_BOARD:(i+1)*BOARD_CARDS_PER_BOARD]
            for i in range(NUM_BOARDS)
        ]

        # Calculate total pot
        pot_total = sum(self.bets)

        # Valid actions
        valid_actions = self._get_valid_actions(player_num)

        # Build observation matching PlayerObservation type
        obs = {
            'seat': player_num,
            'acting_seat': self.acting_agent,
            'street': self.street,
            'hole_cards': list(self.player_cards[player_num]),
            'community_cards': community_cards,
            'bets': list(self.bets),
            'my_stack': self.stacks[player_num],
            'all_stacks': list(self.stacks),
            'button_position': self.button_position,
            'pot_total': pot_total,
            'min_raise': self.min_raise,
            'max_raise': min(BET_CAP, self.MAX_PLAYER_BET - max(self.bets)),
            'valid_actions': valid_actions,
            'time_used': 0.0,  # TODO: Implement time tracking
            'time_left': 300.0,  # TODO: Implement time tracking
            'hand_number': self.current_hand,
        }

        # Add kept_cards if card selection phase is complete
        if self.kept_cards[player_num]:
            obs['kept_cards'] = list(self.kept_cards[player_num])

        # All-in situation (cap min_raise to max_raise)
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]

        return obs

    def _get_obs(self, winner, invalid_action=False):
        """
        Returns the observation for both players.
        """
        obs0, info0 = self._get_single_player_obs(0)
        obs1, info1 = self._get_single_player_obs(1)
        if winner == 0:
            reward = (min(self.bets), -min(self.bets))
        elif winner == 1:
            reward = (-min(self.bets), min(self.bets))
        else:
            reward = (0, 0)
        terminated = winner is not None
        truncated = False

        is_showdown = terminated and self.street > 3
        info = (
            {
                "player_0_cards": info0["player_cards"],
                "player_1_cards": info1["player_cards"],
                "community_cards": info0["community_cards"],
                "invalid_action": invalid_action,
            }
            if is_showdown
            else {}
        )

        return (obs0, obs1), reward, terminated, truncated, info

    def _draw_card(self):
        drawn_card = self.cards[0]
        self.cards = self.cards[1:]
        return drawn_card

    def reset(self, *, seed=None, options=None):
        """
        Resets the game for a new hand.

        For 6-player triple-board bomb-pot:
        - Deals 5 hole cards to each active player
        - Deals 15 community cards (3 boards × 5 cards)
        - Forces $1 bomb-pot ante from all players
        - Starts at street 0 (card selection phase)

        Args:
            seed: Random seed for reproducibility
            options: Dict with optional keys:
                - 'cards': List of card integers to rig the deck
                - 'button_seat': Override button position

        Returns:
            observations: Tuple of 6 PlayerObservation dicts (None for empty seats)
            info: Dict with game metadata
        """
        from poker_types import NUM_SEATS, NUM_BOARDS, HOLE_CARDS_PER_PLAYER, BOMB_POT_ANTE

        super().reset(seed=seed)

        # Button position (rotates each hand)
        self.button_position = self.current_hand % NUM_SEATS

        # Initialize game state
        self.street = 0  # 0 = card selection, 1 = betting, 2+ = showdown
        self.bets = [0] * NUM_SEATS
        self.stacks = [0] * NUM_SEATS  # Net gain/loss tracking (infinite bankroll)
        self.folded_players = set()  # Track who has folded
        self.kept_cards = [[] for _ in range(NUM_SEATS)]  # Track card selection

        # Set up deck
        DECK_SIZE = 52
        self.cards = np.arange(DECK_SIZE)
        np.random.shuffle(self.cards)

        # Override options if provided
        if options is not None:
            self.cards = options.get("cards", self.cards)
            self.button_position = options.get("button_seat", self.button_position)

        # Deal hole cards (5 per player, only to active players)
        self.player_cards = []
        for seat in range(NUM_SEATS):
            if seat < self.num_players:
                self.player_cards.append([self._draw_card() for _ in range(HOLE_CARDS_PER_PLAYER)])
            else:
                self.player_cards.append([])  # Empty for inactive seats

        # Deal community cards (15 total = 3 boards × 5 cards)
        self.community_cards = [self._draw_card() for _ in range(15)]

        # Apply bomb-pot ante for active players
        for seat in range(self.num_players):
            self.bets[seat] = BOMB_POT_ANTE
            self.stacks[seat] -= BOMB_POT_ANTE  # Deduct from stack

        # First action is card selection, starts left of button
        self.acting_agent = (self.button_position + 1) % NUM_SEATS
        self.min_raise = BOMB_POT_ANTE
        self.last_street_bet = BOMB_POT_ANTE

        # Generate observations for all seats
        observations = []
        for seat in range(NUM_SEATS):
            if seat < self.num_players:
                obs = self._get_single_player_obs(seat)
                observations.append(obs)
            else:
                observations.append(None)  # Null observation for empty seat

        info = {
            'hand_number': self.current_hand,
            'button_position': self.button_position,
        }

        # Increment hand counter for next reset (skip first reset in __init__)
        self._reset_count += 1
        if self._reset_count > 1:
            self.current_hand += 1

        return tuple(observations), info

    def _next_street(self):
        """
        Update to the next street of the game.
        """
        self.street += 1
        self.min_raise = self.big_blind_amount
        assert self.bets[0] == self.bets[1], self.logger.log(f"Bet amounts are not equal: {self.bets}")
        self.last_street_bet = self.bets[0]
        self.acting_agent = self.small_blind_player

    def _get_winner(self):
        """
        Returns the winner of the game.
        """
        board_cards = list(map(self.int_to_card, self.community_cards))
        player_0_cards = list(map(self.int_to_card, [c for c in self.player_cards[0] if c != -1]))
        player_1_cards = list(map(self.int_to_card, [c for c in self.player_cards[1] if c != -1]))
        assert len(player_0_cards) == 2 and len(player_1_cards) == 2 and len(board_cards) == 5
        player_0_hand_rank = self.evaluator.evaluate(
            player_0_cards,
            board_cards,
        )
        player_1_hand_rank = self.evaluator.evaluate(
            player_1_cards,
            board_cards,
        )

        self.logger.debug(f"(get winner) Player 0 cards: {list(map(Card.int_to_str, player_0_cards))}; Player 1 cards: {list(map(Card.int_to_str, player_1_cards))}")
        self.logger.debug(f"Determined winner based on hand scores; p0 score: {player_0_hand_rank}; p1 score: {player_1_hand_rank}")

        # showdown
        if player_0_hand_rank == player_1_hand_rank:
            winner = -1  # tie
        elif player_1_hand_rank < player_0_hand_rank:
            winner = 1
        else:
            winner = 0
        return winner

    def step(self, action: tuple[int, int, int]):
        """
        Takes a step in the game for 6-player bomb-pot.

        Action format depends on street:
        - Street 0 (card selection): action = (card_idx_1, card_idx_2, -1)
          - card_idx_1, card_idx_2: indices (0-4) of cards to KEEP from hole_cards
        - Street 1 (betting): action = (action_type, raise_amount, -1)
          - action_type: FOLD, CHECK, CALL, or RAISE
          - raise_amount: amount to raise (if RAISE action)

        Returns:
            observations: Tuple of 6 PlayerObservation dicts (None for empty seats)
            rewards: Tuple of 6 floats (0 until hand completes)
            terminated: bool, True if hand is over
            truncated: bool, always False
            info: dict with hand metadata
        """
        from poker_types import NUM_SEATS, HOLE_CARDS_PER_PLAYER, KEPT_CARDS_PER_PLAYER

        # Unpack action
        param1, param2, param3 = action

        invalid_action = False
        winner = None
        terminated = False

        # ===================================================================
        # STREET 0: CARD SELECTION PHASE
        # ===================================================================
        if self.street == 0:
            # Action format: (card_idx_1, card_idx_2, -1)
            card_idx_1, card_idx_2 = param1, param2

            acting_player = self.acting_agent
            player_hole_cards = self.player_cards[acting_player]

            # Validate card selection
            valid_indices = list(range(HOLE_CARDS_PER_PLAYER))

            # Check for invalid indices
            if card_idx_1 not in valid_indices or card_idx_2 not in valid_indices:
                self.logger.error(
                    f"Player {acting_player} selected invalid card indices: {card_idx_1}, {card_idx_2}. "
                    f"Valid indices: {valid_indices}"
                )
                invalid_action = True
                self.folded_players.add(acting_player)

            # Check for duplicate selection
            elif card_idx_1 == card_idx_2:
                self.logger.error(
                    f"Player {acting_player} selected same card twice: index {card_idx_1}"
                )
                invalid_action = True
                self.folded_players.add(acting_player)

            # Valid selection
            else:
                # Store the kept cards (actual card values, not indices)
                kept_card_1 = player_hole_cards[card_idx_1]
                kept_card_2 = player_hole_cards[card_idx_2]
                self.kept_cards[acting_player] = [kept_card_1, kept_card_2]

                self.logger.debug(
                    f"Player {acting_player} kept cards at indices {card_idx_1}, {card_idx_2}: "
                    f"cards {kept_card_1}, {kept_card_2}"
                )

            # Advance to next player
            self.acting_agent = (self.acting_agent + 1) % NUM_SEATS

            # Skip empty seats
            while self.acting_agent >= self.num_players:
                self.acting_agent = (self.acting_agent + 1) % NUM_SEATS

            # Check if all active players have selected
            players_selected = sum(1 for i in range(self.num_players) if self.kept_cards[i])
            if players_selected >= (self.num_players - len(self.folded_players)):
                # All active players have selected, advance to betting round
                self.street = 1
                self.acting_agent = (self.button_position + 1) % NUM_SEATS
                # Skip folded/empty players
                while self.acting_agent >= self.num_players or self.acting_agent in self.folded_players:
                    self.acting_agent = (self.acting_agent + 1) % NUM_SEATS

                self.logger.info(f"Card selection complete. Advancing to street 1 (betting)")

        # ===================================================================
        # STREET 1: BETTING ROUND
        # ===================================================================
        elif self.street == 1:
            # Action format: (action_type, raise_amount, -1)
            action_type, raise_amount = param1, param2

            acting_player = self.acting_agent
            valid_actions = self._get_valid_actions(acting_player)

            # Validate action type
            if action_type not in range(len(self.ActionType) - 1):
                self.logger.error(f"Player {acting_player} invalid action type: {action_type}")
                invalid_action = True
                self.folded_players.add(acting_player)

            elif not valid_actions[action_type]:
                action_name = self.ActionType(action_type).name
                valid_action_names = [self.ActionType(i).name for i, is_valid in enumerate(valid_actions) if is_valid]
                self.logger.error(
                    f"Player {acting_player} attempted invalid action: {action_name}. "
                    f"Valid actions: {valid_action_names}"
                )
                invalid_action = True
                self.folded_players.add(acting_player)

            # Handle each action type
            elif action_type == self.ActionType.FOLD.value:
                self.logger.debug(f"Player {acting_player} folded")
                self.folded_players.add(acting_player)

                # Check if only one player remains
                active_players = [i for i in range(self.num_players) if i not in self.folded_players]
                if len(active_players) == 1:
                    # Winner by default
                    winner = active_players[0]
                    terminated = True
                    self.logger.info(f"Player {winner} wins by elimination")

            elif action_type == self.ActionType.CHECK.value:
                self.logger.debug(f"Player {acting_player} checked")
                # Check is valid (validated by _get_valid_actions)

            elif action_type == self.ActionType.CALL.value:
                max_bet = max(self.bets)
                call_amount = max_bet - self.bets[acting_player]
                self.bets[acting_player] = max_bet
                self.stacks[acting_player] -= call_amount
                self.logger.debug(f"Player {acting_player} called {call_amount} (total bet: {max_bet})")

            elif action_type == self.ActionType.RAISE.value:
                from poker_types import BET_CAP

                # Validate raise amount
                max_bet = max(self.bets)
                call_amount = max_bet - self.bets[acting_player]
                new_bet = max_bet + raise_amount

                # Check bet cap
                if new_bet > BET_CAP:
                    self.logger.error(
                        f"Player {acting_player} attempted to raise to ${new_bet}, "
                        f"exceeds BET_CAP ${BET_CAP}"
                    )
                    invalid_action = True
                    self.folded_players.add(acting_player)

                # Check min raise
                elif raise_amount < self.min_raise:
                    self.logger.error(
                        f"Player {acting_player} attempted to raise ${raise_amount}, "
                        f"min_raise is ${self.min_raise}"
                    )
                    invalid_action = True
                    self.folded_players.add(acting_player)

                else:
                    # Valid raise
                    total_contribution = call_amount + raise_amount
                    self.bets[acting_player] = new_bet
                    self.stacks[acting_player] -= total_contribution
                    self.min_raise = raise_amount  # Update min raise for next raiser
                    self.logger.debug(
                        f"Player {acting_player} raised ${raise_amount} "
                        f"(total bet: ${new_bet})"
                    )

            # Advance to next active player
            if not terminated:
                original_actor = self.acting_agent
                self.acting_agent = (self.acting_agent + 1) % NUM_SEATS

                # Skip folded and empty players
                while (self.acting_agent >= self.num_players or
                       self.acting_agent in self.folded_players):
                    self.acting_agent = (self.acting_agent + 1) % NUM_SEATS

                    # Safety check: prevent infinite loop
                    if self.acting_agent == original_actor:
                        self.logger.error("Infinite loop in action order!")
                        break

                # Check if betting round is complete
                # Round is complete when all active players have equal bets
                active_players = [i for i in range(self.num_players) if i not in self.folded_players]
                if active_players:
                    active_bets = [self.bets[i] for i in active_players]
                    if len(set(active_bets)) == 1:  # All bets are equal
                        # Betting round complete, advance to showdown
                        self.street = 2
                        self.logger.info(f"Betting round complete. Advancing to street 2 (showdown)")

        # ===================================================================
        # STREET 2+: SHOWDOWN (TODO: Implement in next phase)
        # ===================================================================
        else:
            # TODO: Implement showdown logic
            self.logger.error(f"Showdown not yet implemented (street {self.street})")
            terminated = True

        # Build observations and rewards
        observations = []
        for seat in range(NUM_SEATS):
            if seat < self.num_players:
                obs = self._get_single_player_obs(seat)
                observations.append(obs)
            else:
                observations.append(None)

        # Calculate rewards
        if terminated and winner is not None:
            # Winner takes the entire pot
            pot_total = sum(self.bets)
            rewards_list = [0.0] * NUM_SEATS

            for seat in range(NUM_SEATS):
                if seat == winner:
                    # Winner gets pot minus their contribution
                    rewards_list[seat] = pot_total - self.bets[seat]
                else:
                    # Losers lose their contribution (already deducted from stack)
                    rewards_list[seat] = -self.bets[seat]

            rewards = tuple(rewards_list)
        else:
            # Rewards are 0 until hand completes
            rewards = tuple([0.0] * NUM_SEATS)

        # Info
        info = {
            'invalid_action': invalid_action,
            'eliminated_seats': list(self.folded_players),
        }

        truncated = False

        return tuple(observations), rewards, terminated, truncated, info
