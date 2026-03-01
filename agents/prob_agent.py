"""
ProbabilityAgent for the tournament variant:
- 27-card deck, 5 hole cards, mandatory discard to 2 on the flop.
- Uses Monte Carlo equity for discard choice (which 2 to keep) and for betting.
"""
import random
from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
DECK_SIZE = 27


class ProbabilityAgent(Agent):
    def __name__(self):
        return "ProbabilityAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.evaluator = PokerEnv().evaluator

    def _compute_equity(
        self,
        my_cards,
        community_cards,
        opp_discarded_cards,
        num_simulations=400,
    ):
        """
        Monte Carlo equity: win probability with given hole cards and board.
        my_cards: list of 2 ints (our hole cards after discard, or 2 of 5 during discard eval).
        community_cards: list of 0–5 ints (visible board).
        opp_discarded_cards: list of 3 ints (opponent's discards; -1 entries ignored).
        """
        shown = set(my_cards)
        for c in community_cards:
            if c != -1:
                shown.add(c)
        for c in opp_discarded_cards:
            if c != -1:
                shown.add(c)

        non_shown = [i for i in range(DECK_SIZE) if i not in shown]
        opp_needed = 2
        board_needed = 5 - len(community_cards)

        wins = 0
        valid = 0
        for _ in range(num_simulations):
            sample_size = opp_needed + board_needed
            if sample_size > len(non_shown):
                continue
            sample = random.sample(non_shown, sample_size)
            opp_cards = sample[:opp_needed]
            full_board = list(community_cards) + sample[opp_needed : opp_needed + board_needed]
            if len(full_board) != 5:
                continue

            my_hand = list(map(int_to_card, my_cards))
            opp_hand = list(map(int_to_card, opp_cards))
            board = list(map(int_to_card, full_board))

            my_rank = self.evaluator.evaluate(my_hand, board)
            opp_rank = self.evaluator.evaluate(opp_hand, board)
            if my_rank < opp_rank:
                wins += 1
            valid += 1

        return wins / valid if valid > 0 else 0.0

    def act(self, observation, reward, terminated, truncated, info):
        my_cards_raw = observation["my_cards"]
        my_cards = [c for c in my_cards_raw if c != -1]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_discarded_cards = list(observation.get("opp_discarded_cards", [-1, -1, -1]))
        valid_actions = observation["valid_actions"]

        # --- Discard phase (flop): we have 5 cards, must keep 2 ---
        if valid_actions[action_types.DISCARD.value]:
            assert len(my_cards) == 5, "Discard phase should have 5 hole cards"
            # Evaluate all 10 ways to choose 2 cards to keep
            best_keep = (0, 1)
            best_equity = -1.0
            for i in range(5):
                for j in range(i + 1, 5):
                    keep_pair = [my_cards[i], my_cards[j]]
                    eq = self._compute_equity(
                        keep_pair,
                        community_cards,
                        opp_discarded_cards,
                        num_simulations=200,
                    )
                    if eq > best_equity:
                        best_equity = eq
                        best_keep = (i, j)
            self.logger.debug(f"Discard: keeping indices {best_keep}, equity={best_equity:.2f}")
            return (action_types.DISCARD.value, 0, best_keep[0], best_keep[1])

        # --- Betting: we have 2 hole cards ---
        if len(my_cards) != 2:
            my_cards = my_cards[:2]
        equity = self._compute_equity(
            my_cards,
            community_cards,
            opp_discarded_cards,
            num_simulations=400,
        )

        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        self.logger.debug(f"Street {observation['street']}: equity={equity:.2f}, pot_odds={pot_odds:.2f}")

        if equity > 0.75 and valid_actions[action_types.RAISE.value]:
            raise_amount = int(pot_size * 0.75)
            raise_amount = max(raise_amount, observation["min_raise"])
            raise_amount = min(raise_amount, observation["max_raise"])
            return (action_types.RAISE.value, raise_amount, 0, 0)
        elif equity >= pot_odds and valid_actions[action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)
        elif valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        else:
            return (action_types.FOLD.value, 0, 0, 0)

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand ended with reward: {reward}")
        if "player_0_cards" in info:
            self.logger.info(
                f"Showdown: {info['player_0_cards']} vs {info['player_1_cards']} "
                f"board {info['community_cards']}"
            )
