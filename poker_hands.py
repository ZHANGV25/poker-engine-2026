from enum import Enum
from collections import Counter


class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class Card:
    RANKS = "23456789TJQKA"
    SUITS = "CDHS"

    def __init__(self, card_int):
        self.rank = self.RANKS[card_int % 13]
        self.suit = self.SUITS[card_int // 13]

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class PokerHand:
    def __init__(self, my_cards: list[int], community_cards: list[int]):
        assert len(my_cards) == 2 and len(community_cards) == 5
        self.my_cards = [Card(card) for card in my_cards]
        self.community_cards = [Card(card) for card in community_cards]
        self.all_cards = self.my_cards + self.community_cards
        self.hand_rank, self.high_cards = self._evaluate_hand()

    def _evaluate_hand(self):
        # Implementation of hand evaluation logic
        # This is a simplified version and doesn't cover all edge cases
        ranks = [card.rank for card in self.all_cards]
        suits = [card.suit for card in self.all_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Check for flush
        flush = max(suit_counts.values()) >= 5

        # Check for straight
        rank_values = [Card.RANKS.index(r) for r in ranks]
        rank_values.sort(reverse=True)
        straight = False
        for i in range(len(rank_values) - 4):
            if rank_values[i] - rank_values[i + 4] == 4:
                straight = True
                break

        # Determine hand rank
        if straight and flush:
            return (HandRank.STRAIGHT_FLUSH, rank_values[:5])
        elif 4 in rank_counts.values():
            four_kind = max(rank_counts, key=rank_counts.get)
            return (
                HandRank.FOUR_OF_A_KIND,
                [Card.RANKS.index(four_kind)] + rank_values[:1],
            )
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            three_kind = max(
                rank for rank, count in rank_counts.items() if count == 3
            )
            pair = max(
                rank for rank, count in rank_counts.items() if count == 2
            )
            return (
                HandRank.FULL_HOUSE,
                [Card.RANKS.index(three_kind), Card.RANKS.index(pair)],
            )
        elif flush:
            flush_ranks = [
                Card.RANKS.index(card.rank)
                for card in self.all_cards
                if card.suit == max(suit_counts, key=suit_counts.get)
            ]
            return (HandRank.FLUSH, sorted(flush_ranks, reverse=True)[:5])
        elif straight:
            return (HandRank.STRAIGHT, rank_values[:5])
        elif 3 in rank_counts.values():
            three_kind = max(
                rank for rank, count in rank_counts.items() if count == 3
            )
            return (
                HandRank.THREE_OF_A_KIND,
                [Card.RANKS.index(three_kind)] + rank_values[:2],
            )
        elif list(rank_counts.values()).count(2) == 2:
            pairs = sorted(
                [
                    Card.RANKS.index(rank)
                    for rank, count in rank_counts.items()
                    if count == 2
                ],
                reverse=True,
            )
            return (HandRank.TWO_PAIR, pairs + rank_values[:1])
        elif 2 in rank_counts.values():
            pair = max(
                rank for rank, count in rank_counts.items() if count == 2
            )
            return (HandRank.PAIR, [Card.RANKS.index(pair)] + rank_values[:3])
        else:
            return (HandRank.HIGH_CARD, rank_values[:5])

    def __lt__(self, other):
        if self.hand_rank.value != other.hand_rank.value:
            return self.hand_rank.value < other.hand_rank.value
        return self.high_cards < other.high_cards

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __eq__(self, other):
        return (
            self.hand_rank == other.hand_rank
            and self.high_cards == other.high_cards
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"PokerHand(my_cards={self.my_cards}, community_cards={self.community_cards}, rank={self.hand_rank.name})"
