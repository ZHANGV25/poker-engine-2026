from typing import Literal, NotRequired, TypeAlias, TypedDict
from dataclasses import dataclass

NUM_SEATS = 6
NUM_BOARDS = 3
HOLE_CARDS_PER_PLAYER = 5
KEPT_CARDS_PER_PLAYER = 2
BOARD_CARDS_PER_BOARD = 5

SeatIndex: TypeAlias = int
CardIndex: TypeAlias = int
CardString: TypeAlias = str

ActionMask: TypeAlias = list[int]            # aligned with PokerEnv.ActionType enum
BoardCards: TypeAlias = list[CardIndex]      # up to BOARD_CARDS_PER_BOARD entries, -1 padded
BoardMatrix: TypeAlias = list[BoardCards]    # NUM_BOARDS inner lists
BetLedger: TypeAlias = list[int]             # per-seat chip contributions
SeatActionHistory: TypeAlias = list[str | None]

@dataclass
class PlayerObservation(TypedDict):
    seat: SeatIndex                      # this seat's index (0..NUM_SEATS-1)
    acting_seat: SeatIndex               # which seat must act now
    street: int                          # numeric street marker (0 = pre-select, 1 = betting, etc.)
    kept_cards: NotRequired[list[CardIndex]]   # resolved card values matching the indices
    community_cards: BoardMatrix         # three boards; unrevealed cards padded with -1
    bets: BetLedger                      # current betting round contributions per seat
    min_raise: int
    max_raise: int
    valid_actions: ActionMask
    time_used: float
    time_left: float
    opp_last_action: NotRequired[str]    # last action taken by any opposing seat
    last_actions: NotRequired[SeatActionHistory]  # action history per seat (names or None)


TableObservation: TypeAlias = tuple[PlayerObservation, ...]
Rewards: TypeAlias = tuple[int, ...]

@dataclass
class ShowdownBoardResult(TypedDict):
    board_index: int
    winning_seats: list[SeatIndex]       # seats that share this board
    hand_description: list[CardString]   # cards forming the winning hand
    evaluator_rank: int                  # treys-style rank (lower is better)
    pot_awarded: int                     # chips paid out from this board's pot

@dataclass
class StepInfo(TypedDict, total=False):
    invalid_action: bool                 # true if we treated the action as a fold/penalty
    hand_number: int                     # match runner injects this for logging
    next_street: int                     # upcoming street if we just advanced
    eliminated_seats: list[SeatIndex]    # seats folded out
    board_results: list[ShowdownBoardResult]
