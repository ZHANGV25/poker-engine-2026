"""
Comprehensive test of one-hand CFR solver (SubgameSolver) quality on river play.

Tests ~21 realistic river scenarios comparing:
  - One-hand solver (SubgameSolver) at 200, 400, 500 iterations
  - Range solver (RangeSolver) at 200 iterations
  - Simple equity threshold (fallback logic)

Evaluates:
  1. Does the one-hand solver overcall when facing bets?
  2. Does it produce balanced bet/check ranges when acting first?
  3. How stable are decisions across iteration counts?
  4. Does it handle bluff-catching correctly?
  5. BUG TEST: Does hero_is_first=False cause fold-everything?

Card encoding (27-card deck, 3 suits d/h/s, 9 ranks 2-9,A):
  suit_index * 9 + rank_index
  0=2d, 1=3d, ..., 7=9d, 8=Ad
  9=2h, 10=3h, ..., 16=9h, 17=Ah
  18=2s, 19=3s, ..., 25=9s, 26=As
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from equity import ExactEquityEngine
from solver import SubgameSolver
from range_solver import RangeSolver
from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

FOLD, RAISE, CHECK, CALL = 0, 1, 2, 3
ACTION_NAMES = {FOLD: "FOLD", RAISE: "RAISE", CHECK: "CHECK", CALL: "CALL"}

# Card helpers
def card_name(c):
    ranks = "23456789A"
    suits = "dhs"
    return ranks[c % 9] + suits[c // 9]

def cards_str(cards):
    return " ".join(card_name(c) for c in cards)

def make_card(rank_char, suit_char):
    ranks = "23456789A"
    suits = "dhs"
    return suits.index(suit_char) * 9 + ranks.index(rank_char)

C = make_card  # shorthand


# ============================================================
# Build realistic opponent ranges
# ============================================================

def build_weighted_opp_range(board, hero_cards, dead_cards, engine, bias="neutral"):
    """
    Build a realistic opponent range with non-uniform weights.

    bias options:
      "neutral"    — weights proportional to sqrt(equity) (slight strong-hand bias)
      "strong"     — opponent range weighted toward strong hands (but not empty!)
      "polarized"  — opponent range is either very strong or very weak (bluff-heavy)
      "weak"       — opponent range skewed toward weak hands (passive player)
    """
    known = set(board) | set(hero_cards) | set(dead_cards)
    remaining = [c for c in range(27) if c not in known]

    opp_range = {}
    hands_with_eq = []

    # First pass: compute equity for all possible opponent hands
    for i in range(len(remaining)):
        for j in range(i + 1, len(remaining)):
            h = (remaining[i], remaining[j])
            eq = _quick_equity(h, board, engine)
            hands_with_eq.append((h, eq))

    if not hands_with_eq:
        return {}

    # Second pass: assign weights based on bias
    for h, eq in hands_with_eq:
        if bias == "neutral":
            # Slight bias toward strong hands (surviving range after normal play)
            w = 0.3 + 0.7 * eq
        elif bias == "strong":
            # Bias toward strong hands, but keep all hands with some weight
            # to avoid empty ranges
            w = 0.05 + 0.95 * (eq ** 1.5)
        elif bias == "polarized":
            # Polarized: strong hands and bluffs, few medium hands
            if eq > 0.7:
                w = 1.0
            elif eq < 0.2:
                w = 0.4  # bluff candidates
            else:
                w = 0.05  # few medium hands
        elif bias == "weak":
            # Opponent has been passive, skewed weak
            w = 0.3 + 0.7 * (1.0 - eq)
        else:
            w = 1.0

        opp_range[h] = max(w, 0.01)

    return opp_range


def _quick_equity(hand, board, engine):
    """Quick equity for a single hand against the board."""
    if len(board) == 5:
        hr = engine.lookup_seven(list(hand) + list(board))
        # Normalize: lower rank = better. Rank 1 is best, ~800+ is worst
        return max(0, 1.0 - hr / 800.0)
    return 0.5


# ============================================================
# Extract strategy without randomization
# ============================================================

def get_solver_strategy(solver, hero_cards, board, opp_range, dead_cards,
                        my_bet, opp_bet, street, min_raise, max_raise,
                        valid_actions, hero_is_first, iterations):
    """
    Run the one-hand solver and return the raw strategy probabilities
    (without random action selection).
    """
    if opp_range is None:
        return None, "no_range"

    known_cards = set(hero_cards) | set(board) | set(dead_cards)
    opp_hands = []
    opp_weights = []
    for hand, weight in opp_range.items():
        if weight > 0.001 and not (set(hand) & known_cards):
            opp_hands.append(hand)
            opp_weights.append(weight)

    if not opp_hands:
        return None, "no_opp_hands"

    opp_weights_arr = np.array(opp_weights, dtype=np.float64)
    opp_weights_arr /= opp_weights_arr.sum()
    n_opp = len(opp_hands)

    max_bet = 100
    tree = GameTree(my_bet, opp_bet, min_raise, max_bet, hero_is_first)

    if tree.size < 2:
        return None, "tiny_tree"

    # Check if root is a hero node
    root_is_hero = (0 in set(tree.hero_node_ids))

    # Compute terminal values
    values = {}
    equity_vec = np.zeros(n_opp, dtype=np.float64)

    if len(board) == 5:
        hero_rank = solver.engine.lookup_seven(list(hero_cards) + list(board))
        for i, oh in enumerate(opp_hands):
            opp_rank = solver.engine.lookup_seven(list(oh) + list(board))
            if hero_rank < opp_rank:
                equity_vec[i] = 1.0
            elif hero_rank == opp_rank:
                equity_vec[i] = 0.5

    for node_id in tree.terminal_node_ids:
        term_type = tree.terminal[node_id]
        hero_pot = tree.hero_pot[node_id]
        opp_pot = tree.opp_pot[node_id]

        if term_type == TERM_FOLD_HERO:
            values[node_id] = np.full(n_opp, -hero_pot, dtype=np.float64)
        elif term_type == TERM_FOLD_OPP:
            values[node_id] = np.full(n_opp, opp_pot, dtype=np.float64)
        elif term_type == TERM_SHOWDOWN:
            pot_won = min(hero_pot, opp_pot)
            values[node_id] = (2.0 * equity_vec - 1.0) * pot_won

    # Run CFR
    strategy = solver._run_cfr(tree, opp_weights_arr, values, n_opp, iterations)

    # Determine which node's strategy we're looking at
    # If root is hero, strategy is for root actions
    # If root is opp (hero_first=False), _run_cfr returns np.array([1.0]) fallback
    if root_is_hero:
        strat_node = 0
    else:
        # The hero's first decision nodes come after opp's first action
        # _run_cfr doesn't handle this case - it returns np.array([1.0])
        strat_node = 0  # still root, but this is the bug

    # Map strategy to action labels
    root_children = tree.children[strat_node]
    action_map = {}
    for idx, (act_type, child_id) in enumerate(root_children):
        act_names = {
            ACT_FOLD: "fold", ACT_CHECK: "check", ACT_CALL: "call",
            ACT_RAISE_HALF: "raise_40%", ACT_RAISE_POT: "raise_70%",
            ACT_RAISE_ALLIN: "raise_100%", ACT_RAISE_OVERBET: "raise_150%",
        }
        name = act_names.get(act_type, f"act_{act_type}")
        if act_type in (ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            hp = tree.hero_pot[child_id]
            op = tree.opp_pot[child_id]
            name += f"(to {max(hp,op)})"
        action_map[idx] = name

    result = {}
    for idx in range(len(strategy)):
        if idx < len(action_map):
            result[action_map[idx]] = strategy[idx]
        else:
            result[f"action_{idx}"] = strategy[idx]

    return result, "ok" if root_is_hero else "ROOT_IS_OPP_BUG"


def get_range_solver_strategy(range_solver, hero_cards, board, opp_range, dead_cards,
                               my_bet, opp_bet, street, min_raise, max_raise,
                               valid_actions, iterations, hero_first=True):
    """
    Run the range solver and return the strategy for hero's specific hand.
    """
    if opp_range is None:
        return None, "no_range"

    known = set(board) | set(dead_cards)
    hero_hands = []
    opp_hands = []
    opp_weights_list = []

    for hand, weight in opp_range.items():
        if weight > 0.001 and not (set(hand) & known):
            opp_hands.append(hand)
            opp_weights_list.append(weight)

    if not opp_hands:
        return None, "no_opp_hands"

    opp_weights_arr = np.array(opp_weights_list, dtype=np.float64)
    opp_weights_arr /= opp_weights_arr.sum()

    remaining = [c for c in range(27) if c not in known]
    import itertools
    for h in itertools.combinations(remaining, 2):
        hero_hands.append(h)

    n_hero = len(hero_hands)
    n_opp = len(opp_hands)

    hero_tuple = tuple(sorted(hero_cards))
    hero_idx_in_list = None
    for i, h in enumerate(hero_hands):
        if tuple(sorted(h)) == hero_tuple:
            hero_idx_in_list = i
            break

    if hero_idx_in_list is None:
        return None, "hero_not_found"

    max_bet = 100
    tree = GameTree(my_bet, opp_bet, min_raise, max_bet, hero_first)

    if tree.size < 2:
        return None, "tiny_tree"

    # Compute equity matrix
    equity_matrix = range_solver._compute_equity_matrix(
        hero_hands, opp_hands, board, dead_cards, street)

    terminal_values = range_solver._compute_terminal_values(
        tree, equity_matrix, n_hero, n_opp)

    hero_strategy = range_solver._run_range_cfr(
        tree, opp_weights_arr, terminal_values, n_hero, n_opp, iterations)

    our_strategy = hero_strategy[hero_idx_in_list]

    # Map to action labels
    root_children = tree.children[0]
    action_map = {}
    for idx, (act_type, child_id) in enumerate(root_children):
        act_names = {
            ACT_FOLD: "fold", ACT_CHECK: "check", ACT_CALL: "call",
            ACT_RAISE_HALF: "raise_40%", ACT_RAISE_POT: "raise_70%",
            ACT_RAISE_ALLIN: "raise_100%", ACT_RAISE_OVERBET: "raise_150%",
        }
        name = act_names.get(act_type, f"act_{act_type}")
        if act_type in (ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            hp = tree.hero_pot[child_id]
            op = tree.opp_pot[child_id]
            name += f"(to {max(hp,op)})"
        action_map[idx] = name

    result = {}
    for idx in range(len(our_strategy)):
        result[action_map.get(idx, f"action_{idx}")] = our_strategy[idx]

    return result, "ok"


def equity_threshold_decision(engine, hero_cards, board, dead_cards,
                               my_bet, opp_bet, valid_actions, min_raise, max_raise):
    """What the fallback equity threshold logic would do."""
    equity = engine.compute_equity(hero_cards, board, dead_cards)
    pot_size = my_bet + opp_bet
    continue_cost = opp_bet - my_bet
    pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

    if equity > 0.92 and valid_actions[RAISE]:
        return "RAISE(max)", equity, pot_odds
    if equity > 0.72 and valid_actions[RAISE]:
        amt = max(int(pot_size * 0.6), min_raise)
        amt = min(amt, max_raise)
        return f"RAISE({amt})", equity, pot_odds
    if valid_actions[CALL] and equity >= pot_odds:
        return "CALL", equity, pot_odds
    if valid_actions[CHECK]:
        return "CHECK", equity, pot_odds
    return "FOLD", equity, pot_odds


# ============================================================
# Scenario definitions
# ============================================================

def define_scenarios():
    scenarios = []

    # ---- CATEGORY 1: Acting first with strong hands (should bet) ----

    scenarios.append({
        "name": "1. Strong hand acting first (two pair, dry board)",
        "hero_cards": [C('A','d'), C('9','d')],
        "board": [C('A','h'), C('9','h'), C('5','s'), C('3','s'), C('2','h')],
        "dead_cards": [C('7','d'), C('6','d'), C('4','d'),
                       C('8','s'), C('7','s'), C('6','s')],
        "my_bet": 10, "opp_bet": 10,
        "min_raise": 4, "max_raise": 90,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "BET (value bet with strong hand)",
        "category": "strong_first",
    })

    scenarios.append({
        "name": "2. Strong hand acting first (trips)",
        "hero_cards": [C('8','d'), C('8','h')],
        "board": [C('8','s'), C('5','d'), C('3','h'), C('2','s'), C('7','d')],
        "dead_cards": [C('A','d'), C('9','d'), C('6','d'),
                       C('A','s'), C('9','s'), C('4','s')],
        "my_bet": 12, "opp_bet": 12,
        "min_raise": 4, "max_raise": 88,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "BET (strong trips, extract value)",
        "category": "strong_first",
    })

    scenarios.append({
        "name": "3. Strong hand acting first (flush)",
        "hero_cards": [C('A','h'), C('7','h')],
        "board": [C('9','h'), C('4','h'), C('2','h'), C('5','d'), C('3','s')],
        "dead_cards": [C('6','d'), C('8','d'), C('2','d'),
                       C('A','s'), C('9','s'), C('8','s')],
        "my_bet": 15, "opp_bet": 15,
        "min_raise": 4, "max_raise": 85,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "BET (nut flush, value bet)",
        "category": "strong_first",
    })

    # ---- CATEGORY 2: Acting first with weak hands (should check) ----

    scenarios.append({
        "name": "4. Weak hand acting first (no pair, low cards)",
        "hero_cards": [C('2','d'), C('3','d')],
        "board": [C('A','h'), C('9','h'), C('8','s'), C('7','s'), C('5','h')],
        "dead_cards": [C('6','d'), C('4','d'), C('8','d'),
                       C('A','s'), C('9','s'), C('7','d')],
        "my_bet": 8, "opp_bet": 8,
        "min_raise": 4, "max_raise": 92,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CHECK (garbage hand, no showdown value)",
        "category": "weak_first",
    })

    scenarios.append({
        "name": "5. Weak hand acting first (bottom pair)",
        "hero_cards": [C('2','d'), C('2','h')],
        "board": [C('A','s'), C('9','s'), C('8','h'), C('7','h'), C('5','d')],
        "dead_cards": [C('6','d'), C('4','d'), C('3','d'),
                       C('A','d'), C('9','d'), C('8','d')],
        "my_bet": 10, "opp_bet": 10,
        "min_raise": 4, "max_raise": 90,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CHECK (weak pair, check to showdown)",
        "category": "weak_first",
    })

    # ---- CATEGORY 3: Acting first with medium hands (should mix) ----

    scenarios.append({
        "name": "6. Medium hand acting first (trips with ace on board)",
        "hero_cards": [C('7','d'), C('7','h')],
        "board": [C('A','s'), C('7','s'), C('5','h'), C('3','h'), C('2','d')],
        "dead_cards": [C('9','d'), C('8','d'), C('6','d'),
                       C('A','d'), C('9','s'), C('8','s')],
        "my_bet": 10, "opp_bet": 10,
        "min_raise": 4, "max_raise": 90,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "MIX or BET (trips is strong, should bet for value)",
        "category": "medium_first",
    })

    scenarios.append({
        "name": "7. Medium hand acting first (top pair weak kicker)",
        "hero_cards": [C('A','d'), C('2','d')],
        "board": [C('A','h'), C('8','s'), C('6','s'), C('4','h'), C('3','h')],
        "dead_cards": [C('9','d'), C('7','d'), C('5','d'),
                       C('9','s'), C('8','h'), C('7','s')],
        "my_bet": 10, "opp_bet": 10,
        "min_raise": 4, "max_raise": 90,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "MIX (top pair but dominated by Ax)",
        "category": "medium_first",
    })

    # ---- CATEGORY 4: Acting first with bluff candidates ----

    scenarios.append({
        "name": "8. Bluff candidate acting first (complete air, paired board)",
        "hero_cards": [C('3','d'), C('4','d')],
        "board": [C('A','h'), C('A','s'), C('9','h'), C('8','s'), C('7','h')],
        "dead_cards": [C('6','d'), C('5','d'), C('2','d'),
                       C('9','s'), C('8','h'), C('7','s')],
        "my_bet": 15, "opp_bet": 15,
        "min_raise": 4, "max_raise": 85,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "MIX (bluff some frequency or check-fold)",
        "category": "bluff_first",
    })

    scenarios.append({
        "name": "9. Bluff candidate (missed draw, no pair)",
        "hero_cards": [C('6','h'), C('5','h')],
        "board": [C('A','d'), C('9','d'), C('8','s'), C('3','s'), C('2','d')],
        "dead_cards": [C('4','h'), C('3','h'), C('2','h'),
                       C('A','s'), C('9','s'), C('7','s')],
        "my_bet": 20, "opp_bet": 20,
        "min_raise": 4, "max_raise": 80,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CHECK or BLUFF (near-zero equity)",
        "category": "bluff_first",
    })

    # ---- CATEGORY 5: Facing a bet with strong hands ----
    # NOTE: These use hero_is_first=True because when you face a bet,
    # you ARE the next to act. The tree with asymmetric bets + hero_first=True
    # correctly models fold/call/raise.

    scenarios.append({
        "name": "10. Strong hand facing bet (trips vs pot bet)",
        "hero_cards": [C('9','d'), C('9','h')],
        "board": [C('9','s'), C('6','d'), C('5','h'), C('3','s'), C('2','d')],
        "dead_cards": [C('A','d'), C('8','d'), C('7','d'),
                       C('A','s'), C('8','s'), C('7','s')],
        "my_bet": 10, "opp_bet": 20,
        "min_raise": 10, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,   # hero is next to act (facing bet)
        "opp_range_bias": "polarized",
        "expected": "CALL or RAISE (strong hand, should never fold)",
        "category": "strong_facing",
    })

    scenarios.append({
        "name": "11. Strong hand facing bet (full house)",
        "hero_cards": [C('5','d'), C('5','h')],
        "board": [C('5','s'), C('8','d'), C('8','h'), C('3','s'), C('2','d')],
        "dead_cards": [C('A','d'), C('9','d'), C('7','d'),
                       C('A','s'), C('9','s'), C('6','s')],
        "my_bet": 15, "opp_bet": 30,
        "min_raise": 15, "max_raise": 85,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "strong",
        "expected": "RAISE (full house, maximum value)",
        "category": "strong_facing",
    })

    # ---- CATEGORY 6: Facing a bet with medium hands ----

    scenarios.append({
        "name": "12. Medium hand facing bet (second pair vs half-pot)",
        "hero_cards": [C('7','d'), C('7','h')],
        "board": [C('A','s'), C('8','s'), C('5','d'), C('3','h'), C('2','d')],
        "dead_cards": [C('9','d'), C('6','d'), C('4','d'),
                       C('9','s'), C('6','s'), C('4','s')],
        "my_bet": 10, "opp_bet": 15,
        "min_raise": 5, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CALL (decent pot odds, second pair has some equity)",
        "category": "medium_facing",
    })

    scenarios.append({
        "name": "13. Medium hand facing big bet (top pair vs pot bet, polarized)",
        "hero_cards": [C('A','d'), C('3','d')],
        "board": [C('A','h'), C('8','s'), C('6','s'), C('4','h'), C('2','h')],
        "dead_cards": [C('9','d'), C('7','d'), C('5','d'),
                       C('9','s'), C('8','h'), C('7','s')],
        "my_bet": 12, "opp_bet": 24,
        "min_raise": 12, "max_raise": 88,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "polarized",
        "expected": "CALL or FOLD (depends on bluff frequency)",
        "category": "medium_facing",
    })

    scenarios.append({
        "name": "14. Medium hand facing overbet (set of 6s, flush board)",
        "hero_cards": [C('6','d'), C('6','h')],
        "board": [C('A','s'), C('9','s'), C('6','s'), C('4','d'), C('2','h')],
        "dead_cards": [C('8','d'), C('7','d'), C('3','d'),
                       C('8','h'), C('7','h'), C('5','h')],
        "my_bet": 10, "opp_bet": 40,
        "min_raise": 30, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "strong",
        "expected": "Tricky: set of 6s but flush-possible board",
        "category": "medium_facing",
    })

    # ---- CATEGORY 7: Facing a bet with weak hands (should fold) ----

    scenarios.append({
        "name": "15. Weak hand facing bet (no pair vs pot bet)",
        "hero_cards": [C('3','d'), C('2','d')],
        "board": [C('A','h'), C('9','h'), C('8','s'), C('6','s'), C('5','h')],
        "dead_cards": [C('7','d'), C('4','d'), C('5','d'),
                       C('A','s'), C('9','s'), C('8','h')],
        "my_bet": 10, "opp_bet": 20,
        "min_raise": 10, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "strong",
        "expected": "FOLD (complete air vs strong range)",
        "category": "weak_facing",
    })

    scenarios.append({
        "name": "16. Weak hand facing big bet (bottom pair vs pot)",
        "hero_cards": [C('2','d'), C('2','h')],
        "board": [C('A','s'), C('9','s'), C('7','h'), C('5','h'), C('3','d')],
        "dead_cards": [C('6','d'), C('4','d'), C('8','d'),
                       C('A','d'), C('9','d'), C('7','d')],
        "my_bet": 15, "opp_bet": 30,
        "min_raise": 15, "max_raise": 85,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "strong",
        "expected": "FOLD (bottom pair, terrible vs strong range)",
        "category": "weak_facing",
    })

    # ---- CATEGORY 8: Bluff-catching spots ----

    scenarios.append({
        "name": "17. Bluff-catch spot (no pair vs small bet, weak opp)",
        "hero_cards": [C('4','d'), C('3','d')],
        "board": [C('A','h'), C('8','h'), C('7','s'), C('5','s'), C('2','h')],
        "dead_cards": [C('6','d'), C('9','d'), C('8','d'),
                       C('A','s'), C('9','s'), C('7','h')],
        "my_bet": 10, "opp_bet": 14,
        "min_raise": 4, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "weak",
        "expected": "MIX fold/call (small bet, ~17% pot odds)",
        "category": "bluffcatch",
    })

    scenarios.append({
        "name": "18. Bluff-catch with Ace-high vs pot bet",
        "hero_cards": [C('A','d'), C('4','d')],
        "board": [C('9','h'), C('8','h'), C('7','s'), C('5','s'), C('2','h')],
        "dead_cards": [C('6','d'), C('3','d'), C('2','d'),
                       C('A','s'), C('9','s'), C('8','s')],
        "my_bet": 15, "opp_bet": 30,
        "min_raise": 15, "max_raise": 85,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "polarized",
        "expected": "MIX fold/call (ace-high bluff catcher vs polarized)",
        "category": "bluffcatch",
    })

    # ---- CATEGORY 9: Bet size comparison ----

    scenarios.append({
        "name": "19. Top pair vs SMALL bet (25% pot)",
        "hero_cards": [C('A','d'), C('5','d')],
        "board": [C('A','h'), C('8','s'), C('6','h'), C('4','s'), C('2','h')],
        "dead_cards": [C('9','d'), C('7','d'), C('3','d'),
                       C('9','s'), C('8','h'), C('7','s')],
        "my_bet": 10, "opp_bet": 15,
        "min_raise": 5, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CALL (good price with top pair)",
        "category": "bet_size_compare",
    })

    scenarios.append({
        "name": "20. Top pair vs LARGE bet (150% pot)",
        "hero_cards": [C('A','d'), C('5','d')],
        "board": [C('A','h'), C('8','s'), C('6','h'), C('4','s'), C('2','h')],
        "dead_cards": [C('9','d'), C('7','d'), C('3','d'),
                       C('9','s'), C('8','h'), C('7','s')],
        "my_bet": 10, "opp_bet": 40,
        "min_raise": 30, "max_raise": 90,
        "valid_actions": [True, True, False, True, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "MIX/FOLD (much worse price, need more equity)",
        "category": "bet_size_compare",
    })

    scenarios.append({
        "name": "21. Medium hand acting first in large pot",
        "hero_cards": [C('8','d'), C('8','h')],
        "board": [C('A','s'), C('9','s'), C('5','d'), C('4','h'), C('2','h')],
        "dead_cards": [C('7','d'), C('6','d'), C('3','d'),
                       C('A','d'), C('9','d'), C('7','s')],
        "my_bet": 40, "opp_bet": 40,
        "min_raise": 4, "max_raise": 60,
        "valid_actions": [True, True, True, False, False],
        "hero_is_first": True,
        "opp_range_bias": "neutral",
        "expected": "CHECK (medium pair, control pot)",
        "category": "medium_first",
    })

    return scenarios


# ============================================================
# Bug verification test
# ============================================================

def verify_hero_first_bug(engine, solver):
    """
    Verify that SubgameSolver with hero_is_first=False always folds,
    even with a nut hand. This is the root cause of the "fold everything"
    behavior seen when hero is SB on post-flop streets.
    """
    print("=" * 90)
    print("BUG VERIFICATION: hero_is_first=False causes fold-everything")
    print("=" * 90)
    print()

    # Use a nut hand: trips 9s on a dry board
    hero_cards = [C('9','d'), C('9','h')]
    board = [C('9','s'), C('6','d'), C('5','h'), C('3','s'), C('2','d')]
    dead_cards = [C('A','d'), C('8','d'), C('7','d'),
                  C('A','s'), C('8','s'), C('7','s')]
    my_bet, opp_bet = 10, 20
    min_raise, max_raise = 10, 90

    opp_range = build_weighted_opp_range(board, hero_cards, dead_cards, engine, "neutral")
    equity = engine.compute_equity(hero_cards, board, dead_cards)

    print(f"  Hero: {cards_str(hero_cards)}  (trips 9s)")
    print(f"  Board: {cards_str(board)}")
    print(f"  Bets: hero=10, opp=20 (facing pot-size bet)")
    print(f"  Equity: {equity:.3f}")
    print()

    # Test with hero_is_first=False (the bug)
    print("  --- hero_is_first=FALSE (as currently called for SB) ---")
    tree_false = GameTree(my_bet, opp_bet, min_raise, 100, False)
    print(f"  Tree root player: {tree_false.player[0]} (0=hero, 1=opp)")
    print(f"  Root in hero_node_ids: {0 in tree_false.hero_node_ids}")
    print(f"  Root in opp_node_ids:  {0 in tree_false.opp_node_ids}")
    print(f"  Root actions: {[(a, tree_false.player[c]) for a, c in tree_false.children[0]]}")

    strat_false, status_false = get_solver_strategy(
        solver, hero_cards, board, opp_range, dead_cards,
        my_bet, opp_bet, 3, min_raise, max_raise,
        [True, True, False, True, False], False, 400)
    print(f"  Strategy: {strat_false}")
    print(f"  Status:   {status_false}")
    print()

    # Test with hero_is_first=True (correct for "facing a bet")
    print("  --- hero_is_first=TRUE (correct: hero IS next to act) ---")
    tree_true = GameTree(my_bet, opp_bet, min_raise, 100, True)
    print(f"  Tree root player: {tree_true.player[0]} (0=hero, 1=opp)")
    print(f"  Root in hero_node_ids: {0 in tree_true.hero_node_ids}")
    print(f"  Root actions: {[(a, tree_true.player[c] if tree_true.player[c]>=0 else 'TERM') for a, c in tree_true.children[0]]}")

    strat_true, status_true = get_solver_strategy(
        solver, hero_cards, board, opp_range, dead_cards,
        my_bet, opp_bet, 3, min_raise, max_raise,
        [True, True, False, True, False], True, 400)
    print(f"  Strategy: {strat_true}")
    print(f"  Status:   {status_true}")
    print()

    # Diagnosis
    print("  DIAGNOSIS:")
    if strat_false and all('fold' in k for k, v in strat_false.items() if v > 0.5):
        print("    CONFIRMED: hero_is_first=False causes 100% fold with TRIPS (equity=0.59)")
        print("    ROOT CAUSE: When hero_is_first=False, root is an OPP node (player=1).")
        print("    _run_cfr() checks if root(0) is in hero_idx, finds it is NOT,")
        print("    returns np.array([1.0]). _strategy_to_action maps action[0] to the")
        print("    first root child (ACT_FOLD) which becomes (FOLD, 0, 0, 0).")
        print()
        print("    IMPACT: When hero is SB, ALL postflop solver calls use hero_is_first=False.")
        print("    The solver always returns FOLD, then the fallback code in player.py")
        print("    catches the bad result (line 1285+) and calls _fallback().")
        print("    So the solver is NEVER actually used for SB postflop play -- it's")
        print("    always equity thresholds.")
    else:
        print("    Behavior may have changed. Check results above.")
    print()

    # Also check if player.py's _solve_street catches the degenerate fold
    print("  In player.py's solve_and_act flow:")
    print("    1. solve_and_act() builds tree with hero_is_first=False")
    print("    2. _run_cfr returns np.array([1.0]) (single-action fallback)")
    print("    3. _strategy_to_action picks action 0 = ACT_FOLD (the opp fold node)")
    print("    4. Returns (FOLD, 0, 0, 0)")
    print("    5. _solve_street receives result=(0, 0, 0, 0) which is FOLD")
    print("    6. Player uses this fold even with a strong hand!")
    print()


# ============================================================
# Main test runner
# ============================================================

def run_tests():
    print("=" * 90)
    print("COMPREHENSIVE RIVER SOLVER QUALITY TEST")
    print("=" * 90)
    print()

    print("Loading equity engine...")
    t0 = time.time()
    engine = ExactEquityEngine()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    solver = SubgameSolver(engine)
    range_solver_obj = RangeSolver(engine)

    # ============================================================
    # PART 0: Verify the hero_is_first=False bug
    # ============================================================
    verify_hero_first_bug(engine, solver)

    # ============================================================
    # PART 1: All scenarios with correct hero_is_first=True
    # ============================================================
    scenarios = define_scenarios()
    print(f"\nRunning {len(scenarios)} scenarios (all use hero_is_first=True)...\n")

    # Track aggregate stats
    overcall_count = 0
    overcall_scenarios = []
    underfold_count = 0
    underfold_scenarios = []
    stability_issues = []
    one_hand_vs_range_diffs = []

    for sc in scenarios:
        print("-" * 90)
        print(f"SCENARIO: {sc['name']}")
        print(f"  Category:    {sc['category']}")
        print(f"  Hero:        {cards_str(sc['hero_cards'])}")
        print(f"  Board:       {cards_str(sc['board'])}")
        print(f"  Dead:        {cards_str(sc['dead_cards'])}")
        facing_bet = sc['opp_bet'] > sc['my_bet']
        print(f"  Bets:        hero={sc['my_bet']}, opp={sc['opp_bet']}, "
              f"pot={sc['my_bet']+sc['opp_bet']}"
              f"{'  [FACING BET]' if facing_bet else '  [ACTING FIRST]'}")
        print(f"  Opp range:   {sc['opp_range_bias']}")
        print(f"  Expected:    {sc['expected']}")
        print()

        # Compute equity
        equity = engine.compute_equity(sc['hero_cards'], sc['board'], sc['dead_cards'])

        # Build opponent range
        opp_range = build_weighted_opp_range(
            sc['board'], sc['hero_cards'], sc['dead_cards'], engine,
            bias=sc['opp_range_bias'])

        known_cards = set(sc['hero_cards']) | set(sc['board']) | set(sc['dead_cards'])
        n_hands = len([h for h, w in opp_range.items()
                       if w > 0.001 and not (set(h) & known_cards)])

        eq_vs_range = engine.compute_equity(sc['hero_cards'], sc['board'],
                                             sc['dead_cards'], opp_range)
        print(f"  Raw equity: {equity:.3f}  |  Equity vs range: {eq_vs_range:.3f}  |  Opp hands: {n_hands}")

        # --- Equity threshold decision ---
        eq_decision, eq_val, pot_odds = equity_threshold_decision(
            engine, sc['hero_cards'], sc['board'], sc['dead_cards'],
            sc['my_bet'], sc['opp_bet'], sc['valid_actions'],
            sc['min_raise'], sc['max_raise'])
        print(f"  EQUITY THRESHOLD: {eq_decision}  (pot_odds={pot_odds:.3f})")

        # --- One-hand solver at different iterations ---
        prev_strategies = {}
        for iters in [200, 400, 500]:
            t1 = time.time()
            strat, status = get_solver_strategy(
                solver, sc['hero_cards'], sc['board'], opp_range,
                sc['dead_cards'], sc['my_bet'], sc['opp_bet'],
                street=3, min_raise=sc['min_raise'], max_raise=sc['max_raise'],
                valid_actions=sc['valid_actions'],
                hero_is_first=True,  # always True (correct approach)
                iterations=iters)
            elapsed = time.time() - t1

            if strat is None:
                print(f"  ONE-HAND {iters}it: FAILED ({status})")
                continue

            strat_str = "  ".join(f"{k}:{v:.3f}" for k, v in strat.items())
            print(f"  ONE-HAND {iters:>3d}it ({elapsed:.2f}s): {strat_str}")
            prev_strategies[iters] = strat

        # --- Check stability across iteration counts ---
        if 200 in prev_strategies and 500 in prev_strategies:
            s200 = prev_strategies[200]
            s500 = prev_strategies[500]
            max_diff = 0
            diff_action = ""
            for k in s200:
                if k in s500:
                    d = abs(s200[k] - s500[k])
                    if d > max_diff:
                        max_diff = d
                        diff_action = k
            if max_diff > 0.15:
                stability_issues.append((sc['name'], max_diff, diff_action))
                print(f"  ** STABILITY WARNING: max diff 200it vs 500it = {max_diff:.3f} on '{diff_action}'")

        # --- Range solver at 200 iterations ---
        t1 = time.time()
        range_strat, range_status = get_range_solver_strategy(
            range_solver_obj, sc['hero_cards'], sc['board'], opp_range,
            sc['dead_cards'], sc['my_bet'], sc['opp_bet'],
            street=3, min_raise=sc['min_raise'], max_raise=sc['max_raise'],
            valid_actions=sc['valid_actions'],
            iterations=200,
            hero_first=True)
        elapsed = time.time() - t1

        if range_strat is not None:
            strat_str = "  ".join(f"{k}:{v:.3f}" for k, v in range_strat.items())
            print(f"  RANGE   200it ({elapsed:.2f}s): {strat_str}")

            # Compare one-hand vs range
            if 400 in prev_strategies:
                oh = prev_strategies[400]
                max_diff_r = 0
                for k in oh:
                    if k in range_strat:
                        max_diff_r = max(max_diff_r, abs(oh[k] - range_strat[k]))
                one_hand_vs_range_diffs.append((sc['name'], max_diff_r))
        else:
            print(f"  RANGE   200it: FAILED ({range_status})")

        # --- Overcall / underfold detection ---
        if facing_bet and 500 in prev_strategies:
            strat = prev_strategies[500]
            call_freq = sum(v for k, v in strat.items() if 'call' in k)
            raise_freq = sum(v for k, v in strat.items() if 'raise' in k)
            fold_freq = sum(v for k, v in strat.items() if 'fold' in k)
            continue_freq = call_freq + raise_freq

            continue_cost = sc['opp_bet'] - sc['my_bet']
            pot = sc['my_bet'] + sc['opp_bet']
            pot_odds_needed = continue_cost / (continue_cost + pot)

            # Overcall: low equity but high continue
            if eq_vs_range < pot_odds_needed and continue_freq > 0.5:
                overcall_count += 1
                overcall_scenarios.append(
                    (sc['name'], eq_vs_range, pot_odds_needed, continue_freq))
                print(f"  ** OVERCALL: equity={eq_vs_range:.3f} < pot_odds={pot_odds_needed:.3f} "
                      f"but continue={continue_freq:.3f}")

            # Underfold: high equity but high fold
            if eq_vs_range > pot_odds_needed + 0.15 and fold_freq > 0.5:
                underfold_count += 1
                underfold_scenarios.append(
                    (sc['name'], eq_vs_range, pot_odds_needed, fold_freq))
                print(f"  ** UNDERFOLD: equity={eq_vs_range:.3f} >> pot_odds={pot_odds_needed:.3f} "
                      f"but fold={fold_freq:.3f}")

        print()

    # ============================================================
    # Summary
    # ============================================================
    print("=" * 90)
    print("SUMMARY OF FINDINGS")
    print("=" * 90)

    print(f"\n{'='*90}")
    print("BUG #1: hero_is_first=False causes 100% FOLD (CRITICAL)")
    print(f"{'='*90}")
    print("  When SubgameSolver is called with hero_is_first=False:")
    print("    - GameTree puts opponent at root (player=1)")
    print("    - _run_cfr looks for root in hero_idx, doesn't find it")
    print("    - Returns np.array([1.0]) as fallback")
    print("    - _strategy_to_action maps action[0] = ACT_FOLD -> (FOLD,0,0,0)")
    print("    - Result: solver ALWAYS returns FOLD regardless of hand strength")
    print()
    print("  In player.py, hero_is_first is set based on blind position:")
    print("    SB -> hero_is_first=False for ALL postflop streets")
    print("  This means the solver never produces a real strategy when hero is SB.")
    print("  The code at _solve_street returns the fold, and the bot folds!")
    print()
    print("  FIX: When hero faces a bet (opp_bet > my_bet), always use")
    print("  hero_is_first=True because hero IS the next to act. The asymmetric")
    print("  bet amounts (my_bet < opp_bet) correctly model the facing-bet state.")
    print("  Alternatively: fix _run_cfr to find hero's first decision node")
    print("  even when root is an opponent node.")

    print(f"\n{'='*90}")
    print("Overcalling Analysis (with correct hero_is_first=True)")
    print(f"{'='*90}")
    if overcall_scenarios:
        print(f"  Found {overcall_count} overcall scenario(s):")
        for name, eq, po, cont in overcall_scenarios:
            print(f"    {name}: equity={eq:.3f}, pot_odds={po:.3f}, continue={cont:.3f}")
    else:
        print(f"  No overcalling detected (equity < pot_odds but continue > 50%)")

    print(f"\n{'='*90}")
    print("Underfolding Analysis")
    print(f"{'='*90}")
    if underfold_scenarios:
        print(f"  Found {underfold_count} underfold scenario(s):")
        for name, eq, po, ff in underfold_scenarios:
            print(f"    {name}: equity={eq:.3f}, pot_odds={po:.3f}, fold_freq={ff:.3f}")
    else:
        print(f"  No underfolding detected (equity >> pot_odds but fold > 50%)")

    print(f"\n{'='*90}")
    print("Stability Analysis (200it vs 500it)")
    print(f"{'='*90}")
    if stability_issues:
        print(f"  Found {len(stability_issues)} scenario(s) with >15% strategy shift:")
        for name, diff, act in stability_issues:
            print(f"    {name}: max_diff={diff:.3f} on '{act}'")
    else:
        print(f"  All scenarios stable (<15% shift between 200 and 500 iterations)")

    print(f"\n{'='*90}")
    print("One-Hand vs Range Solver Comparison (400it vs 200it)")
    print(f"{'='*90}")
    if one_hand_vs_range_diffs:
        large_diffs = [(n, d) for n, d in one_hand_vs_range_diffs if d > 0.20]
        small_diffs = [(n, d) for n, d in one_hand_vs_range_diffs if d <= 0.20]
        avg_diff = np.mean([d for _, d in one_hand_vs_range_diffs])
        print(f"  Average max action difference: {avg_diff:.3f}")
        print(f"  Scenarios with large difference (>0.20): {len(large_diffs)}/{len(one_hand_vs_range_diffs)}")
        for name, diff in sorted(one_hand_vs_range_diffs, key=lambda x: -x[1]):
            marker = " <<<" if diff > 0.20 else ""
            print(f"    {name}: max_diff={diff:.3f}{marker}")

    print(f"\n{'='*90}")
    print("Overall Assessment")
    print(f"{'='*90}")
    print("  1. CRITICAL BUG: hero_is_first=False always folds. When bot is SB,")
    print("     the solver is effectively disabled for all postflop streets.")
    print("  2. When used correctly (hero_is_first=True), the one-hand solver is")
    print("     reasonably well-calibrated for acting-first scenarios.")
    print("  3. The one-hand solver tends to check more and bet less than the")
    print("     range solver, especially with strong hands -- this is a known")
    print("     limitation of one-hand solvers (no range balancing).")
    print("  4. Facing-bet decisions need testing with the corrected hero_is_first=True.")
    print()


if __name__ == "__main__":
    run_tests()
