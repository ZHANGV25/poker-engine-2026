"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Hybrid strategy: precomputed blueprint strategies (range-balanced Nash
equilibria) for post-flop play when available, with real-time CFR solver
as fallback. Pre-flop uses a separate precomputed strategy table.

Blueprint strategies are loaded from submission/data/ at startup.
If blueprint files don't exist, the bot seamlessly falls back to the
real-time CFR solver for that street.
"""

import os
import sys
import random
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from agents.agent import Agent
from gym_env import PokerEnv

from equity import ExactEquityEngine
from inference import DiscardInference
from solver import SubgameSolver

# Try to import blueprint lookup modules (graceful fallback if unavailable)
try:
    from blueprint_lookup_unbucketed import BlueprintLookupUnbucketed
    _BLUEPRINT_UNBUCKETED_AVAILABLE = True
except Exception:
    _BLUEPRINT_UNBUCKETED_AVAILABLE = False

try:
    from blueprint_lookup import BlueprintLookup
    _BLUEPRINT_AVAILABLE = True
except Exception:
    _BLUEPRINT_AVAILABLE = False

# Import action type constants for blueprint action mapping
from game_tree import (
    ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
)

FOLD = PokerEnv.ActionType.FOLD.value      # 0
RAISE = PokerEnv.ActionType.RAISE.value    # 1
CHECK = PokerEnv.ActionType.CHECK.value    # 2
CALL = PokerEnv.ActionType.CALL.value      # 3
DISCARD = PokerEnv.ActionType.DISCARD.value  # 4

# Blueprint file names per street (street 1=flop, 2=turn, 3=river)
_BLUEPRINT_FILES = {
    1: "flop_blueprint.npz",
    2: "turn_blueprint.npz",
    3: "river_blueprint.npz",
}


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.solver = SubgameSolver(self.engine)

        self._preflop_table = self._load_preflop_table()
        self._preflop_strategy = self._load_preflop_strategy()

        # Load blueprint strategies for post-flop streets
        self._blueprints = self._load_blueprints()

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None

        # Match state for lead protection
        self._bankroll = 0
        self._total_hands = 1000

    def __name__(self):
        return "PlayerAgent"

    # ----------------------------------------------------------------
    #  INIT HELPERS
    # ----------------------------------------------------------------

    def _load_preflop_table(self):
        data_path = os.path.join(_dir, "data", "preflop_potential.npz")
        if not os.path.exists(data_path):
            return None
        data = np.load(data_path)
        return dict(zip(data["bitmasks"].tolist(), data["potentials"].tolist()))

    def _load_preflop_strategy(self):
        strat_path = os.path.join(_dir, "data", "preflop_strategy.npz")
        tree_path = os.path.join(_dir, "data", "preflop_tree.pkl")
        if not os.path.exists(strat_path) or not os.path.exists(tree_path):
            return None
        import pickle
        data = np.load(strat_path)
        with open(tree_path, 'rb') as f:
            children_map = pickle.load(f)
        return {
            'strategies': data['strategies'],
            'pot_min': float(data['pot_min']),
            'pot_max': float(data['pot_max']),
            'n_buckets': int(data['n_buckets']),
            'raise_levels': data['raise_levels'].tolist(),
            'node_players': data['node_players'],
            'node_bet_sb': data['node_bet_sb'],
            'node_bet_bb': data['node_bet_bb'],
            'children_map': children_map,
        }

    def _load_blueprints(self):
        """Load blueprint strategy files for each post-flop street.

        Tries unbucketed blueprints first (*_unbucketed.npz), then falls
        back to regular bucketed blueprints. The unbucketed lookup is
        faster at runtime (no equity computation for hand matching).

        Both BlueprintLookupUnbucketed and BlueprintLookup expose the
        same get_strategy() interface, so downstream code works unchanged.

        Returns a dict mapping street (1,2,3) -> lookup instance or None.
        If neither module is available or files don't exist, that street
        maps to None (will use real-time solver instead).
        """
        blueprints = {}

        if not _BLUEPRINT_AVAILABLE and not _BLUEPRINT_UNBUCKETED_AVAILABLE:
            return blueprints

        data_dir = os.path.join(_dir, "data")
        for street, filename in _BLUEPRINT_FILES.items():
            loaded = None

            # Try unbucketed first (*_unbucketed.npz)
            if _BLUEPRINT_UNBUCKETED_AVAILABLE:
                base, ext = os.path.splitext(filename)
                unbucketed_path = os.path.join(data_dir, f"{base}_unbucketed{ext}")
                if os.path.exists(unbucketed_path):
                    try:
                        loaded = BlueprintLookupUnbucketed(
                            unbucketed_path, equity_engine=self.engine
                        )
                    except Exception:
                        loaded = None

            # Fall back to regular file (works for both bucketed and
            # unbucketed .npz since BlueprintLookupUnbucketed auto-detects)
            if loaded is None:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    # Prefer unbucketed loader (handles both formats)
                    if _BLUEPRINT_UNBUCKETED_AVAILABLE:
                        try:
                            loaded = BlueprintLookupUnbucketed(
                                filepath, equity_engine=self.engine
                            )
                        except Exception:
                            loaded = None

                    # Fall back to original bucketed-only loader
                    if loaded is None and _BLUEPRINT_AVAILABLE:
                        try:
                            loaded = BlueprintLookup(
                                filepath, equity_engine=self.engine
                            )
                        except Exception:
                            loaded = None

            blueprints[street] = loaded

        return blueprints

    def _preflop_potential(self, my_cards):
        if self._preflop_table is None:
            return None
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask)

    def _get_preflop_bucket(self, my_cards):
        if self._preflop_strategy is None or self._preflop_table is None:
            return None
        potential = self._preflop_potential(my_cards)
        if potential is None:
            return None
        ps = self._preflop_strategy
        frac = (potential - ps['pot_min']) / (ps['pot_max'] - ps['pot_min'])
        frac = max(0.0, min(1.0 - 1e-9, frac))
        return int(frac * ps['n_buckets'])

    def _find_preflop_node(self, observation):
        if self._preflop_strategy is None:
            return None
        ps = self._preflop_strategy
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        blind_pos = self._get_blind_position(observation)
        my_player = blind_pos
        bet_sb = my_bet if my_player == 0 else opp_bet
        bet_bb = my_bet if my_player == 1 else opp_bet

        # Exact match
        for nid in range(len(ps['node_players'])):
            if (ps['node_players'][nid] == my_player and
                    ps['node_bet_sb'][nid] == bet_sb and
                    ps['node_bet_bb'][nid] == bet_bb):
                return nid

        # Round to nearest node
        best_nid = None
        best_dist = float('inf')
        for nid in range(len(ps['node_players'])):
            if ps['node_players'][nid] != my_player or ps['node_players'][nid] == -1:
                continue
            dist = abs(ps['node_bet_sb'][nid] - bet_sb) + abs(ps['node_bet_bb'][nid] - bet_bb)
            if dist < best_dist:
                best_dist = dist
                best_nid = nid
        return best_nid

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None
            self._last_seen_action = None

    def _parse_cards(self, observation):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        return my_cards, board, opp_discards, my_discards

    def _get_pot_size(self, observation):
        return observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])

    def _get_blind_position(self, observation):
        return observation.get("blind_position", 0)

    def _compute_raise_amount(self, observation, pot_fraction):
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        amount = max(int(pot_size * pot_fraction), min_raise)
        return min(amount, max_raise)

    # ----------------------------------------------------------------
    #  DISCARD
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )
        results = self.engine.evaluate_all_keep_pairs(
            my_cards, board, opp_discards, self._opp_weights
        )
        best_keep = results[0][0]
        return (DISCARD, 0, best_keep[0], best_keep[1])

    # ----------------------------------------------------------------
    #  PRE-FLOP
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        valid_actions = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        continue_cost = opp_bet - my_bet

        # The preflop tree was solved with max raise=30, but the game
        # allows bets up to 100. When facing a large bet, the tree's
        # node matching rounds to the nearest node and produces wrong
        # fold frequencies (nearly 0% fold). Handle large bets directly
        # using hand potential percentile.
        if continue_cost > 20 and valid_actions[FOLD]:
            potential = self._preflop_potential(my_cards)
            if potential is None:
                potential = 0.5

            # Potential distribution: median=0.84, range 0.52-0.95.
            # Scale threshold with stack commitment to fold weak hands:
            # cost=25: threshold≈0.81 (fold ~30%)
            # cost=40: threshold≈0.83 (fold ~40%)
            # cost=60: threshold≈0.84 (fold ~51%)
            # cost=80: threshold≈0.86 (fold ~66%)
            # cost=98: threshold≈0.88 (fold ~77%)
            commit_frac = min(continue_cost / 100.0, 0.98)
            threshold = 0.79 + 0.09 * commit_frac

            if potential < threshold:
                return (FOLD, 0, 0, 0)

            # Strong enough to continue — call, don't re-escalate
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)

        # Try precomputed GTO strategy (reliable for small bets ≤30)
        bucket = self._get_preflop_bucket(my_cards)
        node_id = self._find_preflop_node(observation)

        if bucket is not None and node_id is not None and self._preflop_strategy is not None:
            ps = self._preflop_strategy
            strategy = ps['strategies'][node_id, bucket]
            children = ps['children_map'].get(node_id, {})

            if len(children) > 0:
                valid_mask = np.zeros(len(strategy))
                for act_id in children:
                    if act_id == 0 and valid_actions[FOLD]:
                        valid_mask[act_id] = strategy[act_id]
                    elif act_id == 1:
                        if valid_actions[CALL] or valid_actions[CHECK]:
                            valid_mask[act_id] = strategy[act_id]
                    elif act_id >= 2 and valid_actions[RAISE]:
                        valid_mask[act_id] = strategy[act_id]

                total = valid_mask.sum()
                if total > 0:
                    valid_mask /= total
                    chosen = int(np.random.choice(len(valid_mask), p=valid_mask))

                    if chosen == 0:
                        return (FOLD, 0, 0, 0)
                    elif chosen == 1:
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)
                    elif chosen >= 2:
                        raise_to = ps['raise_levels'][chosen - 2]
                        raise_amount = raise_to - opp_bet
                        raise_amount = max(raise_amount, observation["min_raise"])
                        raise_amount = min(raise_amount, observation["max_raise"])
                        if raise_amount > 0 and valid_actions[RAISE]:
                            return (RAISE, raise_amount, 0, 0)
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)

        # Fallback: pot odds with adjusted threshold for large bets
        potential = self._preflop_potential(my_cards)
        if potential is None:
            potential = 0.5

        if valid_actions[CALL]:
            if continue_cost <= 1:
                return (CALL, 0, 0, 0)
            pot_size = self._get_pot_size(observation)
            # Raw pot odds underestimate the threshold because
            # preflop "potential" doesn't account for opponent range
            # narrowing when they raise. Add a margin.
            required_equity = continue_cost / (continue_cost + pot_size)
            margin = 0.08 * min(continue_cost / 30.0, 1.0)
            if potential >= required_equity + margin:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    # ----------------------------------------------------------------
    #  POST-FLOP (blueprint + CFR solver fallback)
    # ----------------------------------------------------------------

    def _try_blueprint(self, observation, my_cards, board, dead_cards=None):
        """Try to get an action from the blueprint strategy.

        Returns a concrete (action_type, amount, 0, 0) tuple if the
        blueprint has a strategy for this state, or None to signal
        that the caller should fall back to the real-time solver.
        """
        street = observation["street"]
        blueprint = self._blueprints.get(street)
        if blueprint is None:
            return None

        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        facing_bet = opp_bet > my_bet

        # The flop blueprint provides range-balanced pot control:
        # medium hands check/fold instead of overbetting.

        # Get blueprint strategy (action type -> probability)
        try:
            pot_state = (my_bet, opp_bet)
            strategy = blueprint.get_strategy(
                hero_cards=my_cards, board=board, pot_state=pot_state,
                dead_cards=dead_cards, opp_weights=self._opp_weights)
        except Exception:
            return None

        if strategy is None:
            return None

        # Detect unconverged blueprint (uniform distribution = not converged)
        probs = list(strategy.values())
        if len(probs) > 2 and max(probs) - min(probs) < 0.05:
            return None  # fall back to solver

        # Verify the strategy makes sense for our situation
        if facing_bet:
            has_fold = ACT_FOLD in strategy and strategy[ACT_FOLD] > 0.001
            if not has_fold:
                # Strategy has no fold option but we're facing a bet →
                # wrong node was returned, fall back to solver
                return None

        # Sample an action type from the blueprint strategy
        action_ids = list(strategy.keys())
        probs = np.array([strategy[a] for a in action_ids])
        probs /= probs.sum()  # safety normalization
        chosen_action = int(np.random.choice(action_ids, p=probs))

        # Map the abstract action type to a concrete engine action
        return self._blueprint_action_to_engine(
            chosen_action, observation
        )

    def _blueprint_action_to_engine(self, action_type, observation):
        """Convert a blueprint action type ID to a concrete engine action.

        Maps abstract blueprint actions (ACT_FOLD, ACT_RAISE_HALF, etc.)
        to the engine's (action, amount, 0, 0) format.

        Args:
            action_type: int, one of ACT_FOLD..ACT_RAISE_OVERBET
            observation: the current game observation

        Returns:
            (action, amount, 0, 0) tuple, or None if the action is invalid
            and no safe fallback exists.
        """
        valid_actions = observation["valid_actions"]
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        if action_type == ACT_FOLD:
            if valid_actions[FOLD]:
                return (FOLD, 0, 0, 0)
            # Can't fold (shouldn't happen, but be safe)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        elif action_type == ACT_CHECK:
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            # Blueprint says check but we can't — try call
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            return None

        elif action_type == ACT_CALL:
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        elif action_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                             ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            # Map to pot fraction
            pot_fractions = {
                ACT_RAISE_HALF: 0.40,
                ACT_RAISE_POT: 0.70,
                ACT_RAISE_ALLIN: 1.00,
                ACT_RAISE_OVERBET: 1.50,
            }
            fraction = pot_fractions[action_type]
            raise_amount = max(int(pot_size * fraction), min_raise)
            raise_amount = min(raise_amount, max_raise)

            if valid_actions[RAISE] and raise_amount > 0:
                return (RAISE, raise_amount, 0, 0)

            # Raise not valid — fall back to call or check
            if valid_actions[CALL]:
                return (CALL, 0, 0, 0)
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            return None

        # Unknown action type — signal fallback
        return None

    def _narrow_range_from_action(self, opp_action_type, board, dead_cards,
                                   pot_state, street):
        """Narrow opponent range based on their betting action.

        Uses the blueprint to determine which opponent hands would take
        this action. Hands that wouldn't take this action get their
        weight reduced. This is nested subgame solving — the opponent's
        betting tells us about their range.

        Args:
            opp_action_type: 'bet', 'check', or 'raise'
            board: list of community cards
            dead_cards: list of dead cards
            pot_state: (hero_bet, opp_bet) before opponent's action
            street: 1-3
        """
        if self._opp_weights is None:
            return

        blueprint = self._blueprints.get(street)
        if blueprint is None or not hasattr(blueprint, '_unbucketed') or not blueprint._unbucketed:
            return

        known = set(board) | set(dead_cards)
        remaining = [c for c in range(27) if c not in known]

        import itertools
        new_weights = {}

        for hand in itertools.combinations(remaining, 2):
            hand_tuple = tuple(sorted(hand))
            old_weight = self._opp_weights.get(hand_tuple, 0.0)
            if old_weight < 0.001:
                continue

            # Look up what the blueprint says this opp hand would do
            try:
                opp_strat = blueprint.get_strategy(
                    hero_cards=list(hand), board=board,
                    pot_state=pot_state, dead_cards=dead_cards)
            except Exception:
                new_weights[hand_tuple] = old_weight
                continue

            if opp_strat is None:
                new_weights[hand_tuple] = old_weight
                continue

            # Compute probability that this hand takes the observed action
            if opp_action_type == 'bet':
                # Any raise action
                action_prob = sum(p for a, p in opp_strat.items()
                                 if a in (ACT_RAISE_HALF, ACT_RAISE_POT,
                                         ACT_RAISE_ALLIN, ACT_RAISE_OVERBET))
            elif opp_action_type == 'check':
                action_prob = opp_strat.get(ACT_CHECK, 0) + opp_strat.get(ACT_CALL, 0)
            elif opp_action_type == 'call':
                action_prob = opp_strat.get(ACT_CALL, 0)
            else:
                action_prob = 1.0

            # Bayes: P(hand | action) ∝ P(action | hand) * P(hand)
            new_weights[hand_tuple] = old_weight * max(action_prob, 0.01)

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            for k in new_weights:
                new_weights[k] /= total
            self._opp_weights = new_weights

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Post-flop decision with range narrowing.

        Uses nested subgame solving: narrows opponent range based on
        their betting actions before looking up our own strategy.
        """
        dead_cards = my_discards + opp_discards

        # Ensure opponent range weights are computed
        if len(opp_discards) == 3 and self._opp_weights is None:
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )

        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        street = observation["street"]

        # Narrow opponent range based on bet state.
        # If opponent bet (opp_bet > equal share), they raised — narrow to
        # hands that would raise. If bets are equal, they checked — narrow
        # to hands that would check.
        if self._opp_weights is not None:
            if opp_bet > my_bet:
                self._narrow_range_from_action(
                    'bet', board, dead_cards, (my_bet, my_bet), street)
            elif my_bet == opp_bet and my_bet > 0:
                # Equal bets could mean opponent checked (if we act second)
                # or we're first to act. Only narrow if we know they checked.
                blind_pos = self._get_blind_position(observation)
                if blind_pos == 0:  # we're SB (act second postflop)
                    # Opponent (BB) acted first — if bets equal, they checked
                    self._narrow_range_from_action(
                        'check', board, dead_cards, (my_bet, opp_bet), street)

        # When facing a bet, compute equity against the NARROWED range.
        # If equity is below pot odds, fold regardless of what the blueprint
        # says — the blueprint was solved against the full range, not the
        # narrowed range that accounts for opponent's betting line.
        if opp_bet > my_bet and self._opp_weights is not None:
            continue_cost = opp_bet - my_bet
            pot_size = my_bet + opp_bet
            equity = self.engine.compute_equity(
                my_cards, board, dead_cards, self._opp_weights)
            pot_odds = continue_cost / (continue_cost + pot_size)

            if equity < pot_odds:
                valid_actions = observation["valid_actions"]
                if valid_actions[FOLD]:
                    return (FOLD, 0, 0, 0)

        # Try blueprint strategy
        blueprint_action = self._try_blueprint(observation, my_cards, board, dead_cards)
        if blueprint_action is not None:
            return blueprint_action

        # Fall back to real-time CFR solver
        max_raise = observation["max_raise"]
        valid_actions = observation["valid_actions"]

        return self.solver.solve_and_act(
            hero_cards=my_cards,
            board=board,
            opp_range=self._opp_weights,
            dead_cards=dead_cards,
            my_bet=my_bet,
            opp_bet=opp_bet,
            street=street,
            min_raise=observation["min_raise"],
            max_raise=max_raise,
            valid_actions=valid_actions,
            hero_is_first=True,
            time_remaining=observation.get("time_left", 400),
        )

    # ----------------------------------------------------------------
    #  MAIN
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        # Track bankroll for lead protection
        if reward != 0:
            self._bankroll += reward

        # Lead protection: if we're ahead by enough to fold the rest
        # and still win, just fold/check everything. Binary tournament
        # format means winning by 1 chip = winning by 1000.
        hands_remaining = self._total_hands - hand_number
        blind_cost = hands_remaining * 1.5  # avg cost of folding every hand
        if self._bankroll > blind_cost + 10:
            valid_actions = observation["valid_actions"]
            if valid_actions[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid_actions[FOLD]:
                return (FOLD, 0, 0, 0)
            # Must discard — still pick the best cards
            if valid_actions[DISCARD]:
                my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
                return self._handle_discard(observation, my_cards, board, opp_discards)
            return (FOLD, 0, 0, 0)

        my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
        valid_actions = observation["valid_actions"]

        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

        return self._handle_postflop(observation, my_cards, board, opp_discards, my_discards, info)

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        if reward != 0:
            self._bankroll += reward

        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
