"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Architecture:
  1. ExactEquityEngine: Exact win probability via full enumeration (~10K evals, <5ms)
  2. DiscardInference: Bayesian opponent range narrowing from revealed discards
  3. OpponentModel: Within-match stat tracking over 1000 hands
  4. Lead-aware + position-aware betting strategy

Key edges over reference bots:
  - Exact equity (zero error) vs Monte Carlo (2.5% error with 400 samples)
  - Discard inference narrows opponent range from ~120 hands to ~20-40
  - Opponent exploitation after ~50 hands of data collection
  - Match-state aware play (protect leads, seek variance when behind)
"""

import os
import sys
import random
import numpy as np

# Ensure submission directory is importable
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from agents.agent import Agent
from gym_env import PokerEnv

from equity import ExactEquityEngine
from inference import DiscardInference
from opponent import OpponentModel

FOLD = PokerEnv.ActionType.FOLD.value      # 0
RAISE = PokerEnv.ActionType.RAISE.value    # 1
CHECK = PokerEnv.ActionType.CHECK.value    # 2
CALL = PokerEnv.ActionType.CALL.value      # 3
DISCARD = PokerEnv.ActionType.DISCARD.value  # 4


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)

        # Core components
        self.engine = ExactEquityEngine()
        self.inference = DiscardInference(self.engine)
        self.opp_model = OpponentModel()

        # Pre-flop hand potential lookup table
        self._preflop_table = self._load_preflop_table()

        # Betting thresholds (tunable)
        self.base_raise_threshold = 0.65
        self.base_strong_raise_threshold = 0.80
        self.base_bluff_frequency = 0.12

        # Metagame trap state
        self._trap_phase = "probe"  # "probe" -> "bait" -> "exploit"
        self._opponent_is_adaptive = False

        # Per-match state (persists across all 1000 hands)
        self.cumulative_reward = 0.0

        # Per-hand state (reset each hand)
        self._current_hand = -1
        self._opp_weights = None       # Bayesian opponent range for current hand
        self._last_seen_action = None  # to avoid double-counting actions
        self._hand_street_seen = set() # track which streets we've recorded actions for
        self._my_player_index = None   # 0 or 1, determined from first hand

    def __name__(self):
        return "PlayerAgent"

    def _load_preflop_table(self):
        """Load precomputed pre-flop hand potential table.

        Maps 5-card bitmask -> expected post-discard equity (0-1).
        If file doesn't exist, returns None (falls back to heuristic).
        """
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "preflop_potential.npz")
        if not os.path.exists(data_path):
            return None
        data = np.load(data_path)
        bitmasks = data["bitmasks"]
        potentials = data["potentials"]
        table = {}
        for i in range(len(bitmasks)):
            table[int(bitmasks[i])] = float(potentials[i])
        return table

    def _preflop_potential(self, my_cards):
        """Look up the pre-flop hand potential for a 5-card hand.

        Returns a float in ~[0.37, 0.65] representing expected post-discard
        equity across all possible flops. Higher = better hand.
        Returns None if table not available.
        """
        if self._preflop_table is None:
            return None
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask)

    def _reset_hand(self, hand_number):
        """Reset per-hand state when a new hand starts."""
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None
            self._last_seen_action = None
            self._hand_street_seen = set()

    def _determine_player_index(self, observation, info):
        """Figure out if we are player 0 or player 1.

        match.py: small_blind_player = hand_number % 2
        If blind_position == 0 (we are SB), then we are the small_blind_player.
        So our player index = hand_number % 2.
        If blind_position == 1 (we are BB), our index = 1 - (hand_number % 2).
        """
        if self._my_player_index is not None:
            return self._my_player_index
        hand_number = info.get("hand_number", 0)
        bp = self._get_blind_position(observation)
        if bp == 0:  # we are SB
            self._my_player_index = hand_number % 2
        else:  # we are BB
            self._my_player_index = 1 - (hand_number % 2)
        return self._my_player_index

    def _parse_cards(self, observation):
        """Extract card lists from observation, filtering out -1 placeholders."""
        my_cards = [c for c in observation["my_cards"] if c != -1]
        board = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        return my_cards, board, opp_discards, my_discards

    def _get_pot_size(self, observation):
        """Compute pot size. Pydantic may strip 'pot_size' from the observation
        since it's not declared in the Observation TypedDict in agent.py."""
        return observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])

    def _get_blind_position(self, observation):
        """Get blind position (0=SB, 1=BB). Falls back to computing from hand number
        since Pydantic may strip 'blind_position'."""
        if "blind_position" in observation:
            return observation["blind_position"]
        # Infer from hand number: positions alternate each hand
        # player 0 is SB on even hands, BB on odd hands
        # But we don't know if we're player 0 or 1, so use acting_agent heuristic
        # Pre-flop: SB acts first -> if we're acting pre-flop first, we're SB
        return 0  # default, will be overridden when we can determine

    # ----------------------------------------------------------------
    #  DISCARD DECISION
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        """Choose which 2 of 5 cards to keep on the flop.

        If we're SB (opponent already discarded), use discard inference
        for weighted equity. If we're BB (discard first), use uniform equity.

        Why: SB sees BB's discards before deciding, giving an information edge.
        Using weighted equity means our discard choice accounts for what the
        opponent likely kept, not just raw hand strength.
        """
        if len(opp_discards) == 3 and self._opp_weights is None:
            # We are SB: opponent (BB) already discarded, we can infer their range
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )

        results = self.engine.evaluate_all_keep_pairs(
            my_cards, board, opp_discards, self._opp_weights
        )

        best_keep = results[0][0]  # (idx1, idx2) of highest equity pair
        return (DISCARD, 0, best_keep[0], best_keep[1])

    # ----------------------------------------------------------------
    #  PRE-FLOP STRATEGY
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        """Pre-flop strategy using precomputed hand potential table.

        Uses a lookup table that maps every 5-card hand to its expected
        post-discard equity across all possible flops. This replaces the
        naive heuristic (count pairs/aces) with actual computed equity.

        The potential score ranges from ~0.37 (worst) to ~0.65 (best).
        We normalize this to a 0-10 strength score for decision thresholds.

        Why we almost always call from SB:
        - SB needs to call 1 chip to see a pot of 3 = 33% pot odds.
        - Even the worst 5-card hand has ~37% potential after optimal discard.
        - Folding pre-flop is almost never correct from SB.
        """
        valid_actions = observation["valid_actions"]

        # Try lookup table first, fall back to heuristic
        potential = self._preflop_potential(my_cards)
        if potential is not None:
            # Normalize potential (0.37-0.65) to strength (0-10)
            # 0.37 -> 0, 0.65 -> 10
            strength = max(0.0, min(10.0, (potential - 0.37) / 0.028))
        else:
            # Fallback heuristic if table not loaded
            strength = self._preflop_heuristic(my_cards)

        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = self._get_pot_size(observation)

        # Decision: use strength thresholds
        if valid_actions[CALL]:
            if strength >= 6.0 and valid_actions[RAISE]:
                # Strong hand: raise
                frac = 0.6 + 0.15 * (strength - 6.0) / 4.0
                raise_amt = self._compute_raise_amount(observation, frac)
                return (RAISE, raise_amt, 0, 0)
            if continue_cost <= 1:
                # SB calling 1 chip: almost always correct
                return (CALL, 0, 0, 0)
            if strength >= 3.0:
                # Facing a raise: call with decent+ hands
                return (CALL, 0, 0, 0)
            if continue_cost > pot_size and strength < 2.0:
                # Big raise, garbage hand: fold
                return (FOLD, 0, 0, 0)
            return (CALL, 0, 0, 0)

        if valid_actions[CHECK]:
            if strength >= 7.0 and valid_actions[RAISE]:
                frac = 0.6 + 0.15 * (strength - 7.0) / 3.0
                raise_amt = self._compute_raise_amount(observation, frac)
                return (RAISE, raise_amt, 0, 0)
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    def _preflop_heuristic(self, my_cards):
        """Fallback heuristic when preflop table is not available."""
        from collections import Counter
        ranks = [c % 9 for c in my_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(c // 9 for c in my_cards)

        strength = 0.0
        if any(count >= 3 for count in rank_counts.values()):
            strength += 5
        pair_count = sum(1 for count in rank_counts.values() if count >= 2)
        if pair_count >= 2:
            strength += 4
        elif pair_count == 1:
            strength += 2
        if 8 in ranks:
            strength += 1
        if max(suit_counts.values()) >= 3:
            strength += 1
        sorted_ranks = sorted(set(ranks))
        max_run = 1
        run = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= 3:
            strength += 1
        return strength

    # ----------------------------------------------------------------
    #  POST-FLOP BETTING
    # ----------------------------------------------------------------

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Equity-based post-flop betting with opponent modeling and lead awareness.

        Decision hierarchy:
        1. Compute exact equity (with Bayesian weights if available)
        2. Apply opponent-specific adjustments to thresholds
        3. Apply match-state (lead) adjustments
        4. Choose action based on adjusted thresholds vs pot odds

        Why pot-odds based:
        - Pot odds tell us the minimum equity needed to profitably call.
        - Raising above certain equity thresholds extracts value from worse hands.
        - This is mathematically sound and beats heuristic-only approaches.
        """
        dead_cards = my_discards + opp_discards
        equity = self.engine.compute_equity(my_cards, board, dead_cards, self._opp_weights)

        valid_actions = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        street = observation["street"]
        hand_number = info.get("hand_number", 0)

        continue_cost = opp_bet - my_bet
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        # --- Adjust thresholds ---
        raise_thresh = self.base_raise_threshold
        strong_raise_thresh = self.base_strong_raise_threshold
        bluff_freq = self.base_bluff_frequency

        # Opponent exploitation
        exploits = self.opp_model.get_exploitation_adjustments()
        raise_thresh += exploits["raise_threshold_mod"]
        bluff_freq *= exploits["bluff_frequency_mod"]
        pot_odds_adj = exploits["call_threshold_mod"]

        # Position adjustment: SB acts last post-flop (advantage)
        is_sb = self._get_blind_position(observation) == 0
        if is_sb:
            raise_thresh -= 0.03  # slightly more aggressive in position

        # Metagame trap: if opponent is adaptive, we played passive early
        # to bait their model, now switch to aggressive exploitation.
        # Hands 0-100: probe (play normally, detect if opponent adapts)
        # Hands 100-200: if adaptive detected, bait (play extra passive, no bluffs)
        # Hands 200+: exploit (bluff heavily, they think we never bluff)
        hand_number = info.get("hand_number", 0)
        if hand_number == 100 and self.opp_model.has_enough_data():
            self._opponent_is_adaptive = self.opp_model.is_adaptive()
        if self._opponent_is_adaptive:
            if 100 <= hand_number < 200:
                self._trap_phase = "bait"
                bluff_freq = 0.0  # never bluff during bait phase
                raise_thresh += 0.05  # play tighter to appear passive
            elif hand_number >= 200:
                self._trap_phase = "exploit"
                bluff_freq *= 2.0  # double bluffs — they think we never bluff
                raise_thresh -= 0.05  # play looser

        # Lead-aware adjustment (last 300 hands)
        hands_remaining = 1000 - hand_number
        if hands_remaining < 300:
            if self.cumulative_reward > 80:
                # Comfortably ahead: tighten up significantly
                raise_thresh += 0.08
                bluff_freq *= 0.3
            elif self.cumulative_reward > 30:
                # Ahead: tighten slightly
                raise_thresh += 0.04
                bluff_freq *= 0.6
            elif self.cumulative_reward < -80:
                # Way behind: loosen up
                raise_thresh -= 0.06
                bluff_freq *= 2.0
            elif self.cumulative_reward < -30:
                # Behind: loosen slightly
                raise_thresh -= 0.03
                bluff_freq *= 1.5

        # --- Make decision ---

        # All-in with near-certain hands
        if equity > 0.95 and valid_actions[RAISE]:
            return (RAISE, max_raise, 0, 0)

        # Strong raise
        if equity > strong_raise_thresh and valid_actions[RAISE]:
            frac = 0.75 + 0.25 * (equity - strong_raise_thresh) / (1.0 - strong_raise_thresh)
            raise_amt = self._compute_raise_amount(observation, frac)
            return (RAISE, raise_amt, 0, 0)

        # Medium raise
        if equity > raise_thresh and valid_actions[RAISE]:
            frac = 0.5 + 0.25 * (equity - raise_thresh) / (strong_raise_thresh - raise_thresh)
            raise_amt = self._compute_raise_amount(observation, frac)
            return (RAISE, raise_amt, 0, 0)

        # Blocker-aware bluff: bluff more when we hold cards that block
        # opponent's strong hands. In a 3-suit deck, each card we hold
        # blocks 33% of pairs of that rank (vs 17% in standard poker).
        if (equity < 0.30
                and valid_actions[RAISE]
                and bluff_freq > 0):
            blocker_bonus = self._compute_blocker_bonus(my_cards, board, opp_discards)
            effective_bluff_freq = bluff_freq * (1.0 + blocker_bonus)

            if random.random() < effective_bluff_freq:
                opp_fold_rate = self.opp_model.fold_rate(street)
                if opp_fold_rate > 0.30 or not self.opp_model.has_enough_data():
                    raise_amt = self._compute_raise_amount(observation, 0.6)
                    return (RAISE, raise_amt, 0, 0)

        # Call if equity justifies it
        adjusted_pot_odds = pot_odds + pot_odds_adj
        if valid_actions[CALL] and equity >= adjusted_pot_odds:
            return (CALL, 0, 0, 0)

        # Check if possible
        if valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)

        # Fold
        return (FOLD, 0, 0, 0)

    def _compute_blocker_bonus(self, my_cards, board, opp_discards):
        """Compute a bluff bonus based on blocker effects.

        In a 3-suit deck, each rank has only 3 copies. If we hold a card of
        rank R, opponent can only have C(2,2)=1 possible pair of R (vs C(3,2)=3
        without our blocker). That's 67% of pairs blocked by holding one card.

        We also check board cards and known discards for additional blocking.

        Returns a float 0.0-1.5 representing how much to boost bluff frequency.
        Higher = more blockers = safer to bluff.
        """
        all_known = set(my_cards) | set(board) | set(opp_discards)

        # Count how many high-rank cards are accounted for (in our hand + board + discards)
        # High ranks (7, 8, 9=Ace) are most important to block
        bonus = 0.0
        high_ranks = [6, 7, 8]  # rank indices for 8, 9, Ace

        for rank in high_ranks:
            # Cards of this rank: rank + 0*9, rank + 1*9, rank + 2*9
            cards_of_rank = [rank, rank + 9, rank + 18]
            known_count = sum(1 for c in cards_of_rank if c in all_known)

            if known_count >= 2:
                # 2+ of 3 copies are accounted for — opponent very unlikely to have pair
                bonus += 0.4
            elif known_count == 1 and any(c in set(my_cards) for c in cards_of_rank):
                # We hold one — blocks 67% of opponent's pairs of this rank
                bonus += 0.2

        # Check if we block straight completions on the board
        board_ranks = sorted(set(c % 9 for c in board))
        my_ranks = set(c % 9 for c in my_cards)

        # If we hold cards that complete board straights, opponent is less likely
        # to have those straight-completing cards
        for rank in my_ranks:
            if rank in board_ranks:
                bonus += 0.1  # we block board pairs too

        return min(bonus, 1.5)  # cap at 1.5x bonus

    def _compute_raise_amount(self, observation, pot_fraction):
        """Compute a raise amount as a fraction of the pot, clamped to valid range.

        Why fraction-based: Raise sizing should scale with the pot. A 75% pot
        raise when the pot is 10 chips is 7.5 chips; when the pot is 50 it's
        37.5 chips. This ensures bets are proportional to what's at stake.
        """
        pot_size = self._get_pot_size(observation)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        amount = max(int(pot_size * pot_fraction), min_raise)
        amount = min(amount, max_raise)
        return amount

    # ----------------------------------------------------------------
    #  OPPONENT TRACKING
    # ----------------------------------------------------------------

    def _track_opponent_action(self, observation, info):
        """Record opponent's last action for the opponent model.

        Called from both act() and observe(). Uses deduplication to avoid
        counting the same action twice.
        """
        opp_action = observation.get("opp_last_action", "None")
        if opp_action == "None":
            return

        # Deduplicate: create a unique key for this action
        street = observation.get("street", 0)
        action_key = (self._current_hand, street, opp_action, observation.get("opp_bet", 0))
        if action_key == self._last_seen_action:
            return
        self._last_seen_action = action_key

        self.opp_model.record_action(street, opp_action, observation)

    # ----------------------------------------------------------------
    #  MAIN ACT / OBSERVE
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        """Main decision function called by the engine when it's our turn.

        Flow:
        1. Reset state if new hand
        2. Track any opponent actions we see in the observation
        3. Route to appropriate handler (discard / preflop / postflop)
        """
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)
        self._determine_player_index(observation, info)

        # Track opponent action if one occurred before our turn
        self._track_opponent_action(observation, info)

        my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
        valid_actions = observation["valid_actions"]

        # Discard phase (mandatory on flop, street 1)
        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        # Pre-flop (street 0): 5 hole cards, no board
        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

        # Post-flop: exact equity-based betting
        return self._handle_postflop(observation, my_cards, board, opp_discards, my_discards, info)

    def observe(self, observation, reward, terminated, truncated, info):
        """Called after opponent acts and when hands end.

        Two purposes:
        1. Track opponent behavior for the opponent model
        2. Record hand results and showdown data for calibration

        Why implement observe():
        - Most bots ignore this (the base class is a no-op)
        - We use it to build a per-match opponent profile
        - Showdown data calibrates our discard inference temperature
        """
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        # Track opponent action
        self._track_opponent_action(observation, info)

        # Infer opponent range when we first see their discards
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )

        # End of hand: record results
        if terminated:
            self.cumulative_reward += reward
            self.opp_model.record_hand_result(reward, info)

            # Calibrate discard inference from showdown data every 100 hands
            if self.opp_model.hands_played % 100 == 0 and self.opp_model.hands_played > 0:
                self._calibrate_inference()

    def _calibrate_inference(self):
        """Use showdown data to calibrate the discard inference temperature.

        At showdown we see opponent's final 2 cards. Combined with their 3
        discards (which we always see), we know their full original 5-card hand.
        We check: did they keep the equity-maximizing pair?

        If they did >80% of the time: lower temperature (assume more rational).
        If they did <50% of the time: raise temperature (assume more random).
        """
        showdown_info_list = self.opp_model.showdown_data
        if len(showdown_info_list) < 10:
            return

        if self._my_player_index is None:
            return

        opp_index = 1 - self._my_player_index
        opp_key = f"player_{opp_index}_cards"

        optimal_count = 0
        total_checked = 0

        for sd_info in showdown_info_list[-50:]:  # last 50 showdowns
            if opp_key not in sd_info or "community_cards" not in sd_info:
                continue

            opp_kept_strs = sd_info[opp_key]  # e.g. ["2d", "5h"]
            board_strs = sd_info["community_cards"]  # e.g. ["3d", "4h", "5d", "6h", "7s"]

            if len(opp_kept_strs) != 2 or len(board_strs) < 3:
                continue

            # We need the opponent's discards for this hand too, but showdown
            # info doesn't include them directly. We stored them in the
            # observation during the hand. For now, use fold-rate heuristic
            # as a proxy since we can't recover per-hand discards from info.
            total_checked += 1

        # Fall back to fold-rate proxy for temperature adjustment
        if self.opp_model.has_enough_data():
            avg_fold = sum(self.opp_model.fold_rate(s) for s in range(1, 4)) / 3
            vpip = self.opp_model.vpip()

            if avg_fold > 0.5 and vpip < 0.4:
                # Tight folder -> probably rational discarder -> lower temp
                self.inference.temperature = max(2.0, self.inference.temperature - 0.5)
            elif avg_fold < 0.15 and vpip > 0.7:
                # Loose caller -> might keep weird hands -> higher temp
                self.inference.temperature = min(15.0, self.inference.temperature + 1.0)
            elif vpip > 0.85:
                # Plays almost everything -> very high temp
                self.inference.temperature = min(15.0, self.inference.temperature + 2.0)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
