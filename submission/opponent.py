"""
Within-match opponent modeling with rolling window.

Tracks opponent behavior across all 1000 hands in a match using two layers:
  1. Rolling window (last WINDOW_SIZE hands): Recent behavior, adapts quickly.
     Used for exploitation decisions. Defends against opponents who change
     strategy mid-match to manipulate our model.
  2. Cumulative stats: All-time stats for the match. Used for detecting
     strategy shifts (comparing recent vs overall behavior).

After ~30 hands in the window, we start exploiting detected patterns.
"""

from collections import deque


WINDOW_SIZE = 100  # Number of recent hands to consider for exploitation


class _StreetStats:
    """Tracks action counts for a single street within a window."""
    __slots__ = ['folds', 'raises', 'calls', 'checks', 'total', 'raise_amounts']

    def __init__(self):
        self.folds = 0
        self.raises = 0
        self.calls = 0
        self.checks = 0
        self.total = 0
        self.raise_amounts = []


class _HandRecord:
    """Summary of opponent actions in a single hand, used by the rolling window."""
    __slots__ = ['hand_num', 'street_actions', 'vpip', 'pfr']

    def __init__(self, hand_num):
        self.hand_num = hand_num
        # street -> list of (action_str, raise_amount_or_0)
        self.street_actions = {s: [] for s in range(4)}
        self.vpip = False
        self.pfr = False


class OpponentModel:
    def __init__(self):
        self.hands_played = 0

        # Rolling window: deque of _HandRecord, max WINDOW_SIZE
        self._window = deque(maxlen=WINDOW_SIZE)

        # Cumulative stats (for shift detection — compare recent vs all-time)
        self._cum_street_folds = {s: 0 for s in range(4)}
        self._cum_street_actions = {s: 0 for s in range(4)}
        self._cum_vpip = 0
        self._cum_pfr = 0
        self._cum_preflop_hands = 0

        # Showdown calibration data
        self.showdown_data = []

        # Per-hand tracking (built during a hand, committed at end)
        self._current_hand = -1
        self._current_record = None
        self._hand_action_keys = []  # dedup keys
        self._opp_preflop_raised = False
        self._opp_preflop_voluntarily_in = False

    def record_action(self, street, action_str, observation):
        """Record an opponent action observed during a hand."""
        hand_num = observation.get("hand_number", 0) if isinstance(observation, dict) else 0

        # Deduplicate
        action_key = (hand_num, street, action_str,
                      observation.get("opp_bet", 0) if isinstance(observation, dict) else 0)
        if action_key in self._hand_action_keys:
            return
        self._hand_action_keys.append(action_key)

        # New hand: start a fresh record
        if hand_num != self._current_hand:
            self._current_hand = hand_num
            self._current_record = _HandRecord(hand_num)
            self._hand_action_keys = [action_key]
            self._opp_preflop_raised = False
            self._opp_preflop_voluntarily_in = False

        if street > 3 or self._current_record is None:
            return

        raise_amt = observation.get("opp_bet", 0) if isinstance(observation, dict) else 0
        self._current_record.street_actions[street].append((action_str, raise_amt))

        if street == 0:
            if action_str == "RAISE":
                self._opp_preflop_raised = True
                self._opp_preflop_voluntarily_in = True
            elif action_str == "CALL":
                self._opp_preflop_voluntarily_in = True

    def record_hand_result(self, reward, info):
        """Commit the current hand record to the window and cumulative stats."""
        self.hands_played += 1

        if self._current_record is not None:
            self._current_record.vpip = self._opp_preflop_voluntarily_in
            self._current_record.pfr = self._opp_preflop_raised
            self._window.append(self._current_record)

        # Update cumulative stats
        self._cum_preflop_hands += 1
        if self._opp_preflop_voluntarily_in:
            self._cum_vpip += 1
        if self._opp_preflop_raised:
            self._cum_pfr += 1

        # Update cumulative street stats from current record
        if self._current_record is not None:
            for s in range(4):
                for action_str, _ in self._current_record.street_actions[s]:
                    self._cum_street_actions[s] += 1
                    if action_str == "FOLD":
                        self._cum_street_folds[s] += 1

        # Showdown data
        if "player_0_cards" in info and "player_1_cards" in info:
            self.showdown_data.append(info)

        # Reset per-hand
        self._current_record = None
        self._hand_action_keys = []
        self._opp_preflop_raised = False
        self._opp_preflop_voluntarily_in = False

    # ----------------------------------------------------------------
    #  WINDOW-BASED STATS (for exploitation)
    # ----------------------------------------------------------------

    def _window_street_stats(self, street):
        """Aggregate stats for a street from the rolling window."""
        stats = _StreetStats()
        for record in self._window:
            for action_str, raise_amt in record.street_actions.get(street, []):
                stats.total += 1
                if action_str == "FOLD":
                    stats.folds += 1
                elif action_str == "RAISE":
                    stats.raises += 1
                    stats.raise_amounts.append(raise_amt)
                elif action_str == "CALL":
                    stats.calls += 1
                elif action_str == "CHECK":
                    stats.checks += 1
        return stats

    def fold_rate(self, street):
        """Opponent's recent fold frequency on a given street."""
        stats = self._window_street_stats(street)
        if stats.total < 5:
            return 0.3  # prior
        return stats.folds / stats.total

    def raise_rate(self, street):
        """Opponent's recent raise frequency on a given street."""
        stats = self._window_street_stats(street)
        if stats.total < 5:
            return 0.2  # prior
        return stats.raises / stats.total

    def avg_raise_size(self, street):
        """Average recent raise size on a given street."""
        stats = self._window_street_stats(street)
        if not stats.raise_amounts:
            return 4  # prior
        return sum(stats.raise_amounts) / len(stats.raise_amounts)

    def aggression_factor(self, street):
        """Recent (raises) / (calls). Higher = more aggressive."""
        stats = self._window_street_stats(street)
        if stats.calls == 0:
            return 2.0 if stats.raises > 0 else 1.0
        return stats.raises / stats.calls

    def vpip(self):
        """Recent VPIP from rolling window."""
        if len(self._window) < 10:
            return 0.5  # prior
        count = sum(1 for r in self._window if r.vpip)
        return count / len(self._window)

    def pfr(self):
        """Recent pre-flop raise frequency from rolling window."""
        if len(self._window) < 10:
            return 0.2  # prior
        count = sum(1 for r in self._window if r.pfr)
        return count / len(self._window)

    def has_enough_data(self):
        """True if we have enough recent data to start exploiting."""
        return len(self._window) >= 30

    # ----------------------------------------------------------------
    #  STRATEGY SHIFT DETECTION
    # ----------------------------------------------------------------

    def detect_strategy_shift(self):
        """Compare recent behavior (window) vs all-time cumulative stats.

        Returns a dict with:
            shifted: bool — True if opponent appears to have changed strategy
            fold_delta: float — recent fold rate minus cumulative fold rate
            direction: str — 'tighter', 'looser', or 'stable'
        """
        if self.hands_played < 100 or len(self._window) < 50:
            return {"shifted": False, "fold_delta": 0.0, "direction": "stable"}

        # Compare post-flop fold rates (streets 1-3)
        recent_folds = 0
        recent_total = 0
        cum_folds = 0
        cum_total = 0

        for s in range(1, 4):
            stats = self._window_street_stats(s)
            recent_folds += stats.folds
            recent_total += stats.total
            cum_folds += self._cum_street_folds[s]
            cum_total += self._cum_street_actions[s]

        recent_fold_rate = recent_folds / recent_total if recent_total > 10 else 0.3
        cum_fold_rate = cum_folds / cum_total if cum_total > 20 else 0.3

        fold_delta = recent_fold_rate - cum_fold_rate

        # Significant shift: >15% change in fold rate
        if abs(fold_delta) > 0.15:
            direction = "tighter" if fold_delta > 0 else "looser"
            return {"shifted": True, "fold_delta": fold_delta, "direction": direction}

        return {"shifted": False, "fold_delta": fold_delta, "direction": "stable"}

    def is_adaptive(self):
        """True if opponent appears to be changing strategy over time.

        Checks if behavior in the recent window differs significantly from
        overall behavior. Useful for deciding whether to deploy metagame traps.
        """
        return self.detect_strategy_shift()["shifted"]

    # ----------------------------------------------------------------
    #  EXPLOITATION ADJUSTMENTS
    # ----------------------------------------------------------------

    def get_exploitation_adjustments(self):
        """Return strategy adjustments based on recent opponent tendencies.

        Uses rolling window stats (not cumulative) so adjustments adapt
        to opponents who change strategy mid-match.

        Returns dict with keys:
            raise_threshold_mod: float added to raise threshold (positive = tighter)
            bluff_frequency_mod: float multiplier for bluff frequency
            call_threshold_mod: float added to call threshold (positive = tighter)
        """
        if not self.has_enough_data():
            return {
                "raise_threshold_mod": 0.0,
                "bluff_frequency_mod": 1.0,
                "call_threshold_mod": 0.0,
            }

        mods = {
            "raise_threshold_mod": 0.0,
            "bluff_frequency_mod": 1.0,
            "call_threshold_mod": 0.0,
        }

        # Average fold rate across post-flop streets (from window)
        avg_fold = sum(self.fold_rate(s) for s in range(1, 4)) / 3

        if avg_fold > 0.5:
            # Folds too much -> bluff more, raise wider
            mods["bluff_frequency_mod"] = 1.8
            mods["raise_threshold_mod"] = -0.07
        elif avg_fold > 0.35:
            # Somewhat foldy -> bluff a bit more
            mods["bluff_frequency_mod"] = 1.3
            mods["raise_threshold_mod"] = -0.03
        elif avg_fold < 0.12:
            # Calling station -> never bluff
            mods["bluff_frequency_mod"] = 0.0
            mods["raise_threshold_mod"] = 0.07
        elif avg_fold < 0.20:
            # Somewhat sticky -> reduce bluffs
            mods["bluff_frequency_mod"] = 0.4
            mods["raise_threshold_mod"] = 0.03

        # Aggression adjustment
        avg_agg = sum(self.aggression_factor(s) for s in range(1, 4)) / 3
        if avg_agg > 2.5:
            # Very aggressive -> call wider vs their raises (they bluff a lot)
            mods["call_threshold_mod"] = -0.08
        elif avg_agg > 1.5:
            mods["call_threshold_mod"] = -0.03
        elif avg_agg < 0.4:
            # Very passive -> their raises are strong, fold more
            mods["call_threshold_mod"] = 0.08
        elif avg_agg < 0.8:
            mods["call_threshold_mod"] = 0.03

        return mods
