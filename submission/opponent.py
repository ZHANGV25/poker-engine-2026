"""
Within-match opponent modeling.

Tracks opponent behavior across all 1000 hands in a match. The bot instance
persists for the entire match, so self variables carry state between hands.

After ~50 hands, we have enough data to identify patterns and exploit them.
That leaves 950 hands of exploitative play.
"""


class OpponentModel:
    def __init__(self):
        self.hands_played = 0

        # Per-street action tracking
        self.street_actions = {s: [] for s in range(4)}  # street -> list of action strings
        self.street_folds = {s: 0 for s in range(4)}
        self.street_raises = {s: 0 for s in range(4)}
        self.street_calls = {s: 0 for s in range(4)}
        self.street_checks = {s: 0 for s in range(4)}
        self.street_action_counts = {s: 0 for s in range(4)}
        self.raise_amounts = {s: [] for s in range(4)}  # street -> list of raise sizes

        # Pre-flop specific
        self.preflop_hands = 0
        self.vpip_count = 0  # voluntarily put money in
        self.pfr_count = 0   # pre-flop raise

        # Showdown calibration data
        self.showdown_data = []  # for discard inference calibration

        # Per-hand tracking (reset each hand)
        self._current_hand = -1
        self._hand_actions = []  # actions seen this hand
        self._opp_preflop_raised = False
        self._opp_preflop_voluntarily_in = False

    def record_action(self, street, action_str, observation):
        """Record an opponent action observed during a hand.

        Called from both act() and observe() when we see opp_last_action.
        """
        hand_num = observation.get("hand_number", 0) if isinstance(observation, dict) else 0

        # Avoid double-counting: track per-hand
        action_key = (hand_num, street, action_str, observation.get("opp_bet", 0) if isinstance(observation, dict) else 0)
        if action_key in self._hand_actions:
            return
        self._hand_actions.append(action_key)

        # New hand reset
        if hand_num != self._current_hand:
            self._current_hand = hand_num
            self._hand_actions = [action_key]
            self._opp_preflop_raised = False
            self._opp_preflop_voluntarily_in = False

        if street > 3:
            return

        self.street_action_counts[street] += 1

        if action_str == "FOLD":
            self.street_folds[street] += 1
        elif action_str == "RAISE":
            self.street_raises[street] += 1
            if isinstance(observation, dict):
                self.raise_amounts[street].append(observation.get("opp_bet", 0))
            if street == 0:
                self._opp_preflop_raised = True
                self._opp_preflop_voluntarily_in = True
        elif action_str == "CALL":
            self.street_calls[street] += 1
            if street == 0:
                self._opp_preflop_voluntarily_in = True
        elif action_str == "CHECK":
            self.street_checks[street] += 1

    def record_hand_result(self, reward, info):
        """Record end-of-hand data. Called from observe() when terminated=True."""
        self.hands_played += 1

        # Track pre-flop stats
        self.preflop_hands += 1
        if self._opp_preflop_voluntarily_in:
            self.vpip_count += 1
        if self._opp_preflop_raised:
            self.pfr_count += 1

        # Record showdown data for discard inference calibration
        if "player_0_cards" in info and "player_1_cards" in info:
            self.showdown_data.append(info)

        # Reset per-hand tracking
        self._hand_actions = []
        self._opp_preflop_raised = False
        self._opp_preflop_voluntarily_in = False

    def fold_rate(self, street):
        """Opponent's fold frequency on a given street."""
        total = self.street_action_counts.get(street, 0)
        if total < 5:
            return 0.3  # prior: assume moderate fold rate
        return self.street_folds[street] / total

    def raise_rate(self, street):
        """Opponent's raise frequency on a given street."""
        total = self.street_action_counts.get(street, 0)
        if total < 5:
            return 0.2  # prior
        return self.street_raises[street] / total

    def avg_raise_size(self, street):
        """Average raise size on a given street."""
        amounts = self.raise_amounts.get(street, [])
        if not amounts:
            return 4  # prior: min raise
        return sum(amounts) / len(amounts)

    def vpip(self):
        """Voluntarily put money in pre-flop (how loose they are)."""
        if self.preflop_hands < 10:
            return 0.5  # prior
        return self.vpip_count / self.preflop_hands

    def pfr(self):
        """Pre-flop raise frequency."""
        if self.preflop_hands < 10:
            return 0.2  # prior
        return self.pfr_count / self.preflop_hands

    def aggression_factor(self, street):
        """(raises) / (calls). Higher = more aggressive."""
        calls = self.street_calls.get(street, 0)
        raises = self.street_raises.get(street, 0)
        if calls == 0:
            return 2.0 if raises > 0 else 1.0  # prior
        return raises / calls

    def has_enough_data(self):
        """True if we have enough hands to start exploiting."""
        return self.hands_played >= 50

    def get_exploitation_adjustments(self):
        """Return a dict of strategy adjustments based on opponent tendencies.

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

        # Average fold rate across post-flop streets
        avg_fold = sum(self.fold_rate(s) for s in range(1, 4)) / 3

        if avg_fold > 0.5:
            # Opponent folds too much -> bluff more, raise wider
            mods["bluff_frequency_mod"] = 1.5
            mods["raise_threshold_mod"] = -0.05
        elif avg_fold < 0.15:
            # Calling station -> never bluff, only value bet
            mods["bluff_frequency_mod"] = 0.0
            mods["raise_threshold_mod"] = 0.05

        # Aggression adjustment
        avg_agg = sum(self.aggression_factor(s) for s in range(1, 4)) / 3
        if avg_agg > 2.0:
            # Very aggressive opponent -> call wider vs their raises
            mods["call_threshold_mod"] = -0.05
        elif avg_agg < 0.5:
            # Very passive -> their raises mean strength, fold more
            mods["call_threshold_mod"] = 0.05

        return mods
