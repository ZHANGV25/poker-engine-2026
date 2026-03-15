"""
Competitive poker bot for CMU DSC Poker Bot Competition 2026.

Strategy: GTO via real-time CFR subgame solving. No exploitation.

Architecture:
  1. ExactEquityEngine: Exact win probability via full enumeration (~5ms)
  2. DiscardInference: Bayesian opponent range narrowing from revealed discards
  3. SubgameSolver: CFR+ solver computes Nash equilibrium for post-flop betting
     in real-time (~90-130ms per decision)

Key edges:
  - Exact equity (zero error) vs Monte Carlo (2.5% error)
  - Discard inference narrows opponent range from ~120 to ~20-40 hands
  - CFR solver computes provably optimal bet/check/fold/call frequencies
    instead of heuristic equity thresholds
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
        self.solver = SubgameSolver(self.engine)

        # Pre-flop hand potential lookup table
        self._preflop_table = self._load_preflop_table()
        # Sorted list of (potential, bitmask) for all hands — used for opponent range estimation
        self._all_hand_potentials = self._build_potential_index()
        # Precomputed preflop GTO strategy (mixed strategies from CFR)
        self._preflop_strategy = self._load_preflop_strategy()

        # Per-hand state
        self._current_hand = -1
        self._opp_weights = None
        self._last_seen_action = None

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
        bitmasks = data["bitmasks"]
        potentials = data["potentials"]
        table = {}
        for i in range(len(bitmasks)):
            table[int(bitmasks[i])] = float(potentials[i])
        return table

    def _preflop_potential(self, my_cards):
        if self._preflop_table is None:
            return None
        mask = 0
        for c in my_cards:
            mask |= 1 << c
        return self._preflop_table.get(mask)

    def _load_preflop_strategy(self):
        """Load precomputed GTO preflop strategy from CFR solve."""
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

    def _get_preflop_bucket(self, my_cards):
        """Map hand potential to a strategy bucket index."""
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
        """Find the current node in the preflop game tree based on bet state.

        Matches the current (my_bet, opp_bet, who_acts) to a node in the
        precomputed tree.
        """
        if self._preflop_strategy is None:
            return None

        ps = self._preflop_strategy
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        blind_pos = self._get_blind_position(observation)

        # Walk the tree to find a matching node
        # We are SB (blind_pos=0) or BB (blind_pos=1)
        # In the tree, player 0 = SB, player 1 = BB
        my_player = blind_pos  # 0=SB, 1=BB

        # Find node where player matches and bets match
        bet_sb = my_bet if my_player == 0 else opp_bet
        bet_bb = my_bet if my_player == 1 else opp_bet

        for nid in range(len(ps['node_players'])):
            if (ps['node_players'][nid] == my_player and
                    ps['node_bet_sb'][nid] == bet_sb and
                    ps['node_bet_bb'][nid] == bet_bb):
                return nid

        return None

    def _build_potential_index(self):
        """Build sorted array of (potential, card_list) for opponent range estimation."""
        if self._preflop_table is None:
            return None
        entries = []
        for mask, pot in self._preflop_table.items():
            cards = [i for i in range(27) if mask & (1 << i)]
            entries.append((pot, cards))
        entries.sort(key=lambda x: -x[0])  # highest potential first
        return entries

    def _estimate_preflop_equity_vs_range(self, my_cards, raise_size):
        """Compute our equity against the opponent's estimated raising range.

        Instead of using raw hand potential (which is equity vs random hands),
        this estimates what hands would raise this amount, samples matchups
        against those hands, and returns our actual equity.

        Args:
            my_cards: list of 5 card ints
            raise_size: how much opponent raised above the blind (opp_bet - 2)

        Returns:
            float equity (0-1) against opponent's estimated range
        """
        if self._all_hand_potentials is None:
            return self._preflop_potential(my_cards) or 0.5

        # Estimate what fraction of hands would raise this amount
        # Bigger raise → tighter range (fewer, stronger hands)
        # Map raise size to range percentile:
        #   raise 2-4:   top 60% of hands
        #   raise 10-20: top 35% of hands
        #   raise 30-50: top 20% of hands
        #   raise 60-80: top 10% of hands
        #   raise 80+:   top 5% of hands
        if raise_size <= 4:
            range_pct = 0.60
        elif raise_size <= 20:
            range_pct = 0.60 - (raise_size - 4) / 16 * 0.25  # 0.60 → 0.35
        elif raise_size <= 50:
            range_pct = 0.35 - (raise_size - 20) / 30 * 0.15  # 0.35 → 0.20
        else:
            range_pct = 0.20 - (raise_size - 50) / 50 * 0.15  # 0.20 → 0.05
        range_pct = max(range_pct, 0.05)

        # Get opponent hands in this range (top range_pct by potential)
        my_card_set = set(my_cards)
        total_hands = len(self._all_hand_potentials)
        cutoff = int(total_hands * range_pct)

        # Collect valid opponent hands (no card overlap with ours)
        opp_hands = []
        for pot, cards in self._all_hand_potentials:
            if not (set(cards) & my_card_set):
                opp_hands.append(cards)
            if len(opp_hands) >= cutoff:
                break

        if not opp_hands:
            return self._preflop_potential(my_cards) or 0.5

        # Sample matchups: pick N opponent hands, M random flops, simulate
        rng = random.Random(hash(tuple(my_cards)))  # deterministic per hand
        n_opp_sample = min(len(opp_hands), 150)
        n_flop_sample = 30

        opp_sample = rng.sample(opp_hands, n_opp_sample)

        # Available cards for flop (not in our hand)
        available_for_flop_base = [c for c in range(27) if c not in my_card_set]

        KEEP_PAIRS = [(i,j) for i in range(5) for j in range(i+1, 5)]
        five_lookup = self.engine.lookup_five
        wins = 0.0
        total = 0.0

        for opp_cards in opp_sample:
            opp_set = set(opp_cards)
            available = [c for c in available_for_flop_base if c not in opp_set]

            # Sample random flops
            for _ in range(n_flop_sample):
                flop = rng.sample(available, 3)

                # Find our best keep
                best_hero_rank = float('inf')
                for i, j in KEEP_PAIRS:
                    rank = five_lookup([my_cards[i], my_cards[j]] + flop)
                    if rank < best_hero_rank:
                        best_hero_rank = rank

                # Find opponent's best keep
                best_opp_rank = float('inf')
                for i, j in KEEP_PAIRS:
                    rank = five_lookup([opp_cards[i], opp_cards[j]] + flop)
                    if rank < best_opp_rank:
                        best_opp_rank = rank

                # Compare
                if best_hero_rank < best_opp_rank:
                    wins += 1.0
                elif best_hero_rank == best_opp_rank:
                    wins += 0.5
                total += 1.0

        return wins / total if total > 0 else 0.5

    def _reset_hand(self, hand_number):
        if hand_number != self._current_hand:
            self._current_hand = hand_number
            self._opp_weights = None
            self._last_seen_action = None
            self._streets_raised = 0
            self._current_street = -1
            self._raised_this_street = False

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
        amount = min(amount, max_raise)
        return amount

    # ----------------------------------------------------------------
    #  ACTION-BASED RANGE NARROWING
    # ----------------------------------------------------------------

    def _narrow_range_by_action(self, opp_action, my_cards, board, dead_cards):
        """Update opponent weights based on their betting action.

        When opponent takes an aggressive action, they likely have a stronger
        hand. Filter the bottom portion of their range and recompute weights.

        This is not exploitation — it's Bayesian inference. Their action IS
        evidence about their hand strength.
        """
        if self._opp_weights is None or len(board) < 3:
            return

        hand_equities = {}
        for opp_pair in self._opp_weights:
            if self._opp_weights[opp_pair] <= 0:
                continue
            if len(board) >= 3:
                five = list(opp_pair) + list(board[:3])
                rank = self.engine.lookup_five(five)
                hand_equities[opp_pair] = rank

        if not hand_equities:
            return

        sorted_hands = sorted(hand_equities.items(), key=lambda x: x[1])
        n = len(sorted_hands)

        if opp_action == "RAISE":
            cutoff_idx = int(n * 0.4)
        elif opp_action == "CALL":
            cutoff_idx = int(n * 0.7)
        else:
            return

        if cutoff_idx < n:
            weak_hands = {h for h, _ in sorted_hands[cutoff_idx:]}
            for hand in weak_hands:
                self._opp_weights[hand] = 0.0

        total = sum(self._opp_weights.values())
        if total > 0:
            for k in self._opp_weights:
                self._opp_weights[k] /= total

    # ----------------------------------------------------------------
    #  DISCARD DECISION
    # ----------------------------------------------------------------

    def _handle_discard(self, observation, my_cards, board, opp_discards):
        """Choose which 2 of 5 cards to keep on the flop."""
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
    #  PRE-FLOP STRATEGY
    # ----------------------------------------------------------------

    def _handle_preflop(self, observation, my_cards):
        """Pre-flop strategy using precomputed GTO mixed strategies from CFR.

        Looks up our hand strength bucket and current game tree node,
        then samples an action from the Nash equilibrium distribution.
        Falls back to pot-odds-based logic if precomputed strategy unavailable.
        """
        valid_actions = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        continue_cost = opp_bet - my_bet
        pot_size = self._get_pot_size(observation)

        # Try precomputed GTO strategy first
        bucket = self._get_preflop_bucket(my_cards)
        node_id = self._find_preflop_node(observation)

        if bucket is not None and node_id is not None and self._preflop_strategy is not None:
            ps = self._preflop_strategy
            strategy = ps['strategies'][node_id, bucket]  # action probabilities
            children = ps['children_map'].get(node_id, {})

            if len(children) > 0:
                # Filter to valid actions and renormalize
                valid_mask = np.zeros(len(strategy))
                for act_id in children:
                    if act_id == 0 and valid_actions[FOLD]:
                        valid_mask[act_id] = strategy[act_id]
                    elif act_id == 1:
                        # "call" in tree = CALL or CHECK depending on context
                        if valid_actions[CALL] or valid_actions[CHECK]:
                            valid_mask[act_id] = strategy[act_id]
                    elif act_id >= 2 and valid_actions[RAISE]:
                        valid_mask[act_id] = strategy[act_id]

                total = valid_mask.sum()
                if total > 0:
                    valid_mask /= total
                    # Sample from distribution
                    chosen = np.random.choice(len(valid_mask), p=valid_mask)

                    if chosen == 0:
                        return (FOLD, 0, 0, 0)
                    elif chosen == 1:
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)
                    elif chosen >= 2:
                        raise_to = ps['raise_levels'][chosen - 2]
                        # Compute raise amount (increment above matching)
                        raise_amount = raise_to - opp_bet
                        raise_amount = max(raise_amount, observation["min_raise"])
                        raise_amount = min(raise_amount, observation["max_raise"])
                        if raise_amount > 0 and valid_actions[RAISE]:
                            return (RAISE, raise_amount, 0, 0)
                        # Raise not possible, call instead
                        if valid_actions[CALL]:
                            return (CALL, 0, 0, 0)
                        return (CHECK, 0, 0, 0)

        # Fallback: pot-odds based with opponent range estimation
        potential = self._preflop_potential(my_cards)
        if potential is not None:
            strength = max(0.0, min(10.0, (potential - 0.37) / 0.028))
        else:
            strength = self._preflop_heuristic(my_cards)

        if valid_actions[CALL]:
            if continue_cost <= 1:
                if strength >= 7.0 and valid_actions[RAISE]:
                    raise_amt = self._compute_raise_amount(observation, 0.65)
                    return (RAISE, raise_amt, 0, 0)
                return (CALL, 0, 0, 0)

            # Facing a raise: use pot odds
            required_equity = continue_cost / (continue_cost + pot_size)
            raise_size = opp_bet - 2
            if raise_size >= 6:
                equity_vs_range = self._estimate_preflop_equity_vs_range(my_cards, raise_size)
                if equity_vs_range < required_equity:
                    return (FOLD, 0, 0, 0)
                if equity_vs_range > required_equity + 0.15 and valid_actions[RAISE]:
                    raise_amt = self._compute_raise_amount(observation, 0.65)
                    return (RAISE, raise_amt, 0, 0)
                return (CALL, 0, 0, 0)
            else:
                if strength >= 7.0 and valid_actions[RAISE]:
                    raise_amt = self._compute_raise_amount(observation, 0.65)
                    return (RAISE, raise_amt, 0, 0)
                if strength >= 2.0:
                    return (CALL, 0, 0, 0)
                return (FOLD, 0, 0, 0)

        if valid_actions[CHECK]:
            if strength >= 8.0 and valid_actions[RAISE]:
                raise_amt = self._compute_raise_amount(observation, 0.65)
                return (RAISE, raise_amt, 0, 0)
            return (CHECK, 0, 0, 0)

        return (FOLD, 0, 0, 0)

    def _preflop_heuristic(self, my_cards):
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
        run = 1
        max_run = 1
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
    #  POST-FLOP BETTING (CFR Solver)
    # ----------------------------------------------------------------

    def _handle_postflop(self, observation, my_cards, board, opp_discards, my_discards, info):
        """Post-flop decisions via real-time CFR subgame solving.

        Instead of equity thresholds, we solve a small game tree (~100-200 nodes)
        using CFR+ for 75-150 iterations. This computes the Nash equilibrium
        action frequencies for our specific hand against the opponent's range.

        The solver handles bluffing, pot control, check-raising, and bet sizing
        implicitly — these all emerge from the equilibrium computation.
        """
        dead_cards = my_discards + opp_discards
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]

        # Determine if we're initiating action or responding to a bet.
        # This determines the tree shape: initiating = CHECK/BET options,
        # responding = FOLD/CALL/RAISE options.
        # We're "first" if bets are equal (no outstanding bet to respond to).
        hero_is_first = (my_bet == opp_bet)

        return self.solver.solve_and_act(
            hero_cards=my_cards,
            board=board,
            opp_range=self._opp_weights,
            dead_cards=dead_cards,
            my_bet=my_bet,
            opp_bet=opp_bet,
            street=observation["street"],
            min_raise=observation["min_raise"],
            max_raise=observation["max_raise"],
            valid_actions=observation["valid_actions"],
            hero_is_first=hero_is_first,
            time_remaining=observation.get("time_left", 400),
        )

    # ----------------------------------------------------------------
    #  MAIN ACT / OBSERVE
    # ----------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        my_cards, board, opp_discards, my_discards = self._parse_cards(observation)
        valid_actions = observation["valid_actions"]

        # Narrow opponent range based on their action (if we have weights)
        opp_action = observation.get("opp_last_action", "None")
        if opp_action in ("RAISE", "CALL") and self._opp_weights is not None:
            self._narrow_range_by_action(opp_action, my_cards, board, my_discards + opp_discards)

        # Discard phase
        if valid_actions[DISCARD]:
            return self._handle_discard(observation, my_cards, board, opp_discards)

        # Pre-flop
        if observation["street"] == 0:
            return self._handle_preflop(observation, my_cards)

        # Post-flop
        return self._handle_postflop(observation, my_cards, board, opp_discards, my_discards, info)

    def observe(self, observation, reward, terminated, truncated, info):
        hand_number = info.get("hand_number", 0)
        self._reset_hand(hand_number)

        # Infer opponent range when we first see their discards
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        if len(opp_discards) == 3 and self._opp_weights is None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            self._opp_weights = self.inference.infer_opponent_weights(
                opp_discards, board, my_cards
            )

        # Narrow range based on opponent's action
        opp_action = observation.get("opp_last_action", "None")
        if opp_action in ("RAISE", "CALL") and self._opp_weights is not None:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            board = [c for c in observation["community_cards"] if c != -1]
            my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
            opp_disc = [c for c in observation["opp_discarded_cards"] if c != -1]
            self._narrow_range_by_action(opp_action, my_cards, board, my_discards + opp_disc)


if __name__ == "__main__":
    PlayerAgent.run(stream=True, port=8000, player_id="poker_bot")
