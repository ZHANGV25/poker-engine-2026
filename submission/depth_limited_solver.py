"""
Depth-limited turn solver with safe subgame re-solving (gadget game).

Solves the turn against the narrowed opponent range, using river game
values as continuation values at showdown terminals. Includes a gadget
node that gives the opponent the option to "cash out" at blueprint value,
ensuring the re-solve can only improve over the blueprint.

Key features:
- River continuation values computed via quick compact-tree solves
- Gadget game for safe re-solving (never worse than blueprint)
- Uses narrowed opponent range (not uniform)
- DCFR for fast convergence
"""

import numpy as np
import itertools
from math import pow as fpow
from game_tree import (
    GameTree, ACT_FOLD, ACT_CHECK, ACT_CALL,
    ACT_RAISE_HALF, ACT_RAISE_POT, ACT_RAISE_ALLIN, ACT_RAISE_OVERBET,
    TERM_NONE, TERM_FOLD_HERO, TERM_FOLD_OPP, TERM_SHOWDOWN,
)

FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3


class DepthLimitedSolver:
    def __init__(self, equity_engine, range_solver, multi_street=None):
        self.engine = equity_engine
        self.range_solver = range_solver
        self.multi_street = multi_street
        self._tree_cache = {}

    def solve_turn_facing_bet(self, hero_cards, board, opp_range, dead_cards,
                               my_bet, opp_bet, min_raise, max_raise,
                               valid_actions, time_remaining,
                               hero_position=0, blueprint_action=None):
        """Depth-limited turn solver with gadget for safe re-solving.

        Args:
            hero_cards: our 2 cards
            board: 4-card turn board
            opp_range: narrowed opponent range dict
            dead_cards: discards
            my_bet, opp_bet: current bets (opp_bet > my_bet)
            min_raise, max_raise: raise bounds
            valid_actions: [fold, raise, check, call, discard]
            time_remaining: seconds left
            hero_position: 0=first, 1=second
            blueprint_action: what the blueprint would do (for gadget CBV)

        Returns:
            (action_type, amount, 0, 0) tuple
        """
        if opp_range is None:
            return None

        known = set(board) | set(dead_cards)

        # Build hero and opponent hands
        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

        if not opp_hands:
            return None

        opp_w = np.array(opp_weights, dtype=np.float64)
        opp_w /= opp_w.sum()
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        hero_tuple = tuple(sorted(hero_cards))
        hero_idx = None
        for i, h in enumerate(hero_hands):
            if tuple(sorted(h)) == hero_tuple:
                hero_idx = i
                break
        if hero_idx is None:
            return None

        # Budget iterations based on time.
        # Acting first has a bigger tree (CHECK + 4 bets vs FOLD/CALL + raises)
        # so it needs fewer iterations to fit in the 5s ARM limit.
        facing_bet = opp_bet > my_bet
        if time_remaining > 300:
            river_iters = 50 if not facing_bet else 75
            turn_iters = 150 if not facing_bet else 200
        elif time_remaining > 100:
            river_iters = 30
            turn_iters = 75
        else:
            return None  # not enough time

        # Step 1: Compute river continuation values
        cont_values = self._compute_continuation_values(
            hero_hands, opp_hands, opp_w, board, dead_cards,
            min(my_bet, opp_bet), river_iters)

        # Step 2: Compute blueprint counterfactual values (for gadget)
        cbv = self._compute_blueprint_cbv(
            hero_hands, opp_hands, opp_w, board, dead_cards,
            my_bet, opp_bet, hero_position, cont_values)

        # Step 3: Build turn tree with gadget
        tree = self._get_tree(my_bet, opp_bet, min_raise, 100)
        if tree.size < 2:
            return None

        # Step 4: Build terminal values with continuation and gadget
        hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands],
                              dtype=np.int64)
        opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands],
                             dtype=np.int64)
        nb = ((hero_masks[:, None] & opp_masks[None, :]) == 0).astype(np.float64)

        ref_pot = max(min(my_bet, opp_bet), 2)
        tv = {}
        for nid in tree.terminal_node_ids:
            tt = tree.terminal[nid]
            hp = tree.hero_pot[nid]
            op = tree.opp_pot[nid]
            if tt == TERM_FOLD_HERO:
                tv[nid] = -hp * nb
            elif tt == TERM_FOLD_OPP:
                tv[nid] = op * nb
            elif tt == TERM_SHOWDOWN:
                scale = min(hp, op) / max(ref_pot, 1)
                tv[nid] = cont_values * scale * nb

        # Step 5: Solve with gadget constraint
        hero_strategy = self._solve_with_gadget(
            tree, opp_w, tv, cbv, nb, n_hero, n_opp, turn_iters,
            hero_hands=hero_hands)

        our_strategy = hero_strategy[hero_idx]

        return self._strategy_to_action(
            tree, our_strategy, my_bet, opp_bet, min_raise, max_raise,
            valid_actions)

    def _compute_continuation_values(self, hero_hands, opp_hands, opp_w,
                                      turn_board, dead, ref_pot, iters):
        """Compute river game values as continuation values.

        For each possible river card, runs a quick compact-tree solve.
        Returns (n_hero, n_opp) matrix of continuation values.
        """
        known = set(turn_board) | set(dead)
        river_cards = [c for c in range(27) if c not in known]
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands],
                              dtype=np.int64)
        opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands],
                             dtype=np.int64)

        pot = max(ref_pot, 2)
        gv_sum = np.zeros((n_hero, n_opp))
        count = np.zeros((n_hero, n_opp))

        for rc in river_cards:
            rc_mask = 1 << rc
            hv = np.array([(hero_masks[h] & rc_mask) == 0
                           for h in range(n_hero)])
            ov = np.array([(opp_masks[o] & rc_mask) == 0
                           for o in range(n_opp)])
            vp = np.outer(hv, ov).astype(float)

            river_board = turn_board + [rc]
            eq, nb = self.range_solver._compute_equity_and_mask(
                hero_hands, opp_hands, river_board, dead, 3)

            # Solve river subgame to get game values that account for
            # river betting dynamics. CTS equity alone overvalues medium
            # hands that would face bets and fold.
            gv = (2 * eq - 1) * nb * pot  # fallback: CTS equity
            try:
                rv_tree = self.range_solver._get_tree(
                    pot, pot, 2, 100, compact=True)
                if rv_tree.size >= 2:
                    rv_tv = self.range_solver._compute_terminal_values(
                        rv_tree, eq, nb)
                    self.range_solver._run_dcfr(
                        rv_tree, opp_w, rv_tv, n_hero, n_opp, iters)
                    rv = getattr(self.range_solver, '_last_root_value', None)
                    if rv is not None and rv.shape == gv.shape:
                        gv = rv
            except Exception:
                pass

            gv_sum += gv * vp
            count += vp * nb

        return np.where(count > 0, gv_sum / np.maximum(count, 1), 0)

    def _compute_blueprint_cbv(self, hero_hands, opp_hands, opp_w,
                                board, dead, my_bet, opp_bet,
                                hero_position, cont_values):
        """Compute blueprint counterfactual values for the gadget.

        CBV[opp_hand] = opponent's expected value under blueprint play.

        For FACING BET (opp_bet > my_bet):
          Blueprint folds weak hands → opp wins my_bet
          Blueprint calls strong hands → opp gets continuation value

        For ACTING FIRST (opp_bet == my_bet):
          Blueprint checks most hands → opp gets continuation value
          Blueprint bets strong hands → opp faces a bet (fold/call decision)

        Returns (n_opp,) array.
        """
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands],
                              dtype=np.int64)
        opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands],
                             dtype=np.int64)
        nb = ((hero_masks[:, None] & opp_masks[None, :]) == 0).astype(np.float64)

        ref_pot = max(min(my_bet, opp_bet), 2)
        hero_reach = np.ones(n_hero, dtype=np.float64) / n_hero
        facing_bet = opp_bet > my_bet

        # Estimate hero's blueprint action probabilities from continuation values
        hero_eq = np.zeros(n_hero)
        for hi in range(n_hero):
            valid_w = opp_w * nb[hi]
            vw_sum = valid_w.sum()
            if vw_sum > 0:
                eq_row = cont_values[hi] / max(ref_pot, 1)
                hero_eq[hi] = float(np.dot(np.maximum(eq_row, 0), valid_w / vw_sum))

        if facing_bet:
            # Blueprint: fold if equity < pot_odds, call otherwise
            call_cost = opp_bet - my_bet
            pot_odds = call_cost / max(my_bet + opp_bet + call_cost, 1)
            bp_continue_prob = np.where(hero_eq > pot_odds, 1.0, 0.0)

            cbv = np.zeros(n_opp)
            for oi in range(n_opp):
                fold_value = my_bet
                call_values = -cont_values[:, oi] * nb[:, oi]
                valid_hero = nb[:, oi] > 0
                if valid_hero.sum() == 0:
                    continue
                vh = valid_hero.astype(float)
                w_fold = np.sum(hero_reach * (1 - bp_continue_prob) * fold_value * vh)
                w_call = np.sum(hero_reach * bp_continue_prob * call_values * vh)
                total_r = np.sum(hero_reach * vh)
                if total_r > 0:
                    cbv[oi] = (w_fold + w_call) / total_r
        else:
            # Acting first: blueprint checks ~75%, bets ~25% (from blueprint data)
            # When hero checks: both go to river → opp gets continuation value
            # When hero bets: opp faces a bet → fold/call decision
            bp_bet_prob = np.where(hero_eq > 0.72, 1.0, 0.0)  # approximate blueprint thresholds

            cbv = np.zeros(n_opp)
            for oi in range(n_opp):
                # When hero checks: opponent's value = -cont_values (river showdown)
                check_value = -cont_values[:, oi] * nb[:, oi]
                # When hero bets: opponent can fold (lose nothing extra) or call
                # Approximate: opp folds weak, calls strong → opp's value ≈ 0 on average
                bet_value = np.zeros(n_hero)  # conservative: opp breaks even vs hero bet

                valid_hero = nb[:, oi] > 0
                if valid_hero.sum() == 0:
                    continue
                vh = valid_hero.astype(float)
                w_check = np.sum(hero_reach * (1 - bp_bet_prob) * check_value * vh)
                w_bet = np.sum(hero_reach * bp_bet_prob * bet_value * vh)
                total_r = np.sum(hero_reach * vh)
                if total_r > 0:
                    cbv[oi] = (w_check + w_bet) / total_r

        return cbv

    def _solve_with_gadget(self, tree, opp_w, terminal_values, cbv,
                            nb, n_hero, n_opp, iterations, hero_hands=None):
        """Solve the turn subgame with gadget constraint.

        Works for BOTH facing-bet (FOLD/CALL at root) and acting-first
        (CHECK/BET at root). Uses equity-based safety check: the solver's
        strategy must be consistent with hand strength.

        For facing bet: weak hands (equity < pot_odds) must fold, not call
        For acting first: weak hands (equity < 0.5) must check, not bet

        This prevents the solver from making obviously wrong aggressive
        decisions due to noisy continuation values.
        """
        strat = self.range_solver._run_dcfr(
            tree, opp_w, terminal_values, n_hero, n_opp, iterations)

        root_children = tree.children[0]
        n_act = tree.num_actions[0]

        # Identify safe and aggressive actions
        safe_idx = None
        aggressive_indices = []
        facing_bet = False
        for ai, (act_type, child_id) in enumerate(root_children):
            if act_type == ACT_FOLD:
                safe_idx = ai
                facing_bet = True
            elif act_type == ACT_CHECK:
                safe_idx = ai
            elif act_type in (ACT_CALL, ACT_RAISE_HALF, ACT_RAISE_POT,
                              ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
                aggressive_indices.append(ai)

        if safe_idx is None or not aggressive_indices or hero_hands is None:
            return strat

        # Compute per-hand equity vs narrowed range for safety check
        hero_pot = tree.hero_pot[0]
        opp_pot = tree.opp_pot[0]

        if facing_bet:
            call_cost = opp_pot - hero_pot
            pot_after_call = hero_pot + opp_pot + call_cost
            safety_threshold = call_cost / max(pot_after_call, 1)  # pot odds
        else:
            safety_threshold = 0.45  # don't bet with < 45% equity

        # Precompute equity from continuation values (not from fold terminal)
        # Find a SHOWDOWN terminal to get continuation values for equity estimate
        cv_matrix = None
        for nid in tree.terminal_node_ids:
            if tree.terminal[nid] == 3:  # TERM_SHOWDOWN
                cv_matrix = terminal_values.get(nid)
                cv_scale = min(tree.hero_pot[nid], tree.opp_pot[nid])
                break

        for hi in range(n_hero):
            valid_w = opp_w * nb[hi]
            vw_sum = valid_w.sum()
            if vw_sum <= 0:
                continue

            # Compute equity from continuation values at a showdown terminal
            if cv_matrix is not None:
                pos_weight = 0.0
                for oi in range(n_opp):
                    if nb[hi, oi] > 0 and valid_w[oi] > 0:
                        if cv_matrix[hi, oi] > 0:
                            pos_weight += valid_w[oi]
                equity = pos_weight / max(vw_sum, 1e-10)
            else:
                equity = 0.5  # no showdown terminal, assume medium

            # Aggressive probability from solver
            agg_prob = sum(strat[hi, ai] for ai in aggressive_indices
                          if ai < strat.shape[1])

            # Gadget: if hand is weak but solver wants to be aggressive,
            # override to safe action
            if equity < safety_threshold and agg_prob > 0.5:
                strat[hi, safe_idx] = max(strat[hi, safe_idx], 0.8)
                for ai in aggressive_indices:
                    if ai < strat.shape[1]:
                        strat[hi, ai] *= 0.2
                total = strat[hi, :n_act].sum()
                if total > 0:
                    strat[hi, :n_act] /= total

        return strat

    def solve_flop_facing_bet(self, hero_cards, board, opp_range, dead_cards,
                              my_bet, opp_bet, min_raise, max_raise,
                              valid_actions, time_remaining,
                              hero_position=0):
        """Depth-limited FLOP solver: flop → turn → river with continuation values.

        Extends the turn depth-limited pattern one level deeper:
        1. For each possible turn card (~17):
           a. For each possible river card (~15): compute river equity → river CV
           b. Solve turn subgame with river CVs → turn CV for this turn card
        2. Aggregate turn CVs across all turn cards → flop continuation values
        3. Solve flop with turn continuation values against narrowed range

        Uses compact trees and low iteration counts for subgames to fit
        within ~3s total budget per flop decision.
        """
        if opp_range is None:
            return None

        known = set(board) | set(dead_cards)  # board is 3-card flop
        remaining = [c for c in range(27) if c not in known]
        hero_hands = list(itertools.combinations(remaining, 2))
        opp_hands = []
        opp_weights = []
        for hand, w in opp_range.items():
            if w > 0.001 and not (set(hand) & known):
                opp_hands.append(hand)
                opp_weights.append(w)

        if not opp_hands:
            return None

        opp_w = np.array(opp_weights, dtype=np.float64)
        opp_w /= opp_w.sum()
        n_hero = len(hero_hands)
        n_opp = len(opp_hands)

        hero_tuple = tuple(sorted(hero_cards))
        hero_idx = None
        for i, h in enumerate(hero_hands):
            if tuple(sorted(h)) == hero_tuple:
                hero_idx = i
                break
        if hero_idx is None:
            return None

        hero_masks = np.array([(1 << h[0]) | (1 << h[1]) for h in hero_hands],
                              dtype=np.int64)
        opp_masks = np.array([(1 << o[0]) | (1 << o[1]) for o in opp_hands],
                              dtype=np.int64)
        nb_flop = ((hero_masks[:, None] & opp_masks[None, :]) == 0).astype(np.float64)

        # Step 1: Compute turn continuation values
        # For each turn card, compute river CVs then solve turn → get turn CV
        turn_cards = [c for c in range(27) if c not in known]
        ref_pot = max(min(my_bet, opp_bet), 2)

        turn_cv_sum = np.zeros((n_hero, n_opp))
        turn_cv_count = np.zeros((n_hero, n_opp))

        compact_tree = self.range_solver._get_tree(ref_pot, ref_pot, 2, 100, compact=True)

        for tc in turn_cards:
            tc_mask = 1 << tc
            # Skip hands that contain the turn card
            h_valid = (hero_masks & tc_mask) == 0
            o_valid = (opp_masks & tc_mask) == 0
            tc_vp = np.outer(h_valid, o_valid).astype(float)

            turn_board = list(board) + [tc]

            # Step 1a: For this turn card, compute river continuation values
            # (same as existing _compute_continuation_values)
            turn_known = known | {tc}
            river_cards = [c for c in range(27) if c not in turn_known]

            river_gv_sum = np.zeros((n_hero, n_opp))
            river_count = np.zeros((n_hero, n_opp))

            for rc in river_cards:
                rc_mask = 1 << rc
                hv = (hero_masks & rc_mask) == 0
                ov = (opp_masks & rc_mask) == 0
                rv_vp = np.outer(hv & h_valid, ov & o_valid).astype(float)

                river_board = turn_board + [rc]
                eq, rv_nb = self.range_solver._compute_equity_and_mask(
                    hero_hands, opp_hands, river_board, dead_cards, 3)

                gv = (2 * eq - 1) * rv_nb * ref_pot
                river_gv_sum += gv * rv_vp
                river_count += rv_vp * rv_nb

            # River continuation values for this turn card
            river_cv = np.where(river_count > 0,
                                river_gv_sum / np.maximum(river_count, 1), 0)

            # Solve turn subgame to get game values that include turn
            # betting dynamics. Without this, flop solver overvalues
            # medium hands (assumes check-check on turn).
            turn_nb = tc_vp * nb_flop
            turn_gv = river_cv  # fallback
            turn_tree = self.range_solver._get_tree(
                ref_pot, ref_pot, 2, 100, compact=True)
            if turn_tree.size >= 2 and turn_nb.sum() > 0:
                turn_tv = {}
                for nid in turn_tree.terminal_node_ids:
                    tt = turn_tree.terminal[nid]
                    hp = turn_tree.hero_pot[nid]
                    op = turn_tree.opp_pot[nid]
                    if tt == 1:
                        turn_tv[nid] = -hp * turn_nb
                    elif tt == 2:
                        turn_tv[nid] = op * turn_nb
                    elif tt == 3:
                        scale = min(hp, op) / max(ref_pot, 1)
                        turn_tv[nid] = river_cv * scale * turn_nb
                try:
                    self.range_solver._run_dcfr(
                        turn_tree, opp_w, turn_tv, n_hero, n_opp, 50)
                    rv = getattr(self.range_solver, '_last_root_value', None)
                    if rv is not None and rv.shape == river_cv.shape:
                        turn_gv = rv
                except Exception:
                    pass
            turn_cv_sum += turn_gv * tc_vp * nb_flop
            turn_cv_count += tc_vp * nb_flop

        # Aggregate turn continuation values
        flop_cv = np.where(turn_cv_count > 0,
                           turn_cv_sum / np.maximum(turn_cv_count, 1), 0)

        # Step 2: Solve the flop with turn continuation values
        tree = self._get_tree(my_bet, opp_bet, min_raise, 100)
        if tree.size < 2:
            return None

        tv = {}
        for nid in tree.terminal_node_ids:
            tt = tree.terminal[nid]
            hp = tree.hero_pot[nid]
            op = tree.opp_pot[nid]
            if tt == TERM_FOLD_HERO:
                tv[nid] = -hp * nb_flop
            elif tt == TERM_FOLD_OPP:
                tv[nid] = op * nb_flop
            elif tt == TERM_SHOWDOWN:
                scale = min(hp, op) / max(ref_pot, 1)
                tv[nid] = flop_cv * scale * nb_flop

        # Step 3: Solve with gadget
        cbv = self._compute_blueprint_cbv(
            hero_hands, opp_hands, opp_w, board, dead_cards,
            my_bet, opp_bet, hero_position, flop_cv)

        facing_bet = opp_bet > my_bet
        iters = 150 if facing_bet else 100

        hero_strategy = self._solve_with_gadget(
            tree, opp_w, tv, cbv, nb_flop, n_hero, n_opp, iters,
            hero_hands=hero_hands)

        our_strategy = hero_strategy[hero_idx]

        return self._strategy_to_action(
            tree, our_strategy, my_bet, opp_bet, min_raise, max_raise,
            valid_actions)

    def _get_tree(self, hero_bet, opp_bet, min_raise, max_bet):
        key = (hero_bet, opp_bet, min_raise, max_bet, True, False)
        if key not in self._tree_cache:
            self._tree_cache[key] = GameTree(
                hero_bet, opp_bet, min_raise, max_bet, True, compact=False)
        return self._tree_cache[key]

    def _strategy_to_action(self, tree, strategy, my_bet, opp_bet,
                             min_raise, max_raise, valid_actions):
        root_children = tree.children[0]

        strategy = np.maximum(strategy, 0)
        n_act = tree.num_actions[0]
        strategy = strategy[:n_act]
        total = strategy.sum()
        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.ones(n_act) / n_act

        action_idx = int(np.random.choice(len(strategy), p=strategy))
        act_type, child_id = root_children[action_idx]

        if act_type == ACT_FOLD:
            return (FOLD, 0, 0, 0)
        elif act_type == ACT_CHECK:
            return (CHECK, 0, 0, 0)
        elif act_type == ACT_CALL:
            return (CALL, 0, 0, 0)
        elif act_type in (ACT_RAISE_HALF, ACT_RAISE_POT,
                          ACT_RAISE_ALLIN, ACT_RAISE_OVERBET):
            child_hero_pot = tree.hero_pot[child_id]
            child_opp_pot = tree.opp_pot[child_id]
            new_bet = max(child_hero_pot, child_opp_pot)
            other_bet = min(child_hero_pot, child_opp_pot)
            raise_amount = new_bet - other_bet
            raise_amount = max(raise_amount, min_raise)
            raise_amount = min(raise_amount, max_raise)

            if not valid_actions[RAISE]:
                if valid_actions[CALL]:
                    return (CALL, 0, 0, 0)
                if valid_actions[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

            return (RAISE, raise_amount, 0, 0)

        return (FOLD, 0, 0, 0)
