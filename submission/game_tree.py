"""
Game tree representation for the CFR subgame solver.

Builds a betting tree for a single street of poker. The tree represents
all possible action sequences from a given starting state (current bets,
who acts first, available bet sizes).

The tree is stored as flat arrays for cache-friendly CFR traversal.
"""

# Action types in the tree
ACT_FOLD = 0
ACT_CHECK = 1
ACT_CALL = 2
ACT_RAISE_HALF = 3
ACT_RAISE_POT = 4
ACT_RAISE_ALLIN = 5
ACT_RAISE_OVERBET = 6
NUM_ACTIONS = 7

# Terminal types
TERM_NONE = 0
TERM_FOLD_HERO = 1   # hero folded -> hero loses
TERM_FOLD_OPP = 2    # opponent folded -> hero wins
TERM_SHOWDOWN = 3    # street ends, compare hands (or use continuation value)

MAX_RAISES_PER_STREET = 2
MAX_NODES = 500  # upper bound on tree size


class GameTree:
    """Betting tree for a single street.

    Built once per unique game state configuration, then reused across
    CFR iterations. Stores tree structure as parallel lists for fast access.
    """

    def __init__(self, hero_bet, opp_bet, min_raise, max_bet, hero_first,
                 compact=False, lean=False):
        """Build the game tree.

        Args:
            hero_bet: hero's current cumulative bet
            opp_bet: opponent's current cumulative bet
            min_raise: minimum raise increment
            max_bet: maximum total bet per player (100)
            hero_first: True if hero acts first on this street
            compact: if True, use 2 bet sizes + 1 raise max (for range solver)
            lean: if True, use blocking bet (25% pot) + pot bet (100%) only
        """
        self._compact = compact
        self._lean = lean
        # Node data (parallel lists, indexed by node_id)
        self.player = []        # 0=hero, 1=opp
        self.terminal = []      # TERM_* constant
        self.hero_pot = []      # hero's total bet at this node
        self.opp_pot = []       # opp's total bet at this node
        self.children = []      # list of (action_id, child_node_id) pairs
        self.parent = []        # parent node_id (-1 for root)
        self.num_actions = []   # number of valid actions at this node

        # Separate indices for hero and opp decision nodes
        self.hero_node_ids = []  # indices into the main arrays where hero decides
        self.opp_node_ids = []   # indices into the main arrays where opp decides
        self.terminal_node_ids = []

        # Build the tree
        self._min_raise = min_raise
        self._max_bet = max_bet
        self._build(hero_bet, opp_bet, hero_first)

    def _add_node(self, player, terminal, hero_pot, opp_pot, parent):
        node_id = len(self.player)
        self.player.append(player)
        self.terminal.append(terminal)
        self.hero_pot.append(hero_pot)
        self.opp_pot.append(opp_pot)
        self.children.append([])
        self.parent.append(parent)
        self.num_actions.append(0)

        if terminal != TERM_NONE:
            self.terminal_node_ids.append(node_id)
        elif player == 0:
            self.hero_node_ids.append(node_id)
        else:
            self.opp_node_ids.append(node_id)

        return node_id

    def _compute_raise_sizes(self, current_pot, acting_bet, other_bet, raises_so_far):
        """Compute discrete raise sizes for bet abstraction.

        Returns list of (action_id, new_acting_bet) pairs.
        """
        max_raises = 1 if self._compact else MAX_RAISES_PER_STREET
        if raises_so_far >= max_raises:
            return []

        remaining = self._max_bet - max(acting_bet, other_bet)
        if remaining <= 0 or self._min_raise > remaining:
            return []

        pot = acting_bet + other_bet
        call_amount = other_bet - acting_bet  # amount to match first
        sizes = []

        if self._compact:
            # Compact mode: 2 bet sizes only (60% pot + all-in)
            # Smaller tree for range solver — ~6x fewer nodes.
            med = max(self._min_raise, int(0.6 * pot))
            med = min(med, remaining)
            new_bet_med = other_bet + med
            if new_bet_med <= self._max_bet:
                sizes.append((ACT_RAISE_POT, new_bet_med))

            allin = self._max_bet
            if allin > other_bet and allin != new_bet_med:
                sizes.append((ACT_RAISE_ALLIN, allin))
        elif self._lean:
            # Lean mode: blocking bet (25% pot) + pot bet (100%).
            # 39 nodes vs 123 (3x faster). Solver uses blocking bet
            # for medium hands that would otherwise check.
            block = max(self._min_raise, int(0.25 * pot))
            block = min(block, remaining)
            new_bet_block = other_bet + block
            if new_bet_block <= self._max_bet:
                sizes.append((ACT_RAISE_HALF, new_bet_block))

            large = max(self._min_raise, pot)
            large = min(large, remaining)
            new_bet_large = other_bet + large
            new_bet_large = min(new_bet_large, self._max_bet)
            if new_bet_large > other_bet and new_bet_large != new_bet_block:
                sizes.append((ACT_RAISE_ALLIN, new_bet_large))
        else:
            # Full mode: 4 bet sizes for maximum expressiveness.

            # Small: 40% pot
            small = max(self._min_raise, int(0.4 * pot))
            small = min(small, remaining)
            new_bet_small = other_bet + small
            if new_bet_small <= self._max_bet:
                sizes.append((ACT_RAISE_HALF, new_bet_small))

            # Medium: 70% pot
            med = max(self._min_raise, int(0.7 * pot))
            med = min(med, remaining)
            new_bet_med = other_bet + med
            if new_bet_med <= self._max_bet:
                sizes.append((ACT_RAISE_POT, new_bet_med))

            # Large: 100% pot
            large = max(self._min_raise, pot)
            large = min(large, remaining)
            new_bet_large = other_bet + large
            new_bet_large = min(new_bet_large, self._max_bet)
            if new_bet_large > other_bet:
                sizes.append((ACT_RAISE_ALLIN, new_bet_large))

            # Overbet: 150% pot
            overbet = max(self._min_raise, int(1.5 * pot))
            overbet = min(overbet, remaining)
            new_bet_over = other_bet + overbet
            new_bet_over = min(new_bet_over, self._max_bet)
            if new_bet_over > other_bet:
                sizes.append((ACT_RAISE_OVERBET, new_bet_over))

        # Deduplicate (if sizes overlap, keep unique values)
        seen = set()
        deduped = []
        for act_id, new_bet in sizes:
            if new_bet not in seen:
                seen.add(new_bet)
                deduped.append((act_id, new_bet))

        return deduped

    def _build(self, hero_bet, opp_bet, hero_first):
        """Recursively build the game tree."""
        first_player = 0 if hero_first else 1
        root = self._add_node(first_player, TERM_NONE, hero_bet, opp_bet, -1)
        self._expand(root, hero_bet, opp_bet, first_player,
                     first_acted=False, raises=0)

    def _expand(self, node_id, hero_bet, opp_bet, acting_player,
                first_acted, raises):
        """Expand a decision node with all valid actions."""
        if len(self.player) > MAX_NODES:
            # Safety: convert to showdown terminal if tree gets too large
            self.terminal[node_id] = TERM_SHOWDOWN
            self.player[node_id] = -1
            if node_id in self.hero_node_ids:
                self.hero_node_ids.remove(node_id)
            elif node_id in self.opp_node_ids:
                self.opp_node_ids.remove(node_id)
            self.terminal_node_ids.append(node_id)
            return

        is_hero = (acting_player == 0)
        my_bet = hero_bet if is_hero else opp_bet
        other_bet = opp_bet if is_hero else hero_bet
        actions = []

        if my_bet == other_bet:
            # No bet to respond to: CHECK or BET
            # CHECK
            if first_acted:
                # Both checked -> street ends (showdown/continuation)
                check_child = self._add_node(-1, TERM_SHOWDOWN, hero_bet, opp_bet, node_id)
                actions.append((ACT_CHECK, check_child))
            else:
                # First player checks, second player acts
                check_child = self._add_node(
                    1 - acting_player, TERM_NONE,
                    hero_bet, opp_bet, node_id)
                actions.append((ACT_CHECK, check_child))
                self._expand(check_child, hero_bet, opp_bet,
                            1 - acting_player, first_acted=True, raises=0)

            # BET (= raise from 0)
            for act_id, new_bet in self._compute_raise_sizes(
                    hero_bet + opp_bet, my_bet, other_bet, raises):
                if is_hero:
                    new_hero, new_opp = new_bet, opp_bet
                else:
                    new_hero, new_opp = hero_bet, new_bet

                bet_child = self._add_node(
                    1 - acting_player, TERM_NONE,
                    new_hero, new_opp, node_id)
                actions.append((act_id, bet_child))
                self._expand(bet_child, new_hero, new_opp,
                            1 - acting_player, first_acted=True, raises=raises + 1)
        else:
            # Facing a bet: FOLD, CALL, or RAISE
            # FOLD
            if is_hero:
                fold_child = self._add_node(-1, TERM_FOLD_HERO, hero_bet, opp_bet, node_id)
            else:
                fold_child = self._add_node(-1, TERM_FOLD_OPP, hero_bet, opp_bet, node_id)
            actions.append((ACT_FOLD, fold_child))

            # CALL
            if is_hero:
                call_hero, call_opp = other_bet, opp_bet
            else:
                call_hero, call_opp = hero_bet, other_bet
            # After call, street ends (bets are equal, both have acted)
            call_child = self._add_node(-1, TERM_SHOWDOWN,
                                        call_hero if is_hero else hero_bet,
                                        call_opp if not is_hero else opp_bet,
                                        node_id)
            # Fix: ensure call matches the other player's bet
            if is_hero:
                call_child_hero = opp_bet
                call_child_opp = opp_bet
            else:
                call_child_hero = hero_bet
                call_child_opp = hero_bet
            self.hero_pot[call_child] = call_child_hero
            self.opp_pot[call_child] = call_child_opp
            actions.append((ACT_CALL, call_child))

            # RAISE
            for act_id, new_bet in self._compute_raise_sizes(
                    hero_bet + opp_bet, my_bet, other_bet, raises):
                if is_hero:
                    new_hero, new_opp = new_bet, opp_bet
                else:
                    new_hero, new_opp = hero_bet, new_bet

                raise_child = self._add_node(
                    1 - acting_player, TERM_NONE,
                    new_hero, new_opp, node_id)
                actions.append((act_id, raise_child))
                self._expand(raise_child, new_hero, new_opp,
                            1 - acting_player, first_acted=True, raises=raises + 1)

        self.children[node_id] = actions
        self.num_actions[node_id] = len(actions)

    @property
    def size(self):
        return len(self.player)
