"""
Card abstraction utilities for the blueprint solver.

Provides two levels of abstraction:
1. Hand bucketing: Groups 2-card hands by equity into ~50 buckets.
2. Board clustering: Groups boards by structural features into ~200 clusters.

These abstractions make the blueprint computation tractable by solving one
strategy per cluster instead of per-board.

(Copied from blueprint/abstraction.py for self-contained deployment.)
"""

import os
import sys
import itertools
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)


def compute_hand_equity(hand, board, dead_cards, equity_engine):
    """
    Compute raw equity of a 2-card hand against uniform opponent range.

    Args:
        hand: tuple of 2 card ints
        board: list of card ints (3-5 cards)
        dead_cards: list of card ints removed from play
        equity_engine: ExactEquityEngine instance

    Returns:
        float in [0, 1]
    """
    return equity_engine.compute_equity(list(hand), board, dead_cards)


def compute_hand_bucket(hand, board, dead_cards, n_buckets, equity_engine):
    """
    Map a 2-card hand to an equity bucket.

    Buckets are evenly spaced: bucket i covers equity in [i/n, (i+1)/n).
    Bucket 0 = weakest hands, bucket n-1 = strongest.

    Args:
        hand: tuple of 2 card ints
        board: list of card ints (3-5 cards)
        dead_cards: list of card ints removed from play
        n_buckets: number of equity buckets
        equity_engine: ExactEquityEngine instance

    Returns:
        int bucket_id in [0, n_buckets-1]
    """
    eq = compute_hand_equity(hand, board, dead_cards, equity_engine)
    bucket = int(eq * n_buckets)
    return min(bucket, n_buckets - 1)


def enumerate_hands(board, dead_cards):
    """
    Enumerate all possible 2-card hands given known board and dead cards.

    Args:
        board: list of card ints
        dead_cards: list of card ints

    Returns:
        list of (c1, c2) tuples with c1 < c2
    """
    known = set(board) | set(dead_cards)
    remaining = [c for c in range(27) if c not in known]
    return list(itertools.combinations(remaining, 2))


def enumerate_hand_buckets(board, dead_cards, n_buckets, equity_engine):
    """
    Map every possible 2-card hand to an equity bucket for a given board.

    Args:
        board: list of card ints (3-5 cards)
        dead_cards: list of card ints
        n_buckets: number of equity buckets
        equity_engine: ExactEquityEngine instance

    Returns:
        dict mapping (c1, c2) tuple -> bucket_id
    """
    hands = enumerate_hands(board, dead_cards)
    hand_to_bucket = {}

    for hand in hands:
        eq = compute_hand_equity(hand, board, dead_cards, equity_engine)
        bucket = int(eq * n_buckets)
        bucket = min(bucket, n_buckets - 1)
        hand_to_bucket[hand] = bucket

    return hand_to_bucket


def enumerate_hand_equities(board, dead_cards, equity_engine):
    """
    Compute equity for every possible 2-card hand on a given board.

    Args:
        board: list of card ints (3-5 cards)
        dead_cards: list of card ints
        equity_engine: ExactEquityEngine instance

    Returns:
        dict mapping (c1, c2) tuple -> equity float
    """
    hands = enumerate_hands(board, dead_cards)
    hand_to_equity = {}

    for hand in hands:
        eq = compute_hand_equity(hand, board, dead_cards, equity_engine)
        hand_to_equity[hand] = eq

    return hand_to_equity


# ---------------------------------------------------------------------------
# Board clustering
# ---------------------------------------------------------------------------

# Card encoding: rank = card_int % 9, suit = card_int // 9
# Ranks: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=A
# Suits: 0=d, 1=h, 2=s

RANK_COUNT = 9
SUIT_COUNT = 3


def card_rank(c):
    return c % RANK_COUNT


def card_suit(c):
    return c // RANK_COUNT


def compute_board_features(board):
    """
    Extract strategic features from a board for clustering.

    Features (12-dimensional):
        [0] flush_type: 0=rainbow, 1=two-tone, 2=monotone
        [1] n_pairs: number of paired ranks on board (0, 1, 2)
        [2] high_card_rank: highest rank on board (0-8, 8=Ace)
        [3] low_card_rank: lowest rank on board
        [4] connectivity: sum of gaps between sorted ranks (lower = more connected)
        [5] n_straight_draws: number of 3-card straight subsets
        [6] n_board_cards: 3, 4, or 5
        [7] suit_concentration: max number of cards sharing one suit
        [8] rank_spread: max_rank - min_rank
        [9] mid_card_rank: median rank
        [10] n_high_cards: cards with rank >= 7 (i.e., 9 or A)
        [11] n_wheel_cards: cards with rank in {0,1,2,3,8} (A,2,3,4,5)

    Returns:
        np.array of shape (12,) dtype float32
    """
    ranks = sorted([card_rank(c) for c in board])
    suits = [card_suit(c) for c in board]
    n = len(board)

    # Flush structure
    suit_counts = [0, 0, 0]
    for s in suits:
        suit_counts[s] += 1
    max_suit = max(suit_counts)

    if max_suit >= 3:
        flush_type = 2  # monotone (3+ same suit)
    elif max_suit == 2:
        flush_type = 1  # two-tone
    else:
        flush_type = 0  # rainbow

    # Pairs
    unique_ranks = set(ranks)
    n_pairs = n - len(unique_ranks)  # number of duplicate rank cards

    # Ranks
    high_card = ranks[-1]
    low_card = ranks[0]
    mid_card = ranks[n // 2]
    rank_spread = high_card - low_card

    # Connectivity: sum of gaps between consecutive sorted unique ranks
    sorted_unique = sorted(unique_ranks)
    connectivity = 0
    for i in range(1, len(sorted_unique)):
        gap = sorted_unique[i] - sorted_unique[i - 1] - 1
        connectivity += gap

    # Straight draws: count 3-card subsets that form 3 consecutive ranks
    n_straight_draws = 0
    for combo in itertools.combinations(sorted_unique, min(3, len(sorted_unique))):
        if len(combo) == 3:
            if combo[2] - combo[0] <= 4:  # within a 5-card straight window
                n_straight_draws += 1

    # High cards (9 or A)
    n_high = sum(1 for r in ranks if r >= 7)

    # Wheel cards (A,2,3,4,5 = ranks 8,0,1,2,3)
    wheel_ranks = {0, 1, 2, 3, 8}
    n_wheel = sum(1 for r in ranks if r in wheel_ranks)

    return np.array([
        flush_type,
        n_pairs,
        high_card,
        low_card,
        connectivity,
        n_straight_draws,
        n,
        max_suit,
        rank_spread,
        mid_card,
        n_high,
        n_wheel,
    ], dtype=np.float32)


def compute_board_cluster(board, n_clusters=200):
    """
    Map a board to a cluster ID using a deterministic hash of its features.

    This is a lightweight clustering that doesn't require precomputation.
    Boards with the same feature vector get the same cluster ID.

    For production, you could replace this with KMeans on all boards'
    feature vectors. But the deterministic approach is simpler and avoids
    needing to precompute cluster assignments.

    Args:
        board: list of card ints (3-5 cards)
        n_clusters: number of clusters

    Returns:
        int cluster_id in [0, n_clusters-1]
    """
    features = compute_board_features(board)
    # Deterministic hash: quantize features and combine
    # This groups boards with identical (or very similar) strategic structure
    key = tuple(int(f) for f in features)
    return hash(key) % n_clusters


def enumerate_all_flops():
    """
    Enumerate all C(27,3) = 2925 possible flop boards.

    Returns:
        list of (c1, c2, c3) tuples with c1 < c2 < c3
    """
    return list(itertools.combinations(range(27), 3))


def cluster_all_boards(boards, n_clusters=200):
    """
    Assign cluster IDs to a list of boards.

    Args:
        boards: list of board tuples
        n_clusters: number of clusters

    Returns:
        dict mapping cluster_id -> list of boards in that cluster
    """
    clusters = {}
    for board in boards:
        cid = compute_board_cluster(list(board), n_clusters)
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(board)
    return clusters


def get_representative_boards(boards, n_clusters=200):
    """
    Select one representative board per cluster for solving.

    Args:
        boards: list of board tuples
        n_clusters: number of clusters

    Returns:
        list of (cluster_id, representative_board) pairs
    """
    clusters = cluster_all_boards(boards, n_clusters)
    representatives = []
    for cid in sorted(clusters.keys()):
        # Pick the first board in each cluster as representative
        representatives.append((cid, clusters[cid][0]))
    return representatives


def get_bucket_boundaries(n_buckets):
    """
    Return the equity boundaries for hand bucketing.

    Bucket i covers equity in [boundaries[i], boundaries[i+1]).

    Returns:
        np.array of shape (n_buckets+1,) with values from 0.0 to 1.0
    """
    return np.linspace(0.0, 1.0, n_buckets + 1)
