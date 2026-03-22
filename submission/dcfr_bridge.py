"""
Bridge between Python range_solver and C DCFR core.
Serializes the game tree and calls the C solver for ~5x speedup.
Falls back to Python if C library unavailable.
Compiles from source at import time if .so not found.
"""
import os
import sys
import subprocess
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

MAX_ACT = 7

_lib = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_DIR, "dcfr_core.so")
_C_PATH = os.path.join(_DIR, "dcfr_core.c")


def _try_compile():
    """Try to compile the C solver. Returns True on success."""
    if not os.path.exists(_C_PATH):
        return False
    # Try with BLAS first (macOS Accelerate or Linux OpenBLAS)
    for flags in [
        ["-DUSE_BLAS", "-framework", "Accelerate"],  # macOS
        ["-DUSE_BLAS", "-lopenblas"],                 # Linux with OpenBLAS
        ["-DUSE_BLAS", "-lblas"],                     # Linux with system BLAS
        ["-lm"],                                       # No BLAS fallback
    ]:
        try:
            cmd = ["gcc", "-O3", "-march=native", "-ffast-math",
                   "-shared", "-fPIC", "-o", _SO_PATH, _C_PATH] + flags
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _try_load():
    """Try to load the compiled .so file."""
    global _lib
    try:
        _lib = ctypes.CDLL(_SO_PATH)
        _lib.run_dcfr_c.restype = None
        _lib.run_dcfr_c.argtypes = [
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ]
        return True
    except (OSError, Exception):
        _lib = None
        return False


# Try loading existing .so, then compile if needed
if not _try_load():
    if _try_compile():
        _try_load()


def c_available():
    return _lib is not None


def run_dcfr_c(tree, opp_weights, terminal_values, n_hero, n_opp, iterations):
    """Run DCFR using the C solver. Returns hero strategy at root (n_hero, n_act)."""
    n_nodes = tree.size

    # Serialize tree to flat arrays
    player_arr = np.array(tree.player, dtype=np.int32)
    n_actions_arr = np.array(tree.num_actions, dtype=np.int32)

    # Children: [n_nodes * MAX_ACT], storing child node IDs
    children_arr = np.zeros(n_nodes * MAX_ACT, dtype=np.int32)
    for nid in range(n_nodes):
        for a, (act_type, child_id) in enumerate(tree.children[nid]):
            children_arr[nid * MAX_ACT + a] = child_id

    # Node index maps
    hero_idx_arr = np.full(n_nodes, -1, dtype=np.int32)
    for i, nid in enumerate(tree.hero_node_ids):
        hero_idx_arr[nid] = i

    opp_idx_arr = np.full(n_nodes, -1, dtype=np.int32)
    for i, nid in enumerate(tree.opp_node_ids):
        opp_idx_arr[nid] = i

    term_idx_arr = np.full(n_nodes, -1, dtype=np.int32)
    for i, nid in enumerate(tree.terminal_node_ids):
        term_idx_arr[nid] = i

    # Terminal values: [n_terminals * n_hero * n_opp]
    n_terminals = len(tree.terminal_node_ids)
    tv_arr = np.zeros(n_terminals * n_hero * n_opp, dtype=np.float64)
    for i, nid in enumerate(tree.terminal_node_ids):
        tv = terminal_values[nid]  # (n_hero, n_opp) array
        tv_arr[i * n_hero * n_opp:(i + 1) * n_hero * n_opp] = tv.ravel()

    n_hero_nodes = len(tree.hero_node_ids)
    n_opp_nodes = len(tree.opp_node_ids)

    # Ensure opp_weights is contiguous float64
    opp_w = np.ascontiguousarray(opp_weights, dtype=np.float64)

    # Output buffer
    hero_strat_sum = np.zeros(n_hero_nodes * n_hero * MAX_ACT, dtype=np.float64)

    # Call C solver
    _lib.run_dcfr_c(
        player_arr, n_actions_arr, children_arr,
        hero_idx_arr, opp_idx_arr, term_idx_arr,
        n_nodes, tv_arr,
        n_hero, n_opp, n_hero_nodes, n_opp_nodes,
        opp_w, iterations, hero_strat_sum
    )

    # Extract root strategy
    root = 0
    root_hero_idx = hero_idx_arr[root]
    if root_hero_idx < 0:
        return np.ones((n_hero, 1)) / 1

    n_act = tree.num_actions[root]
    strat = hero_strat_sum.reshape(n_hero_nodes, n_hero, MAX_ACT)
    strat_slice = strat[root_hero_idx, :, :n_act].copy()
    totals = strat_slice.sum(axis=1, keepdims=True)
    result = np.where(totals > 0, strat_slice / np.maximum(totals, 1e-10),
                      np.full_like(strat_slice, 1.0 / n_act))
    return result
