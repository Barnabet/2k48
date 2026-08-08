"""ctypes bindings to the C++ engine and trained agents.

Python drives the same compiled code the trainer uses, so the reference
environment in ``env.py`` can be validated against it and tooling can play with
a real checkpoint at full speed.

Build the library first::

    make bin/lib2048.so
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

_LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "lib2048.so")


class NativeUnavailable(RuntimeError):
    pass


def _load() -> ctypes.CDLL:
    if not os.path.exists(_LIB_PATH):
        raise NativeUnavailable(
            f"{_LIB_PATH} not found — run `make bin/lib2048.so` from the repo root"
        )
    lib = ctypes.CDLL(_LIB_PATH)

    lib.g2048_move.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
    lib.g2048_move.restype = ctypes.c_uint64
    lib.g2048_has_move.argtypes = [ctypes.c_uint64]
    lib.g2048_has_move.restype = ctypes.c_int
    lib.g2048_count_empty.argtypes = [ctypes.c_uint64]
    lib.g2048_count_empty.restype = ctypes.c_int
    lib.g2048_max_tile.argtypes = [ctypes.c_uint64]
    lib.g2048_max_tile.restype = ctypes.c_int
    lib.g2048_place.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
    lib.g2048_place.restype = ctypes.c_uint64

    lib.g2048_agent_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    lib.g2048_agent_load.restype = ctypes.c_void_p
    lib.g2048_agent_free.argtypes = [ctypes.c_void_p]
    lib.g2048_agent_free.restype = None
    lib.g2048_agent_value.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.g2048_agent_value.restype = ctypes.c_float
    lib.g2048_agent_stages.argtypes = [ctypes.c_void_p]
    lib.g2048_agent_stages.restype = ctypes.c_int
    lib.g2048_agent_act.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.g2048_agent_act.restype = ctypes.c_int
    return lib


_lib: Optional[ctypes.CDLL] = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


# --- engine ---------------------------------------------------------------


def move(board: int, action: int) -> tuple[int, int]:
    """Applies `action` to a bitboard. Returns (afterstate, reward)."""
    reward = ctypes.c_uint32(0)
    out = lib().g2048_move(ctypes.c_uint64(board), action, ctypes.byref(reward))
    return int(out), int(reward.value)


def has_move(board: int) -> bool:
    return bool(lib().g2048_has_move(ctypes.c_uint64(board)))


def count_empty(board: int) -> int:
    return int(lib().g2048_count_empty(ctypes.c_uint64(board)))


def max_tile(board: int) -> int:
    return int(lib().g2048_max_tile(ctypes.c_uint64(board)))


def place(board: int, nth_empty: int, exponent: int) -> int:
    return int(lib().g2048_place(ctypes.c_uint64(board), nth_empty, exponent))


# --- agent ----------------------------------------------------------------


class Agent:
    """A trained n-tuple agent.

    Not thread safe — the search keeps a mutable transposition table, so give
    each thread its own instance.

    Args:
        path: checkpoint written by ``bin/train``.
        depth: expectimax depth; 0 selects the 1-ply greedy policy.
        cutoff: probability below which spawn branches are pruned.
        adaptive: extend depth on crowded boards.
    """

    def __init__(self, path: str, depth: int = 0, cutoff: float = 1e-4, adaptive: bool = True):
        handle = lib().g2048_agent_load(
            path.encode(), int(depth), float(cutoff), 1 if adaptive else 0
        )
        if not handle:
            raise RuntimeError(f"failed to load model: {path}")
        self._handle = handle
        self.path = path
        self.depth = depth

    @property
    def stages(self) -> int:
        return int(lib().g2048_agent_stages(self._handle))

    def value(self, board: int) -> float:
        return float(lib().g2048_agent_value(self._handle, ctypes.c_uint64(board)))

    def act(self, board: int) -> tuple[int, int, int]:
        """Returns (action, afterstate, reward); action is -1 when terminal."""
        after = ctypes.c_uint64(0)
        reward = ctypes.c_uint32(0)
        a = lib().g2048_agent_act(
            self._handle, ctypes.c_uint64(board), ctypes.byref(after), ctypes.byref(reward)
        )
        return int(a), int(after.value), int(reward.value)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            lib().g2048_agent_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
