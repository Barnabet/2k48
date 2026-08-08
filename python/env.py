"""A 2048 environment with a Gymnasium-style API.

This is a self-contained NumPy implementation, independent of the C++ engine so
it can serve as a readable reference and be used without building anything.
`python/test_env.py` cross-checks it move-for-move against the C++ engine, which
is what the trainer actually runs.

Tiles are stored as exponents: 0 is an empty cell and `k` means a tile of value
2**k. The largest representable tile is 2**15 = 32768, and two of those do not
merge.

    env = Env2048(seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(Action.LEFT)
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np

MAX_EXPONENT = 15
SIZE = 4


class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def _slide_merge_left(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Slides one row toward index 0. Each tile merges at most once."""
    tiles = [v for v in line if v]
    out: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] != MAX_EXPONENT:
            merged = int(tiles[i]) + 1
            out.append(merged)
            # int() keeps rewards and scores plain Python integers rather than
            # NumPy scalars, which callers (and json) expect.
            reward += int(1 << merged)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return np.array(out, dtype=np.int8), reward


def apply_move(grid: np.ndarray, action: int) -> tuple[np.ndarray, int]:
    """Returns the afterstate and merge reward. The grid is unchanged if illegal.

    Every direction is expressed as "slide left" on a suitably oriented view, so
    there is exactly one merge implementation to get right.
    """
    g = grid.copy()
    reward = 0
    if action == Action.LEFT:
        for r in range(SIZE):
            g[r], sc = _slide_merge_left(g[r])
            reward += sc
    elif action == Action.RIGHT:
        for r in range(SIZE):
            row, sc = _slide_merge_left(g[r][::-1])
            g[r] = row[::-1]
            reward += sc
    elif action == Action.UP:
        for c in range(SIZE):
            col, sc = _slide_merge_left(g[:, c])
            g[:, c] = col
            reward += sc
    elif action == Action.DOWN:
        for c in range(SIZE):
            col, sc = _slide_merge_left(g[:, c][::-1])
            g[:, c] = col[::-1]
            reward += sc
    else:
        raise ValueError(f"invalid action: {action}")
    return g, reward


def legal_actions(grid: np.ndarray) -> list[int]:
    return [a for a in range(4) if not np.array_equal(apply_move(grid, a)[0], grid)]


def is_terminal(grid: np.ndarray) -> bool:
    return not legal_actions(grid)


def to_bitboard(grid: np.ndarray) -> int:
    """Packs the grid into the same uint64 layout the C++ engine uses."""
    b = 0
    for i in range(16):
        b |= int(grid.flat[i]) << (4 * i)
    return b


def from_bitboard(board: int) -> np.ndarray:
    return np.array([(board >> (4 * i)) & 0xF for i in range(16)], dtype=np.int8).reshape(4, 4)


class Env2048:
    """Standard 2048: a 4x4 board, a 2 or 4 spawning after every successful move.

    Args:
        seed: RNG seed for tile spawns.
        obs_type: ``"exponent"`` gives a 4x4 int8 grid of exponents;
            ``"onehot"`` gives a (16, 4, 4) float32 stack, the usual input
            format for a neural policy; ``"grid"`` gives actual tile values.
        illegal_move: ``"noop"`` returns reward 0 and leaves the board alone;
            ``"raise"`` raises; ``"terminate"`` ends the episode.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: int | None = None,
        obs_type: str = "exponent",
        illegal_move: str = "noop",
    ) -> None:
        if obs_type not in ("exponent", "onehot", "grid"):
            raise ValueError(f"unknown obs_type: {obs_type}")
        if illegal_move not in ("noop", "raise", "terminate"):
            raise ValueError(f"unknown illegal_move: {illegal_move}")
        self.obs_type = obs_type
        self.illegal_move = illegal_move
        self._rng = np.random.default_rng(seed)
        self.grid = np.zeros((SIZE, SIZE), dtype=np.int8)
        self.score = 0
        self.moves = 0
        self._done = True

    # -- core API ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[Any, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.grid = np.zeros((SIZE, SIZE), dtype=np.int8)
        self.score = 0
        self.moves = 0
        self._done = False
        self._spawn()
        self._spawn()
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        after, reward = apply_move(self.grid, action)
        if np.array_equal(after, self.grid):
            if self.illegal_move == "raise":
                raise ValueError(f"illegal action {action} for board:\n{self.render()}")
            if self.illegal_move == "terminate":
                self._done = True
                return self._obs(), 0.0, True, False, self._info(illegal=True)
            return self._obs(), 0.0, False, False, self._info(illegal=True)

        self.grid = after
        self.score += reward
        self.moves += 1
        self._spawn()
        self._done = is_terminal(self.grid)
        return self._obs(), float(reward), self._done, False, self._info()

    def render(self) -> str:
        bar = "+" + "------+" * SIZE
        lines = [bar]
        for r in range(SIZE):
            cells = "".join(
                f"{(1 << v) if v else '':>5} |" for v in self.grid[r]
            )
            lines.append("|" + cells)
            lines.append(bar)
        lines.append(f"score {self.score}  moves {self.moves}  max {self.max_tile}")
        return "\n".join(lines)

    # -- helpers ----------------------------------------------------------

    @property
    def max_tile(self) -> int:
        m = int(self.grid.max())
        return (1 << m) if m else 0

    @property
    def bitboard(self) -> int:
        return to_bitboard(self.grid)

    def legal_actions(self) -> list[int]:
        return legal_actions(self.grid)

    def _spawn(self) -> None:
        empty = np.flatnonzero(self.grid == 0)
        if empty.size == 0:
            return
        idx = int(self._rng.choice(empty))
        # 90% a 2, 10% a 4 — the standard distribution.
        value = 2 if self._rng.random() < 0.1 else 1
        self.grid.flat[idx] = value

    def _obs(self) -> Any:
        if self.obs_type == "exponent":
            return self.grid.copy()
        if self.obs_type == "grid":
            return np.where(self.grid > 0, 1 << self.grid.astype(np.int32), 0)
        onehot = np.zeros((MAX_EXPONENT + 1, SIZE, SIZE), dtype=np.float32)
        rows, cols = np.indices((SIZE, SIZE))
        onehot[self.grid.reshape(SIZE, SIZE), rows, cols] = 1.0
        return onehot

    def _info(self, illegal: bool = False) -> dict:
        return {
            "score": self.score,
            "moves": self.moves,
            "max_tile": self.max_tile,
            "legal_actions": self.legal_actions(),
            "illegal_move": illegal,
        }
