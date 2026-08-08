"""Tests for the Python environment, cross-checked against the C++ engine.

The point of the cross-check is that the trainer learns against the C++ rules.
If the Python reference env drifted from them, anything measured through it
would be measuring a different game.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env as E  # noqa: E402
import native  # noqa: E402


def _native_or_skip():
    try:
        native.lib()
    except native.NativeUnavailable as exc:
        pytest.skip(str(exc))


def random_grid(rng: np.random.Generator) -> np.ndarray:
    g = np.zeros(16, dtype=np.int8)
    for i in range(16):
        # Weighted toward empty and small tiles, like a real board.
        g[i] = 0 if rng.random() < 0.45 else rng.integers(1, 13)
    return g.reshape(4, 4)


# --- bitboard packing -----------------------------------------------------


def test_bitboard_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        g = random_grid(rng)
        assert np.array_equal(E.from_bitboard(E.to_bitboard(g)), g)


# --- rules ----------------------------------------------------------------


def test_merge_each_tile_once():
    g = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    after, reward = E.apply_move(g, E.Action.LEFT)
    assert list(after[0]) == [2, 2, 0, 0], "[2,2,2,2] must give two 4s, not one 8"
    assert reward == 8


def test_merge_direction_matters():
    g = np.array([[1, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    left, _ = E.apply_move(g, E.Action.LEFT)
    right, _ = E.apply_move(g, E.Action.RIGHT)
    assert list(left[0]) == [2, 2, 0, 0]
    assert list(right[0]) == [0, 0, 2, 2]


def test_max_tile_does_not_merge():
    g = np.zeros((4, 4), dtype=np.int8)
    g[0, 0] = g[0, 1] = 15
    after, reward = E.apply_move(g, E.Action.LEFT)
    assert np.array_equal(after, g), "32768 tiles must not merge past the representable range"
    assert reward == 0


def test_terminal_detection():
    checker = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.int8)
    assert E.is_terminal(checker)
    assert E.legal_actions(checker) == []
    # Make the bottom row [2, 1, 2, 2] so the last two cells can merge. That
    # also puts a matching pair in the last column, so every direction opens up.
    almost = checker.copy()
    almost[3, 3] = 2
    assert not E.is_terminal(almost)
    assert set(E.legal_actions(almost)) == {0, 1, 2, 3}


# --- cross-check against the C++ engine -----------------------------------


def test_moves_match_native():
    _native_or_skip()
    rng = np.random.default_rng(1234)
    for _ in range(4000):
        g = random_grid(rng)
        board = E.to_bitboard(g)
        for a in range(4):
            py_after, py_reward = E.apply_move(g, a)
            c_after, c_reward = native.move(board, a)
            assert E.to_bitboard(py_after) == c_after, (
                f"afterstate mismatch, action={a}\n{g}\npython:\n{py_after}\n"
                f"native:\n{E.from_bitboard(c_after)}"
            )
            assert py_reward == c_reward, f"reward mismatch, action={a}: {py_reward} vs {c_reward}"


def test_terminal_matches_native():
    _native_or_skip()
    rng = np.random.default_rng(99)
    for _ in range(4000):
        g = random_grid(rng)
        board = E.to_bitboard(g)
        assert E.is_terminal(g) == (not native.has_move(board))
        assert int((g == 0).sum()) == native.count_empty(board)


# --- environment mechanics ------------------------------------------------


def test_reset_starts_with_two_tiles():
    e = E.Env2048(seed=7)
    e.reset()
    assert int((e.grid > 0).sum()) == 2
    assert e.score == 0


def test_step_spawns_and_scores():
    e = E.Env2048(seed=3)
    e.reset()
    before_tiles = int((e.grid > 0).sum())
    for a in e.legal_actions():
        _, reward, _, _, _ = e.step(a)
        # One tile merged away at most, and exactly one spawned.
        assert int((e.grid > 0).sum()) >= before_tiles - 1
        assert reward >= 0
        break


# A single column of distinct tiles: UP and DOWN are blocked, LEFT is a no-op,
# only RIGHT does anything.
_STUCK = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]], dtype=np.int8)


def test_illegal_move_is_a_noop():
    e = E.Env2048(seed=5, illegal_move="noop")
    e.reset()
    e.grid = _STUCK.copy()
    assert E.Action.UP not in e.legal_actions()
    before = e.grid.copy()
    obs, reward, term, trunc, info = e.step(E.Action.UP)
    assert np.array_equal(e.grid, before), "an illegal move must not spawn a tile"
    assert reward == 0
    assert info["illegal_move"]
    assert not term


def test_illegal_move_can_raise():
    e = E.Env2048(seed=5, illegal_move="raise")
    e.reset()
    e.grid = _STUCK.copy()
    with pytest.raises(ValueError):
        e.step(E.Action.UP)


def test_illegal_move_can_terminate():
    e = E.Env2048(seed=5, illegal_move="terminate")
    e.reset()
    e.grid = _STUCK.copy()
    _, reward, term, _, info = e.step(E.Action.UP)
    assert term and reward == 0 and info["illegal_move"]


def test_episode_terminates_and_score_matches_rewards():
    e = E.Env2048(seed=11)
    e.reset()
    total = 0.0
    for _ in range(100000):
        legal = e.legal_actions()
        if not legal:
            break
        _, r, term, _, _ = e.step(legal[0])
        total += r
        if term:
            break
    assert e._done
    assert total == e.score
    assert E.is_terminal(e.grid)


def test_step_after_done_raises():
    e = E.Env2048(seed=13)
    e.reset()
    while True:
        legal = e.legal_actions()
        if not legal:
            break
        _, _, term, _, _ = e.step(legal[0])
        if term:
            break
    with pytest.raises(RuntimeError):
        e.step(0)


def test_observation_shapes():
    for obs_type, shape in (("exponent", (4, 4)), ("onehot", (16, 4, 4)), ("grid", (4, 4))):
        e = E.Env2048(seed=1, obs_type=obs_type)
        obs, _ = e.reset()
        assert obs.shape == shape
        if obs_type == "onehot":
            # Exactly one channel is hot per cell.
            assert np.allclose(obs.sum(axis=0), np.ones((4, 4)))


def test_seeding_is_reproducible():
    a = E.Env2048(seed=42)
    b = E.Env2048(seed=42)
    a.reset()
    b.reset()
    assert np.array_equal(a.grid, b.grid)
    for _ in range(50):
        legal = a.legal_actions()
        if not legal:
            break
        act = legal[0]
        a.step(act)
        b.step(act)
        assert np.array_equal(a.grid, b.grid)


def test_spawn_distribution_is_90_10():
    e = E.Env2048(seed=2026)
    twos = fours = 0
    for _ in range(4000):
        e.grid = np.zeros((4, 4), dtype=np.int8)
        e._spawn()
        v = int(e.grid.max())
        twos += v == 1
        fours += v == 2
    frac4 = fours / (twos + fours)
    assert 0.08 < frac4 < 0.12, f"expected ~10% fours, got {frac4:.3f}"
