#!/usr/bin/env python3
"""Watch a trained agent play, or play 2048 yourself.

    python3 python/play.py --model runs/model.bin            # watch, greedy
    python3 python/play.py --model runs/model.bin --depth 3  # watch, with search
    python3 python/play.py --human                           # play it yourself
    python3 python/play.py --model runs/model.bin --games 50 # batch stats

The agent drives the C++ value function through ctypes, while the board itself
is the Python reference environment, so what you see is the real policy playing
the real game.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env as E  # noqa: E402

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# 256-colour codes keyed by tile exponent, dim for small tiles and hot for large.
_TILE_COLOURS = {
    0: 238, 1: 250, 2: 252, 3: 215, 4: 209, 5: 203, 6: 197, 7: 220,
    8: 226, 9: 190, 10: 118, 11: 46, 12: 51, 13: 45, 14: 39, 15: 201,
}


def colour_board(grid, score: int, moves: int, use_colour: bool = True) -> str:
    lines = []
    for r in range(4):
        cells = []
        for c in range(4):
            v = int(grid[r, c])
            text = f"{1 << v:>6}" if v else "     ."
            if use_colour:
                cells.append(f"\033[38;5;{_TILE_COLOURS.get(v, 15)}m{text}\033[0m")
            else:
                cells.append(text)
        lines.append(" ".join(cells))
    m = int(grid.max())
    lines.append(f"\n  score {score:>10}   moves {moves:>6}   max {(1 << m) if m else 0:>6}")
    return "\n".join(lines)


def watch(model: str, depth: int, delay: float, seed: int, colour: bool) -> None:
    import native

    with native.Agent(model, depth=depth) as agent:
        e = E.Env2048(seed=seed)
        e.reset()
        label = f"expectimax depth {depth}" if depth > 0 else "greedy 1-ply"
        print(f"model: {model}   policy: {label}   stages: {agent.stages}\n")
        while True:
            action, _after, _reward = agent.act(e.bitboard)
            if action < 0:
                break
            _, _, terminated, _, _ = e.step(action)
            if delay > 0:
                # Redraw in place so the board animates rather than scrolling.
                sys.stdout.write("\033[H\033[J")
            print(colour_board(e.grid, e.score, e.moves, colour))
            print(f"  -> {ACTION_NAMES[action]}   V = {agent.value(e.bitboard):,.0f}")
            if delay > 0:
                time.sleep(delay)
            if terminated:
                break
        print(f"\ngame over — score {e.score:,}  max tile {e.max_tile}  moves {e.moves}")


def batch(model: str, depth: int, games: int, seed: int) -> None:
    import native

    with native.Agent(model, depth=depth) as agent:
        scores, tiles = [], []
        t0 = time.time()
        for i in range(games):
            e = E.Env2048(seed=seed + i)
            e.reset()
            while True:
                action, _, _ = agent.act(e.bitboard)
                if action < 0:
                    break
                _, _, terminated, _, _ = e.step(action)
                if terminated:
                    break
            scores.append(e.score)
            tiles.append(e.max_tile)
            print(f"\r  game {i + 1}/{games}", end="", flush=True)
        elapsed = time.time() - t0

    scores.sort()
    print(f"\n\n{games} games in {elapsed:.1f}s")
    print(f"  mean score   {sum(scores) / len(scores):>12,.0f}")
    print(f"  median score {scores[len(scores) // 2]:>12,}")
    print(f"  max score    {scores[-1]:>12,}")
    for t in (1024, 2048, 4096, 8192, 16384, 32768):
        n = sum(1 for x in tiles if x >= t)
        if n or t <= 4096:
            print(f"  reached {t:>6}: {100.0 * n / games:>6.2f}%")


def human(seed: int, colour: bool) -> None:
    e = E.Env2048(seed=seed)
    e.reset()
    keys = {"w": 0, "d": 1, "s": 2, "a": 3}
    print("WASD to move, q to quit.\n")
    while True:
        print(colour_board(e.grid, e.score, e.moves, colour))
        if not e.legal_actions():
            print("\ngame over")
            break
        try:
            k = input("  move> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if k == "q":
            break
        if k not in keys:
            print("  use w/a/s/d")
            continue
        if keys[k] not in e.legal_actions():
            print("  that move does nothing here")
            continue
        _, _, terminated, _, _ = e.step(keys[k])
        if terminated:
            print(colour_board(e.grid, e.score, e.moves, colour))
            print("\ngame over")
            break
    print(f"final score {e.score:,}  max tile {e.max_tile}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="checkpoint from bin/train")
    p.add_argument("--depth", type=int, default=0, help="expectimax depth (0 = greedy)")
    p.add_argument("--games", type=int, default=0, help="play N games and print stats")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between moves when watching")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--human", action="store_true", help="play yourself")
    p.add_argument("--no-colour", action="store_true")
    args = p.parse_args()

    if args.human:
        human(args.seed, not args.no_colour)
        return 0
    if not args.model:
        p.error("--model is required unless --human is given")
    if not os.path.exists(args.model):
        p.error(f"model not found: {args.model}")

    if args.games:
        batch(args.model, args.depth, args.games, args.seed)
    else:
        watch(args.model, args.depth, args.delay, args.seed, not args.no_colour)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
