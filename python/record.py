#!/usr/bin/env python3
"""Record agent games into a compact trace for the web viewer.

Two passes: play a batch of games cheaply to find a spread of outcomes, then
replay a selected handful with full per-move detail. Seeds make the replay
identical to the scan, so the selected games really are the ones measured.

The per-move trace records the agent's actual decision criterion — the 1-ply
total ``r + V(afterstate)`` for every action — so the viewer can show why each
move was chosen, not just what was chosen.

    python3 python/record.py --model runs/model.bin --out web/games.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env as E  # noqa: E402
import native  # noqa: E402

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def play_scan(agent: native.Agent, seed: int) -> tuple[int, int, int]:
    """Plays one game, returning (score, max_tile_exponent, moves)."""
    e = E.Env2048(seed=seed)
    e.reset()
    while True:
        action, _, _ = agent.act(e.bitboard)
        if action < 0:
            break
        _, _, terminated, _, _ = e.step(action)
        if terminated:
            break
    m = int(e.grid.max())
    return e.score, m, e.moves


def play_traced(agent: native.Agent, seed: int) -> dict:
    """Replays a game recording every move and every action's evaluation."""
    e = E.Env2048(seed=seed)
    e.reset()
    lines = []
    while True:
        board = e.bitboard
        # Evaluate all four actions exactly as the greedy policy does.
        totals: list[str] = []
        for a in range(4):
            after, reward = native.move(board, a)
            if after == board:
                totals.append("-")           # illegal
            else:
                totals.append(f"{reward + agent.value(after):.0f}")

        action, after, reward = agent.act(board)
        if action < 0:
            break
        # board | chosen action | reward | the four totals
        lines.append(f"{board:016x} {action} {reward} " + " ".join(totals))

        _, _, terminated, _, _ = e.step(action)
        if terminated:
            break

    # Final board, so the viewer can show the position the game ended in.
    lines.append(f"{e.bitboard:016x} - 0 - - - -")
    return {
        "seed": seed,
        "score": e.score,
        "maxTile": e.max_tile,
        "moves": e.moves,
        "trace": "\n".join(lines),
    }


def pick_seeds(results: list[tuple[int, int, int, int]], count: int) -> list[int]:
    """Chooses a spread of games: the best, the worst, and evenly spaced between.

    Showing only the best games would misrepresent the agent, so the selection
    deliberately spans the outcome distribution.
    """
    ordered = sorted(results, key=lambda r: r[1])  # by score
    if count >= len(ordered):
        return [r[0] for r in ordered]
    idx = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(ordered[i][0])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/model.bin")
    p.add_argument("--out", default="web/games.json")
    p.add_argument("--scan", type=int, default=40, help="games to play when looking for a spread")
    p.add_argument("--keep", type=int, default=7, help="games to record in full")
    p.add_argument("--depth", type=int, default=0, help="search depth (0 = greedy, matches the panel)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}")
        return 1

    t0 = time.time()
    with native.Agent(args.model, depth=args.depth) as agent:
        print(f"scanning {args.scan} games with {args.model} ...")
        results = []
        for i in range(args.scan):
            seed = args.seed + i
            score, mexp, moves = play_scan(agent, seed)
            results.append((seed, score, mexp, moves))
            print(f"\r  {i + 1}/{args.scan}", end="", flush=True)
        print()

        scores = sorted(r[1] for r in results)
        dist: dict[int, int] = {}
        for _, _, mexp, _ in results:
            dist[1 << mexp] = dist.get(1 << mexp, 0) + 1
        print(f"  scan: mean {sum(scores) / len(scores):,.0f}  median {scores[len(scores) // 2]:,}  "
              f"max {scores[-1]:,}")
        print(f"  max-tile distribution: {dict(sorted(dist.items()))}")

        seeds = pick_seeds(results, args.keep)
        print(f"recording {len(seeds)} games in full ...")
        games = []
        for i, seed in enumerate(seeds):
            games.append(play_traced(agent, seed))
            print(f"\r  {i + 1}/{len(seeds)}", end="", flush=True)
        print()
        stages = agent.stages

    games.sort(key=lambda g: g["score"])
    payload = {
        "model": os.path.basename(args.model),
        "policy": f"expectimax depth {args.depth}" if args.depth else "greedy 1-ply",
        "stages": stages,
        "scan": {
            "games": len(results),
            "mean": sum(r[1] for r in results) / len(results),
            "median": scores[len(scores) // 2],
            "max": scores[-1],
            "tileDistribution": {str(k): v for k, v in sorted(dist.items())},
        },
        "games": games,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size = os.path.getsize(args.out)
    total_moves = sum(g["moves"] for g in games)
    print(f"wrote {args.out} — {size / 1e6:.2f} MB, {len(games)} games, "
          f"{total_moves:,} moves, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
