#!/usr/bin/env python3
"""Where do training updates actually land, by max-tile regime?

Multi-stage training splits the weight table on the largest tile present. The
boundaries should not be hand-picked: a stage that sees very few updates cannot
train its own 134M-weight table, and a stage that sees most of them gains
nothing from being split off.

What matters is the share of *moves* in each regime, not the share of games
that reach it — a game that reaches 16384 spends a long endgame there, so the
regime carries far more update mass than the game-level reach rate suggests.

    python3 python/stage_mass.py web/games.json --stages 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def max_exp(hex_board: str) -> int:
    return max(int(ch, 16) for ch in hex_board)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="?", default="web/games.json")
    ap.add_argument("--stages", type=int, default=3, help="how many stages to propose")
    ap.add_argument("--floor", type=float, default=4.0, help="minimum %% of moves a stage must see")
    args = ap.parse_args()

    with open(args.games) as f:
        data = json.load(f)

    moves = Counter()
    for g in data["games"]:
        for line in g["trace"].split("\n"):
            hexb = line.split(" ", 1)[0]
            moves[max_exp(hexb)] += 1

    total = sum(moves.values())
    print(f"{args.games}: {len(data['games'])} games, {total:,} moves")
    print(f"model {data.get('model')} — {data.get('policy')}\n")

    print(f"{'max tile':>10} {'moves':>10} {'share':>8} {'cumulative':>11}")
    cum = 0.0
    order = sorted(moves)
    for e in order:
        share = 100.0 * moves[e] / total
        cum += share
        print(f"{1 << e:>10,} {moves[e]:>10,} {share:>7.2f}% {cum:>10.2f}%")

    # Choose boundaries by brute force over every combination of cut points,
    # minimising the largest stage's share. A greedy "cut where the cumulative
    # share crosses k/N" pass looks simpler but silently returns fewer stages
    # than asked for whenever two targets fall inside the same tile bucket —
    # and this distribution is lumpy enough that they routinely do.
    from itertools import combinations

    candidates = [e for e in range(1, 16) if any(x >= e for x in order)]
    best = None
    for cuts in combinations(candidates, args.stages - 1):
        bounds = [0] + list(cuts) + [16]
        masses = [
            sum(moves[e] for e in order if bounds[i] <= e < bounds[i + 1]) / total
            for i in range(len(bounds) - 1)
        ]
        # A stage below the floor cannot train its own table and is worse than
        # not splitting at all.
        if min(masses) < args.floor / 100.0:
            continue
        # Minimise squared deviation from an equal split. Minimising the
        # *largest* stage instead is degenerate here: the 8192 bucket is a
        # single tile value holding ~40% of moves and cannot be subdivided, so
        # every candidate ties on that measure and the choice becomes arbitrary.
        target = 1.0 / args.stages
        score = sum((m - target) ** 2 for m in masses)
        if best is None or score < best[0]:
            best = (score, list(cuts))

    if best is None:
        print(f"\ncannot form {args.stages} non-empty stages from this distribution")
        return 1
    cuts = best[1]
    print(f"\nbest-balanced boundaries for {args.stages} stages:")
    print(f"  --stages {','.join(str(c) for c in cuts)}   "
          f"(new stage when the largest tile reaches "
          f"{', '.join(f'{1 << c:,}' for c in cuts)})")

    # Report the resulting split so an unbalanced proposal is visible.
    bounds = [0] + cuts + [16]
    print("\n  resulting stage mass:")
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        m = sum(moves[e] for e in order if lo <= e < hi)
        label = f"<{1 << hi:,}" if lo == 0 else (f">={1 << lo:,}" if hi == 16 else f"{1 << lo:,}")
        print(f"    stage {i} ({label:>10}): {100.0 * m / total:>6.2f}%  {m:>9,} moves")

    print("\nNote: a stage below a few percent of moves will train slowly and may")
    print("end up worse than the shared table it was split out of.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
