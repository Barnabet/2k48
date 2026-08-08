#!/usr/bin/env python3
"""Check whether the value function is calibrated.

V(afterstate) is trained to predict the remaining score from that board. Every
recorded game contains both the prediction and, in hindsight, what actually
happened — so the two can be compared directly.

This separates two very different explanations for "mean V rises while mean
score doesn't":

  * overestimation — V predicts more remaining return than the policy realises,
    the classic maximisation bias of greedy TD control;
  * a shape effect — V is averaged over moves while score is per game, so a
    change in how score accrues over a game moves one and not the other.

    python3 python/calibration.py web/games.json
"""

from __future__ import annotations

import argparse
import json
import sys


def analyse(path: str, buckets: int = 8) -> int:
    with open(path) as f:
        data = json.load(f)

    pairs: list[tuple[float, float]] = []   # (predicted, realised) remaining return
    for g in data["games"]:
        rows = g["trace"].split("\n")
        # Rebuild the running score, then walk it again knowing the final total.
        score = 0
        frames = []
        for line in rows:
            p = line.split(" ")
            reward = int(p[2]) if p[2] != "-" else 0
            action = -1 if p[1] == "-" else int(p[1])
            total = None if action < 0 or p[3 + action] == "-" else float(p[3 + action])
            frames.append((action, reward, total, score))
            score += reward
        final = score

        for action, reward, total, before in frames:
            if action < 0 or total is None:
                continue           # terminal marker carries no prediction
            predicted = total - reward          # V(afterstate), excludes this move's reward
            realised = final - (before + reward)  # what actually followed
            pairs.append((predicted, realised))

    if not pairs:
        print("no usable frames")
        return 1

    n = len(pairs)
    mp = sum(p for p, _ in pairs) / n
    mr = sum(r for _, r in pairs) / n
    print(f"{path}: {len(data['games'])} games, {n:,} moves")
    print(f"model {data.get('model')} — {data.get('policy')}\n")
    print(f"mean predicted remaining : {mp:12,.0f}")
    print(f"mean realised remaining  : {mr:12,.0f}")
    print(f"bias (pred - real)       : {mp - mr:+12,.0f}   ({100 * (mp - mr) / max(1.0, mr):+.1f}%)\n")

    # Calibration by predicted value: within each bucket the mean prediction
    # should match the mean outcome.
    pairs.sort(key=lambda x: x[0])
    size = n // buckets
    print(f"{'predicted range':>26} {'n':>8} {'mean pred':>11} {'mean real':>11} {'bias':>11}")
    for b in range(buckets):
        lo = b * size
        hi = n if b == buckets - 1 else (b + 1) * size
        ch = pairs[lo:hi]
        if not ch:
            continue
        bp = sum(p for p, _ in ch) / len(ch)
        br = sum(r for _, r in ch) / len(ch)
        print(f"{ch[0][0]:>11,.0f}..{ch[-1][0]:>12,.0f} {len(ch):>8,} "
              f"{bp:>11,.0f} {br:>11,.0f} {bp - br:>+11,.0f}")

    print("\nNote: a positive bias at every level means V is systematically optimistic.")
    print("A bias that flips sign across buckets is a fit problem, not a drift problem.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="?", default="web/games.json")
    ap.add_argument("--buckets", type=int, default=8)
    args = ap.parse_args()
    return analyse(args.games, args.buckets)


if __name__ == "__main__":
    sys.exit(main())
