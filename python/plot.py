#!/usr/bin/env python3
"""Plot training curves from one or more train_log.csv files.

    python3 python/plot.py runs/train_log.csv -o runs/curve.png
    python3 python/plot.py runs/sweep/*.csv -o runs/sweep.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TILE_COLUMNS = [
    ("r1024", "1024"),
    ("r2048", "2048"),
    ("r4096", "4096"),
    ("r8192", "8192"),
    ("r16384", "16384"),
    ("r32768", "32768"),
]


def read_log(path: str) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k is None or v is None or v == "":
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    pass
    return cols


def smooth(ys: list[float], window: int) -> list[float]:
    """Centred moving average; TD score curves are very noisy episode to episode."""
    if window <= 1 or len(ys) < window:
        return ys
    out, acc = [], 0.0
    from collections import deque

    q: deque[float] = deque()
    for y in ys:
        q.append(y)
        acc += y
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="one or more train_log.csv files")
    p.add_argument("-o", "--out", default="runs/curve.png")
    p.add_argument("--smooth", type=int, default=15, help="moving-average window")
    p.add_argument("--x", default="episodes", choices=["episodes", "elapsed_s"])
    args = p.parse_args()

    logs = [(os.path.splitext(os.path.basename(p_))[0], read_log(p_)) for p_ in args.logs]
    logs = [(n, c) for n, c in logs if c.get("mean_score")]
    if not logs:
        print("no usable rows found")
        return 1

    multi = len(logs) > 1
    fig, axes = plt.subplots(1, 3 if not multi else 2, figsize=(16 if not multi else 12, 4.5))
    fig.suptitle("2048 — TD(0) n-tuple training", fontsize=13)

    ax = axes[0]
    for name, c in logs:
        x = c.get(args.x, c["episodes"])
        n = min(len(x), len(c["mean_score"]))
        ax.plot(x[:n], smooth(c["mean_score"], args.smooth)[:n], label=name, lw=1.2)
    ax.set_xlabel(args.x)
    ax.set_ylabel("mean score (self-play, greedy)")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    if multi:
        ax.legend(fontsize=8)

    ax = axes[1]
    if multi:
        for name, c in logs:
            x = c.get(args.x, c["episodes"])
            ys = c.get("r2048", [])
            n = min(len(x), len(ys))
            ax.plot(x[:n], smooth(ys, args.smooth)[:n], label=name, lw=1.2)
        ax.set_ylabel("% of games reaching 2048")
        ax.set_title("2048 reach rate")
        ax.legend(fontsize=8)
    else:
        # Stacked share of games *ending* on each tile. Cumulative reach rates
        # saturate and hide the real story; this shows the probability mass
        # marching to the right as the agent improves.
        name, c = logs[0]
        x = c.get(args.x, c["episodes"])
        keys = [k for k, _ in TILE_COLUMNS]
        n = min([len(x)] + [len(c[k]) for k in keys if k in c])
        series, labels = [], []
        for i, (key, label) in enumerate(TILE_COLUMNS):
            if key not in c:
                continue
            cum = smooth(c[key], args.smooth)[:n]
            nxt_key = TILE_COLUMNS[i + 1][0] if i + 1 < len(TILE_COLUMNS) else None
            nxt = smooth(c[nxt_key], args.smooth)[:n] if nxt_key in c else [0.0] * n
            exact = [max(0.0, a - b) for a, b in zip(cum, nxt)]
            if max(exact) <= 0.5:
                continue
            series.append(exact)
            labels.append(label)
        if series:
            colours = plt.cm.viridis([i / max(1, len(series) - 1) for i in range(len(series))])
            ax.stackplot(x[:n], *series, labels=labels, colors=colours, alpha=0.9)
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.set_ylabel("% of games ending on tile")
        ax.set_title("Where games actually end")
    ax.set_xlabel(args.x)
    ax.grid(alpha=0.3)

    if not multi:
        # Mean |TD error| is the honest convergence signal: it should fall and
        # stay finite. A blow-up here is divergence, whatever the score does.
        # The two series differ by ~300x, so they get their own linear axes.
        # A shared log axis makes steady growth in V look like convergence,
        # which is exactly the question this panel is meant to answer.
        ax = axes[2]
        name, c = logs[0]
        x = c.get(args.x, c["episodes"])
        ys = c.get("mean_value")
        if ys:
            n = min(len(x), len(ys))
            ax.plot(x[:n], smooth(ys, args.smooth)[:n], color="#1f77b4", lw=1.4, label="mean V")
            ax.set_ylabel("mean V", color="#1f77b4")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
            ax.set_ylim(bottom=0)
        ys2 = c.get("mean_abs_td_error")
        if ys2:
            ax2 = ax.twinx()
            n = min(len(x), len(ys2))
            ax2.plot(x[:n], smooth(ys2, args.smooth)[:n], color="#d62728", lw=1.4, label="mean |TD error|")
            ax2.set_ylabel("mean |TD error|", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax2.set_ylim(bottom=0)
        ax.set_xlabel(args.x)
        ax.set_title("Value scale / TD error (has it converged?)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
