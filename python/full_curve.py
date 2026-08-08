#!/usr/bin/env python3
"""Reconstruct the FULL training curve across every phase and round.

Training happened as a chain of resumed runs — phase 1..3, then one A/B round
after another — so no single CSV holds the whole story. This script stitches
the champion lineage back together in order, offsetting episodes and elapsed
time cumulatively, and draws the same three panels as python/plot.py's
single-log mode, with segment boundaries marked. The arms of the newest round
(usually still running) are drawn as branches from the tip of the trunk.

The lineage is reconstructed from the ground truth of each round's judgement:
runs/round{N}_selection.txt names the SELECTED checkpoint, which maps back to
the arm's CSV (keeper picks like r6_a15_best.bin map to r6_a15.csv — the
keeper snapshot came from that same training run, just mid-flight).

    python3 python/full_curve.py                 # auto: full lineage + live arms
    python3 python/full_curve.py -o runs/full_curve.png --x elapsed_s
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import TILE_COLUMNS, read_log, smooth  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

# The pre-round trunk: three sequential training phases on the same network.
PHASES = ["train_log.csv", "train_log_p2.csv", "train_log_p3.csv"]


def selected_csv(selection_path: str) -> str | None:
    """Map a round's SELECTED checkpoint to the training CSV it came from."""
    try:
        with open(selection_path) as f:
            m = re.search(r"SELECTED:\s*runs/(\S+?)\.bin", f.read())
    except OSError:
        return None
    if not m:
        return None
    stem = re.sub(r"_best$", "", m.group(1))
    csv_path = os.path.join(RUNS, stem + ".csv")
    return csv_path if os.path.isfile(csv_path) else None  # incumbent won -> no new segment


def discover() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (trunk, branches) as (label, path) lists."""
    trunk = [(f"phase{i+1}", os.path.join(RUNS, p)) for i, p in enumerate(PHASES)
             if os.path.isfile(os.path.join(RUNS, p))]

    rounds = sorted(
        {int(m.group(1)) for p in glob.glob(os.path.join(RUNS, "r*_*.csv")) + glob.glob(os.path.join(RUNS, "r*.csv"))
         if (m := re.match(r"r(\d+)(_|\.)", os.path.basename(p)))}
    )
    branches: list[tuple[str, str]] = []
    for n in rounds:
        judged = selected_csv(os.path.join(RUNS, f"round{n}_selection.txt"))
        arms = sorted(
            p for p in glob.glob(os.path.join(RUNS, f"r{n}_*.csv")) + glob.glob(os.path.join(RUNS, f"r{n}.csv"))
            if "hist" not in os.path.basename(p) and "_best" not in os.path.basename(p)
        )
        if judged is not None:
            trunk.append((os.path.splitext(os.path.basename(judged))[0], judged))
        elif os.path.isfile(os.path.join(RUNS, f"round{n}_selection.txt")):
            continue  # judged, incumbent kept: arms are dead ends, skip
        else:
            # Newest, unjudged round: its arms are the live branches.
            branches = [(os.path.splitext(os.path.basename(p))[0], p) for p in arms]
    return trunk, branches


def stitched(segments: list[tuple[str, str]]) -> tuple[list[dict], float, float]:
    """Load segments, offsetting episodes/elapsed cumulatively. Returns
    (list of {label, cols with offsets applied}, end_episodes, end_elapsed)."""
    out, ep0, el0 = [], 0.0, 0.0
    for label, path in segments:
        c = read_log(path)
        if not c.get("mean_score"):
            continue
        c["episodes"] = [e + ep0 for e in c.get("episodes", [])]
        c["elapsed_s"] = [e + el0 for e in c.get("elapsed_s", [])]
        out.append({"label": label, "cols": c, "start_ep": ep0, "start_el": el0})
        ep0 = c["episodes"][-1] if c["episodes"] else ep0
        el0 = c["elapsed_s"][-1] if c["elapsed_s"] else el0
    return out, ep0, el0


def concat(parts: list[dict], key: str) -> list[float]:
    ys: list[float] = []
    for p in parts:
        ys.extend(p["cols"].get(key, []))
    return ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="runs/full_curve.png")
    ap.add_argument("--smooth", type=int, default=25, help="moving-average window")
    ap.add_argument("--x", default="episodes", choices=["episodes", "elapsed_s"])
    args = ap.parse_args()

    trunk_specs, branch_specs = discover()
    if not trunk_specs:
        print("no trunk segments found")
        return 1
    trunk, end_ep, end_el = stitched(trunk_specs)

    # Every branch starts where the trunk ends (they all resumed its endpoint).
    branches = []
    for label, path in branch_specs:
        c = read_log(path)
        if not c.get("mean_score"):
            continue
        c["episodes"] = [e + end_ep for e in c.get("episodes", [])]
        c["elapsed_s"] = [e + end_el for e in c.get("elapsed_s", [])]
        branches.append({"label": label, "cols": c})
    # Primary branch (used in panels 2-3): the one currently scoring highest.
    branches.sort(key=lambda b: -b["cols"]["mean_score"][-1])

    xkey = args.x
    w = args.smooth
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    chain = " → ".join(p["label"] for p in trunk)
    live = ", ".join(b["label"] for b in branches) or "none"
    fig.suptitle(f"2048 — full reconstructed training history   [{chain}]   live: {live}", fontsize=11)

    # Panel 1: the learning curve across everything, boundaries marked.
    ax = axes[0]
    x_all = concat(trunk, xkey)
    y_all = smooth(concat(trunk, "mean_score"), w)
    n = min(len(x_all), len(y_all))
    ax.plot(x_all[:n], y_all[:n], color="#1f77b4", lw=1.2, label="champion lineage")
    for b, col in zip(branches, ["#d62728", "#7f7f7f", "#2ca02c", "#9467bd"]):
        xb, yb = b["cols"][xkey], smooth(b["cols"]["mean_score"], min(w, 9))
        m = min(len(xb), len(yb))
        ax.plot(xb[:m], yb[:m], color=col, lw=1.3, label=b["label"])
    for p in trunk[1:]:
        bx = p["start_ep"] if xkey == "episodes" else p["start_el"]
        ax.axvline(bx, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax.annotate(p["label"], (bx, ax.get_ylim()[0]), rotation=90, fontsize=6.5,
                    va="bottom", ha="right", alpha=0.75)
    ax.set_xlabel(xkey)
    ax.set_ylabel("mean score (self-play, greedy)")
    ax.set_title("Learning curve — all phases and rounds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: where games end, trunk + primary branch stitched.
    ax = axes[1]
    parts2 = trunk + branches[:1]
    x2 = concat(parts2, xkey)
    keys = [k for k, _ in TILE_COLUMNS]
    cols2 = {k: concat(parts2, k) for k in keys}
    n2 = min([len(x2)] + [len(v) for v in cols2.values() if v])
    series, labels = [], []
    for i, (key, label) in enumerate(TILE_COLUMNS):
        if not cols2.get(key):
            continue
        cum = smooth(cols2[key], w)[:n2]
        nxt_key = TILE_COLUMNS[i + 1][0] if i + 1 < len(TILE_COLUMNS) else None
        nxt = smooth(cols2[nxt_key], w)[:n2] if nxt_key and cols2.get(nxt_key) else [0.0] * n2
        exact = [max(0.0, a - b) for a, b in zip(cum, nxt)]
        if max(exact, default=0.0) <= 0.5:
            continue
        series.append(exact)
        labels.append(label)
    if series:
        colours = plt.cm.viridis([i / max(1, len(series) - 1) for i in range(len(series))])
        ax.stackplot(x2[:n2], *series, labels=labels, colors=colours, alpha=0.9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    tip = f" (+{branches[0]['label']})" if branches else ""
    ax.set_xlabel(xkey)
    ax.set_ylabel("% of games ending on tile")
    ax.set_title(f"Where games actually end — lineage{tip}")
    ax.grid(alpha=0.3)

    # Panel 3: value scale and TD error over the whole history.
    ax = axes[2]
    x3 = x2
    ys = smooth(concat(parts2, "mean_value"), w)
    if ys:
        m = min(len(x3), len(ys))
        ax.plot(x3[:m], ys[:m], color="#1f77b4", lw=1.4)
        ax.set_ylabel("mean V", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_ylim(bottom=0)
    ys2 = smooth(concat(parts2, "mean_abs_td_error"), w)
    if ys2:
        ax2 = ax.twinx()
        m = min(len(x3), len(ys2))
        ax2.plot(x3[:m], ys2[:m], color="#d62728", lw=1.4)
        ax2.set_ylabel("mean |TD error|", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylim(bottom=0)
    ax.set_xlabel(xkey)
    ax.set_title("Value scale / TD error (has it converged?)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"trunk: {chain}")
    print(f"live branches: {live}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
