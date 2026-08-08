#!/usr/bin/env python3
"""Champion lineage vs a fresh run, aligned at the origin.

full_curve.py answers "what happened"; this answers "is the new recipe
better". Both histories are drawn from episode zero on a shared axis: the
stitched champion lineage (phases 1-3 plus every judged round winner, from
full_curve's reconstruction) against a challenger training run — by default
the round-10 fresh retrain. The third panel plots the challenger's lead at
equal episode counts, which is the honest form of the question: not "has it
caught the champion yet" but "is it ahead of where the champion's recipe was
at this point in its own history".

Wall-clock caveat: lineage segments ran on 7 or 14 threads at different
moves/s, so elapsed_s across segments is not comparable — episodes is the
default and recommended axis.

    python3 python/compare_curve.py                       # lineage vs r10_fresh
    python3 python/compare_curve.py --challenger runs/r11.csv -o runs/cmp.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_curve import concat, discover, stitched  # noqa: E402
from plot import read_log, smooth  # noqa: E402

LINEAGE_COLOR = "#1f77b4"
FRESH_COLOR = "#d62728"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenger", default="runs/r10_fresh.csv")
    ap.add_argument("--label", default=None, help="challenger legend label (default: file stem)")
    ap.add_argument("-o", "--out", default="runs/compare_curve.png")
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()

    trunk_specs, _branches = discover()
    trunk, _end_ep, _end_el = stitched(trunk_specs)
    if not trunk:
        print("no lineage segments found")
        return 1
    ch = read_log(args.challenger)
    if not ch.get("mean_score"):
        print(f"no usable rows in {args.challenger}")
        return 1
    ch_label = args.label or os.path.splitext(os.path.basename(args.challenger))[0]

    w = args.smooth
    lx = concat(trunk, "episodes")
    ly = smooth(concat(trunk, "mean_score"), w)
    n = min(len(lx), len(ly))
    lx, ly = lx[:n], ly[:n]
    cx = ch["episodes"]
    cy = smooth(ch["mean_score"], w)
    m = min(len(cx), len(cy))
    cx, cy = cx[:m], cy[:m]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    chain = " → ".join(p["label"] for p in trunk)
    fig.suptitle(f"2048 — champion lineage vs {ch_label} (aligned at episode 0)", fontsize=12)

    # Panel 1: both learning curves from the origin.
    ax = axes[0]
    ax.plot(lx, ly, color=LINEAGE_COLOR, lw=1.2, label=f"champion lineage [{chain}]")
    ax.plot(cx, cy, color=FRESH_COLOR, lw=1.4, label=ch_label)
    ax.axhline(ly[-1], color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.annotate(f"lineage today ≈ {ly[-1]:,.0f}", (0, ly[-1]), fontsize=7.5,
                va="bottom", ha="left", color="grey")
    ax.set_xlabel("episodes")
    ax.set_ylabel("mean score (self-play, greedy)")
    ax.set_title("Learning curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: the tile race — solid 16384, dashed 8192.
    ax = axes[1]
    for cols, x, colour, label in ((dict((k, concat(trunk, k)) for k in ("r8192", "r16384")), lx, LINEAGE_COLOR, "lineage"),
                                   (ch, cx, FRESH_COLOR, ch_label)):
        for key, style in (("r16384", "-"), ("r8192", "--")):
            ys = smooth(cols.get(key, []), w)
            k = min(len(x), len(ys))
            if k:
                ax.plot(x[:k], ys[:k], style, color=colour, lw=1.2,
                        label=f"{label} {key[1:]}")
    ax.set_xlabel("episodes")
    ax.set_ylabel("% of games reaching tile")
    ax.set_title("8192 (dashed) and 16384 (solid) reach rates")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    # Panel 3: challenger's lead at equal episodes. Positive = the new recipe
    # is ahead of where the champion's own history was at this many episodes.
    ax = axes[2]
    upto = min(cx[-1], lx[-1])
    grid = np.linspace(0, upto, 400)
    lead = np.interp(grid, cx, cy) - np.interp(grid, lx, ly)
    ax.plot(grid, lead, color="#2ca02c", lw=1.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.fill_between(grid, lead, 0, where=lead >= 0, color="#2ca02c", alpha=0.15)
    ax.fill_between(grid, lead, 0, where=lead < 0, color="#d62728", alpha=0.15)
    ax.set_xlabel("episodes")
    ax.set_ylabel(f"{ch_label} − lineage (mean score)")
    ax.set_title("Challenger's lead at equal episode count")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"lineage: {chain} ({lx[-1]:,.0f} episodes, ends ≈ {ly[-1]:,.0f})")
    print(f"challenger: {ch_label} ({cx[-1]:,.0f} episodes, now ≈ {cy[-1]:,.0f})")
    if cx[-1] <= lx[-1]:
        at = np.interp(cx[-1], lx, ly)
        print(f"lineage at the same point was ≈ {at:,.0f}  (lead: {cy[-1] - at:+,.0f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
