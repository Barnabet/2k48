#!/usr/bin/env python3
"""Decide whether phase 1 is recovering from the 16384 regression.

Captures a trailing-average baseline, waits, then recomputes and prints a
verdict. The criterion is fixed before the wait so the call is not made by
eyeballing a noisy curve after the fact.
"""

from __future__ import annotations

import csv
import sys
import time

LOG = "runs/train_log.csv"
WINDOWS = 20          # trailing windows to average (20 * 2000 = 40k episodes)
WAIT_S = 1200         # 20 minutes
MIN_16K_GAIN = 1.5    # percentage points
MIN_MEAN_GAIN = 2000  # score


def trailing(n: int = WINDOWS) -> dict:
    with open(LOG) as f:
        rows = list(csv.DictReader(f))
    chunk = rows[-n:]
    g = lambda k: sum(float(r[k]) for r in chunk) / len(chunk)
    return {
        "episodes": int(float(chunk[-1]["episodes"])),
        "elapsed_min": float(chunk[-1]["elapsed_s"]) / 60,
        "mean": g("mean_score"),
        "r4096": g("r4096"),
        "r8192": g("r8192"),
        "r16384": g("r16384"),
        "V": g("mean_value"),
    }


def show(tag: str, s: dict) -> None:
    print(f"{tag}: ep {s['episodes']:,} ({s['elapsed_min']:.0f} min)  "
          f"mean {s['mean']:,.0f}  4096 {s['r4096']:.1f}%  8192 {s['r8192']:.1f}%  "
          f"16384 {s['r16384']:.1f}%  V {s['V']:,.0f}", flush=True)


def main() -> int:
    base = trailing()
    show("BASELINE", base)
    print(f"waiting {WAIT_S // 60} min; recovery needs 16384 +{MIN_16K_GAIN} pts "
          f"AND mean +{MIN_MEAN_GAIN}", flush=True)
    time.sleep(WAIT_S)

    now = trailing()
    show("NOW     ", now)

    d16 = now["r16384"] - base["r16384"]
    dmean = now["mean"] - base["mean"]
    print(f"\ndelta: 16384 {d16:+.1f} pts   mean {dmean:+,.0f}   "
          f"episodes +{now['episodes'] - base['episodes']:,}", flush=True)

    if d16 >= MIN_16K_GAIN and dmean >= MIN_MEAN_GAIN:
        print("VERDICT: RECOVERING — let phase 1 continue")
    else:
        print("VERDICT: NOT RECOVERING — cut phase 1 and move to multi-stage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
