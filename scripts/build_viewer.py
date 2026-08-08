#!/usr/bin/env python3
"""Splice recorded game data into the viewer template.

Keeping the multi-megabyte payload out of the template means the page source
stays readable and the data can be regenerated for a new checkpoint without
touching the markup.

    python3 python/record.py --model runs/model.bin --out web/games.json
    python3 scripts/build_viewer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=os.path.join(ROOT, "web/viewer_template.html"))
    p.add_argument("--data", default=os.path.join(ROOT, "web/games.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "web/viewer.html"))
    args = p.parse_args()

    for path in (args.template, args.data):
        if not os.path.exists(path):
            print(f"missing: {path}")
            return 1

    with open(args.template) as f:
        html = f.read()
    with open(args.data) as f:
        raw = f.read()

    if "__DATA__" not in html:
        print("template has no __DATA__ placeholder")
        return 1

    # Validate before embedding — a malformed payload would fail silently in
    # the browser, and JSON.parse errors there are far harder to trace.
    data = json.loads(raw)
    if not data.get("games"):
        print("no games in payload")
        return 1

    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that can break out of it is a literal "</script>".
    safe = raw.replace("</", "<\\/")
    html = html.replace("__DATA__", safe)

    with open(args.out, "w") as f:
        f.write(html)

    size = os.path.getsize(args.out)
    moves = sum(g["moves"] for g in data["games"])
    print(f"wrote {args.out} — {size / 1e6:.2f} MB, {len(data['games'])} games, {moves:,} moves")
    if size > 15_000_000:
        print("WARNING: approaching the 16 MB artifact limit; record fewer games")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
