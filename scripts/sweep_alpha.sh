#!/usr/bin/env bash
# Short single-stage runs at several step sizes, to pick alpha for the long run.
# Sequential so each run gets the full thread budget and a comparable episode count.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/sweep

MINUTES="${MINUTES:-3}"
THREADS="${THREADS:-14}"

for a in 0.5 0.2 0.1 0.05; do
  echo "=== alpha=$a ==="
  ./bin/train --threads "$THREADS" --alpha "$a" --time-min "$MINUTES" \
    --out "runs/sweep/a$a.bin" --log "runs/sweep/a$a.csv" \
    --ckpt-min 999 --log-every 20000
  echo "exit=$?"
done
echo "SWEEP COMPLETE"
