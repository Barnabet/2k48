#!/usr/bin/env bash
# Overnight chain: round 8 judgement -> depth ladder on the winner -> round 9.
# Each stage starts only when the previous one's cores are free.
set -u
cd "$(dirname "$0")/.."
LOG=runs/chain.log
say() { echo "$(date -Is) $*" >> "$LOG"; }

say "chain armed: waiting for round8_capacity.sh"
while ps -eo cmd | grep -q '[r]ound8_capacity.sh'; do sleep 120; done
say "round 8 done; selection: $(grep SELECTED runs/round8_selection.txt 2>/dev/null)"

if [ ! -s runs/round8_selection.txt ] || ! grep -q SELECTED runs/round8_selection.txt; then
  say "ABORT: round 8 selection missing or failed; not chaining further"
  exit 1
fi

say "starting depth ladder on runs/final.bin"
./scripts/depth_ladder.sh >> "$LOG" 2>&1
say "depth ladder exit=$?"

say "starting round 9 (restart training A/B)"
./scripts/round9_restart.sh
say "round 9 exit=$?"
say "=== CHAIN COMPLETE ==="
