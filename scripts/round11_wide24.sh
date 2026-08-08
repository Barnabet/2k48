#!/usr/bin/env bash
# Round 11: 24-pattern widening vs 16-pattern continuation.
#
# Capacity is the only lever that has paid on this base: round 8's 8 -> 16
# widening beat a live control by ~9k on half the samples, while every tuning
# knob, multi-stage (twice), restart training, and a full fresh retrain
# (round 10 — per-episode lead decayed from +7.7k to +2.9k and it finished
# 45k behind) came up null or worse. This round asks the same question one
# step up: does 16 -> 24 patterns beat continuing at 16 on equal compute?
#
# The widened arm's 8 new tables start at zero with fresh TC stats — full
# plasticity bolted onto a trained network, exactly the round-8 recipe. The
# cost is 192 feature lookups per board instead of 128, so the wide arm sees
# about a third fewer episodes; whether capacity outruns lost episodes inside
# a fixed budget is what the A/B answers.
#
# Disk notes: 24-pattern checkpoints are 4.5 GiB and saves hold file + .tmp
# at once against a ~21 GiB MooseFS quota, so the arms use coprime-ish
# checkpoint intervals (30 vs 41 min) and staggered end times (the control
# ends 15 min early) so final saves never overlap. Keeper bests go to local
# /tmp/2m48 as usual.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-735}"              # wide arm minutes; control gets MIN-15
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/round9_final.bin}"
SEED="${SEED:-37}"
LOG=runs/round11.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

say "round 11: wide24 vs ctl16, ${ARM_THREADS} threads each, wide $MIN min / ctl $((MIN - 15)) min, src=$SRC, judge seed $SEED"

for a in wide24 ctl; do
  GAMES=600 THREADS=1 INTERVAL=300 ./scripts/keep_arm.sh "runs/r11_$a.bin" "runs/r11_${a}_best" \
    > "runs/r11_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --extend-patterns 24 \
  --seed 1101 --out runs/r11_wide24.bin --log runs/r11_wide24.csv --ckpt-min 30 \
  > runs/r11_wide24.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$((MIN - 15))" --resume "$SRC" \
  --seed 1102 --out runs/r11_ctl.bin --log runs/r11_ctl.csv --ckpt-min 41 \
  > runs/r11_ctl.out 2>&1 &
say "arms launched"

deadline=$(( $(date +%s) + (MIN + 40) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

for a in wide24 ctl; do
  [ -f "runs/r11_$a.bin" ] || say "WARNING: arm $a produced no checkpoint"
  say "arm $a episodes: $(tail -1 "runs/r11_$a.csv" 2>/dev/null | cut -d, -f1)"
done

touch runs/r11_wide24_best.stop runs/r11_ctl_best.stop; sleep 5

CANDIDATES="runs/round9_final.bin runs/r11_wide24.bin runs/r11_ctl.bin runs/r11_wide24_best.bin runs/r11_ctl_best.bin" \
  SEED="$SEED" OUT=runs/round11_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 11 COMPLETE ==="
