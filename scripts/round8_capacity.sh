#!/usr/bin/env bash
# Round 8: does doubling the pattern set beat spending the same time on the
# network we already have?
#
# Every other lever is measured flat. Alpha is indifferent across 0.1-0.15 and
# only hurts at 0.05. Thread count is indifferent between 7 and 14. Multi-stage
# returned a null twice, the second time on a base reaching 8192 in 90.7% of
# games with a live control — 16 points of difference after four hours. Search
# depth and cutoff were already at their frontier. Continued single-stage
# training has decayed from ~3,200 points/hour to ~650. Capacity is what is
# left untested.
#
# The widened arm keeps everything already learned: the default eight patterns
# are a prefix of the sixteen, so trained tables copy into the front and the new
# tables start at zero. Verified rather than assumed — evaluated before and
# after widening, both score 255,774 / median 287,516 / max 483,716 across 2,000
# deterministic games, identical to the digit.
#
# The cost is real and is the whole question: 64 -> 128 features per board means
# roughly half the episodes per hour. Double the capacity against half the
# samples is not obviously a win inside a fixed budget, so both arms get equal
# wall-clock from the same checkpoint and the judgement decides.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-240}"
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
SEED="${SEED:-19}"
LOG=runs/round8.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -e runs/round7_final.bin ]; then
  cp "$SRC" runs/round7_final.bin.tmp && mv runs/round7_final.bin.tmp runs/round7_final.bin
fi
say "incumbent preserved: runs/round7_final.bin (255650 at seed 17)"
say "round 8: 16-pattern widened vs 8-pattern control, ${ARM_THREADS} threads each, $MIN min"

for a in wide ctl; do
  GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "runs/r8_$a.bin" "runs/r8_${a}_best" \
    > "runs/r8_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --extend-patterns --seed 801 --out runs/r8_wide.bin --log runs/r8_wide.csv --ckpt-min 20 \
  > runs/r8_wide.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --seed 802 --out runs/r8_ctl.bin --log runs/r8_ctl.csv --ckpt-min 20 \
  > runs/r8_ctl.out 2>&1 &
say "arms launched"

deadline=$(( $(date +%s) + (MIN + 30) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

for a in wide ctl; do
  [ -f "runs/r8_$a.bin" ] || say "WARNING: arm $a produced no checkpoint"
done

# Episodes reached matter as much as the score here: the widened arm is expected
# to see far fewer, and the size of that gap is what makes the result readable.
for a in wide ctl; do
  say "arm $a episodes: $(tail -1 "runs/r8_$a.csv" 2>/dev/null | cut -d, -f1)"
done

touch runs/r8_wide_best.stop runs/r8_ctl_best.stop; sleep 5

CANDIDATES="runs/round7_final.bin runs/r8_wide.bin runs/r8_ctl.bin runs/r8_wide_best.bin runs/r8_ctl_best.bin" \
  SEED="$SEED" OUT=runs/round8_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 8 COMPLETE ==="
