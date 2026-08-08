#!/usr/bin/env bash
# Round 6: is alpha 0.1 the ceiling, or is higher still better?
#
# Round 4 established 0.1 > 0.05 (z = 3.75). It left the other direction open,
# and there is now a reason to test it cheaply: rounds 4 and 5 gained the same
# amount on 7 threads as on 14 (+7,896 vs +7,840), so a second arm costs almost
# nothing in progress. Two 7-thread arms for 150 minutes should each advance
# about as far as one 14-thread arm would have.
#
# 0.15 can destabilise — TD with TC step sizes can diverge if the global rate is
# too high. The trainer aborts on non-finite TD error, so divergence shows up as
# a dead arm rather than silent garbage, and the control arm carries the round.
# A dead 0.15 arm is a real answer to the question being asked.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-150}"
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
SEED="${SEED:-13}"
LOG=runs/round6.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -f runs/round5_final.bin ]; then
  cp "$SRC" runs/round5_final.bin.tmp && mv runs/round5_final.bin.tmp runs/round5_final.bin
fi
say "incumbent preserved: runs/round5_final.bin (249483 at seed 11)"
say "round 6 A/B: alpha 0.1 vs 0.15, ${ARM_THREADS} threads each, $MIN min, from $SRC"

for a in a10 a15; do
  GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "runs/r6_$a.bin" "runs/r6_${a}_best" \
    > "runs/r6_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --seed 601 --out runs/r6_a10.bin --log runs/r6_a10.csv --ckpt-min 15 > runs/r6_a10.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.15 --time-min "$MIN" --resume "$SRC" \
  --seed 602 --out runs/r6_a15.bin --log runs/r6_a15.csv --ckpt-min 15 > runs/r6_a15.out 2>&1 &
say "arms launched"

deadline=$(( $(date +%s) + (MIN + 25) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

# A diverged arm leaves no usable checkpoint; note it rather than letting the
# selection pass silently skip a missing file.
for a in a10 a15; do
  [ -f "runs/r6_$a.bin" ] || say "WARNING: arm $a produced no checkpoint (diverged?)"
  grep -qiE "non-finite|diverg|abort" "runs/r6_$a.out" 2>/dev/null && say "arm $a reported divergence"
done

touch runs/r6_a10_best.stop runs/r6_a15_best.stop; sleep 5

# Last checkpoints first: round 5 showed the keeper's banked maximum losing to
# the checkpoint it had rejected, by z = 3.6.
CANDIDATES="runs/round5_final.bin runs/r6_a10.bin runs/r6_a15.bin runs/r6_a10_best.bin runs/r6_a15_best.bin" \
  SEED="$SEED" OUT=runs/round6_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 6 COMPLETE ==="
