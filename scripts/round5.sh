#!/usr/bin/env bash
# Round 5: continue at the alpha round 4 established, with every thread on it.
#
# Round 4 answered the open question — alpha 0.1 beat 0.05 by 3,806 at z = 3.75,
# and both beat the incumbent, so the model is not saturated. There is no second
# arm here because there is no longer a question worth half the threads: pushing
# alpha higher still is speculative, the downside (instability) is real, and
# round 4's own rate says a full-width continuation is worth more than the
# experiment would be. Round 4 gained 8,293 in 150 min on 7 threads while still
# climbing at the buzzer; this gets 14.
#
# The incumbent is still judged alongside the challengers on a seed none of them
# has been scored on, so a regression is caught rather than assumed away.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-150}"
ALPHA="${ALPHA:-0.1}"
THREADS="${THREADS:-14}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
SEED="${SEED:-11}"
LOG=runs/round5.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

# Preserved under its own name before anything can overwrite runs/final.bin,
# which is where the selection pass writes its winner.
if [ ! -f runs/round4_final.bin ]; then
  cp "$SRC" runs/round4_final.bin.tmp && mv runs/round4_final.bin.tmp runs/round4_final.bin
fi
say "incumbent preserved: runs/round4_final.bin (242091 at seed 7)"
say "round 5: alpha $ALPHA, $THREADS threads, $MIN min, from $SRC"

GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh runs/r5.bin runs/r5_best \
  > runs/r5.keeper.out 2>&1 &
say "keeper pid $!"

./bin/train --threads "$THREADS" --alpha "$ALPHA" --time-min "$MIN" --resume "$SRC" \
  --seed 501 --out runs/r5.bin --log runs/r5.csv --ckpt-min 15 > runs/r5.out 2>&1
say "round 5 exit=$?"

touch runs/r5_best.stop; sleep 5

CANDIDATES="runs/round4_final.bin runs/r5.bin runs/r5_best.bin" \
  SEED="$SEED" OUT=runs/round5_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 5 COMPLETE ==="
