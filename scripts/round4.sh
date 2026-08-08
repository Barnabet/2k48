#!/usr/bin/env bash
# Round 4: is the slowdown caused by the low alpha, or by saturation?
#
# Round 2 gained 18,359 in 160 min at alpha 0.1 (115/min). Round 3 gained
# 1,275 +/- 1,015 in 70 min at alpha 0.05 (18/min) — not significant. Two
# explanations, opposite prescriptions: either dropping to 0.05 was premature
# and the model still has room at 0.1, or the model is saturating and no alpha
# will help. Continuing at 0.025 assumes the second without testing it.
#
# So both arms resume from the same checkpoint and differ only in alpha. Round 1
# split three ways and returned a null because stage layout genuinely does not
# matter much; alpha demonstrably does — an 18k gain against a ~1k standard
# error. If alpha is driving the slowdown this design will see it clearly, and
# if both arms land together that is the saturation answer, which is equally
# worth knowing.
#
# The incumbent runs in the judgement, not just the two challengers: "more
# training made it worse" has to be an available outcome.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-150}"
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
# A seed the candidates have not been selected on before. Every evaluation so
# far used the default seed 1, and picking repeatedly from one fixed set of
# 20,000 games starts fitting the game set rather than the game.
SEED="${SEED:-7}"
LOG=runs/round4.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

# Preserved before anything can overwrite runs/final.bin, which is where the
# selection pass writes its winner.
if [ ! -f runs/round3_final.bin ]; then
  cp "$SRC" runs/round3_final.bin.tmp && mv runs/round3_final.bin.tmp runs/round3_final.bin
fi
say "incumbent preserved: runs/round3_final.bin (234160 at seed 1)"
say "round 4 A/B: alpha 0.1 vs 0.05, ${ARM_THREADS} threads each, $MIN min, from $SRC"

# 2 arms x 7 training threads + 2 intermittent keeper threads = 16 against a
# 15.375 core quota. The keepers are idle most of the time; oversubscribing the
# trainers themselves would make the arms unequal, which is the one thing this
# design cannot tolerate.
for a in hi lo; do
  case $a in hi) al=0.1 ;; lo) al=0.05 ;; esac
  GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "runs/r4_$a.bin" "runs/r4_${a}_best" \
    > "runs/r4_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

# Launched inline, not through a helper: a $(...) wrapper would background the
# trainer inside a subshell, so it would not be a child of this shell and the
# wait below would return immediately and judge two unfinished arms.
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --seed 401 --out runs/r4_hi.bin --log runs/r4_hi.csv --ckpt-min 15 > runs/r4_hi.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.05 --time-min "$MIN" --resume "$SRC" \
  --seed 402 --out runs/r4_lo.bin --log runs/r4_lo.csv --ckpt-min 15 > runs/r4_lo.out 2>&1 &
say "arms launched"

# Polled rather than waited on, so one arm dying early cannot be mistaken for
# both finishing. The deadline is a backstop against a hung trainer.
deadline=$(( $(date +%s) + (MIN + 25) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

touch runs/r4_hi_best.stop runs/r4_lo_best.stop; sleep 5

CANDIDATES="runs/round3_final.bin runs/r4_hi.bin runs/r4_hi_best.bin runs/r4_lo.bin runs/r4_lo_best.bin" \
  SEED="$SEED" OUT=runs/round4_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 4 COMPLETE ==="
