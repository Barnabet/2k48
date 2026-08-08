#!/usr/bin/env bash
# Round 7: re-test multi-stage, on a base strong enough for it to matter.
#
# Round 1 split the network into stages and returned a null. That result was
# honest but badly timed. The split arms were judged on a model that reached
# 16384 in 44% of games, so a stage boundary at 16384 was training its top table
# on fewer than half the episodes — starved, and unable to show a benefit even
# if the idea is sound. The base now reaches 4096 in 96.8% of games, 8192 in
# 90.7% and 16384 in 62.0%, so every stage below gets real traffic.
#
# The published multi-stage TD agents report roughly 380-400k greedy against
# 250-300k for single-stage networks of this size, which is exactly the gap
# between where we are and where the literature says this architecture tops out.
# If multi-stage is going to pay anywhere, it pays here.
#
# Both arms get equal thread-time from the same checkpoint. The control is not a
# formality: continued single-stage training is still gaining ~1,900 per 150 min,
# so multi-stage has to beat a live alternative, not a straw man.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-240}"
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
SEED="${SEED:-17}"
# 8192 / 16384 — three stages. Chosen so each sees a well-populated slice of play
# rather than by copying a published split; the round-1 failure was a population
# problem, so population is what picks the boundaries. The base reaches 8192 in
# 90.7% of games and 16384 in 62.0%, so both upper tables get real traffic.
#
# Three rather than four is a disk decision as much as a modelling one. Each
# stage is 1.5 GiB with TC accumulators, and a save holds the old file and the
# .tmp at once: four stages would put 12 GiB transient against a ~21 GiB quota,
# which is the exact pressure that silently corrupted checkpoints overnight.
# Three costs 9 GiB at peak and still puts the capacity where the agent fails.
STAGES="${STAGES:-13,14}"
LOG=runs/round7.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

# Symlinked to the local-disk copy rather than copied: /tmp/2m48/r6_a15_best.bin
# is already byte-identical to the shipped model (verified with cmp), so a second
# copy would spend 1.6 GiB of a quota this round is deliberately staying clear of.
if [ ! -e runs/round6_final.bin ]; then
  ln -sfn /tmp/2m48/r6_a15_best.bin runs/round6_final.bin
fi
say "incumbent preserved: runs/round6_final.bin (251474 seed 1 / 252313 seed 13)"
say "round 7: multi-stage($STAGES) vs single-stage control, ${ARM_THREADS} threads each, $MIN min"

for a in ms ss; do
  GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "runs/r7_$a.bin" "runs/r7_${a}_best" \
    > "runs/r7_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

# The multi-stage arm passes --stages; the control deliberately does not, so the
# layout travels inside its own checkpoint unchanged.
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --stages "$STAGES" --seed 701 --out runs/r7_ms.bin --log runs/r7_ms.csv --ckpt-min 20 \
  > runs/r7_ms.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --seed 702 --out runs/r7_ss.bin --log runs/r7_ss.csv --ckpt-min 20 \
  > runs/r7_ss.out 2>&1 &
say "arms launched"

deadline=$(( $(date +%s) + (MIN + 30) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

for a in ms ss; do
  [ -f "runs/r7_$a.bin" ] || say "WARNING: arm $a produced no checkpoint"
done

touch runs/r7_ms_best.stop runs/r7_ss_best.stop; sleep 5

CANDIDATES="runs/round6_final.bin runs/r7_ms.bin runs/r7_ss.bin runs/r7_ms_best.bin runs/r7_ss_best.bin" \
  SEED="$SEED" OUT=runs/round7_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 7 COMPLETE ==="
