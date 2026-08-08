#!/usr/bin/env bash
# Round 9: late-game restart training vs plain continuation.
#
# The training distribution is the last untested explanation for the gap to the
# literature. Greedy self-play reaches 16384 in only ~65% of games, so the
# weights that decide the 16k -> 32k transition see a starved slice of the
# data. Every published agent that reaches 32768 consistently oversamples the
# late game during training — Jaskowski's carousel shaping, Yeh's multi-stage
# splits that restart from saved stage-boundary boards. It also explains this
# project's two multi-stage nulls: rounds 1 and 7 split the network without
# changing what it trained on, which starves the same distribution through
# more parameters. This round tests the mechanism itself, without the split.
#
# The restart arm harvests afterstates the first time an episode reaches
# 16384 and starts half its episodes from the pool. The control continues
# plain training with the same budget. Both from the same checkpoint; a 20k
# fresh-seed greedy judgement decides, with the 32768 column watched as
# closely as the mean — a late-game gain can move the tail before the mean.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-240}"
ARM_THREADS="${ARM_THREADS:-7}"
GAMES="${GAMES:-20000}"
SRC="${SRC:-runs/final.bin}"
SEED="${SEED:-23}"
FRAC="${FRAC:-0.5}"
TILE="${TILE:-14}"
LOG=runs/round9.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -e runs/round8_final.bin ]; then
  cp "$SRC" runs/round8_final.bin.tmp && mv runs/round8_final.bin.tmp runs/round8_final.bin
fi
say "incumbent preserved: runs/round8_final.bin"

# Quota pre-flight: the judged round-8 arms are superseded by round8_final.bin;
# their keeper bests live on /tmp. Freeing them makes room for this round's
# checkpoints (each save holds file + .tmp at once against a ~21 GiB quota).
rm -f runs/r8_wide.bin runs/r8_ctl.bin
say "round 9: restart(frac=$FRAC tile=$TILE) vs control, ${ARM_THREADS} threads each, $MIN min, src=$SRC"

for a in rst ctl; do
  GAMES=600 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "runs/r9_$a.bin" "runs/r9_${a}_best" \
    > "runs/r9_$a.keeper.out" 2>&1 &
  say "keeper $a pid $!"
done

./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --restart-frac "$FRAC" --restart-tile "$TILE" --restart-pool runs/r9_pool.bin \
  --seed 901 --out runs/r9_rst.bin --log runs/r9_rst.csv --ckpt-min 20 \
  > runs/r9_rst.out 2>&1 &
./bin/train --threads "$ARM_THREADS" --alpha 0.1 --time-min "$MIN" --resume "$SRC" \
  --seed 902 --out runs/r9_ctl.bin --log runs/r9_ctl.csv --ckpt-min 20 \
  > runs/r9_ctl.out 2>&1 &
say "arms launched"

deadline=$(( $(date +%s) + (MIN + 30) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { say "DEADLINE passed with trainers alive"; break; }
  sleep 60
done
say "arms finished"

for a in rst ctl; do
  [ -f "runs/r9_$a.bin" ] || say "WARNING: arm $a produced no checkpoint"
  say "arm $a episodes: $(tail -1 "runs/r9_$a.csv" 2>/dev/null | cut -d, -f1)"
done
# The restart arm's own diagnostics: how many episodes restarted, how many of
# those reached 32768, and the pool size at the end.
say "restart diag (episodes,restart_eps,restart_32k,pool): $(tail -1 runs/r9_rst.csv 2>/dev/null | cut -d, -f1,14,15,16)"

touch runs/r9_rst_best.stop runs/r9_ctl_best.stop; sleep 5

CANDIDATES="runs/round8_final.bin runs/r9_rst.bin runs/r9_ctl.bin runs/r9_rst_best.bin runs/r9_ctl_best.bin" \
  SEED="$SEED" OUT=runs/round9_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 9 COMPLETE ==="
