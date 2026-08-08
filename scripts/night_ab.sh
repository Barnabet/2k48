#!/usr/bin/env bash
# Unattended overnight schedule: settle the multi-stage question by experiment,
# then spend the remaining time on whatever won.
#
# Two things were unproven when phase 1 was running:
#
#   1. whether splitting the weight table by game phase pays for itself at all
#      at this budget — it triples memory and divides the update stream, and
#      the phase-1 table was still improving when we left it;
#   2. whether the boundaries derived from move-mass (4096 / 16384) beat the
#      conventional ones (2048 / 8192).
#
# Both were arguments from structure, not measurements. Round 1 runs the three
# candidates concurrently from a common ancestor with equal thread-time, judges
# them on a 20k-game evaluation, and rounds 2 and 3 give the winner the rest of
# the night at full width.
#
# Every arm gets its own keeper, so an arm that ends mid-dip is still judged on
# its best model rather than its last one.
set -u
cd "$(dirname "$0")/.."

AB=runs/ab
mkdir -p "$AB"
DLOG="$AB/driver.log"
say() { echo "$(date -Is) $*" | tee -a "$DLOG"; }

R1_MIN="${R1_MIN:-150}"     # per-arm A/B
R2_MIN="${R2_MIN:-160}"     # winner, full width, alpha 0.1
R3_MIN="${R3_MIN:-70}"      # polish, alpha 0.05
ARM_THREADS="${ARM_THREADS:-4}"
FULL_THREADS="${FULL_THREADS:-14}"
JUDGE_GAMES="${JUDGE_GAMES:-20000}"

# 4 * 3 training threads + 3 intermittent keeper threads = 15, against a 15.375
# core quota. Oversubscribing here would make the arms unequal, not just slow.

say "=== night driver up ==="

# ---------------------------------------------------------------- phase 1 wait
if [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; then
  say "waiting for phase 1 to finish"
  while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do sleep 30; done
fi
say "phase 1 finished"

# The phase-1 keeper writes fixed paths and follows the newest checkpoint; it
# would start chasing the A/B arms and mixing their results together.
pkill -f '[k]eep_best\.sh' 2>/dev/null && say "stopped phase-1 keeper"
sleep 3

# ------------------------------------------------------------- common ancestor
# best.bin, not phase1.bin: phase 1 oscillates by a few percent and its final
# checkpoint may land in a trough, which would be inherited by all three arms.
SRC=runs/best.bin
[ -f "$SRC" ] || SRC=runs/phase1.bin
cp "$SRC" "$AB/base.bin.tmp" && mv "$AB/base.bin.tmp" "$AB/base.bin"
say "common ancestor: $SRC -> $AB/base.bin"
nice -n 5 ./bin/eval --model "$AB/base.bin" --games 5000 --threads "$FULL_THREADS" \
  > "$AB/base_eval.txt" 2>&1
say "base mean = $(awk '/^mean score/{print $3}' "$AB/base_eval.txt")"

# --------------------------------------------------------------------- round 1
# Arms differ only in the stage split. Distinct seeds keep them from sharing a
# lucky trajectory; the eval below is far larger than the seed noise.
# Launched inline rather than through a helper: a `$(...)` wrapper would put the
# trainer in a subshell, the pid would not be a child of this shell, and `wait`
# would return immediately and judge three unfinished arms.
say "round 1: 3 arms x ${ARM_THREADS} threads x ${R1_MIN} min"
ARGS=(--threads "$ARM_THREADS" --alpha 0.1 --time-min "$R1_MIN" --resume "$AB/base.bin" --ckpt-min 15)

# control: keep the single shared table
./bin/train "${ARGS[@]}" --seed 101 \
  --out "$AB/single.bin"  --log "$AB/single.csv"  > "$AB/single.log" 2>&1 &
# boundaries derived from move-mass: 4096 / 16384
./bin/train "${ARGS[@]}" --seed 102 --stages 12,14 \
  --out "$AB/derived.bin" --log "$AB/derived.csv" > "$AB/derived.log" 2>&1 &
# conventional boundaries: 2048 / 8192
./bin/train "${ARGS[@]}" --seed 103 --stages 11,13 \
  --out "$AB/classic.bin" --log "$AB/classic.csv" > "$AB/classic.log" 2>&1 &
say "arms launched: $(jobs -p | tr '\n' ' ')"

for a in single derived classic; do
  GAMES=400 THREADS=1 INTERVAL=180 ./scripts/keep_arm.sh "$AB/$a.bin" "$AB/${a}_best" &
done
sleep 10

# Poll rather than `wait`, so one arm dying early cannot be mistaken for all
# three finishing. The deadline is a backstop against a hung trainer.
deadline=$(( $(date +%s) + (R1_MIN + 25) * 60 ))
while [ "$(ps -eo cmd | grep -c '[b]in/train')" -gt 0 ]; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    say "round 1 past deadline; killing stragglers"
    pkill -f '[b]in/train'; sleep 10; break
  fi
  sleep 30
done
say "round 1 training done"
for a in single derived classic; do touch "$AB/${a}_best.stop"; done
sleep 5

# ------------------------------------------------------------------- judgement
# Judge the best checkpoint of each arm, not the last. 20k games puts the
# standard error near 600 points, well under any difference worth acting on.
best_name=""; best_mean=-1
for a in single derived classic; do
  m="$AB/${a}_best.bin"; [ -f "$m" ] || m="$AB/$a.bin"
  if [ ! -f "$m" ]; then say "arm $a produced no checkpoint — skipped"; continue; fi
  nice -n 5 ./bin/eval --model "$m" --games "$JUDGE_GAMES" --threads "$FULL_THREADS" \
    > "$AB/judge_$a.txt" 2>&1
  mean=$(awk '/^mean score/{print $3}' "$AB/judge_$a.txt")
  # Anchored: an unanchored /5th pct/ also matches the "95th pct" line.
  p05=$(awk '/^ *5th pct/{print $3}' "$AB/judge_$a.txt")
  p95=$(awk '/^ *95th pct/{print $3}' "$AB/judge_$a.txt")
  se=$(awk -v a="$p05" -v b="$p95" -v n="$JUDGE_GAMES" 'BEGIN{printf "%.0f",(b-a)/3.2897/sqrt(n)}')
  r16=$(awk '$1==16384{print $2}' "$AB/judge_$a.txt")
  say "JUDGE $a: mean=$mean +/-${se}  16384=$r16  model=$m"
  if [ -n "$mean" ] && [ "$(awk -v x="$mean" -v y="$best_mean" 'BEGIN{print (x>y)?1:0}')" = 1 ]; then
    best_mean=$mean; best_name=$a; BEST_MODEL=$m
  fi
done

if [ -z "$best_name" ]; then
  say "FATAL: no arm survived round 1"
  exit 1
fi
say "WINNER: $best_name (mean $best_mean)"
echo "$best_name $best_mean $BEST_MODEL" > "$AB/winner.txt"

# --------------------------------------------------------------- rounds 2 & 3
# No --stages: the stage layout travels inside the winning checkpoint, so the
# winner keeps its own structure without the driver having to restate it.
cp "$BEST_MODEL" runs/phase2_start.bin

GAMES=600 THREADS=3 INTERVAL=120 ./scripts/keep_arm.sh runs/phase2.bin runs/final_best &
KEEPER=$!

say "round 2: winner=$best_name, ${FULL_THREADS} threads, alpha 0.1, ${R2_MIN} min"
./bin/train --threads "$FULL_THREADS" --alpha 0.1 --time-min "$R2_MIN" \
  --resume runs/phase2_start.bin \
  --out runs/phase2.bin --log runs/train_log_p2.csv --ckpt-min 20 \
  >> runs/pipeline.log 2>&1
say "round 2 exit=$?"

touch runs/final_best.stop; sleep 3; wait "$KEEPER" 2>/dev/null; rm -f runs/final_best.stop

# Polish resumes the best of round 2 for the same reason round 1 did.
P3SRC=runs/final_best.bin; [ -f "$P3SRC" ] || P3SRC=runs/phase2.bin
GAMES=600 THREADS=3 INTERVAL=120 ./scripts/keep_arm.sh runs/model.bin runs/model_best &
KEEPER=$!

say "round 3: polish, alpha 0.05, ${R3_MIN} min, from $P3SRC"
./bin/train --threads "$FULL_THREADS" --alpha 0.05 --time-min "$R3_MIN" \
  --resume "$P3SRC" \
  --out runs/model.bin --log runs/train_log_p3.csv --ckpt-min 15 \
  >> runs/pipeline.log 2>&1
say "round 3 exit=$?"

touch runs/model_best.stop; sleep 3; wait "$KEEPER" 2>/dev/null

FINAL=runs/model_best.bin; [ -f "$FINAL" ] || FINAL=runs/model.bin
cp "$FINAL" runs/final.bin
say "final model: $FINAL -> runs/final.bin"

# ------------------------------------------------------------------ final eval
say "final evaluation"
./bin/eval --model runs/final.bin --games "$JUDGE_GAMES" --threads "$FULL_THREADS" \
  --csv runs/eval_greedy.csv > runs/eval_greedy.txt 2>&1
say "greedy mean = $(awk '/^mean score/{print $3}' runs/eval_greedy.txt)"

# Search costs ~100x per move, so this is a smaller sample by necessity.
./bin/eval --model runs/final.bin --games 400 --threads "$FULL_THREADS" --depth 2 \
  > runs/eval_depth2.txt 2>&1
say "depth-2 mean = $(awk '/^mean score/{print $3}' runs/eval_depth2.txt)"

python3 python/record.py --model runs/final.bin --out web/games.json --scan 400 --keep 6 \
  > runs/record_final.log 2>&1 && say "recorded web/games.json"
python3 python/plot.py runs/train_log.csv -o runs/curve.png >/dev/null 2>&1 || true
python3 python/plot.py "$AB"/single.csv "$AB"/derived.csv "$AB"/classic.csv \
  -o "$AB"/ab_curves.png >/dev/null 2>&1 || true

say "=== NIGHT RUN COMPLETE ==="
echo "PIPELINE COMPLETE" >> runs/pipeline.log
