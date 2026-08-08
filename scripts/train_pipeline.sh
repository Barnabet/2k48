#!/usr/bin/env bash
# Full staged training pipeline.
#
#   Phase 1  single-stage TD(0) at the base step size — learns the bulk of the
#            value function, from an empty board to a strong mid-game.
#   Phase 2  multi-stage fine-tuning at a reduced step size — each stage
#            specialises on one phase of play, seeded from the phase 1 weights.
#   Phase 3  low step size polish — anneals into a stable final policy.
#
# Each phase resumes from the previous checkpoint, so the pipeline can be
# interrupted and restarted at a phase boundary. Ctrl-C during a phase still
# writes a usable checkpoint.
set -u
cd "$(dirname "$0")/.."

THREADS="${THREADS:-14}"
P1_MIN="${P1_MIN:-300}"      # 5 h
P2_MIN="${P2_MIN:-180}"      # 3 h
P3_MIN="${P3_MIN:-60}"       # 1 h
# Step sizes come from scripts/sweep_alpha.sh: 0.1 was decisively best, 0.2 was
# unstable (mean |TD error| ~10x higher) and 0.5 diverged outright.
P1_ALPHA="${P1_ALPHA:-0.1}"
P2_ALPHA="${P2_ALPHA:-0.1}"
P3_ALPHA="${P3_ALPHA:-0.05}"
# Stage boundaries separate game *phases*, not tile magnitudes. Measured on an
# unbiased 30-game sample (225k moves, python/stage_mass.py):
#
#   regime    board sum   2nd tile   V bias
#     2,048       3,528        808  -17,079
#     4,096       6,326      1,436  +10,242   <- value bias flips sign
#     8,192      11,997      2,540  +39,099
#    16,384      21,085      2,977  +45,167   <- 2nd tile stops scaling
#
# 4096 -> 8192 is a smooth doubling of every statistic: the same position at a
# larger scale, so splitting there spends a whole weight table on a distinction
# the value function does not need. 16384 is genuinely different — board sum
# doubles while the second tile barely moves, i.e. a huge dead tile with small
# tiles rebuilding around it.
#
# Stage masses: 26.9% / 62.5% / 10.6% of moves.
#
# 32768 cannot be a boundary: it holds ~0.01% of moves, so that stage would
# never accumulate enough updates to beat the table it was split out of.
STAGES="${STAGES:-12,14}"

mkdir -p runs

run_phase() {
  local name="$1"; shift
  echo ""
  echo "================ $name ================"
  date -Is
  ./bin/train "$@"
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then
    echo "ABORTING: $name failed"
    exit $rc
  fi
}

if [ ! -f runs/phase1.bin ]; then
  run_phase "phase 1: single stage, alpha=$P1_ALPHA, ${P1_MIN}min" \
    --threads "$THREADS" --alpha "$P1_ALPHA" --time-min "$P1_MIN" \
    --out runs/phase1.bin --log runs/train_log.csv --ckpt-min 20
else
  echo "runs/phase1.bin exists, skipping phase 1"
fi

if [ ! -f runs/phase2.bin ]; then
  run_phase "phase 2: stages=$STAGES, alpha=$P2_ALPHA, ${P2_MIN}min" \
    --threads "$THREADS" --alpha "$P2_ALPHA" --time-min "$P2_MIN" \
    --resume runs/phase1.bin --stages "$STAGES" \
    --out runs/phase2.bin --log runs/train_log.csv --ckpt-min 20
else
  echo "runs/phase2.bin exists, skipping phase 2"
fi

run_phase "phase 3: polish, alpha=$P3_ALPHA, ${P3_MIN}min" \
  --threads "$THREADS" --alpha "$P3_ALPHA" --time-min "$P3_MIN" \
  --resume runs/phase2.bin --stages "$STAGES" \
  --out runs/model.bin --log runs/train_log.csv --ckpt-min 20

echo ""
echo "================ evaluation ================"
./bin/eval --model runs/model.bin --games 2000 --threads "$THREADS" \
  --csv runs/eval_greedy.csv | tee runs/eval_greedy.txt

python3 python/plot.py runs/train_log.csv -o runs/curve.png || true
echo "PIPELINE COMPLETE"
date -Is
