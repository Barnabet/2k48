#!/usr/bin/env bash
# Final evaluation of a trained model at several search strengths.
#
# Sample sizes shrink as search gets more expensive, so each number is quoted
# with the count it came from — a 30-game depth-3 mean has a wide error bar and
# should not be compared to a 2000-game greedy mean as if both were precise.
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-runs/model.bin}"
THREADS="${THREADS:-14}"
OUT="${OUT:-runs/eval}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL"
  exit 1
fi
mkdir -p "$OUT"

echo "############ greedy 1-ply, 2000 games ############"
./bin/eval --model "$MODEL" --games 2000 --threads "$THREADS" \
  --csv "$OUT/greedy.csv" | tee "$OUT/greedy.txt"

echo ""
echo "############ expectimax depth 2, 300 games ############"
./bin/eval --model "$MODEL" --games 300 --threads "$THREADS" --depth 2 \
  --csv "$OUT/depth2.csv" | tee "$OUT/depth2.txt"

echo ""
echo "############ expectimax depth 3, 60 games ############"
# Depth 3 needs a coarser probability cutoff to stay tractable: cumulative
# branch probability only falls to ~1e-3 by the third chance layer, so a 1e-4
# cutoff prunes almost nothing there.
./bin/eval --model "$MODEL" --games 60 --threads "$THREADS" --depth 3 --cutoff 3e-3 \
  --csv "$OUT/depth3.csv" | tee "$OUT/depth3.txt"

echo ""
echo "EVALUATION COMPLETE -> $OUT"
