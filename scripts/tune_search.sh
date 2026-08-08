#!/usr/bin/env bash
# Finds a tractable expectimax configuration by measuring cost and strength
# together. Each config gets a hard timeout, so configs that blow up are
# reported as such instead of hanging the run.
#
# Run this only when training is not using the cores, or the timings are
# meaningless.
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-runs/model.bin}"
GAMES="${GAMES:-8}"
THREADS="${THREADS:-14}"
LIMIT="${LIMIT:-180}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL"
  exit 1
fi

printf "%-8s %-8s %-10s %14s %14s\n" depth cutoff adaptive "mean score" "moves/s"
for depth in 1 2 3; do
  for cutoff in 1e-4 1e-3 3e-3 1e-2; do
    for adaptive in yes no; do
      extra=""
      [ "$adaptive" = "no" ] && extra="--no-adaptive"
      out=$(timeout "$LIMIT" ./bin/eval --model "$MODEL" --games "$GAMES" \
            --threads "$THREADS" --depth "$depth" --cutoff "$cutoff" $extra 2>&1)
      if [ -z "$(echo "$out" | grep 'mean score')" ]; then
        printf "%-8s %-8s %-10s %14s %14s\n" "$depth" "$cutoff" "$adaptive" "timeout>${LIMIT}s" "-"
      else
        mean=$(echo "$out" | awk '/mean score/{print $3}')
        mps=$(echo "$out" | awk '/moves\/s/{print $(NF-1)}')
        printf "%-8s %-8s %-10s %14s %14s\n" "$depth" "$cutoff" "$adaptive" "$mean" "$mps"
      fi
    done
  done
done
echo "TUNE COMPLETE"
