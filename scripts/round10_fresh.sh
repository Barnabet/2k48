#!/usr/bin/env bash
# Round 10: the synthesis run — everything learned, retrained from scratch.
#
# Rounds 1-9 were single-variable A/Bs on a shared lineage; that discipline is
# what made their verdicts readable, but it means the shipped model still
# carries its history. TC annealing is one-way: weights that oscillated early
# keep permanently tiny step sizes, locked around decisions made when the
# network was weak. The widened arm trains its new tables against those
# semi-frozen ones; a fresh net optimises all sixteen patterns jointly, and
# restart training from episode one shapes the whole value function around
# late-game truth rather than a few hours of patching at the end.
#
# Config = the sum of every measured verdict: 16 patterns (round 8), TC on
# (phase 1), alpha 0.1 (rounds 4-6: flat 0.1-0.15, worse at 0.05), single
# stage (rounds 1 and 7), restart training (round 9). The pool can be seeded
# from round 9's harvest so the fresh run oversamples the late game before its
# own games get there. The fresh lane gets every core: the lineage's marginal
# gain rate is already measured (~650/hour and falling), so the interesting
# counterfactual is not "lineage + 12 more hours" but "recipe from zero".
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-720}"
THREADS="${THREADS:-14}"
GAMES="${GAMES:-20000}"
SEED="${SEED:-31}"
FRAC="${FRAC:-0.5}"
TILE="${TILE:-14}"
LOG=runs/round10.log

say() { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -e runs/round9_final.bin ]; then
  cp runs/final.bin runs/round9_final.bin.tmp && mv runs/round9_final.bin.tmp runs/round9_final.bin
fi
say "incumbent preserved: runs/round9_final.bin"

# Head start for the harvest: round 9's pool holds boards the incumbent
# reached; a fresh net can train on their continuations from minute one.
if [ -f runs/r9_pool.bin ] && [ ! -f runs/r10_pool.bin ]; then
  cp runs/r9_pool.bin runs/r10_pool.bin
  say "restart pool seeded from round 9 ($(stat -c%s runs/r10_pool.bin) bytes)"
fi

say "round 10: fresh 16-pattern + restart(frac=$FRAC tile=$TILE), $THREADS threads, $MIN min"

GAMES=600 THREADS=1 INTERVAL=300 ./scripts/keep_arm.sh runs/r10_fresh.bin runs/r10_fresh_best \
  > runs/r10_fresh.keeper.out 2>&1 &
say "keeper pid $!"

./bin/train --threads "$THREADS" --alpha 0.1 --time-min "$MIN" \
  --extend-patterns --restart-frac "$FRAC" --restart-tile "$TILE" --restart-pool runs/r10_pool.bin \
  --seed 1001 --out runs/r10_fresh.bin --log runs/r10_fresh.csv --ckpt-min 30 \
  > runs/r10_fresh.out 2>&1
say "trainer exit=$? episodes: $(tail -1 runs/r10_fresh.csv 2>/dev/null | cut -d, -f1)"

touch runs/r10_fresh_best.stop; sleep 5

CANDIDATES="runs/round9_final.bin runs/r10_fresh.bin runs/r10_fresh_best.bin" \
  SEED="$SEED" OUT=runs/round10_selection.txt ./scripts/final_select.sh "$GAMES" >> "$LOG" 2>&1
say "selection exit=$?"
say "=== ROUND 10 COMPLETE ==="
