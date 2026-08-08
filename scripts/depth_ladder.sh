#!/usr/bin/env bash
# Measure the shipped model under real search, at a game count that can rank.
#
# Every selection so far judged candidates 1-ply greedy, where the model scores
# ~255k and reaches 32768 in ~0.1% of games. The one search sweep on record
# (runs/search_tune.txt) was 20 games per cell — SE ~24k, unable to rank
# anything — but even so it put depth-2 adaptive at ~363k. Published agents get
# their "consistent 32768" numbers from exactly this: 3-5 ply expectimax over a
# TD-trained n-tuple net. What the search policy is actually worth, and what
# the real 32k rate is, has never been measured here at scale. This does that.
#
# Costs (from the tune's moves/s at 14 threads): depth 1 ~316s/1000 games,
# depth 2 ~21min/400 games, depth 3 ~23min/200 games. SE at 400 games is ~5k —
# coarse but honest, and the tile-rate columns are what this run is for.
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-runs/final.bin}"
THREADS="${THREADS:-14}"
SEED="${SEED:-29}"
OUT="${OUT:-runs/depth_ladder.txt}"
mkdir -p runs/depth_ladder

: > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }
say "depth ladder on $MODEL, seed $SEED, $(date -Is)"
say ""
printf "%-28s %8s %10s %10s %9s %9s %9s\n" config games mean median 8192 16384 32768 | tee -a "$OUT"

run() { # label games extra-args...
  local label="$1" games="$2"; shift 2
  local o
  o=$(./bin/eval --model "$MODEL" --games "$games" --threads "$THREADS" --seed "$SEED" \
        --csv "runs/depth_ladder/${label// /_}.csv" "$@" 2>&1)
  echo "$o" > "runs/depth_ladder/${label// /_}.txt"
  local mean med r8 r16 r32
  mean=$(echo "$o" | awk '/^mean score/{print $3}')
  [ -n "$mean" ] || { say "$label: EVAL FAILED"; return; }
  med=$(echo "$o" | awk '/^median score/{print $3}')
  r8=$(echo "$o"  | awk '$1==8192{gsub(/%/,"",$2); print $2}')
  r16=$(echo "$o" | awk '$1==16384{gsub(/%/,"",$2); print $2}')
  r32=$(echo "$o" | awk '$1==32768{gsub(/%/,"",$2); print $2}')
  printf "%-28s %8s %10s %10s %8s%% %8s%% %8s%%\n" "$label" "$games" "$mean" "$med" "$r8" "$r16" "$r32" | tee -a "$OUT"
}

run "greedy"                  2000
run "d1 c1e-4 adaptive"       1000 --depth 1
run "d2 c1e-4 adaptive"        400 --depth 2
run "d3 c1e-2 adaptive"        200 --depth 3 --cutoff 1e-2

say ""
say "DEPTH LADDER COMPLETE $(date -Is)"
