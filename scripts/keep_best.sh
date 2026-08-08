#!/usr/bin/env bash
# Retains the best checkpoint seen during a training run, and keeps every
# evaluation it performs.
#
# The trainer overwrites a single path every 20 minutes, so a late dip in
# quality silently replaces a better earlier model. This watches whichever
# checkpoint the pipeline is currently writing, evaluates each new version,
# keeps a copy of the best as runs/best.bin, and records the full result.
#
# Outputs:
#   runs/best.bin            best checkpoint by mean score
#   runs/best.json           its metrics
#   runs/keeper.log          one line per decision
#   runs/evals/<stamp>.txt   full eval output, one file per checkpoint
#   runs/eval_history.csv    parsed time series of independently measured quality
#
# Cost is ~10s of a few niced cores every 20 minutes.
set -u
cd "$(dirname "$0")/.."

GAMES="${GAMES:-500}"
THREADS="${THREADS:-3}"
INTERVAL="${INTERVAL:-60}"
WATCH="${WATCH:-runs/phase1.bin runs/phase2.bin runs/model.bin}"
BEST=runs/best.bin
META=runs/best.json
LOG=runs/keeper.log
EVALDIR=runs/evals
HIST=runs/eval_history.csv

mkdir -p "$EVALDIR"
log() { echo "$(date -Is) $*" >> "$LOG"; }

if [ ! -f "$HIST" ]; then
  echo "timestamp,source,games,mean,median,p05,p95,max,mean_moves,r2048,r4096,r8192,r16384,r32768,kept" > "$HIST"
fi

best_mean=-1
if [ -f "$META" ]; then
  # Resume rather than re-keeping a worse model after a restart.
  best_mean=$(grep -oE '"mean":[0-9.]+' "$META" | head -1 | cut -d: -f2)
  best_mean=${best_mean:--1}
  log "resuming with existing best mean=$best_mean"
fi

last_sig=""
log "keeper started: $GAMES games, $THREADS threads, poll ${INTERVAL}s"

# Reach rate for a tile, percent sign stripped. Reads $out from the caller.
rate() { echo "$out" | awk -v t="$1" '$1==t{gsub(/%/,"",$2); print $2; exit}'; }

while true; do
  if grep -q "PIPELINE COMPLETE" runs/pipeline.log 2>/dev/null; then
    log "pipeline complete; keeper exiting"
    exit 0
  fi

  # Follow whichever watched checkpoint was written most recently, so the
  # keeper tracks the pipeline across phase boundaries by itself.
  current=""
  newest=0
  for f in $WATCH; do
    [ -f "$f" ] || continue
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if [ "$m" -gt "$newest" ]; then newest=$m; current=$f; fi
  done
  if [ -z "$current" ]; then sleep "$INTERVAL"; continue; fi

  sig="$current:$(stat -c %Y-%s "$current" 2>/dev/null)"
  if [ "$sig" = "$last_sig" ]; then sleep "$INTERVAL"; continue; fi

  # Checkpoints are written as .tmp then renamed, so any file we can open is
  # complete, and an open fd survives the next rename.
  out=$(nice -n 15 ./bin/eval --model "$current" --games "$GAMES" --threads "$THREADS" 2>&1)
  mean=$(echo "$out" | awk '/^mean score/{print $3}')

  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  name="$EVALDIR/$stamp-$(basename "$current" .bin)"
  if [ -z "$mean" ]; then
    echo "$out" > "$name-FAILED.txt"
    log "eval FAILED for $current — output in $name-FAILED.txt"
    sleep "$INTERVAL"
    continue
  fi
  echo "$out" > "$name.txt"
  last_sig="$sig"

  median=$(echo "$out" | awk '/^median score/{print $3}')
  p05=$(echo "$out"    | awk '/^ *5th pct/{print $3}')
  p95=$(echo "$out"    | awk '/^ *95th pct/{print $3}')
  maxs=$(echo "$out"   | awk '/^max score/{print $3}')
  mvs=$(echo "$out"    | awk '/^mean moves/{print $3}')

  better=$(awk -v a="$mean" -v b="$best_mean" 'BEGIN{print (a>b)?1:0}')
  if [ "$better" = "1" ]; then
    # Copy then rename, so a kill mid-copy cannot leave a corrupt best.bin.
    cp "$current" "$BEST.tmp" && mv "$BEST.tmp" "$BEST"
    cat > "$META" <<EOF
{"mean":$mean,"median":${median:-0},"r2048":"$(rate 2048)%","r8192":"$(rate 8192)%","r16384":"$(rate 16384)%","source":"$current","eval":"$name.txt","saved":"$(date -Is)"}
EOF
    log "NEW BEST mean=$mean (was $best_mean) from $current  2048=$(rate 2048)% 8192=$(rate 8192)% 16384=$(rate 16384)%"
    best_mean=$mean
    kept=1
  else
    log "kept existing best mean=$best_mean; $current scored $mean"
    kept=0
  fi

  echo "$(date -Is),$current,$GAMES,$mean,${median:-},${p05:-},${p95:-},${maxs:-},${mvs:-},$(rate 2048),$(rate 4096),$(rate 8192),$(rate 16384),$(rate 32768),$kept" >> "$HIST"

  sleep "$INTERVAL"
done
