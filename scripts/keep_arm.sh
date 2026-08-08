#!/usr/bin/env bash
# Best-checkpoint keeper for a single training arm.
#
# scripts/keep_best.sh follows whichever pipeline checkpoint was written most
# recently and writes to fixed paths, so it cannot supervise several concurrent
# runs. This watches exactly one checkpoint and keeps its own outputs, which is
# what an A/B needs: each arm must be judged on its own best model, not on
# whatever it happened to be doing when the clock ran out.
#
#   scripts/keep_arm.sh runs/ab/multi.bin runs/ab/multi_best
#
# writes <prefix>.bin, <prefix>.json, <prefix>.keeper.log, <prefix>.hist.csv
# and exits when <prefix>.stop appears.
set -u
cd "$(dirname "$0")/.."

WATCH="${1:?usage: keep_arm.sh <checkpoint> <out-prefix>}"
PREFIX="${2:?usage: keep_arm.sh <checkpoint> <out-prefix>}"
GAMES="${GAMES:-400}"
THREADS="${THREADS:-1}"
INTERVAL="${INTERVAL:-120}"
# BANK=0 turns the keeper into a pure failure detector: it still evaluates
# every checkpoint and writes the history/log, but never copies a best model.
# For checkpoints too large to double-buffer anywhere (32-pattern = 6.4 GiB,
# peak 2x during copy), banking is not insurance we can afford — and across
# rounds 8-11 a banked best has never once out-judged the final checkpoint.
BANK="${BANK:-1}"

# The kept model goes to local disk, not the shared /workspace mount, which is
# under a size quota that three concurrent multi-GB copies blew through — the
# copies died mid-chunk and the trainers' own checkpoint writes started failing
# silently alongside them. Only the small text outputs stay next to the run.
# Callers that need <prefix>.bin at the usual path get a symlink to BINDIR.
BINDIR="${BINDIR:-/tmp/2m48}"
mkdir -p "$BINDIR"
BEST="$BINDIR/$(basename "$PREFIX").bin"
META="$PREFIX.json"
LOG="$PREFIX.keeper.log"
HIST="$PREFIX.hist.csv"
STOP="$PREFIX.stop"

log() { echo "$(date -Is) $*" >> "$LOG"; }
rate() { echo "$out" | awk -v t="$1" '$1==t{gsub(/%/,"",$2); print $2; exit}'; }

[ -f "$HIST" ] || echo "timestamp,source,games,mean,median,p05,p95,max,mean_moves,r2048,r4096,r8192,r16384,r32768,kept" > "$HIST"
rm -f "$STOP"

# The judgement looks for <prefix>.bin next to the run. Round 9 relied on the
# caller to make that link, no caller did, and both keeper bests were silently
# skipped at selection — so the keeper now creates it itself, before the first
# bank, where it cannot be forgotten.
[ "$BANK" = "0" ] || ln -sfn "$BEST" "$PREFIX.bin"

# Resume the high-water mark across restarts. Without this a restarted keeper
# starts at -1 and banks whatever it evaluates next, overwriting a better model
# it had already saved.
best_mean=-1
if [ -s "$META" ]; then
  best_mean=$(grep -oE '"mean":[0-9.]+' "$META" | head -1 | cut -d: -f2)
  best_mean=${best_mean:--1}
  log "resuming with stored best mean=$best_mean"
fi
last_sig=""
log "keeper started on $WATCH: $GAMES games, $THREADS threads, poll ${INTERVAL}s"

while true; do
  if [ -f "$STOP" ]; then log "stop file seen; exiting"; exit 0; fi
  if [ ! -f "$WATCH" ]; then sleep "$INTERVAL"; continue; fi

  sig="$(stat -c %Y-%s "$WATCH" 2>/dev/null)"
  if [ "$sig" = "$last_sig" ]; then sleep "$INTERVAL"; continue; fi

  # Checkpoints are written as .tmp then renamed, so anything openable is whole.
  out=$(nice -n 15 ./bin/eval --model "$WATCH" --games "$GAMES" --threads "$THREADS" 2>&1)
  mean=$(echo "$out" | awk '/^mean score/{print $3}')
  if [ -z "$mean" ]; then
    echo "$out" > "$PREFIX-FAILED.txt"
    log "eval FAILED — output in $PREFIX-FAILED.txt"
    sleep "$INTERVAL"; continue
  fi
  last_sig="$sig"

  median=$(echo "$out" | awk '/^median score/{print $3}')
  p05=$(echo "$out"    | awk '/^ *5th pct/{print $3}')
  p95=$(echo "$out"    | awk '/^ *95th pct/{print $3}')
  maxs=$(echo "$out"   | awk '/^max score/{print $3}')
  mvs=$(echo "$out"    | awk '/^mean moves/{print $3}')

  kept=0
  if [ "$BANK" = "0" ]; then
    if [ "$(awk -v a="$mean" -v b="$best_mean" 'BEGIN{print (a>b)?1:0}')" = "1" ]; then
      log "high-water mean=$mean (was $best_mean; banking disabled)"
      best_mean=$mean
    else
      log "checkpoint scored $mean; high-water $best_mean (banking disabled)"
    fi
    echo "$(date -Is),$WATCH,$GAMES,$mean,${median:-},${p05:-},${p95:-},${maxs:-},${mvs:-},$(rate 2048),$(rate 4096),$(rate 8192),$(rate 16384),$(rate 32768),$kept" >> "$HIST"
    continue
  fi
  if [ "$(awk -v a="$mean" -v b="$best_mean" 'BEGIN{print (a>b)?1:0}')" = "1" ]; then
    # Copies are serialised across all arms by a shared lock, and the space
    # check happens *inside* it. The arms checkpoint in the same second, so two
    # 4.5 GiB copies would otherwise race for a disk that fits only one: the
    # loser skipped every window, freezing one arm's stored best while the other
    # advanced. That is a bias in the experiment, not just wasted space.
    #
    # Waiting is correct where skipping was not — the space the loser needs is
    # exactly what the winner releases when it finishes.
    #
    # Chained through: a failed copy must not go on to publish metadata for a
    # best model that is not there. That is what happened when the workspace
    # quota filled — the cp died and the run recorded a new best anyway.
    # Run it, then test the status. Writing this as `if ! ( ... ); then rc=$?`
    # captures the status of the *negation*, which is always 0 in the taken
    # branch — every failure logged an unhelpful "rc=0".
    #
    # flock's stdout goes to /dev/null because it is inherited, and an inherited
    # stdout can be broken in ways that have nothing to do with locking. During
    # round 3 the driver's stdout file was wedged by the quota (writes to it
    # returned EDQUOT permanently); flock writes nothing, but its exit-time
    # stdout flush failed, so it exited nonzero and every single copy was logged
    # as a 20-minute lock timeout that never happened. The keeper banked nothing
    # for the whole round.
    (
      flock -w 1200 200 >/dev/null || exit 9
      need=$(stat -c %s "$WATCH")
      avail=$(df -kP "$BINDIR" | awk 'NR==2{print $4}')
      [ "$(( avail * 1024 ))" -lt "$(( need + 1073741824 ))" ] && exit 8
      cp "$WATCH" "$BEST.tmp" && mv "$BEST.tmp" "$BEST"
    ) 200>"$BINDIR/.copylock" 2>>"$PREFIX.copy.err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      rm -f "$BEST.tmp"
      case $rc in
        9) log "COPY LOCK TIMEOUT (mean=$mean) — best unchanged at $best_mean" ;;
        8) log "SKIP keep (mean=$mean): not enough space in $BINDIR even with the lock held" ;;
        *) log "COPY FAILED rc=$rc (mean=$mean) — best unchanged at $best_mean" ;;
      esac
      # Do not advance last_sig: retry this same checkpoint on the next poll
      # rather than waiting 15 minutes for the next one. Deliberately no history
      # row either — the quota janitor reclaims a checkpoint once the history is
      # newer than it, which would delete the very file we intend to retry.
      last_sig=""
      sleep "$INTERVAL"; continue
    fi
    cat > "$META" <<EOF
{"mean":$mean,"median":${median:-0},"r2048":"$(rate 2048)%","r8192":"$(rate 8192)%","r16384":"$(rate 16384)%","source":"$WATCH","saved":"$(date -Is)"}
EOF
    log "NEW BEST mean=$mean (was $best_mean)  2048=$(rate 2048)% 8192=$(rate 8192)% 16384=$(rate 16384)%"
    best_mean=$mean
    kept=1
  else
    log "kept existing best mean=$best_mean; checkpoint scored $mean"
    kept=0
  fi
  echo "$(date -Is),$WATCH,$GAMES,$mean,${median:-},${p05:-},${p95:-},${maxs:-},${mvs:-},$(rate 2048),$(rate 4096),$(rate 8192),$(rate 16384),$(rate 32768),$kept" >> "$HIST"
done
