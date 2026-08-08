#!/usr/bin/env bash
# Low-volume watcher for the training pipeline. Emits one line per phase
# transition, a progress summary every 30 minutes, and any terminal condition.
# Covers the failure paths too — a watcher that only greps for success stays
# silent through a crash, which is indistinguishable from "still running".
set -u
cd "$(dirname "$0")/.."

LOG=runs/pipeline.log
INTERVAL=60
REPORT_EVERY=30      # in units of INTERVAL -> 30 minutes
STALL_TICKS=20       # no episode progress for 20 min -> treat as wedged
tick=0
stalled=0
last_phase=""
last_ep=""

while true; do
  if [ ! -f "$LOG" ]; then
    echo "FAILURE: $LOG does not exist"
    exit 1
  fi

  phase=$(grep -E "^=+ (phase|evaluation)" "$LOG" | tail -1)
  if [ -n "$phase" ] && [ "$phase" != "$last_phase" ]; then
    echo "PHASE: $phase"
    last_phase="$phase"
  fi

  if grep -q "PIPELINE COMPLETE" "$LOG"; then
    echo "PIPELINE COMPLETE — $(grep '^ep ' "$LOG" | tail -1)"
    exit 0
  fi

  fail=$(grep -E "DIVERGED|ABORTING|exit=[1-9]|checkpoint save FAILED|failed to load" "$LOG" | tail -1)
  if [ -n "$fail" ]; then
    echo "FAILURE: $fail"
    exit 1
  fi

  # If neither the driver script nor a trainer is alive, the run ended without
  # printing a completion marker — that is a failure worth waking up for.
  if ! pgrep -f "train_pipeline.sh" >/dev/null 2>&1 && ! pgrep -f "bin/train" >/dev/null 2>&1; then
    if ! grep -q "PIPELINE COMPLETE" "$LOG"; then
      echo "FAILURE: pipeline process vanished; last line: $(tail -1 "$LOG")"
      exit 1
    fi
  fi

  # Stall detection. A crashed trainer is caught above, but a wedged one keeps
  # its process alive and would just re-emit the same progress line forever,
  # which is indistinguishable from healthy. Episodes must keep advancing.
  latest_ep=$(grep '^ep ' "$LOG" | tail -1 | awk '{print $2}')
  latest_ep=${latest_ep:-0}
  if [ "$latest_ep" = "$last_ep" ]; then
    stalled=$((stalled + 1))
  else
    stalled=0
    last_ep="$latest_ep"
  fi
  # Episodes get slower as play improves, and a phase boundary pauses for a
  # multi-GB checkpoint write, so only a long freeze counts.
  if [ "$stalled" -ge "$STALL_TICKS" ]; then
    echo "FAILURE: no new episodes for $((STALL_TICKS * INTERVAL / 60)) min (stuck at ep $last_ep)"
    exit 1
  fi

  if [ $((tick % REPORT_EVERY)) -eq 0 ]; then
    echo "progress: $(grep '^ep ' "$LOG" | tail -1)"
  fi

  tick=$((tick + 1))
  sleep "$INTERVAL"
done
