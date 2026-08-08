#!/usr/bin/env bash
# Keep the A/B inside the workspace quota by dropping checkpoints the keeper has
# already banked.
#
# The quota ceiling is ~21 GiB and a save transiently needs twice the checkpoint
# size (the old file plus its .tmp, until the rename). Three arms holding
# 1.5 + 4.5 + 4.5 GiB steady cannot also write 10.5 GiB of .tmp at once, so
# every save window failed for the two multi-stage arms while the small control
# arm sailed through — silently biasing the comparison in the control's favour.
#
# The raw checkpoint is disposable: the keeper evaluates it and copies the best
# one to local disk, and that copy is what the driver judges and resumes from.
# So once the keeper has recorded a result *newer* than a checkpoint, that
# checkpoint has served its purpose and can go.
#
# Deleting a file another process has open is safe here — eval mmaps it, and an
# unlinked file stays alive for as long as the fd is held.
set -u
cd "$(dirname "$0")/.."
INTERVAL="${INTERVAL:-60}"
STOP=runs/ab/janitor.stop
LOG=runs/ab/janitor.log

rm -f "$STOP"
echo "$(date -Is) janitor started (poll ${INTERVAL}s)" >> "$LOG"

while true; do
  [ -f "$STOP" ] && { echo "$(date -Is) stop file seen" >> "$LOG"; exit 0; }

  for a in single derived classic; do
    b="runs/ab/$a.bin"
    h="runs/ab/${a}_best.hist.csv"
    [ -f "$b" ] || continue
    bm=$(stat -c %Y "$b" 2>/dev/null || echo 0)
    hm=$(stat -c %Y "$h" 2>/dev/null || echo 0)
    # Keeper wrote a result after this checkpoint appeared, so it has been seen.
    if [ "$hm" -gt "$bm" ]; then
      sz=$(stat -c %s "$b" 2>/dev/null || echo 0)
      rm -f "$b" && echo "$(date -Is) reclaimed $b ($(( sz / 1073741824 )) GiB; keeper banked it)" >> "$LOG"
    fi
  done

  sleep "$INTERVAL"
done
