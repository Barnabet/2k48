#!/usr/bin/env bash
# One compact health line every INTERVAL seconds, for unattended overnight runs.
#
# The milestone monitor only fires when the driver writes a line, so a stall
# between milestones — every trainer dead, the driver wedged, the box out of
# memory — would look exactly like normal quiet progress. This makes silence
# impossible: if the heartbeat stops arriving, something ate the heartbeat too.
set -u
cd "$(dirname "$0")/.."
INTERVAL="${INTERVAL:-1200}"

# Per-arm state as "<name> <keeper mean>(ckpt <age>m)".
#
# Read from the keeper's history, not the trainer's own CSV. When the quota
# filled, the trainers' log streams latched a write error and never recovered:
# they still run, but their CSVs are frozen with a block of NULs at EOF. The
# keeper appends with a fresh fd per row, so it kept working. Its mean is the
# better number anyway — an independent evaluation rather than training
# self-play.
#
# Checkpoint age is the load-bearing signal here. A frozen checkpoint is how a
# quota failure actually presents: training looks perfectly healthy at 400% CPU
# while nothing it produces reaches disk.
arm_state() {
  local h="runs/ab/$1_best.hist.csv" b="runs/ab/$1.bin" m age
  m=$(tr -d '\0' < "$h" 2>/dev/null | tail -1 | awk -F, 'NF>3 && $4+0>0{printf "%.0fk", $4/1000}')
  if [ -f "$b" ]; then age=$(( ( $(date +%s) - $(stat -c %Y "$b") ) / 60 )); else age="-"; fi
  echo "$1 ${m:-?}(ckpt ${age}m)"
}

# Whichever checkpoint the current phase is writing, and the newest keeper
# result for it. WATCH_CKPTS is deliberately a list: rounds 2 and 3 write
# different paths, and the heartbeat should follow the run rather than be
# re-pointed by hand at every phase boundary.
WATCH_CKPTS="${WATCH_CKPTS:-runs/phase2.bin runs/model.bin}"
WATCH_HISTS="${WATCH_HISTS:-runs/final_best.hist.csv runs/model_best.hist.csv}"

# Echoes "<age-in-minutes> <state string>"; age is -1 when nothing exists yet,
# which is normal for the first 20 minutes of a phase and must not alert.
phase_state() {
  local f t nt=0 newest="" m="" v age
  for f in $WATCH_CKPTS; do
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if [ "$t" -gt "$nt" ]; then nt=$t; newest=$f; fi
  done
  for f in $WATCH_HISTS; do
    [ -s "$f" ] || continue
    v=$(tr -d '\0' < "$f" | tail -1 | awk -F, 'NF>3 && $4+0>0{printf "%.0fk", $4/1000}')
    [ -n "$v" ] && m="$v"
  done
  if [ -n "$newest" ]; then age=$(( ( $(date +%s) - nt ) / 60 )); else age=-1; fi
  echo "$age $(basename "${newest:-none}")@${age}m keeper=${m:-?}"
}

# Any arm whose checkpoint has not been rewritten in this long is stuck.
# Checkpoints are every 15 min, so 30 means two consecutive misses.
STALE_MIN="${STALE_MIN:-32}"
stale_arms() {
  local a b age out=""
  for a in single derived classic; do
    b="runs/ab/$a.bin"
    [ -f "$b" ] || continue
    age=$(( ( $(date +%s) - $(stat -c %Y "$b") ) / 60 ))
    [ "$age" -gt "$STALE_MIN" ] && out="$out $a(${age}m)"
  done
  echo "$out"
}

# Free space on both filesystems. The /workspace quota filling is what killed
# the first set of keepers and silently dropped a round of checkpoints, so it
# is a first-class health signal, not a footnote.
# df on /workspace reports the shared MooseFS pool (hundreds of TB free), which
# says nothing about our per-directory quota — that is why the first sign of
# trouble was corrupt files rather than a full disk. Probe by writing instead.
disk() {
  local ours lo probe
  ours=$(du -sk /workspace/2M48 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  lo=$(df -kP /tmp 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')
  if dd if=/dev/zero of=/workspace/2M48/.wprobe bs=1M count=64 status=none 2>/dev/null; then
    probe=ok
  else
    probe=QUOTA-FULL
  fi
  rm -f /workspace/2M48/.wprobe
  echo "proj=${ours}G loc=${lo}G write=$probe"
}

while true; do
  ntrain=$(ps -eo cmd | grep -c '[b]in/train')
  ndriver=$(ps -eo cmd | grep -c '[n]ight_ab.sh')
  nkeep=$(ps -eo cmd | grep -c '[k]eep_arm.sh\|[k]eep_best.sh')
  mem=$(awk '{printf "%.0f", $1/1073741824}' /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)
  load=$(awk '{print $1}' /proc/loadavg)

  stage=$(tail -1 runs/ab/driver.log 2>/dev/null | cut -c27- | cut -c1-46)

  d=$(disk)
  ps_out=$(phase_state)
  ckpt_age=${ps_out%% *}
  ckpt_str=${ps_out#* }

  if [ "$ndriver" -eq 0 ]; then
    echo "HEARTBEAT ALERT: night driver is gone (trainers=$ntrain) — last: $stage"
  elif [ "$ntrain" -eq 0 ] && [ "$nkeep" -eq 0 ]; then
    echo "HEARTBEAT ALERT: nothing training and no keepers — last: $stage"
  elif [ "${d##*write=}" != "ok" ]; then
    echo "HEARTBEAT ALERT: workspace quota full — $d | $stage"
  elif [ "$ckpt_age" -gt "$STALE_MIN" ] 2>/dev/null; then
    echo "HEARTBEAT ALERT: checkpoint frozen — $ckpt_str | $d | $stage"
  else
    echo "HEARTBEAT trainers=$ntrain keepers=$nkeep mem=${mem}G $d | $ckpt_str | $stage"
  fi
  sleep "$INTERVAL"
done
