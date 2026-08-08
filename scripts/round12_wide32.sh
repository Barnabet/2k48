#!/usr/bin/env bash
# Round 12: 24 -> 32 pattern widening vs a live 24-pattern control.
#
# Third capacity test. 8->16 paid +11.4k (round 8), 16->24 paid +14.6k
# (round 11); the curve has not bent, so keep pulling until it does. The 8
# added shapes are the first batch chosen by exhaustive enumeration: every
# connected 6-cell subset, canonicalised under the 8 board symmetries, novel
# classes only (which is how we learned the round-8 set carries 4 internal
# symmetry-duplicates — see ntuple.hpp). Widening verified bit-exact against
# the champion before launch (identical 200-game evals, seed 99).
#
# Disk: MooseFS quota on runs/ is ~21 GiB and a 32-pattern checkpoint is
# 6.44 GiB with a 2x transient during the atomic save. Placement:
#   runs/  — xz'd archives (~4.5G) + wide32 ckpt: peak ~18G  (margin ~3G)
#   /tmp/2m48 — ctl 24-pat ckpt (4.83G, peak 9.66G): local disk, ~15G free.
# runs/final.bin (4.83G) is DELETED once both arms have loaded it; the
# champion survives as runs/round11_final.bin.xz and is decompressed to /tmp
# for the judgement. ckpt-mins 30 vs 47 + staggered end times keep the two
# arms' save transients from coinciding.
#
# Keepers run BANK=0 (eval-only failure detectors): a 32-pattern best cannot
# double-buffer anywhere, and across rounds 8-11 a banked best never once
# out-judged the final checkpoint.
set -u
cd "$(dirname "$0")/.."

MIN="${MIN:-900}"          # wide arm; ctl ends 15 min earlier
SEED=41                    # judgement seed — never used before (17,19,23,29,31,37 burned)
SRC=runs/final.bin         # round-11 champion (24-pat, 283,313 +/-645 seed 37)
LOG=runs/round12.log
mkdir -p /tmp/2m48

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
say "=== ROUND 12 LAUNCH: 32-pattern widening vs 24-pattern control ==="
say "wide32: ${MIN}min x 7thr from $SRC; ctl: $((MIN-15))min x 7thr"

./bin/train --resume "$SRC" --extend-patterns 32 --threads 7 --alpha 0.1 \
  --time-min "$MIN" --seed 1201 --ckpt-min 30 \
  --out runs/r12_wide32.bin --log runs/r12_wide32.csv \
  > runs/r12_wide32.out 2>&1 &
W_PID=$!

./bin/train --resume "$SRC" --threads 7 --alpha 0.1 \
  --time-min "$((MIN-15))" --seed 1202 --ckpt-min 47 \
  --out /tmp/2m48/r12_ctl.bin --log runs/r12_ctl.csv \
  > runs/r12_ctl.out 2>&1 &
C_PID=$!

# Both arms read the resume checkpoint at startup; once both have printed
# their config line the 4.83G source file is dead weight against the quota.
for f in runs/r12_wide32.out runs/r12_ctl.out; do
  until grep -q "patterns=" "$f" 2>/dev/null; do sleep 5; done
done
grep -q "patterns=32" runs/r12_wide32.out || { say "ABORT: wide arm did not widen"; kill $W_PID $C_PID; exit 1; }
grep -q "patterns=24" runs/r12_ctl.out    || { say "ABORT: ctl arm wrong width";   kill $W_PID $C_PID; exit 1; }
say "both arms up; removing runs/final.bin (champion preserved as round11_final.bin.xz)"
rm -f runs/final.bin

sleep 240   # let first checkpoints appear before keepers start polling
BANK=0 GAMES=600 THREADS=1 INTERVAL=300 ./scripts/keep_arm.sh runs/r12_wide32.bin    runs/r12_wide32_best > /dev/null 2>&1 &
BANK=0 GAMES=600 THREADS=1 INTERVAL=300 ./scripts/keep_arm.sh /tmp/2m48/r12_ctl.bin  runs/r12_ctl_best    > /dev/null 2>&1 &

DEADLINE=$(( $(date +%s) + (MIN + 40) * 60 ))
while kill -0 $W_PID 2>/dev/null || kill -0 $C_PID 2>/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { say "DEADLINE exceeded; continuing to judgement with what exists"; break; }
  sleep 120
done
say "arms finished: wide32 $(tail -c 400 runs/r12_wide32.out | tr '\n' ' ' | tail -c 120)"
touch runs/r12_wide32_best.stop runs/r12_ctl_best.stop
sleep 10

say "restoring incumbent for judgement"
xz -d -T8 -k -c runs/round11_final.bin.xz > /tmp/2m48/round11_final.bin \
  || { say "ABORT: incumbent decompress failed"; exit 1; }

CANDIDATES="/tmp/2m48/round11_final.bin runs/r12_wide32.bin /tmp/2m48/r12_ctl.bin" \
  SEED=$SEED OUT=runs/round12_selection.txt ./scripts/final_select.sh 20000 \
  >> "$LOG" 2>&1
say "selection exit=$?"
rm -f /tmp/2m48/round11_final.bin
say "=== ROUND 12 COMPLETE ==="
