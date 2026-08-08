#!/usr/bin/env bash
# Pick the final model by a large head-to-head, not by the keeper's choice.
#
# The keeper banks the maximum over many 600-game evaluations, each with a
# standard error near 3,600. Taking a maximum over noisy draws is a selection
# bias: it reliably picks a luckily-*measured* checkpoint rather than a better
# one, and the winner's recorded score is inflated by roughly a standard
# deviation. That is harmless for an A/B where every arm goes through the same
# procedure, but it is the wrong way to choose what actually ships.
#
# So evaluate every candidate again at 20k games — standard error ~600 — and
# take the best. Measured once, honestly, on equal footing.
#
#   scripts/final_select.sh [games]
set -u
cd "$(dirname "$0")/.."

GAMES="${1:-20000}"
THREADS="${THREADS:-14}"
# bin/eval is deterministic: same seed, same games, same score to the digit.
# That makes a re-run reproducible, but it also means repeatedly selecting the
# best of N candidates on the default seed 1 fits that particular set of 20,000
# games. Later rounds pass a seed the candidates have not been scored on.
SEED="${SEED:-1}"
OUT="${OUT:-runs/final_selection.txt}"

# Every plausible endpoint: the latest round's best and last, and the previous
# round's best in case the lower-alpha polish made things worse rather than
# better. Overridable so later rounds can add their own candidates without
# editing the decision rule they will be judged by.
CANDIDATES="${CANDIDATES:-runs/model_best.bin runs/model.bin runs/final_best.bin runs/phase2.bin}"

: > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }

say "final selection — $GAMES games per candidate, seed $SEED, $(date -Is)"
say ""
printf "%-26s %10s %8s %10s %9s %9s\n" candidate mean "+/-" median 16384 32768 | tee -a "$OUT"

best_mean=-1; best_model=""
for m in $CANDIDATES; do
  [ -f "$m" ] || continue
  # Resolve symlinks so two names for one file are not evaluated twice.
  real=$(readlink -f "$m")
  case " ${seen:-} " in *" $real "*) continue ;; esac
  seen="${seen:-} $real"

  o=$(./bin/eval --model "$m" --games "$GAMES" --threads "$THREADS" --seed "$SEED" 2>&1)
  mean=$(echo "$o" | awk '/^mean score/{print $3}')
  [ -n "$mean" ] || { say "  $m: EVAL FAILED"; continue; }
  med=$(echo "$o"  | awk '/^median score/{print $3}')
  p05=$(echo "$o"  | awk '/^ *5th pct/{print $3}')
  p95=$(echo "$o"  | awk '/^ *95th pct/{print $3}')
  r16=$(echo "$o"  | awk '$1==16384{gsub(/%/,"",$2); print $2}')
  r32=$(echo "$o"  | awk '$1==32768{gsub(/%/,"",$2); print $2}')
  se=$(awk -v a="$p05" -v b="$p95" -v n="$GAMES" 'BEGIN{printf "%.0f",(b-a)/3.2897/sqrt(n)}')
  printf "%-26s %10s %8s %10s %8s%% %8s%%\n" "$(basename "$m")" "$mean" "$se" "$med" "$r16" "$r32" | tee -a "$OUT"
  echo "$o" > "runs/final_eval_$(basename "$m" .bin).txt"

  if [ "$(awk -v x="$mean" -v y="$best_mean" 'BEGIN{print (x>y)?1:0}')" = 1 ]; then
    best_mean=$mean; best_model=$m; best_se=$se
  fi
done

say ""
if [ -z "$best_model" ]; then
  say "no candidate evaluated successfully"
  exit 1
fi
say "SELECTED: $best_model  mean=$best_mean +/-${best_se:-?}"
say ""
say "Differences under ~2x the pooled standard error are not real; where"
say "candidates tie, the earlier/simpler one is the safer choice."

cp "$best_model" runs/final.bin.tmp && mv runs/final.bin.tmp runs/final.bin
say "wrote runs/final.bin"
