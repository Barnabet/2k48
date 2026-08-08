# Results

Training ran unattended from 2026-08-05 22:00 through 2026-08-07 on a
15.375-core / 57 GiB box. Every score below is from `bin/eval`, which plays
greedy 1-ply unless a depth is stated.

## Headline

The shipped model is `runs/final.bin` — **24×6-tuple network**, one stage,
402,653,184 weights. It began as the standard 8-pattern set, was widened to 16
in round 8 and to 24 in round 11, each time with the trained tables preserved
exactly; the two widenings are the two largest improvements of the project.

Greedy 1-ply, 20,000 games, seed 37:

| | value |
|---|---|
| mean | **283,313 ± 645** |
| median | 326,640 |
| 5th pct | 80,568 |
| 95th pct | 380,792 |
| max | 585,032 |
| mean moves | 10,879 |
| 2048 | 99.33% |
| 4096 | 98.05% |
| 8192 | 94.28% |
| 16384 | 74.22% |
| 32768 | 0.14% |

Under expectimax search (seed 29; the round-9 champion's ladder on the same
seed is kept in `runs/depth_ladder_round9.txt` for comparison):

| policy | games | mean | 16384 | 32768 |
|---|---|---|---|---|
| greedy | 2,000 | 281,848 | 73.9% | 0.1% |
| depth 1 | 1,000 | 340,887 | 92.1% | 1.4% |
| depth 2 | 400 | 361,086 | 95.8% | 3.3% |
| depth 3, cutoff 1e-2 | 200 | **371,277** | **98.0%** | 4.5% |

One ply of search is worth +59k on identical weights; depth 3 adds another 30k
and makes 16384 essentially routine — 98% of depth-3 games reach it, up from
93% on the previous champion. Search costs orders of magnitude per move
(greedy evaluates ~1.2M moves/s across the machine, depth 3 about 1.4k), which
is why the training and selection metric stays the greedy number. The 32768
column is the sobering one: it did not move (4.5% vs the old 5.0% is inside
noise at 200 games, SE ≈ 1.5pp). The round-11 widening lifted everything *up
to* 16384, but 98% of depth-3 games now carry a 16384 and fewer than one in
twenty convert it. The post-16384 phase is where all remaining 32768 progress
lives, and the value function still plays it poorly — consistent 32768s need
a stronger V there, not more plies (Experiment 6 tested and retired the
obvious distribution fix on a weaker base).

## Trajectory

| stage | budget | α | result |
|---|---|---|---|
| phase 1 | overnight warm start | 0.1 | 206,462 (5,000 games) |
| round 1 winner | 150 min × 4 threads | 0.1 | 214,526 ± 659 |
| round 2 | 160 min × 14 threads | 0.1 | 232,885 ± 718 |
| round 3 | 70 min × 14 threads | 0.05 | 234,160 ± 718 |
| round 4 | 150 min × 7 threads | 0.1 | 240,900 ± 733 |
| round 5 | 150 min × 14 threads | 0.1 | 247,814 ± 725 |
| round 6 | 150 min × 7 threads | 0.15 | 251,474 ± 713 |
| round 7 | 240 min × 7 threads | 0.1 | 255,650 ± 641 (seed 17) |
| round 8 | 240 min × 7 threads, **16 patterns** | 0.1 | 267,318 ± 640 (seed 19) |
| round 9 | 240 min × 7 threads | 0.1 | 270,284 ± 638 (seed 23) |
| round 10 | fresh retrain challenge | 0.1 | champion held, 268,124 vs 222,865 (seed 31) |
| round 11 | 735 min × 7 threads, **24 patterns** | 0.1 | **283,313 ± 645** (seed 37) |

Each round is judged on a fresh seed, so the trajectory is not one model
re-measured on friendlier games; cross-seed checks in rounds 4 and 7 put
seed-to-seed drift inside one SE.

Round 2 gained 18,359 in 160 minutes (115/min). Round 3 gained 1,275 ± 1,015 in
70 minutes (18/min) — **not significant at z = 1.26**. The polish phase did not
measurably help, which is what round 4 was built to explain: it turned out the
alpha drop was the cause, and reverting to 0.1 recovered 6,740 on *half* the
threads. Round 3's annealing schedule was simply premature.

Rounds 4 and 5 also bound the return on parallelism. Both ran 150 minutes at
α=0.1 from the then-current champion, and both gained essentially the same
amount — +7,896 on 7 threads, +7,840 on 14. Doubling the workers bought nothing.
That is either the model approaching saturation or Hogwild's lock-free updates
losing value to staleness at this width; these runs cannot separate the two, and
distinguishing them would need a thread-count sweep at fixed wall-clock.

## Experiment 1 — does splitting the network into stages pay?

Three arms resumed from one common ancestor, equal thread-time (150 min ×
4 threads), differing only in where a new stage begins.

| arm | stage boundaries | mean (20,000 games) | 16384 |
|---|---|---|---|
| single | none | **214,526 ± 659** | 44.35% |
| derived | 4096 / 16384 | 213,855 ± 676 | 44.88% |
| classic | 2048 / 8192 | 212,906 ± 676 | 44.02% |

**Null result.** Maximum separation is 1,620 against a measured SE of the
difference of 1,015 → z = 1.60, p ≈ 0.11. Splitting bought nothing here, and the
two boundary choices are indistinguishable from each other.

Scope: this says splitting does not pay *within 150 minutes of fine-tuning on an
already-warm table*. A fresh multi-stage run trained from scratch is a different
experiment and is not addressed.

The interesting part is where the arms differed at all. Medians agree within 164
points and 95th percentiles within 428 — but the 5th percentile is 60,348
(single) vs 52,220 and 52,040. Both split arms were independently worse in their
bad games, costing ~8,000 points of left tail while the typical and good games
were unchanged. A stage boundary means the weights above it are trained on
strictly fewer episodes; the games that suffer are the ones that get there early
and then need well-trained weights to survive.

## Experiment 2 — was the round-3 slowdown alpha, or saturation?

Two arms from the same checkpoint, 150 min × 7 threads, differing only in alpha
(0.1 vs 0.05). Judged at 20,000 games on **seed 7** against the preserved
incumbent, so "more training made it worse" stayed an available outcome.

| candidate | α | mean (20,000 games, seed 7) | median | 16384 |
|---|---|---|---|---|
| round3_final (incumbent) | — | 233,798 ± 677 | 246,020 | 53.08% |
| **r4_hi_best** | **0.1** | **242,091 ± 670** | 270,200 | 56.88% |
| r4_hi (last) | 0.1 | 241,694 ± 666 | 269,660 | 56.58% |
| r4_lo (last) | 0.05 | 237,888 ± 677 | 261,340 | 55.07% |
| r4_lo_best | 0.05 | 235,635 ± 677 | 248,484 | 54.20% |

**It was the alpha, not saturation.** Comparing last checkpoint to last
checkpoint, α=0.1 beats α=0.05 by 3,806 against SE_diff ≈ 1,015 — z = 3.75. Both
arms beat the incumbent, so the model was never saturated; round 3's flat result
was caused by annealing to 0.05 too early, and that decision cost real progress.

A useful cross-check falls out of the fresh seed: the incumbent scored 233,798 at
seed 7 against 234,160 at seed 1, a gap of 362 well inside one SE. Changing the
game set did not shift the scale, so scores across the two seeds are comparable.

The arms also make a good case study in not reading noise. At the 93-minute mark
α=0.1 had climbed 228,203 → 238,759 while α=0.05 had descended 239,908 → 234,745
across six consecutive readings — a near-monotone decline that looks compelling
and was entirely spurious. Each of those readings is 600 games, SE ≈ 4,100, and
the very next checkpoint had α=0.05 posting its best score of the round. The
600-game keeper series is for *catching failures*, not for ranking arms; only the
20,000-game judgement decides.

## Experiment 3 — is α=0.1 the ceiling, and is there anything left to gain?

Round 4 showed 0.05 < 0.1 and left the other direction untested. Two 7-thread
arms from the round-5 champion, 150 min, α=0.1 against α=0.15, judged at 20,000
games on **seed 13**.

| candidate | α | mean (seed 13) | median | 16384 |
|---|---|---|---|---|
| round5_final (incumbent) | — | 249,909 ± 650 | 281,564 | 61.05% |
| **r6_a15_best** | 0.15 | **252,313 ± 648** | 285,520 | 62.51% |
| r6_a15 (last) | 0.15 | 252,135 ± 642 | 285,072 | 62.34% |
| r6_a10 (last) | 0.1 | 251,776 ± 649 | 285,372 | 62.37% |
| r6_a10_best | 0.1 | 250,102 ± 649 | 283,676 | 61.28% |

**α=0.15 is not better than α=0.1** — 359 apart at z = 0.35. With round 4's
result on the other side, the alpha response is flat across 0.1–0.15 and falls
off by 0.05. There is nothing further to win by tuning it. α=0.15 also never
destabilised; no non-finite TD error, both arms ran the full 150 minutes.

**The more important result is the slowdown.** Round 6's α=0.1 arm ran the
identical configuration to round 4's winning arm — 7 threads, 150 minutes, same
alpha — and gained a quarter as much:

| round | config | gain over its incumbent |
|---|---|---|
| round 4 | 150 min × 7 threads, α=0.1 | +7,896 |
| round 5 | 150 min × 14 threads, α=0.1 | +7,840 |
| round 6 | 150 min × 7 threads, α=0.1 | +1,867 |

This is the saturation signal that round 5's keeper series falsely suggested and
then withdrew — except this one is measured at 20,000 games per candidate rather
than 600, so it is a real bend in the curve rather than a noisy draw. Together
with the flat alpha response and the flat thread-count response, all three knobs
this pipeline exposes are now at their plateau. Further gains need a different
architecture — a larger or differently-shaped tuple set — not more of this run.

**A caveat on the headline number.** `r6_a15_best` was chosen as the maximum of
four challengers, and a max over four draws at SE 648 is inflated by roughly 670
in expectation. The honest gain over round 5 is therefore nearer 1,700 (z ≈ 1.7)
than the nominal 2,404 — marginal rather than solid. It was shipped because the
α=0.15 arm beat the incumbent at *both* of its endpoints, which is more robust
than a single lucky pick, not because the maximum cleared a threshold.

## Experiment 4 — multi-stage, re-tested where it should have worked

Round 1's multi-stage null had an excuse: the base model reached 16384 in only
44% of games, so a stage split at 16384 was starving its top table. Round 7
removed the excuse — the base reached 8192 in 90.7% and 16384 in 62% — and ran
stages 8192/16384 against a live single-stage control, 240 min × 7 threads
each, judged at 20,000 games on seed 17.

| arm | mean (seed 17) | 16384 | 32768 |
|---|---|---|---|
| multi-stage (8192/16384) | 253,033 ± 632 | 64.28% | 0.04% |
| **single-stage control** | **255,650 ± 641** | 64.30% | 0.09% |

**Null again, and this time without the alibi** — 16 points of separation
between the arms' means would be nothing, but the control actually *beat* the
multi-stage arm by 2,617 (z ≈ 2.9), and multi-stage halved the 32768 rate.
Splitting the same data distribution across more parameters trains each
parameter on less. The published multi-stage gains bundle the split with
restart-style oversampling of late-game states; the split alone is worthless
here (and Experiment 6 shows the oversampling alone is too).

## Experiment 5 — capacity: 16 patterns vs 8 (the one that worked)

Every tuning knob had measured flat, and same-config training gains had decayed
from +7,896 (round 4) to +1,867 (round 6). Round 8 widened the architecture
instead: the 8 trained tables were prefix-copied into a 16-pattern network (the
default set is a prefix of the extended set, so V is preserved exactly — the new
tables start at zero and contribute nothing until they learn). The widened arm
ran against an 8-pattern control, equal wall-clock from the same checkpoint,
240 min × 7 threads, judged at 20,000 games on seed 19.

| candidate | mean (seed 19) | 16384 |
|---|---|---|
| incumbent (round 7) | 255,962 ± 641 | 64.44% |
| 8-pattern control | 258,277 ± 640 | 65.83% |
| **16-pattern widened** | **267,318 ± 640** | **68.72%** |

**+9,041 over the live control (z ≈ 10) and +11,356 over the incumbent — the
largest gain since round 2.** The widened arm did this on *half the samples*:
128 feature lookups per board instead of 64 cost it 347k episodes against the
control's 732k. Capacity was the binding constraint all along, which is exactly
why alpha, thread count, and stages had all measured flat: the old network had
nowhere to put what more training taught it. Both widened endpoints (final
checkpoint and keeper best) landed within 300 points of each other, so the win
is the arm's, not one lucky checkpoint's.

## Experiment 6 — late-game restart training: a clean null on its own metric

Only ~65–70% of self-play games reach 16384, so the weights that decide the
16384 → 32768 transition see a starved slice of the data. Published agents that
reach 32768 consistently oversample the late game (carousel shaping, stage
splits that restart from saved boards). Round 9 tested the mechanism directly:
the trainer harvests afterstates the first time an episode reaches 16384 and
starts half of all episodes from the pool (`--restart-frac 0.5
--restart-tile 14`). Restart arm vs plain continuation, 240 min × 7 threads,
judged at 20,000 games on seed 23.

| arm | mean (seed 23) | 16384 | 32768 |
|---|---|---|---|
| restart (frac 0.5) | 265,330 ± 640 | 67.89% | 0.04% |
| **control** | **269,231 ± 639** | 69.42% | 0.10% |

The restart arm ran 230,653 restart episodes (804 of which reached 32768 during
training — far more late-game experience than the control ever saw) and still
lost by 3,901 (z ≈ 4.3). But greedy fresh-start mean is the metric restart
training is *expected* to sacrifice — halving the fresh-episode budget must
cost some early/mid-game accuracy — so the deciding measurement was made on its
target metric: 1,000-game depth-1 search evals of both arms.

| arm | depth-1 mean | 16384 | 32768 |
|---|---|---|---|
| restart | 335,316 | 91.1% | 1.7% |
| control | 335,180 | 91.4% | 1.3% |

**Identical.** 17 vs 13 games in 1,000 reaching 32768 is well inside noise, and
the means agree to 0.04%. The 4k greedy deficit vanishes under search — restart
training reshaped the value function without improving it anywhere. The
oversampled experience updated the same shared tables that ordinary play
already updates through bootstrapping, and bought nothing at 2× the plumbing.
Mechanism retired on evidence from its own home ground.

## Experiment 7 — was the lineage's history holding it back? (round 10)

The champion's 16-pattern network carries visible scar tissue: TC step sizes
anneal one way and never recover, the first 8 tables trained alone for 6.8M
episodes before the second 8 were bolted on, and every early decision was made
by a much weaker value function. Round 10 asked whether any of that matters, by
training a fresh 16-pattern network from zero with everything the lineage
learned along the way — joint training of all 24 tables' worth of capacity from
episode one, α=0.1, TC on, 14 threads for 12 hours. The bar was deliberately
asymmetric: the fresh run got 2.9M episodes against the lineage's 7.1M, so even
a tie would have proven the history was dead weight.

| candidate | mean (seed 31) | 16384 | 32768 |
|---|---|---|---|
| **champion (lineage)** | **268,124 ± 640** | 69.02% | 0.09% |
| fresh retrain | 222,865 ± 679 | 50.12% | 0.03% |
| fresh keeper best | 219,277 ± 679 | 48.66% | 0.02% |

It was not a tie. The more interesting number is the *equal-episode* comparison
(`python/compare_curve.py`): at 1.7M episodes the fresh run led the lineage's
own curve by +7,663 and converted 8192s into 16384s at a visibly higher rate —
joint training from scratch genuinely is more episode-efficient early. But the
lead *decayed* as training went on, down to +2,919 at 2.9M episodes, which is
the signature of two runs converging to the same place rather than one pulling
away. Extrapolating, the fresh run would have needed the lineage's full budget
just to catch up, with no evidence it would pass. The lineage's scar tissue is
real but cheap, and round 8's prefix widening preserved essentially everything
that mattered. Fresh retrains retired.

## Experiment 8 — capacity again: 24 patterns vs 16 (round 11)

Round 8 established that widening pays; round 11 asked whether it *keeps*
paying. Eight new 6-tuple patterns were added to the sixteen (each candidate
checked against all 8 board symmetries of every existing pattern — a third of
hand-picked candidates turned out to be symmetry-duplicates that would have
added exactly nothing), the champion's tables were prefix-copied bit-exactly,
and the widened network trained 735 minutes against a live 16-pattern control
continuing from the same checkpoint.

| candidate | mean (seed 37) | median | 16384 | 32768 |
|---|---|---|---|---|
| incumbent (round 9) | 268,726 ± 640 | 290,508 | 69.45% | 0.07% |
| 16-pattern control | 271,828 ± 639 | 290,596 | 71.18% | 0.07% |
| **24-pattern widened** | **283,313 ± 645** | **326,640** | **74.22%** | **0.14%** |

The second widening paid *more* than the first: +14,587 over the incumbent
(round 8's was +11,356) and +11,485 over the live control, on 36% fewer
episodes than the control ran (618k vs 968k — 24 patterns cost 192 table
lookups per board instead of 128). The control's own +3,102 shows what twelve
more hours at fixed capacity buys; widening bought 4.7× that. The median tells
the sharpest story: +36k, meaning the *typical* game improved, not just the
tail. The 32768 rate doubled.

## Where the remaining gap lives

After rounds 7–11 the map is: capacity pays and keeps paying at an undiminished
rate (Experiments 5 and 8), the distribution tricks do not (Experiments 4 and
6), history is not the problem (Experiment 7), and search converts
value-function quality into score at a steep but known rate (the depth ladder
above). Two widenings in, the capacity curve has not bent — 8→16 gained +11k,
16→24 gained +14.6k — so the next question is where it does. Round 12
(24 → 32 patterns, same prefix-copy recipe) is the obvious next experiment;
the practical constraints are now storage and speed rather than ideas, since a
32-pattern checkpoint is 6.4 GB and each widening cuts episodes/hour by
another quarter.

## Methodology notes

Six things I got wrong or nearly wrong, since they change how the numbers read.

**The round-8 pattern set contains four symmetry-duplicates.** Every pattern
is expanded over all 8 board symmetries with shared weights, so two patterns
that are mirrors or rotations of each other span the same function space —
their two tables sum into one effective table, adding lookup cost and zero
capacity. Nobody checked this when the extended set was hand-designed for
round 8; the check was only built for round 11's additions, and only round
12's exhaustive enumeration ran it against the *existing* set:
`{0,4,8,12,13,14}`, `{1,2,5,6,9,10}`, `{0,1,4,5,8,9}` and `{2,3,6,7,10,11}`
are all images of earlier patterns (the last two are both images of the same
2x3 corner block — that shape is effectively in the network three times). The
"16-pattern" round-8 network therefore spans 12 distinct classes and the
"24-pattern" champion 20 — which means round 8's +11k came from just 4 novel
shapes and round 11's +14.6k from 8, and roughly a sixth of every training
and evaluation cycle is spent maintaining redundant tables. The prefix
contract makes this unfixable in place (removing a table shifts every offset
behind it); the practical lesson is that generated-and-checked beats
hand-picked, and round 12's additions were chosen that way.

**A missing symlink silently dropped candidates from a judgement.** The keeper
banks its best checkpoint to local disk and the selection script looks for
`<prefix>.bin` next to the run; the link between the two was created by hand in
rounds 4–8 and forgotten in round 9, so `final_select.sh`'s `[ -f ] || continue`
skipped both keeper bests without a word — including the eventual champion,
which had banked 4k above everything the judgement did see. The fix is
structural, not procedural: `keep_arm.sh` now creates the symlink itself,
before the first bank, where no caller can forget it. Selection scripts that
silently `continue` past missing candidates earn their silence only if nothing
upstream can quietly fail to produce the file.

**`bin/eval` is deterministic.** `--seed` defaults to 1, so every evaluation
above played the *same* games. That makes results reproducible to the digit, but
it also means repeatedly picking the best of N candidates on seed 1 fits that
particular set of 20,000 games. Round 4 judges on a seed its candidates have
never been scored on.

**Sharing a seed does not create common random numbers.** I expected paired
comparisons to be more sensitive. Measured directly from per-game CSVs, the
paired SE of a difference is 1,015 against 956 for the unpaired estimate — no
better, slightly worse. Two models diverge on a single move, consume the RNG
stream differently from then on, and the games decorrelate immediately. Pairing
buys nothing in a chaotic environment.

**The percentile SE estimator runs ~6% low.** `(p95 − p05)/3.2897` assumes
normality; the score distribution is left-skewed with a heavy low tail, so it
reported 676 where the true SE is 718. Every difference is correspondingly
*less* significant than a percentile-based figure suggests. Both conclusions
above survive re-checking against the true value.

**Selection bias in checkpoint keeping.** The keeper banks the maximum over many
600-game evaluations, each with SE ≈ 4,100. Taking a maximum over noisy draws
reliably picks a luckily-*measured* checkpoint rather than a better one, and
inflates its recorded score by roughly one standard deviation. That is harmless
for an A/B where every arm goes through the same procedure, but it is the wrong
way to choose what ships. `scripts/final_select.sh` therefore re-evaluates every
endpoint at 20,000 games and picks on that.

Round 5 caught the bias in the act. The keeper's banked best measured 251,670 on
its 600 games and was duly saved; the last checkpoint, which the keeper had
*rejected* five times running, measured lower every time. At 20,000 games:

| | mean (seed 11) |
|---|---|
| r5.bin (last checkpoint, rejected by keeper) | **249,483 ± 650** |
| r5_best.bin (keeper's banked maximum) | 245,839 ± 657 |

The keeper's pick is 3,644 worse — z = 3.6, not a coin flip. Its 600-game series
had simply drawn high on one checkpoint and low on the others. Read the keeper as
a failure detector, never as a ranking.

## Operational notes

The workspace sits on a MooseFS mount with a **per-directory quota of ~21 GiB
that `df` does not report** — `df` shows the shared 593 TB pool. Filling it does
not present as a full disk. It presents as corrupt output: abandoned
chunk-aligned `.bin.tmp` partials, blocks of NUL bytes at the tail of CSVs whose
appends failed, and trainers that keep running at full CPU while nothing they
produce reaches disk. Checkpoint saves need **2× the checkpoint size**
transiently, because the old file and the `.tmp` coexist until the rename.

Two consequences worth carrying forward:

- Monitor by **writing a probe file**, not by reading `df`. `scripts/heartbeat.sh`
  does this, and checkpoint *age* is the load-bearing signal — a frozen
  checkpoint is how a quota failure actually looks from outside.
- A wedged file stays wedged. `runs/ab/driver.out` is stuck at exactly 512 bytes
  and returns `EDQUOT` on every write even now, with the workspace writable
  again. The round-3 keeper inherited that file as its stdout; `flock` writes
  nothing to stdout but flushes it on exit, that flush failed, `flock` exited
  nonzero, and the keeper logged 26 consecutive "lock timeouts" for a lock that
  was never contended — banking nothing for the entire round. Tools whose exit
  status you branch on should not inherit a stream you do not control
  (`scripts/keep_arm.sh` now redirects it to `/dev/null`).

The round-3 loss cost nothing only by luck: the driver's
`[ -f "$FINAL" ] || FINAL=runs/model.bin` fallback saw through the dead symlink
and selected the last checkpoint, which the selection pass then confirmed was
the best candidate anyway.
