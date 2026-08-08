# 2048 — environment and reinforcement learning pipeline

A fast 2048 environment plus a complete RL training pipeline that learns to play
the game at a strong level.

The trained agent averages **283,313** over 20,000 games playing greedily and
reaches 16384 in 74.2% of them; with 3-ply expectimax on the same weights that
becomes **~371,000** with 16384 in 98% of games (32768 still converts in only
~5% — that conversion is the open problem). See [RESULTS.md](RESULTS.md)
for the full numbers, the eight A/B experiments behind the training schedule
(including the two capacity widenings that produced the largest gains), and
the measurement caveats that go with them.

## Why this algorithm

The agent is trained with **TD(0) learning over afterstates, using an n-tuple
network** as the value function, with **temporal-coherence (TC) adaptive step
sizes** and **expectimax search** at play time. The shipped network uses 24
6-tuple patterns; it began as the standard 8-pattern set and was widened
mid-training twice — to 16 in round 8, to 24 in round 11 — with the trained
tables preserved each time (see RESULTS.md, Experiments 5 and 8 — each
widening was worth more than every tuning knob combined). Multi-stage
fine-tuning and late-game restart sampling are implemented and were both
measured as nulls on this base (Experiments 1, 4, 6), and a full fresh retrain
with the final recipe could not catch the widened lineage on equal compute
(Experiment 7).

This is deliberately not deep RL. For 2048 specifically:

- The value function is naturally expressed as a **sum of lookups over local
  tile patterns**. An n-tuple network is a giant sparse linear model over those
  patterns — ~403M weights, of which exactly 192 are active for any board. Each
  update touches 192 floats, so it is enormously cheaper per sample than a
  gradient step through a CNN, and it can represent sharp, non-smooth
  distinctions between board configurations that a small CNN blurs together.
- 2048 has **deterministic afterstates**. Learning `V(afterstate)` rather than
  `Q(state, action)` removes the stochastic tile spawn from the learning target,
  which cuts variance substantially and makes 1-ply greedy action selection
  exact.
- It is **memory-bound, not FLOP-bound**, so it uses CPU cores and RAM well and
  needs no GPU at all.

Published DQN/PPO agents on 2048 typically plateau around a 10–30% rate of
reaching the 2048 tile. N-tuple TD agents exceed 95% and reach far larger tiles.
The gap is large enough that picking the policy-gradient method would mean
deliberately choosing a much weaker agent.

## Layout

```
cpp/board.hpp        bitboard engine: 64-bit board, 16 nibbles, table-driven moves
cpp/ntuple.hpp       n-tuple value network, symmetry weight sharing, TC learning
cpp/agent.hpp        greedy policy, expectimax search, game rollout
cpp/train.cpp        parallel TD(0) afterstate trainer
cpp/eval.cpp         evaluation harness (score distribution, tile reach rates)
cpp/capi.cpp         C ABI so Python can drive the engine and trained models
cpp/solve_small.cpp  exact optimal-play ceilings on 2x2 / 3x3 boards
cpp/test_engine.cpp  engine + network tests, incl. a reference implementation

python/env.py        Gymnasium-style environment (pure NumPy, self-contained)
python/native.py     ctypes bindings to the C++ engine and agents
python/play.py       watch the agent play, or play yourself
python/plot.py       training curves for individual runs
python/full_curve.py the whole training history stitched across all rounds
python/compare_curve.py  champion lineage vs a challenger run, aligned at 0
python/test_env.py   env tests, cross-checked against the C++ engine

scripts/             training pipeline and sweeps
```

## Build and test

```bash
make            # builds bin/train, bin/eval, bin/test_engine, bin/lib2048.so
make test       # C++ engine tests + Python env tests
```

## The environment

Two implementations of the same game, kept in sync by a test that compares them
move for move over thousands of random boards:

- **`cpp/board.hpp`** — the fast one, used for training. A board is a `uint64_t`
  of 16 nibbles holding tile *exponents*. Moves are four lookups into a
  precomputed 65536-entry row table, with a bit-twiddling transpose for the
  vertical directions. Roughly 900k moves/second/core including value-function
  evaluation.
- **`python/env.py`** — the readable one, with a Gymnasium-style API:

```python
from env import Env2048, Action

env = Env2048(seed=0, obs_type="onehot")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(Action.LEFT)
print(env.render())
```

Rules: 4x4 board, spawns are 90% a 2 and 10% a 4, reward is the sum of merged
tile values, each tile merges at most once per move. 32768 is the largest
representable tile and two of them do not merge.

## Training

```bash
./scripts/train_pipeline.sh          # the full staged run
```

or directly:

```bash
./bin/train --threads 14 --alpha 0.2 --time-min 240 \
            --out runs/model.bin --log runs/train_log.csv
```

Resume, and add stages, from an existing checkpoint:

```bash
./bin/train --resume runs/model.bin --stages 14,15 --alpha 0.1 ...
```

Key options: `--threads`, `--alpha`, `--time-min`, `--episodes`, `--stages`,
`--resume`, `--ckpt-min`, `--plain` (disable TC), `--extend-patterns [N]`
(widen a trained checkpoint to N=16 or 24 patterns keeping its tables, or
start fresh at that width),
`--restart-frac` / `--restart-tile` / `--restart-pool` (late-game restart
training).

### How the learning works

At each step the agent picks `argmax_a [ r(s,a) + V(afterstate(s,a)) ]`, then
updates the *previous* afterstate toward the next one:

```
V(s'_t)  <-  V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]
```

At game over the final afterstate is regressed toward 0.

Workers share one weight array without locking (Hogwild). Updates touch ~200
of ~400M weights, so collisions are rare and the resulting noise is harmless.

**Step size matters more than it looks.** `alpha` is applied to `V` as a whole
and divided across the active weights. Values near 1.0 make each update
replace `V` with its own bootstrap target, which diverges — and because a NaN
value function loses every `>` comparison, greedy selection silently degenerates
into "always take the first legal action", producing a flat learning curve that
looks like a failure to learn rather than a numerical blow-up. The trainer
detects non-finite TD errors and aborts with a clear message, and logs mean `V`
and mean `|TD error|` so divergence is visible immediately. See
`scripts/sweep_alpha.sh` for the sweep used to pick the value.

### Temporal coherence

Each weight keeps a signed sum `E` and an absolute sum `A` of the updates it has
received, and its effective step size is `|E| / A`. Weights whose updates
consistently point the same direction keep learning fast; weights that oscillate
anneal themselves. The accumulators are periodically halved, which leaves the
ratio unchanged but keeps the sums inside float32's precise range.

### Multi-stage and restart training (implemented, measured as nulls)

Late-game boards are a different problem from early-game ones, and two
mechanisms from the literature target that: **multi-stage** networks split on
the largest tile present (`--stages`), and **restart training** harvests boards
that reach a threshold tile and starts a fraction of episodes from them
(`--restart-frac`, `--restart-tile`, `--restart-pool`). Both are implemented
and both were A/B-tested against live controls on a strong base: multi-stage
lost twice, and restart training changed nothing on its own target metric (the
32768 rate under search) while costing greedy strength. See RESULTS.md
Experiments 1, 4, and 6 before reaching for either. What did pay was widening
the pattern set (`--extend-patterns`), which resumes a trained 8-pattern
checkpoint into the 16-pattern architecture with V preserved exactly.

## Evaluation

```bash
./bin/eval --model runs/model.bin --games 2000              # greedy 1-ply
./bin/eval --model runs/model.bin --games 300 --depth 3     # with search
./bin/eval --model runs/model.bin --render                  # watch one game
```

Reports mean/median/percentile scores and the rate at which each tile is
reached. `--csv` writes per-game results.

## How good can any agent get? (exact ceilings)

Because spawns are random, a fraction of games cannot reach a given tile *no
matter how well they are played*. The ceiling for tile T is

```
P*(T) = max over all policies of P(the game ever produces tile T)
```

`P*(T) < 1` for any reachable T: with k empty cells the spawn lands on the worst
cell with probability at least 1/k and is a 4 with probability 0.1, so any
finite trapping sequence occurs with probability at least (1/160)^n > 0.

This is a finite MDP, and usefully an *acyclic* one — merges preserve the total
tile sum and every spawn adds 2 or 4, so the sum strictly increases along every
trajectory. Backward induction is therefore exact and a memoised recursion
terminates. `bin/solve_small` does this:

```bash
./bin/solve_small --size 2                    # exact, instant
./bin/solve_small --size 3 --max-target 8     # exact, minutes
```

Measured results:

| board | tile | ceiling |
|-------|------|---------|
| 2x2 | 4, 8 | 100% |
| 2x2 | 16 | 96.251125% |
| 2x2 | 32 | 8.288729% |
| 2x2 | 64 | 0% (impossible) |
| 3x3 | 4 … 64 | 100% |
| 3x3 | 128 | 99.999981% |
| 3x3 | 256 | 99.570964% |
| 3x3 | 512 | 73.677418% |

The shape is the interesting part: the ceiling pins at 1, cracks almost
imperceptibly, holds above 99.5% one doubling later, then falls off a cliff —
99.57% to 73.68% in a single step as the target approaches the board's capacity.
There is no gentle decline; 3x3 goes from "essentially always reachable" to
"fails a quarter of the time" between 256 and 512.

An earlier version of this table gave 99.999988% and 99.570972% for 128 and 256.
Those came from a float32 build and were wrong in the last one or two digits —
float32 carries about 7 decimal digits, which is exactly where a ceiling of
0.9999999 runs out of room. The values above are the double-precision results
from the current build. The 2x2 figures are unaffected: they are far enough from
1 that 6 decimal places never depended on the extra precision.

**The full 4x4 game is out of reach** — the reachable state count is on the
order of 1e15+, and the cost per target level grows 2.4-3.7x (914k -> 3.4M ->
9.7M -> 23.1M states on 3x3, the last needing ~10 GB of memo table). What the
small-board results give us is a way to *read* the 4x4
numbers: 2048 is nowhere near 16 cells' capacity, so `P*(2048)` on 4x4 is
essentially 1 and any shortfall is agent skill rather than luck. The collapse on
4x4 lives up around 16384-65536, so a shortfall at 32768 may be the ceiling
itself rather than a weak agent. Measured agent rates are strict *lower bounds*
on the ceiling.

Note that this implementation caps tiles at 32768: cells are 4-bit nibbles
(exponent <= 15) and two 32768s are explicitly not merged, so 65536 is
unreachable here by construction regardless of play.

## Watching it play

```bash
python3 python/play.py --model runs/model.bin --delay 0.05
python3 python/play.py --model runs/model.bin --depth 3
python3 python/play.py --human
```

## Plotting

```bash
python3 python/plot.py runs/train_log.csv -o runs/curve.png   # one run
python3 python/full_curve.py                                  # whole history
python3 python/compare_curve.py                               # lineage vs challenger
```

`full_curve.py` reconstructs the complete training history across every phase
and round by parsing each round's `SELECTED:` verdict, stitching the winning
lineage with cumulative offsets; `compare_curve.py` overlays a challenger run
against that lineage from episode zero and plots its lead at equal episode
count.

## References

- Szubert & Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the
  Game 2048* (CIG 2014) — the afterstate TD formulation used here.
- Jaśkowski, *Mastering 2048 with Delayed Temporal Coherence Learning* (2017) —
  temporal-coherence step sizes and multi-stage training.
- Yeh et al., *Multi-Stage Temporal Difference Learning for 2048* (2016).
