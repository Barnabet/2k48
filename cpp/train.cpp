// Parallel TD(0) afterstate trainer for 2048.
//
// Each worker plays complete self-play games with the greedy policy and applies
// the TD afterstate update of Szubert & Jaskowski (2014):
//
//   V(s'_t)  <-  V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]
//
// where s'_t is the afterstate at step t and r_{t+1} is the merge reward of the
// next move. At game over the last afterstate is regressed toward 0.
//
// Workers share one weight array without locking (Hogwild): updates are sparse
// (64 of ~134M weights per step), so collisions are rare and the resulting
// noise is harmless.

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "agent.hpp"
#include "board.hpp"
#include "ntuple.hpp"

using namespace g2048;

namespace {

std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

struct Options {
  int threads = 14;
  // Step size applied to V as a whole; it is divided across the 64 active
  // weights internally. Values near 1.0 make the update replace V with the
  // bootstrap target outright, which diverges.
  double alpha = 0.1;
  uint64_t episodes = 0;          // 0 = unlimited, bounded by time limit
  double timeLimitMin = 0;        // 0 = unlimited
  std::string out = "runs/model.bin";
  std::string resume;
  std::string log = "runs/train_log.csv";
  double ckptMin = 20.0;
  uint64_t logEvery = 2000;
  std::vector<int> thresholds;    // empty = single stage
  bool broadcast = false;         // seed all stages from stage 0 after loading
  int extendPatterns = 0;         // widen to N patterns (16, 24 or 32), keeping trained tables; 0 = off
  bool plain = false;             // disable TC, use fixed step size
  uint64_t rescaleEvery = 2000000; // episodes between TC accumulator rescales
  uint64_t seed = 12345;
  // Late-game restart training (carousel-lite, after Jaskowski 2017 / Yeh 2016).
  // Boards are harvested the first time an episode pushes its max tile to
  // restartTile or beyond, and a fraction of episodes then start from a
  // harvested board instead of an empty one. Greedy self-play reaches 16384 in
  // only ~2/3 of games and spends a minority of its moves there, so the tables
  // that decide 16k -> 32k are trained on a starved slice of the data; restarts
  // oversample exactly that slice.
  double restartFrac = 0.0;       // fraction of episodes started from the pool
  int restartTile = 14;           // harvest at max tile >= 2^14 = 16384
  std::string restartPool;        // persist/load the pool here (optional)
};

// Shared reservoir of harvested late-game afterstates. Harvests are rare (at
// most a few per episode) so one mutex is not a contention concern.
struct RestartPool {
  std::mutex mu;
  std::vector<Board> boards;
  static constexpr size_t kCap = 1u << 16;
  static constexpr size_t kMinToSample = 256;

  void add(Board b, Rng& rng) {
    std::lock_guard<std::mutex> lk(mu);
    if (boards.size() < kCap) boards.push_back(b);
    else boards[rng.below(static_cast<uint32_t>(kCap))] = b;
  }

  bool sample(Board& out, Rng& rng) {
    std::lock_guard<std::mutex> lk(mu);
    if (boards.size() < kMinToSample) return false;
    out = boards[rng.below(static_cast<uint32_t>(boards.size()))];
    return true;
  }

  size_t size() {
    std::lock_guard<std::mutex> lk(mu);
    return boards.size();
  }

  bool save(const std::string& path) {
    std::lock_guard<std::mutex> lk(mu);
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(boards.data(), sizeof(Board), boards.size(), f) == boards.size();
    ok &= (std::fclose(f) == 0);
    if (!ok) { std::remove(tmp.c_str()); return false; }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
  }

  bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::lock_guard<std::mutex> lk(mu);
    boards.clear();
    Board b;
    while (boards.size() < kCap && std::fread(&b, sizeof(Board), 1, f) == 1)
      boards.push_back(b);
    std::fclose(f);
    return true;
  }
};

// Per-episode diagnostics accumulated by a worker before it takes the lock.
struct EpisodeStats {
  double absDeltaSum = 0;  // mean |TD error| tracks convergence
  double valueSum = 0;     // mean V(afterstate) should track mean return
  uint64_t updates = 0;
};

// Rolling window of finished games, used for the learning curve.
struct Window {
  std::mutex mu;
  uint64_t episodes = 0;
  uint64_t moves = 0;
  double scoreSum = 0;
  uint64_t reach[16] = {0};
  uint64_t maxScore = 0;
  double absDeltaSum = 0;
  double valueSum = 0;
  uint64_t updates = 0;

  void add(const GameResult& g, const EpisodeStats& es) {
    ++episodes;
    moves += g.moves;
    scoreSum += static_cast<double>(g.score);
    if (g.score > maxScore) maxScore = g.score;
    for (int t = 1; t <= g.maxTile && t < 16; ++t) ++reach[t];
    absDeltaSum += es.absDeltaSum;
    valueSum += es.valueSum;
    updates += es.updates;
  }

  void reset() {
    episodes = 0;
    moves = 0;
    scoreSum = 0;
    maxScore = 0;
    std::memset(reach, 0, sizeof(reach));
    absDeltaSum = 0;
    valueSum = 0;
    updates = 0;
  }
};

std::atomic<bool> g_diverged{false};

void usage() {
  std::printf(
      "usage: train [options]\n"
      "  --threads N        worker threads (default 14)\n"
      "  --alpha F          learning rate (default 0.1)\n"
      "  --episodes N       stop after N episodes (default unlimited)\n"
      "  --time-min F       stop after F minutes (default unlimited)\n"
      "  --out PATH         checkpoint path (default runs/model.bin)\n"
      "  --resume PATH      load weights (and TC stats) before training\n"
      "  --log PATH         CSV learning curve (default runs/train_log.csv)\n"
      "  --ckpt-min F       minutes between checkpoints (default 20)\n"
      "  --log-every N      episodes between log rows (default 2000)\n"
      "  --stages A,B       max-tile exponents starting each new stage\n"
      "  --broadcast        copy stage 0 weights into all stages after resume\n"
      "  --extend-patterns [N]  widen to N patterns (16, 24 or 32, default 16),\n"
      "                     keeping trained tables; also picks the set for a\n"
      "                     fresh run\n"
      "  --plain            disable temporal coherence\n"
      "  --restart-frac F   fraction of episodes started from harvested\n"
      "                     late-game boards (default 0 = off)\n"
      "  --restart-tile N   harvest boards on reaching tile 2^N (default 14)\n"
      "  --restart-pool P   load the board pool from P if it exists and save\n"
      "                     it there at every checkpoint\n"
      "  --seed N           RNG seed (default 12345)\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--threads") opt.threads = std::atoi(next().c_str());
    else if (a == "--alpha") opt.alpha = std::atof(next().c_str());
    else if (a == "--episodes") opt.episodes = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--time-min") opt.timeLimitMin = std::atof(next().c_str());
    else if (a == "--out") opt.out = next();
    else if (a == "--resume") opt.resume = next();
    else if (a == "--log") opt.log = next();
    else if (a == "--ckpt-min") opt.ckptMin = std::atof(next().c_str());
    else if (a == "--log-every") opt.logEvery = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--broadcast") opt.broadcast = true;
    else if (a == "--extend-patterns") {
      // Optional count: bare flag means 16 (the original widening), an
      // explicit 16 or 24 picks the target set.
      opt.extendPatterns = 16;
      if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0]))) {
        opt.extendPatterns = std::atoi(next().c_str());
        if (opt.extendPatterns != 16 && opt.extendPatterns != 24 && opt.extendPatterns != 32) {
          std::fprintf(stderr, "--extend-patterns takes 16, 24 or 32, got %d\n", opt.extendPatterns);
          return 1;
        }
      }
    }
    else if (a == "--plain") opt.plain = true;
    else if (a == "--restart-frac") opt.restartFrac = std::atof(next().c_str());
    else if (a == "--restart-tile") opt.restartTile = std::atoi(next().c_str());
    else if (a == "--restart-pool") opt.restartPool = next();
    else if (a == "--seed") opt.seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--stages") {
      std::string s = next();
      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        opt.thresholds.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
      }
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      usage();
      return 1;
    }
  }

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  Network net;
  if (!opt.resume.empty()) {
    std::printf("loading %s ...\n", opt.resume.c_str());
    std::fflush(stdout);
    // Scoped so the source network's mappings are released before training.
    Network loaded;
    if (!loaded.load(opt.resume, true)) {
      std::fprintf(stderr, "failed to load %s\n", opt.resume.c_str());
      return 1;
    }
    std::vector<int> thr = opt.thresholds.empty() ? loaded.thresholds() : opt.thresholds;

    // Widening the pattern set is checked before anything is allocated: a set
    // that is not a superset-by-prefix cannot be prefix-copied, and silently
    // training a mis-aligned table would waste the whole run.
    bool widen = false;
    std::vector<std::vector<int>> target;
    if (opt.extendPatterns) {
      target = opt.extendPatterns == 32   ? wide32Patterns()
               : opt.extendPatterns == 24 ? wide24Patterns()
                                          : extendedPatterns();
      if (!isPatternPrefix(loaded.patterns(), target)) {
        std::fprintf(stderr,
                     "--extend-patterns needs a checkpoint whose patterns are a prefix of the "
                     "target set; %s has %zu pattern(s) that do not match\n",
                     opt.resume.c_str(), loaded.patterns().size());
        return 1;
      }
      widen = loaded.patterns().size() < target.size();
    }
    net.init(widen ? target : loaded.patterns(), thr, true);

    if (widen) {
      // V is preserved exactly: the new tables are zero, so they add nothing
      // until they learn. Verified by evaluating before and after.
      for (int i = 0; i < net.nStages(); ++i)
        net.stage(i).copyPrefixFrom(loaded.stage(std::min(i, loaded.nStages() - 1)));
      std::printf("widened %zu -> %zu patterns, existing tables copied, new tables zero\n",
                  loaded.patterns().size(), net.patterns().size());
    } else if (net.nStages() == loaded.nStages() && !opt.broadcast) {
      for (int i = 0; i < net.nStages(); ++i) {
        net.stage(i).copyWeightsFrom(loaded.stage(i));
        net.stage(i).copyStatsFrom(loaded.stage(i));
      }
      std::printf("resumed %d stage(s)\n", net.nStages());
    } else {
      // Re-staging: every stage starts as a copy of the trained stage 0 and
      // then specialises. TC stats come along so step sizes stay annealed.
      for (int i = 0; i < net.nStages(); ++i) {
        net.stage(i).copyWeightsFrom(loaded.stage(0));
        net.stage(i).copyStatsFrom(loaded.stage(0));
      }
      std::printf("re-staged %d -> %d stage(s), all seeded from stage 0\n",
                  loaded.nStages(), net.nStages());
    }
  } else {
    net.init(opt.extendPatterns == 32   ? wide32Patterns()
             : opt.extendPatterns == 24 ? wide24Patterns()
             : opt.extendPatterns == 16 ? extendedPatterns()
                                        : defaultPatterns(),
             opt.thresholds, true);
  }

  size_t weightsPerStage = net.stage(0).total();
  double gib = static_cast<double>(weightsPerStage) * sizeof(float) * 3 * net.nStages() /
               (1024.0 * 1024.0 * 1024.0);
  std::printf("patterns=%zu  weights/stage=%zu  stages=%d  memory=%.2f GiB  threads=%d  alpha=%.3f  TC=%s\n",
              net.patterns().size(), weightsPerStage, net.nStages(), gib, opt.threads,
              opt.alpha, opt.plain ? "off" : "on");
  std::fflush(stdout);

  // Append across runs so a resumed run extends the same learning curve; the
  // header goes in only when the file is new or empty.
  bool needHeader = true;
  {
    FILE* probe = std::fopen(opt.log.c_str(), "rb");
    if (probe) {
      std::fseek(probe, 0, SEEK_END);
      needHeader = (std::ftell(probe) == 0);
      std::fclose(probe);
    }
  }
  FILE* logf = std::fopen(opt.log.c_str(), "a");
  if (logf && needHeader) {
    std::fprintf(logf,
                 "episodes,elapsed_s,mean_score,max_score,moves_per_s,"
                 "r1024,r2048,r4096,r8192,r16384,r32768,mean_value,mean_abs_td_error,"
                 "restart_episodes,restart_32k,pool_size\n");
    std::fflush(logf);
  }

  Window win;
  RestartPool rpool;
  if (!opt.restartPool.empty() && rpool.load(opt.restartPool))
    std::printf("restart pool: loaded %zu boards from %s\n", rpool.size(), opt.restartPool.c_str());
  if (opt.restartFrac > 0)
    std::printf("restart training: frac=%.2f tile=%d (2^%d = %d)\n", opt.restartFrac,
                opt.restartTile, opt.restartTile, 1 << opt.restartTile);
  // Restart episodes are tracked apart from the Window: their scores start
  // mid-game and would drag the fresh-start learning curve into
  // incomparability with every earlier run's.
  std::atomic<uint64_t> restartEpisodes{0};
  std::atomic<uint64_t> restart32k{0};
  std::atomic<uint64_t> totalEpisodes{0};
  std::atomic<uint64_t> totalMoves{0};
  auto start = std::chrono::steady_clock::now();
  auto lastCkpt = start;
  std::atomic<uint64_t> nextLogAt{opt.logEvery};
  std::atomic<uint64_t> nextRescaleAt{opt.rescaleEvery};
  std::mutex ioMu;

  auto elapsedSec = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  auto worker = [&](int tid) {
    Rng rng(opt.seed * 1000003ULL + static_cast<uint64_t>(tid) * 7919ULL + 1);
    while (!g_stop.load(std::memory_order_relaxed)) {
      if (opt.episodes && totalEpisodes.load(std::memory_order_relaxed) >= opt.episodes) break;

      GameResult g;
      EpisodeStats es;
      Board s = 0;
      bool isRestart = false;
      if (opt.restartFrac > 0 && rng.uniform() < opt.restartFrac) {
        Board b;
        if (rpool.sample(b, rng)) {
          // The pool holds afterstates; spawning here re-randomises the tile so
          // one harvested board seeds many distinct continuations.
          s = spawnTile(b, rng);
          isRestart = true;
        }
      }
      if (!isRestart) s = initialBoard(rng);
      int epMax = maxTile(s);
      bool havePrev = false;
      Board prevAfter = 0;
      float prevValue = 0.0f;

      while (true) {
        Move m = greedyMove(net, s);
        if (!m.legal) break;

        if (havePrev) {
          float delta = (static_cast<float>(m.reward) + m.value) - prevValue;
          // A non-finite error means the value function has blown up. Without
          // this guard NaN silently poisons every weight and greedy selection
          // degenerates to "first legal action", which looks like a flat
          // learning curve rather than a failure.
          if (!std::isfinite(delta)) {
            g_diverged.store(true);
            g_stop.store(true);
            return;
          }
          es.absDeltaSum += std::fabs(delta);
          es.valueSum += prevValue;
          ++es.updates;
          if (opt.plain) net.stage(net.stageOf(prevAfter)).updatePlain(prevAfter, delta, opt.alpha);
          else net.update(prevAfter, delta, opt.alpha);
        }

        g.score += m.reward;
        ++g.moves;
        if (opt.restartFrac > 0) {
          int mt = maxTile(m.afterstate);
          // Harvest each first crossing of a new max at or above the restart
          // tile — from restart episodes too, so a pooled 16k board that merges
          // to 32768 deposits that deeper board back into the pool and the
          // sampled frontier advances with the agent.
          if (mt > epMax) {
            if (mt >= opt.restartTile) rpool.add(m.afterstate, rng);
            epMax = mt;
          }
        }
        prevAfter = m.afterstate;
        // Reuse the value computed during action selection. It is marginally
        // stale after the update above, which is the standard trade-off: a
        // fresh evaluation would cost another full network pass per move.
        prevValue = m.value;
        havePrev = true;
        s = spawnTile(m.afterstate, rng);
      }

      if (havePrev) {
        // Terminal: no further reward is obtainable from the last afterstate.
        float delta = 0.0f - prevValue;
        es.absDeltaSum += std::fabs(delta);
        es.valueSum += prevValue;
        ++es.updates;
        if (opt.plain) net.stage(net.stageOf(prevAfter)).updatePlain(prevAfter, delta, opt.alpha);
        else net.update(prevAfter, delta, opt.alpha);
      }

      g.maxTile = maxTile(s);
      uint64_t ep = totalEpisodes.fetch_add(1, std::memory_order_relaxed) + 1;
      totalMoves.fetch_add(static_cast<uint64_t>(g.moves), std::memory_order_relaxed);

      if (isRestart) {
        restartEpisodes.fetch_add(1, std::memory_order_relaxed);
        if (g.maxTile >= 15) restart32k.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::lock_guard<std::mutex> lk(win.mu);
        win.add(g, es);
      }

      if (ep >= nextLogAt.load(std::memory_order_relaxed)) {
        // Exactly one worker crosses each boundary and does the reporting.
        uint64_t expected = nextLogAt.load(std::memory_order_relaxed);
        if (ep >= expected &&
            nextLogAt.compare_exchange_strong(expected, expected + opt.logEvery)) {
          double el = elapsedSec();
          double mean, mps, meanAbsDelta, meanV;
          uint64_t mx, reach[16];
          uint64_t n;
          {
            std::lock_guard<std::mutex> lk(win.mu);
            n = win.episodes ? win.episodes : 1;
            mean = win.scoreSum / static_cast<double>(n);
            mx = win.maxScore;
            double u = win.updates ? static_cast<double>(win.updates) : 1.0;
            meanAbsDelta = win.absDeltaSum / u;
            meanV = win.valueSum / u;
            std::memcpy(reach, win.reach, sizeof(reach));
            win.reset();
          }
          mps = static_cast<double>(totalMoves.load()) / (el > 0 ? el : 1);
          auto rate = [&](int t) { return 100.0 * static_cast<double>(reach[t]) / static_cast<double>(n); };

          uint64_t rEps = restartEpisodes.load(std::memory_order_relaxed);
          uint64_t r32 = restart32k.load(std::memory_order_relaxed);
          size_t poolN = (opt.restartFrac > 0) ? rpool.size() : 0;

          std::lock_guard<std::mutex> lk(ioMu);
          std::printf(
              "ep %-10llu %6.0fs  mean %9.0f  max %8llu  V %9.0f  |d| %7.0f  %5.0fk mv/s  "
              "1024 %5.1f  2048 %5.1f  4096 %5.1f  8192 %5.1f  16k %5.1f  32k %5.1f",
              static_cast<unsigned long long>(ep), el, mean,
              static_cast<unsigned long long>(mx), meanV, meanAbsDelta, mps / 1000.0,
              rate(10), rate(11), rate(12), rate(13), rate(14), rate(15));
          if (opt.restartFrac > 0)
            std::printf("  rst %llu/%llu@32k pool %zu",
                        static_cast<unsigned long long>(rEps),
                        static_cast<unsigned long long>(r32), poolN);
          std::printf("\n");
          std::fflush(stdout);
          if (logf) {
            std::fprintf(logf,
                         "%llu,%.1f,%.1f,%llu,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,"
                         "%llu,%llu,%zu\n",
                         static_cast<unsigned long long>(ep), el, mean,
                         static_cast<unsigned long long>(mx), mps, rate(10), rate(11),
                         rate(12), rate(13), rate(14), rate(15), meanV, meanAbsDelta,
                         static_cast<unsigned long long>(rEps),
                         static_cast<unsigned long long>(r32), poolN);
            std::fflush(logf);
          }
        }
      }
    }
  };

  std::vector<std::thread> pool;
  for (int t = 0; t < opt.threads; ++t) pool.emplace_back(worker, t);

  // Supervisor: checkpoints, TC rescaling, and the stop conditions.
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    bool done = g_stop.load();
    if (opt.episodes && totalEpisodes.load() >= opt.episodes) done = true;
    if (opt.timeLimitMin > 0 && elapsedSec() >= opt.timeLimitMin * 60.0) {
      g_stop.store(true);
      done = true;
    }

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - lastCkpt).count() >= opt.ckptMin * 60.0 && !done) {
      lastCkpt = now;
      std::lock_guard<std::mutex> lk(ioMu);
      std::printf("checkpoint -> %s (ep %llu)\n", opt.out.c_str(),
                  static_cast<unsigned long long>(totalEpisodes.load()));
      std::fflush(stdout);
      // Written while workers keep training; the resulting snapshot is
      // slightly inconsistent, which is irrelevant for a value table.
      net.save(opt.out, true);
      if (!opt.restartPool.empty() && opt.restartFrac > 0) rpool.save(opt.restartPool);
    }

    uint64_t ep = totalEpisodes.load();
    uint64_t target = nextRescaleAt.load();
    if (!opt.plain && ep >= target && nextRescaleAt.compare_exchange_strong(target, target + opt.rescaleEvery)) {
      std::lock_guard<std::mutex> lk(ioMu);
      std::printf("rescaling TC accumulators at ep %llu\n", static_cast<unsigned long long>(ep));
      std::fflush(stdout);
      net.rescaleStats(0.5f);
    }

    if (done) break;
  }

  for (auto& t : pool) t.join();

  if (g_diverged.load()) {
    std::fprintf(stderr,
                 "\nDIVERGED: the value function produced a non-finite TD error.\n"
                 "Lower --alpha and retrain; the checkpoint below is not usable.\n");
    return 2;
  }

  std::printf("saving final checkpoint -> %s\n", opt.out.c_str());
  std::fflush(stdout);
  if (!net.save(opt.out, true)) {
    std::fprintf(stderr, "checkpoint save FAILED\n");
    return 1;
  }
  if (!opt.restartPool.empty() && opt.restartFrac > 0 && !rpool.save(opt.restartPool))
    std::fprintf(stderr, "restart pool save failed (training result unaffected)\n");
  if (logf) std::fclose(logf);
  std::printf("done: %llu episodes, %llu moves, %.1f s\n",
              static_cast<unsigned long long>(totalEpisodes.load()),
              static_cast<unsigned long long>(totalMoves.load()), elapsedSec());
  return 0;
}
