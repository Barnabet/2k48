// Evaluation harness: plays games with a trained network and reports the score
// distribution and max-tile reach rates.

#include <algorithm>
#include <atomic>
#include <chrono>
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

void renderBoard(Board b, uint64_t score, int moveNo) {
  std::printf("\nmove %d   score %llu\n", moveNo, (unsigned long long)score);
  std::printf("+------+------+------+------+\n");
  for (int r = 0; r < 4; ++r) {
    std::printf("|");
    for (int c = 0; c < 4; ++c) {
      int v = tileAt(b, 4 * r + c);
      if (v == 0) std::printf("      |");
      else std::printf("%5d |", 1 << v);
    }
    std::printf("\n+------+------+------+------+\n");
  }
}

void usage() {
  std::printf(
      "usage: eval --model PATH [options]\n"
      "  --model PATH    checkpoint to evaluate (required)\n"
      "  --games N       number of games (default 1000)\n"
      "  --threads N     parallel workers (default 14)\n"
      "  --depth N       expectimax depth; 0 = 1-ply greedy (default 0)\n"
      "  --cutoff F      expectimax probability cutoff (default 1e-4)\n"
      "  --no-adaptive   disable depth extension on crowded boards\n"
      "  --csv PATH      write per-game score,max_tile,moves\n"
      "  --render        print the board of the first game move by move\n"
      "  --seed N        RNG seed (default 1)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, csv;
  int games = 1000, threads = 14, depth = 0;
  double cutoff = 1e-4;
  bool adaptive = true, render = false;
  uint64_t seed = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--games") games = std::atoi(next().c_str());
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--depth") depth = std::atoi(next().c_str());
    else if (a == "--cutoff") cutoff = std::atof(next().c_str());
    else if (a == "--no-adaptive") adaptive = false;
    else if (a == "--csv") csv = next();
    else if (a == "--render") render = true;
    else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 1; }
  }
  if (model.empty()) { usage(); return 1; }

  Network net;
  if (!net.load(model, false)) {
    std::fprintf(stderr, "failed to load %s\n", model.c_str());
    return 1;
  }
  std::printf("model %s: %zu patterns, %d stage(s), policy=%s\n", model.c_str(),
              net.patterns().size(), net.nStages(),
              depth > 0 ? "expectimax" : "greedy 1-ply");
  if (depth > 0)
    std::printf("search: depth=%d cutoff=%.1e adaptive=%s\n", depth, cutoff,
                adaptive ? "yes" : "no");
  std::fflush(stdout);

  SearchConfig cfg;
  cfg.depth = depth;
  cfg.probCutoff = cutoff;
  cfg.adaptive = adaptive;

  if (render) {
    Rng rng(seed);
    Board s = initialBoard(rng);
    uint64_t score = 0;
    int moveNo = 0;
    Expectimax ex(net, cfg);
    while (true) {
      renderBoard(s, score, moveNo);
      Move m = depth > 0 ? ex.best(s) : greedyMove(net, s);
      if (!m.legal) break;
      static const char* kNames[4] = {"UP", "RIGHT", "DOWN", "LEFT"};
      std::printf("  -> %s (reward %u, V=%.0f)\n", kNames[m.action], m.reward, m.value);
      score += m.reward;
      ++moveNo;
      s = spawnTile(m.afterstate, rng);
    }
    std::printf("\ngame over: score %llu, max tile %d, %d moves\n",
                (unsigned long long)score, 1 << maxTile(s), moveNo);
    return 0;
  }

  std::atomic<int> nextGame{0};
  std::mutex mu;
  std::vector<GameResult> results;
  results.reserve(games);
  std::atomic<uint64_t> totalMoves{0};
  auto start = std::chrono::steady_clock::now();

  auto worker = [&](int tid) {
    // A private search instance per worker: the transposition table is mutable.
    Expectimax ex(net, cfg);
    std::vector<GameResult> local;
    while (true) {
      int idx = nextGame.fetch_add(1);
      if (idx >= games) break;
      Rng rng(seed * 1000003ULL + static_cast<uint64_t>(idx) * 7919ULL + 1);
      GameResult g;
      if (depth > 0) {
        Board s = initialBoard(rng);
        while (true) {
          Move m = ex.best(s);
          if (!m.legal) break;
          g.score += m.reward;
          ++g.moves;
          s = spawnTile(m.afterstate, rng);
        }
        g.maxTile = maxTile(s);
      } else {
        g = playGreedy(net, rng);
      }
      totalMoves.fetch_add(static_cast<uint64_t>(g.moves));
      local.push_back(g);

      int done = idx + 1;
      if (done % 200 == 0) {
        std::lock_guard<std::mutex> lk(mu);
        std::printf("\r  %d/%d games", done, games);
        std::fflush(stdout);
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    results.insert(results.end(), local.begin(), local.end());
    (void)tid;
  };

  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& t : pool) t.join();
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("\r%*s\r", 40, "");

  std::vector<uint64_t> scores;
  scores.reserve(results.size());
  uint64_t reach[16] = {0};
  double sum = 0;
  uint64_t movesTotal = 0;
  for (const auto& g : results) {
    scores.push_back(g.score);
    sum += static_cast<double>(g.score);
    movesTotal += static_cast<uint64_t>(g.moves);
    for (int t = 1; t <= g.maxTile && t < 16; ++t) ++reach[t];
  }
  std::sort(scores.begin(), scores.end());
  const size_t n = scores.size();
  auto pct = [&](double p) { return scores.empty() ? 0ULL : scores[std::min(n - 1, static_cast<size_t>(p * n))]; };

  std::printf("\n=== %zu games ===\n", n);
  std::printf("mean score    %12.0f\n", n ? sum / static_cast<double>(n) : 0.0);
  std::printf("median score  %12llu\n", (unsigned long long)pct(0.50));
  std::printf("  5th pct     %12llu\n", (unsigned long long)pct(0.05));
  std::printf(" 95th pct     %12llu\n", (unsigned long long)pct(0.95));
  std::printf("max score     %12llu\n", n ? (unsigned long long)scores.back() : 0ULL);
  std::printf("mean moves    %12.0f\n", n ? static_cast<double>(movesTotal) / static_cast<double>(n) : 0.0);
  // "reached" is cumulative (>= this tile); "exactly" is the share of games
  // whose final largest tile is this one. Once an agent is strong the two tell
  // very different stories — a 99% reach rate for 2048 can coexist with only a
  // few percent of games actually ending there.
  const double denom = static_cast<double>(n ? n : 1);
  // Every tile up to 32768 is always listed, including the ones at zero: a
  // missing row is ambiguous between "not reached" and "not reported", and a
  // fixed table shape keeps downstream parsing and diffing simple.
  std::printf("\n%8s %10s %10s %9s\n", "tile", "reached", "exactly", "count");
  for (int t = 9; t < 16; ++t) {
    uint64_t next = (t + 1 < 16) ? reach[t + 1] : 0;
    std::printf("%8d %9.2f%% %9.2f%% %9llu\n", 1 << t,
                100.0 * static_cast<double>(reach[t]) / denom,
                100.0 * static_cast<double>(reach[t] - next) / denom,
                (unsigned long long)reach[t]);
  }
  std::printf("\n%.1f s, %.0f moves/s\n", elapsed,
              static_cast<double>(totalMoves.load()) / (elapsed > 0 ? elapsed : 1));

  if (!csv.empty()) {
    FILE* f = std::fopen(csv.c_str(), "w");
    if (f) {
      std::fprintf(f, "score,max_tile,moves\n");
      for (const auto& g : results)
        std::fprintf(f, "%llu,%d,%d\n", (unsigned long long)g.score, 1 << g.maxTile, g.moves);
      std::fclose(f);
      std::printf("wrote %s\n", csv.c_str());
    }
  }
  return 0;
}
