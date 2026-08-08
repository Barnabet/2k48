// Policies built on top of a trained n-tuple network.
//
// The value function is an *afterstate* function: V(s') estimates the remaining
// return from the deterministic board that a move produces, before the random
// tile appears. A 1-ply greedy policy therefore picks argmax_a [ r(s,a) +
// V(s'_a) ], which is already the training policy. Expectimax extends the same
// evaluation over several plies of the random tile spawns.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "board.hpp"
#include "ntuple.hpp"

namespace g2048 {

struct Move {
  int action = -1;
  Board afterstate = 0;
  uint32_t reward = 0;
  float value = 0.0f;   // V(afterstate)
  float total = 0.0f;   // reward + V(afterstate)
  bool legal = false;
};

// 1-ply greedy action selection. Returns an illegal Move when the game is over.
inline Move greedyMove(const Network& net, Board s) {
  Move best;
  best.total = -1e30f;
  for (int a = 0; a < 4; ++a) {
    uint32_t r;
    Board after = applyMove(s, a, r);
    if (after == s) continue;
    float v = net.value(after);
    float t = static_cast<float>(r) + v;
    if (!best.legal || t > best.total) {
      best.action = a;
      best.afterstate = after;
      best.reward = r;
      best.value = v;
      best.total = t;
      best.legal = true;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Expectimax search
// ---------------------------------------------------------------------------

struct SearchConfig {
  int depth = 3;            // chance layers expanded before falling back to V
  double probCutoff = 1e-4; // prune spawn branches below this cumulative weight
  size_t ttBits = 22;       // transposition table size, 2^ttBits entries
  bool adaptive = true;     // deepen when the board is crowded
};

struct SearchStats {
  uint64_t nodes = 0;
  uint64_t ttHits = 0;
};

class Expectimax {
 public:
  Expectimax(const Network& net, const SearchConfig& cfg)
      : net_(net), cfg_(cfg), mask_((size_t(1) << cfg.ttBits) - 1) {
    tt_.assign(size_t(1) << cfg.ttBits, TTEntry{});
  }

  SearchStats stats;

  Move best(Board s) {
    int depth = cfg_.depth;
    if (cfg_.adaptive) {
      // A chance node branches on 2 * (empty cells), and each branch fans out
      // over 4 moves, so one extra ply costs roughly 8 * empty. That is only
      // affordable when the board is nearly full — at 6 empties an extra ply
      // is ~50x, which is what made the old rule (extend up to +2 whenever
      // empty <= 6) thousands of times slower than the fixed-depth search.
      // Crowded boards are also where lookahead actually decides the game.
      if (countEmpty(s) <= 2) depth += 1;
    }

    Move best;
    best.total = -1e30f;
    for (int a = 0; a < 4; ++a) {
      uint32_t r;
      Board after = applyMove(s, a, r);
      if (after == s) continue;
      float v = (depth <= 0) ? net_.value(after) : chance(after, depth, 1.0);
      float t = static_cast<float>(r) + v;
      if (!best.legal || t > best.total) {
        best.action = a;
        best.afterstate = after;
        best.reward = r;
        best.value = v;
        best.total = t;
        best.legal = true;
      }
    }
    return best;
  }

 private:
  struct TTEntry {
    Board key = 0;
    float value = 0.0f;
    int16_t depth = -1;
    bool valid = false;
  };

  // Expected value over the tile that spawns on `after`.
  float chance(Board after, int depth, double prob) {
    ++stats.nodes;
    if (depth <= 0 || prob < cfg_.probCutoff) return net_.value(after);

    size_t slot = (hash(after) ^ (static_cast<size_t>(depth) * 0x9E3779B9u)) & mask_;
    TTEntry& te = tt_[slot];
    if (te.valid && te.key == after && te.depth == depth) {
      ++stats.ttHits;
      return te.value;
    }

    int empty = countEmpty(after);
    if (empty == 0) return net_.value(after);

    const double pCell = 1.0 / static_cast<double>(empty);
    double acc = 0.0;
    for (int i = 0; i < 16; ++i) {
      if (tileAt(after, i) != 0) continue;
      // 2 with probability 0.9, 4 with probability 0.1.
      acc += 0.9 * pCell * maxNode(setTile(after, i, 1), depth, prob * 0.9 * pCell);
      acc += 0.1 * pCell * maxNode(setTile(after, i, 2), depth, prob * 0.1 * pCell);
    }

    float v = static_cast<float>(acc);
    te.key = after;
    te.value = v;
    te.depth = static_cast<int16_t>(depth);
    te.valid = true;
    return v;
  }

  // Best move from a full state. A state with no legal move ends the game and
  // contributes no further reward.
  float maxNode(Board s, int depth, double prob) {
    float best = -1e30f;
    bool any = false;
    for (int a = 0; a < 4; ++a) {
      uint32_t r;
      Board after = applyMove(s, a, r);
      if (after == s) continue;
      any = true;
      float v = static_cast<float>(r) + chance(after, depth - 1, prob);
      if (v > best) best = v;
    }
    return any ? best : 0.0f;
  }

  static size_t hash(Board b) {
    b ^= b >> 33;
    b *= 0xFF51AFD7ED558CCDULL;
    b ^= b >> 33;
    b *= 0xC4CEB9FE1A85EC53ULL;
    b ^= b >> 33;
    return static_cast<size_t>(b);
  }

  const Network& net_;
  SearchConfig cfg_;
  size_t mask_;
  std::vector<TTEntry> tt_;
};

// ---------------------------------------------------------------------------
// Game rollout
// ---------------------------------------------------------------------------

struct GameResult {
  uint64_t score = 0;
  int maxTile = 0;
  int moves = 0;
};

inline GameResult playGreedy(const Network& net, Rng& rng) {
  GameResult g;
  Board s = initialBoard(rng);
  while (true) {
    Move m = greedyMove(net, s);
    if (!m.legal) break;
    g.score += m.reward;
    ++g.moves;
    s = spawnTile(m.afterstate, rng);
  }
  g.maxTile = maxTile(s);
  return g;
}

inline GameResult playSearch(const Network& net, const SearchConfig& cfg, Rng& rng) {
  GameResult g;
  Expectimax ex(net, cfg);
  Board s = initialBoard(rng);
  while (true) {
    Move m = ex.best(s);
    if (!m.legal) break;
    g.score += m.reward;
    ++g.moves;
    s = spawnTile(m.afterstate, rng);
  }
  g.maxTile = maxTile(s);
  return g;
}

}  // namespace g2048
