// N-tuple network value function with temporal-coherence (TC) learning.
//
// The value of a board is the sum of one looked-up weight per (pattern,
// symmetry) pair. All 8 symmetries of a pattern share a single weight table, so
// the value function is symmetric by construction and every training sample
// teaches all 8 orientations at once.
//
// Weights are updated with temporal coherence (Beal & Smith): each weight keeps
// a running signed sum E and absolute sum A of the updates applied to it, and
// its effective step size is |E| / A. Weights whose updates keep pointing the
// same way keep a large step; weights that oscillate anneal themselves.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>

#include "board.hpp"

namespace g2048 {

// Large arrays get transparent huge pages where available; the access pattern
// is random over hundreds of MB, so TLB pressure dominates otherwise.
template <typename T>
T* allocLarge(size_t count) {
  size_t bytes = count * sizeof(T);
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return nullptr;
#ifdef MADV_HUGEPAGE
  madvise(p, bytes, MADV_HUGEPAGE);
#endif
  return static_cast<T*>(p);
}

template <typename T>
void freeLarge(T* p, size_t count) {
  if (p) munmap(p, count * sizeof(T));
}

constexpr int kMaxTupleLen = 8;

// The 8 dihedral symmetries of the board, as permutations of cell indices.
inline void buildSymmetries(int sym[8][16]) {
  auto rc = [](int r, int c) { return 4 * r + c; };
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      int i = rc(r, c);
      sym[0][i] = rc(r, c);              // identity
      sym[1][i] = rc(c, 3 - r);          // rot 90
      sym[2][i] = rc(3 - r, 3 - c);      // rot 180
      sym[3][i] = rc(3 - c, r);          // rot 270
      sym[4][i] = rc(r, 3 - c);          // mirror
      sym[5][i] = rc(c, r);              // mirror + rot 90
      sym[6][i] = rc(3 - r, c);          // mirror + rot 180
      sym[7][i] = rc(3 - c, 3 - r);      // mirror + rot 270
    }
  }
}

// The default 8x6-tuple pattern set: rows, blocks, and L/T shapes covering the
// board corners and edges. Pattern sets of this shape are what published
// n-tuple 2048 agents use; the set is configurable so it can be swapped.
inline std::vector<std::vector<int>> defaultPatterns() {
  return {
      {0, 1, 2, 3, 4, 5},
      {4, 5, 6, 7, 8, 9},
      {0, 1, 2, 4, 5, 6},
      {4, 5, 6, 8, 9, 10},
      {0, 1, 5, 6, 7, 10},
      {0, 1, 2, 5, 9, 10},
      {0, 1, 5, 9, 13, 14},
      {0, 1, 5, 8, 9, 13},
  };
}

// Twice the capacity: the default eight followed by eight more 6-tuples.
//
// The default set must stay a *prefix* of this one. Table offsets are the
// running sum of 16^len in pattern order, so a shared prefix means patterns
// 0-7 land at byte-identical positions in both networks. That is what lets a
// trained 8-pattern checkpoint be copied straight into the front of a
// 16-pattern network: the new tables start at zero (allocLarge hands back
// zeroed pages) and contribute nothing, so V is preserved exactly rather than
// approximately, and no retraining from scratch is needed.
//
// The added shapes deliberately cover regions the default eight under-serve —
// the right and bottom edges, the centre block, and a diagonal — since all
// eight of the originals are anchored in the top-left. The 8-fold symmetry
// expansion reflects each into the other corners regardless, but a pattern set
// whose members are near-duplicates buys less than one that is spread out.
//
// This is not free: features per board go from 64 to 128, so every lookup and
// every update roughly doubles in cost and the trainer sees about half as many
// episodes per hour. Whether the extra capacity outruns the lost episodes
// inside a fixed budget is exactly what an A/B has to answer.
inline std::vector<std::vector<int>> extendedPatterns() {
  std::vector<std::vector<int>> p = defaultPatterns();
  for (const auto& extra : std::vector<std::vector<int>>{
           {0, 1, 2, 3, 7, 11},
           {0, 4, 8, 12, 13, 14},
           {1, 2, 5, 6, 9, 10},
           {0, 1, 4, 5, 8, 9},
           {2, 3, 6, 7, 10, 11},
           {8, 9, 10, 11, 12, 13},
           {0, 1, 2, 5, 10, 15},
           {4, 5, 9, 10, 13, 14},
       })
    p.push_back(extra);
  return p;
}

// Three times the original capacity: the extended sixteen followed by eight
// more 6-tuples, keeping the same prefix contract so a trained 8- or
// 16-pattern checkpoint copies straight into the front of this network.
//
// Every earlier shape is anchored near an edge or the top-left region; these
// eight were chosen by generating candidates and rejecting any whose 8-fold
// symmetry expansion collides with an existing pattern's (a mirrored or
// rotated duplicate shares its weights and adds nothing — 4 of the first 12
// hand candidates failed that test). All eight are connected, and they lean
// toward zigzag/staircase shapes because expert play arranges tiles in a
// snake whose turns are exactly the configurations the row- and block-shaped
// originals straddle.
//
// Cost scales the same way as the first widening: 192 feature lookups per
// board instead of 128, so roughly a third fewer episodes per hour.
inline std::vector<std::vector<int>> wide24Patterns() {
  std::vector<std::vector<int>> p = extendedPatterns();
  for (const auto& extra : std::vector<std::vector<int>>{
           {0, 1, 2, 5, 6, 7},    // zigzag across rows 0-1
           {0, 1, 5, 6, 10, 11},  // staircase to the centre
           {0, 4, 8, 9, 10, 11},  // left edge into row 2
           {0, 1, 4, 5, 9, 10},   // 2x2 corner + shifted 2x2
           {0, 1, 2, 3, 5, 6},    // top row + inner pair below
           {1, 4, 5, 6, 9, 13},   // cross at cell 5 with a tail
           {5, 6, 9, 10, 12, 13}, // centre block + bottom-left pair
           {0, 1, 4, 8, 12, 13},  // corner claw down the left edge
       })
    p.push_back(extra);
  return p;
}

// Four times the original capacity: the wide24 set followed by eight more
// 6-tuples under the same prefix contract.
//
// These eight were picked differently from every earlier batch: a script
// enumerated *all* connected 6-cell subsets of the board, canonicalised each
// under the 8 symmetries, and kept only classes not already spanned — which
// is when it found that the round-8 extended set itself contains four
// symmetry-duplicates ({0,4,8,12,13,14}, {1,2,5,6,9,10}, {0,1,4,5,8,9},
// {2,3,6,7,10,11} are images of earlier patterns), so the "24-pattern"
// champion really spans only 20 distinct classes. Duplicated tables sum into
// one effective table under the linear model: zero added capacity for 1/6th
// of the lookup bill. Nothing to do about it retroactively — the prefix
// contract forbids removal — but it makes checked novelty mandatory here.
// Only 49 novel connected classes remain; these are the 8 best by internal
// adjacency, small bounding box, edge/corner anchoring, and a hard cap of
// ≤4 shared cells (under any symmetry) between any two picks.
//
// Cost: 256 feature lookups per board vs 192 — about a quarter fewer
// episodes per hour, and a 6.4 GiB checkpoint (weights + TC accumulators).
inline std::vector<std::vector<int>> wide32Patterns() {
  std::vector<std::vector<int>> p = wide24Patterns();
  for (const auto& extra : std::vector<std::vector<int>>{
           {0, 1, 2, 4, 5, 8},    // corner triangle
           {0, 1, 2, 5, 6, 9},    // staircase off the top row
           {0, 1, 2, 3, 4, 6},    // top row + comb teeth
           {0, 1, 5, 9, 12, 13},  // left-edge S through the middle
           {1, 2, 4, 5, 6, 7},    // full row 1 + offset pair above
           {0, 1, 5, 8, 9, 10},   // corner pair hooking into row 2
           {1, 5, 6, 9, 10, 13},  // vertical ladder through the centre
           {0, 1, 2, 6, 7, 10},   // Z from the top row to the centre
       })
    p.push_back(extra);
  return p;
}

// True when `sub` is a prefix of `sup` — the condition that makes a weight
// prefix copy between two pattern sets valid.
inline bool isPatternPrefix(const std::vector<std::vector<int>>& sub,
                            const std::vector<std::vector<int>>& sup) {
  if (sub.size() > sup.size()) return false;
  for (size_t i = 0; i < sub.size(); ++i)
    if (sub[i] != sup[i]) return false;
  return true;
}

// One weight table set: the whole value function for a single stage.
class Stage {
 public:
  Stage() = default;
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  ~Stage() {
    freeLarge(w_, total_);
    freeLarge(e_, total_);
    freeLarge(a_, total_);
  }

  // `withStats` allocates the TC accumulators; evaluation-only loads skip them.
  void init(const std::vector<std::vector<int>>& patterns, bool withStats) {
    patterns_ = patterns;
    nPatterns_ = static_cast<int>(patterns.size());

    int sym[8][16];
    buildSymmetries(sym);

    offsets_.assign(nPatterns_, 0);
    lens_.assign(nPatterns_, 0);
    total_ = 0;
    for (int p = 0; p < nPatterns_; ++p) {
      lens_[p] = static_cast<int>(patterns[p].size());
      offsets_[p] = total_;
      total_ += static_cast<size_t>(1) << (4 * lens_[p]);
    }

    // Flatten (pattern, symmetry) -> cell list for branch-free indexing.
    cells_.assign(static_cast<size_t>(nPatterns_) * 8 * kMaxTupleLen, 0);
    for (int p = 0; p < nPatterns_; ++p)
      for (int s = 0; s < 8; ++s)
        for (int j = 0; j < lens_[p]; ++j)
          cells_[(static_cast<size_t>(p) * 8 + s) * kMaxTupleLen + j] =
              static_cast<uint8_t>(sym[s][patterns[p][j]]);

    nFeatures_ = nPatterns_ * 8;

    w_ = allocLarge<float>(total_);
    if (withStats) {
      e_ = allocLarge<float>(total_);
      a_ = allocLarge<float>(total_);
    }
    hasStats_ = withStats;
    // mmap gives zeroed pages, which is the intended zero initialisation.
  }

  size_t total() const { return total_; }
  int nFeatures() const { return nFeatures_; }
  bool hasStats() const { return hasStats_; }
  const std::vector<std::vector<int>>& patterns() const { return patterns_; }
  float* weights() { return w_; }
  float* eStats() { return e_; }
  float* aStats() { return a_; }

  // Splits the board into nibbles once; every lookup indexes this array.
  static inline void nibbles(Board b, uint8_t out[16]) {
    for (int i = 0; i < 16; ++i) out[i] = static_cast<uint8_t>((b >> (4 * i)) & 0xFULL);
  }

  inline size_t featureIndex(const uint8_t nib[16], int p, int s) const {
    const uint8_t* c = &cells_[(static_cast<size_t>(p) * 8 + s) * kMaxTupleLen];
    size_t idx = 0;
    const int L = lens_[p];
    for (int j = 0; j < L; ++j) idx |= static_cast<size_t>(nib[c[j]]) << (4 * j);
    return offsets_[p] + idx;
  }

  float value(Board b) const {
    uint8_t nib[16];
    nibbles(b, nib);
    float v = 0.0f;
    for (int p = 0; p < nPatterns_; ++p)
      for (int s = 0; s < 8; ++s) v += w_[featureIndex(nib, p, s)];
    return v;
  }

  // TC update: distribute `delta` across the active weights, each scaled by its
  // own coherence ratio. Races between threads are benign (Hogwild).
  void update(Board b, float delta, float alpha) {
    uint8_t nib[16];
    nibbles(b, nib);
    const float du = alpha * delta / static_cast<float>(nFeatures_);
    const float adu = std::fabs(du);
    for (int p = 0; p < nPatterns_; ++p) {
      for (int s = 0; s < 8; ++s) {
        size_t i = featureIndex(nib, p, s);
        float a = a_[i];
        float beta = (a > 0.0f) ? std::fabs(e_[i]) / a : 1.0f;
        w_[i] += beta * du;
        e_[i] += du;
        a_[i] += adu;
      }
    }
  }

  // Plain TD update without coherence scaling, for ablation.
  void updatePlain(Board b, float delta, float alpha) {
    uint8_t nib[16];
    nibbles(b, nib);
    const float du = alpha * delta / static_cast<float>(nFeatures_);
    for (int p = 0; p < nPatterns_; ++p)
      for (int s = 0; s < 8; ++s) w_[featureIndex(nib, p, s)] += du;
  }

  // Halves both accumulators. The step size |E|/A is a ratio so it is
  // unchanged, but this keeps the sums inside float32's precise range after
  // billions of updates.
  void rescaleStats(float factor) {
    if (!hasStats_) return;
    for (size_t i = 0; i < total_; ++i) {
      e_[i] *= factor;
      a_[i] *= factor;
    }
  }

  void copyWeightsFrom(const Stage& other) {
    std::memcpy(w_, other.w_, total_ * sizeof(float));
  }

  // Carrying the TC accumulators across a re-stage matters: a stage that starts
  // with A == 0 uses step size 1 everywhere and would undo the trained weights.
  void copyStatsFrom(const Stage& other) {
    if (!hasStats_ || !other.hasStats_) return;
    std::memcpy(e_, other.e_, total_ * sizeof(float));
    std::memcpy(a_, other.a_, total_ * sizeof(float));
  }

  // Copy a smaller network's tables into the front of this one, leaving the
  // rest at zero. Valid only when the source's pattern list is a prefix of this
  // one's; the caller checks that with isPatternPrefix. The plain copy helpers
  // above size the memcpy by the *destination's* total, which would read off
  // the end of a smaller source, so widening needs its own path.
  //
  // The new weights are left with A == 0, which the TC rule reads as step size
  // 1. That is deliberate here and the opposite of the re-staging case: the old
  // weights keep their annealed steps and stay put, while the new tables start
  // at zero and adapt fast, so the added capacity absorbs whatever residual the
  // original patterns could not represent.
  void copyPrefixFrom(const Stage& other) {
    const size_t n = std::min(total_, other.total_);
    std::memcpy(w_, other.w_, n * sizeof(float));
    if (!hasStats_ || !other.hasStats_) return;
    std::memcpy(e_, other.e_, n * sizeof(float));
    std::memcpy(a_, other.a_, n * sizeof(float));
  }

 private:
  std::vector<std::vector<int>> patterns_;
  std::vector<uint8_t> cells_;
  std::vector<size_t> offsets_;
  std::vector<int> lens_;
  int nPatterns_ = 0;
  int nFeatures_ = 0;
  size_t total_ = 0;
  bool hasStats_ = false;
  float* w_ = nullptr;
  float* e_ = nullptr;
  float* a_ = nullptr;
};

// A staged value function. Stage boundaries are keyed on the largest tile on
// the board, which never decreases during a game, so each stage network sees a
// contiguous phase of play and a game visits stages in order.
class Network {
 public:
  // `thresholds` are max-tile exponents. With {14, 15}: stage 0 covers boards
  // whose largest tile is below 16384, stage 1 covers 16384, stage 2 covers
  // 32768 and above.
  void init(const std::vector<std::vector<int>>& patterns,
            const std::vector<int>& thresholds, bool withStats) {
    thresholds_ = thresholds;
    nStages_ = static_cast<int>(thresholds.size()) + 1;
    stages_ = std::vector<Stage>(nStages_);
    for (int i = 0; i < nStages_; ++i) stages_[i].init(patterns, withStats);
    patterns_ = patterns;
  }

  int nStages() const { return nStages_; }
  Stage& stage(int i) { return stages_[i]; }
  const Stage& stage(int i) const { return stages_[i]; }
  const std::vector<int>& thresholds() const { return thresholds_; }
  const std::vector<std::vector<int>>& patterns() const { return patterns_; }

  inline int stageOf(Board b) const {
    if (nStages_ == 1) return 0;
    int m = maxTile(b);
    int s = 0;
    for (size_t i = 0; i < thresholds_.size(); ++i)
      if (m >= thresholds_[i]) s = static_cast<int>(i) + 1;
    return s;
  }

  inline float value(Board b) const { return stages_[stageOf(b)].value(b); }

  inline void update(Board b, float delta, float alpha) {
    stages_[stageOf(b)].update(b, delta, alpha);
  }

  // Seeds every stage with stage 0's weights, so multi-stage fine-tuning starts
  // from the converged single-stage function instead of from zero.
  void broadcastStage0() {
    for (int i = 1; i < nStages_; ++i) stages_[i].copyWeightsFrom(stages_[0]);
  }

  void rescaleStats(float factor) {
    for (int i = 0; i < nStages_; ++i) stages_[i].rescaleStats(factor);
  }

  bool save(const std::string& path, bool withStats) const;
  bool load(const std::string& path, bool wantStats);

 private:
  std::vector<Stage> stages_;
  std::vector<int> thresholds_;
  std::vector<std::vector<int>> patterns_;
  int nStages_ = 1;
};

// ---------------------------------------------------------------------------
// Checkpoint format
//
//   magic "NTUP2048" | version | nStages | nThresholds | thresholds[]
//   | nPatterns | (len, cells[len])* | hasStats | per stage: w[], (e[], a[])
// ---------------------------------------------------------------------------

inline bool Network::save(const std::string& path, bool withStats) const {
  std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return false;

  auto wr = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
  bool ok = true;
  ok &= wr("NTUP2048", 8);
  int32_t version = 1;
  ok &= wr(&version, 4);
  int32_t ns = nStages_;
  ok &= wr(&ns, 4);
  int32_t nt = static_cast<int32_t>(thresholds_.size());
  ok &= wr(&nt, 4);
  for (int t : thresholds_) {
    int32_t v = t;
    ok &= wr(&v, 4);
  }
  int32_t np = static_cast<int32_t>(patterns_.size());
  ok &= wr(&np, 4);
  for (const auto& p : patterns_) {
    int32_t L = static_cast<int32_t>(p.size());
    ok &= wr(&L, 4);
    for (int c : p) {
      int32_t v = c;
      ok &= wr(&v, 4);
    }
  }
  int32_t hs = (withStats && stages_[0].hasStats()) ? 1 : 0;
  ok &= wr(&hs, 4);

  for (int i = 0; i < nStages_ && ok; ++i) {
    const Stage& st = stages_[i];
    size_t n = st.total();
    // Write in chunks; a single 500 MB fwrite is fine but this bounds retries.
    const size_t kChunk = 1u << 22;
    auto dump = [&](const float* src) {
      for (size_t off = 0; off < n && ok; off += kChunk)
        ok &= wr(src + off, sizeof(float) * std::min(kChunk, n - off));
    };
    dump(const_cast<Stage&>(st).weights());
    if (hs) {
      dump(const_cast<Stage&>(st).eStats());
      dump(const_cast<Stage&>(st).aStats());
    }
  }

  ok &= (std::fclose(f) == 0);
  if (!ok) {
    std::remove(tmp.c_str());
    return false;
  }
  // Rename last so a crash mid-write never leaves a truncated checkpoint.
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

inline bool Network::load(const std::string& path, bool wantStats) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;

  auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
  char magic[8];
  bool ok = rd(magic, 8) && std::memcmp(magic, "NTUP2048", 8) == 0;
  int32_t version = 0, ns = 0, nt = 0, np = 0, hs = 0;
  ok = ok && rd(&version, 4) && rd(&ns, 4) && rd(&nt, 4);

  std::vector<int> thr;
  for (int i = 0; i < nt && ok; ++i) {
    int32_t v;
    ok &= rd(&v, 4);
    thr.push_back(v);
  }
  ok = ok && rd(&np, 4);

  std::vector<std::vector<int>> pats;
  for (int i = 0; i < np && ok; ++i) {
    int32_t L;
    ok &= rd(&L, 4);
    std::vector<int> p;
    for (int j = 0; j < L && ok; ++j) {
      int32_t v;
      ok &= rd(&v, 4);
      p.push_back(v);
    }
    pats.push_back(p);
  }
  ok = ok && rd(&hs, 4);
  if (!ok) {
    std::fclose(f);
    return false;
  }

  init(pats, thr, wantStats);
  if (ns != nStages_) {
    std::fclose(f);
    return false;
  }

  for (int i = 0; i < nStages_ && ok; ++i) {
    Stage& st = stages_[i];
    size_t n = st.total();
    const size_t kChunk = 1u << 22;
    auto slurp = [&](float* dst) {
      for (size_t off = 0; off < n && ok; off += kChunk)
        ok &= rd(dst + off, sizeof(float) * std::min(kChunk, n - off));
    };
    auto skip = [&]() { ok &= (std::fseek(f, static_cast<long>(n * sizeof(float)), SEEK_CUR) == 0); };

    slurp(st.weights());
    if (hs) {
      if (wantStats) {
        slurp(st.eStats());
        slurp(st.aStats());
      } else {
        skip();
        skip();
      }
    }
  }

  std::fclose(f);
  return ok;
}

}  // namespace g2048
