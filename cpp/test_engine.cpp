// Correctness tests for the bitboard engine and the n-tuple indexing.
//
// The engine is compared against a straightforward list-based reference
// implementation over random boards, which is the part most likely to hide a
// subtle bug (merge ordering, direction, transpose layout).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "agent.hpp"
#include "board.hpp"
#include "ntuple.hpp"

using namespace g2048;

namespace {

int failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

// ---------------------------------------------------------------------------
// Reference implementation on a plain 4x4 grid of exponents
// ---------------------------------------------------------------------------

using Grid = std::vector<std::vector<int>>;

Grid toGrid(Board b) {
  Grid g(4, std::vector<int>(4, 0));
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) g[r][c] = tileAt(b, 4 * r + c);
  return g;
}

Board fromGrid(const Grid& g) {
  Board b = 0;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) b = setTile(b, 4 * r + c, g[r][c]);
  return b;
}

std::vector<int> refSlideLeft(const std::vector<int>& line, uint32_t& score) {
  std::vector<int> t;
  for (int v : line)
    if (v) t.push_back(v);
  std::vector<int> out;
  for (size_t i = 0; i < t.size();) {
    if (i + 1 < t.size() && t[i] == t[i + 1] && t[i] != 15) {
      out.push_back(t[i] + 1);
      score += 1u << (t[i] + 1);
      i += 2;
    } else {
      out.push_back(t[i]);
      i += 1;
    }
  }
  out.resize(4, 0);
  return out;
}

Board refMove(Board b, int action, uint32_t& score) {
  Grid g = toGrid(b);
  score = 0;
  if (action == LEFT) {
    for (int r = 0; r < 4; ++r) g[r] = refSlideLeft(g[r], score);
  } else if (action == RIGHT) {
    for (int r = 0; r < 4; ++r) {
      std::vector<int> rev(g[r].rbegin(), g[r].rend());
      auto out = refSlideLeft(rev, score);
      std::reverse(out.begin(), out.end());
      g[r] = out;
    }
  } else if (action == UP) {
    for (int c = 0; c < 4; ++c) {
      std::vector<int> col{g[0][c], g[1][c], g[2][c], g[3][c]};
      auto out = refSlideLeft(col, score);
      for (int r = 0; r < 4; ++r) g[r][c] = out[r];
    }
  } else {
    for (int c = 0; c < 4; ++c) {
      std::vector<int> col{g[3][c], g[2][c], g[1][c], g[0][c]};
      auto out = refSlideLeft(col, score);
      for (int r = 0; r < 4; ++r) g[3 - r][c] = out[r];
    }
  }
  return fromGrid(g);
}

Board randomBoard(Rng& rng) {
  Board b = 0;
  for (int i = 0; i < 16; ++i) {
    uint32_t r = rng.below(100);
    int v = 0;
    if (r < 45) v = 0;                     // often empty, like real boards
    else v = 1 + static_cast<int>(rng.below(12));
    b = setTile(b, i, v);
  }
  return b;
}

}  // namespace

int main() {
  Rng rng(20260805);

  // --- transpose -----------------------------------------------------------
  for (int trial = 0; trial < 2000; ++trial) {
    Board b = randomBoard(rng);
    Board t = transpose(b);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        if (tileAt(t, 4 * c + r) != tileAt(b, 4 * r + c)) {
          check(false, "transpose maps (r,c)->(c,r)");
          trial = 1 << 30;
          r = c = 4;
        }
    check(transpose(t) == b, "transpose is self-inverse");
  }

  // --- moves vs reference --------------------------------------------------
  for (int trial = 0; trial < 200000; ++trial) {
    Board b = randomBoard(rng);
    for (int a = 0; a < 4; ++a) {
      uint32_t s1 = 0, s2 = 0;
      Board got = applyMove(b, a, s1);
      Board want = refMove(b, a, s2);
      if (got != want || s1 != s2) {
        std::printf("mismatch action=%d board=%016llx got=%016llx (%u) want=%016llx (%u)\n", a,
                    (unsigned long long)b, (unsigned long long)got, s1,
                    (unsigned long long)want, s2);
        check(false, "applyMove matches reference");
        trial = 1 << 30;
        break;
      }
    }
  }

  // --- specific hand-checked cases ----------------------------------------
  {
    // Top row [2,2,4,4] -> left gives [4,8,0,0] with reward 4 + 8 = 12.
    Board b = 0;
    b = setTile(b, 0, 1);
    b = setTile(b, 1, 1);
    b = setTile(b, 2, 2);
    b = setTile(b, 3, 2);
    uint32_t s;
    Board got = moveLeft(b, s);
    check(tileAt(got, 0) == 2 && tileAt(got, 1) == 3 && tileAt(got, 2) == 0 &&
              tileAt(got, 3) == 0,
          "[2,2,4,4] slides left to [4,8,_,_]");
    check(s == 12, "[2,2,4,4] left scores 12");
  }
  {
    // A row of four 2s makes two 4s, not one 8: each tile merges at most once.
    Board b = 0;
    for (int i = 0; i < 4; ++i) b = setTile(b, i, 1);
    uint32_t s;
    Board got = moveLeft(b, s);
    check(tileAt(got, 0) == 2 && tileAt(got, 1) == 2, "[2,2,2,2] -> [4,4,_,_]");
    check(s == 8, "[2,2,2,2] left scores 8");
  }
  {
    // 32768 is the cap: two of them must not merge into an unrepresentable 65536.
    Board b = 0;
    b = setTile(b, 0, 15);
    b = setTile(b, 1, 15);
    uint32_t s;
    Board got = moveLeft(b, s);
    check(got == b && s == 0, "two 32768 tiles do not merge");
  }
  {
    // A full board with no equal neighbours is terminal.
    Board b = 0;
    int vals[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1};
    // Build a checkerboard-free arrangement by construction below instead.
    for (int i = 0; i < 16; ++i) b = setTile(b, i, vals[i]);
    (void)b;
    Board dead = 0;
    int pattern[16] = {1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1};
    for (int i = 0; i < 16; ++i) dead = setTile(dead, i, pattern[i]);
    check(!hasMove(dead), "checkerboard board has no legal move");
    check(countEmpty(dead) == 0, "checkerboard board is full");
  }

  // --- spawn ---------------------------------------------------------------
  {
    Board b = 0;
    int twos = 0, fours = 0;
    for (int i = 0; i < 20000; ++i) {
      Board s = spawnTile(b, rng);
      check(countEmpty(s) == 15, "spawn fills exactly one cell");
      int v = maxTile(s);
      if (v == 1) ++twos;
      else if (v == 2) ++fours;
      else check(false, "spawn produces only 2 or 4");
    }
    double frac4 = static_cast<double>(fours) / (twos + fours);
    check(frac4 > 0.08 && frac4 < 0.12, "spawn is ~10% fours");
    if (!(frac4 > 0.08 && frac4 < 0.12)) std::printf("  frac4=%.4f\n", frac4);
  }

  // --- n-tuple network -----------------------------------------------------
  {
    Network net;
    net.init(defaultPatterns(), {}, true);

    // Value must be invariant to all 8 board symmetries because the 8
    // orientations of each pattern share one weight table.
    Stage& st = net.stage(0);
    for (int i = 0; i < 200; ++i) {
      Board b = randomBoard(rng);
      st.update(b, 100.0f, 1.0f);
    }

    int sym[8][16];
    buildSymmetries(sym);
    for (int trial = 0; trial < 500; ++trial) {
      Board b = randomBoard(rng);
      float v0 = net.value(b);
      for (int s = 1; s < 8; ++s) {
        Board t = 0;
        for (int i = 0; i < 16; ++i) t = setTile(t, sym[s][i], tileAt(b, i));
        float v = net.value(t);
        if (std::abs(v - v0) > 1e-2f * (1.0f + std::abs(v0))) {
          std::printf("  sym %d: %.4f vs %.4f\n", s, v, v0);
          check(false, "value is invariant under board symmetry");
          trial = 1 << 30;
          break;
        }
      }
    }

    // An update in the direction of the error must move the value that way.
    Board b = randomBoard(rng);
    float before = net.value(b);
    net.update(b, 500.0f, 1.0f);
    float after = net.value(b);
    check(after > before, "positive TD error raises the value");
    net.update(b, -1000.0f, 1.0f);
    check(net.value(b) < after, "negative TD error lowers the value");
  }

  // --- save / load round trip ---------------------------------------------
  {
    Network a;
    a.init(defaultPatterns(), {14}, true);
    Rng r2(7);
    std::vector<Board> probes;
    for (int i = 0; i < 300; ++i) {
      Board b = randomBoard(r2);
      probes.push_back(b);
      a.update(b, 250.0f, 1.0f);
    }
    check(a.save("/tmp/ntuple_test.bin", true), "checkpoint saves");

    Network b;
    check(b.load("/tmp/ntuple_test.bin", true), "checkpoint loads");
    check(b.nStages() == a.nStages(), "stage count round-trips");
    bool same = true;
    for (Board p : probes)
      if (std::abs(a.value(p) - b.value(p)) > 1e-6f) same = false;
    check(same, "values round-trip exactly");
    std::remove("/tmp/ntuple_test.bin");
  }

  // --- staging is monotone over a game ------------------------------------
  {
    Network net;
    net.init(defaultPatterns(), {14, 15}, false);
    check(net.nStages() == 3, "three stages from two thresholds");
    Board b = 0;
    check(net.stageOf(b) == 0, "empty board is stage 0");
    b = setTile(b, 0, 14);
    check(net.stageOf(b) == 1, "16384 tile enters stage 1");
    b = setTile(b, 1, 15);
    check(net.stageOf(b) == 2, "32768 tile enters stage 2");
  }

  // --- a full greedy game runs and terminates -----------------------------
  {
    Network net;
    net.init(defaultPatterns(), {}, true);
    Rng r3(99);
    GameResult g = playGreedy(net, r3);
    check(g.moves > 0, "untrained agent makes moves");
    check(g.maxTile >= 1, "game reaches at least a 2 tile");
    // With an all-zero value function every move ties, so play is effectively
    // random and the score should still be in a sane range.
    check(g.score < 10000000ULL, "score is finite and sane");
  }

  if (failures == 0) {
    std::printf("all engine tests passed\n");
    return 0;
  }
  std::printf("%d test(s) failed\n", failures);
  return 1;
}
