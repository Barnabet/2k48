// Exact optimal-play ceilings for small 2048 boards.
//
// For a target tile T, the ceiling is
//
//     P*(T) = max over all policies of P(the game ever produces tile T)
//
// This is the value of a finite MDP, and the MDP is acyclic: merges preserve
// the total tile sum and every spawn adds 2 or 4, so the sum strictly increases
// along every trajectory. That makes exact backward induction well defined, and
// a memoised recursion terminates without any need for value iteration.
//
// The full 4x4 game has far too many reachable states for this, but 2x2 and 3x3
// are tractable and give real numbers for how often perfect play still loses.
//
//   V_player(s)  = 1 if s already contains T
//                = 0 if no move is legal
//                = max over legal a of V_chance(afterstate(s, a))
//
//   V_chance(s') = 1 if s' already contains T
//                = sum over empty cells i, (1/k) * [ 0.9 * V_player(s' with 2 at i)
//                                                  + 0.1 * V_player(s' with 4 at i) ]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int N = 3;         // board side
int CELLS = 9;     // N * N
int TARGET = 7;    // target exponent; 7 means the 128 tile

using Board = uint64_t;

inline int tileAt(Board b, int i) { return static_cast<int>((b >> (4 * i)) & 0xFULL); }
inline Board setTile(Board b, int i, int v) {
  return (b & ~(0xFULL << (4 * i))) | (static_cast<Board>(v) << (4 * i));
}

int maxTile(Board b) {
  int m = 0;
  for (int i = 0; i < CELLS; ++i) m = std::max(m, tileAt(b, i));
  return m;
}

int countEmpty(Board b) {
  int n = 0;
  for (int i = 0; i < CELLS; ++i)
    if (tileAt(b, i) == 0) ++n;
  return n;
}

// Slides a line toward index 0, merging each tile at most once. Mirrors the
// rules in cpp/board.hpp exactly, including the cap at exponent 15.
void slideMergeLeft(std::vector<int>& line) {
  std::vector<int> t;
  for (int v : line)
    if (v) t.push_back(v);
  std::vector<int> out;
  for (size_t i = 0; i < t.size();) {
    if (i + 1 < t.size() && t[i] == t[i + 1] && t[i] != 15) {
      out.push_back(t[i] + 1);
      i += 2;
    } else {
      out.push_back(t[i]);
      i += 1;
    }
  }
  out.resize(line.size(), 0);
  line = out;
}

// action: 0 up, 1 right, 2 down, 3 left
Board applyMove(Board b, int action) {
  std::vector<std::vector<int>> g(N, std::vector<int>(N, 0));
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) g[r][c] = tileAt(b, r * N + c);

  for (int k = 0; k < N; ++k) {
    std::vector<int> line(N);
    for (int j = 0; j < N; ++j) {
      switch (action) {
        case 3: line[j] = g[k][j]; break;             // left
        case 1: line[j] = g[k][N - 1 - j]; break;     // right
        case 0: line[j] = g[j][k]; break;             // up
        default: line[j] = g[N - 1 - j][k]; break;    // down
      }
    }
    slideMergeLeft(line);
    for (int j = 0; j < N; ++j) {
      switch (action) {
        case 3: g[k][j] = line[j]; break;
        case 1: g[k][N - 1 - j] = line[j]; break;
        case 0: g[j][k] = line[j]; break;
        default: g[N - 1 - j][k] = line[j]; break;
      }
    }
  }

  Board out = 0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) out = setTile(out, r * N + c, g[r][c]);
  return out;
}

// The value function is invariant under the 8 board symmetries, so states are
// memoised by their canonical (smallest) representative. This cuts the table by
// close to 8x.
Board canonical(Board b) {
  Board best = ~0ULL;
  Board cur = b;
  for (int flip = 0; flip < 2; ++flip) {
    for (int rot = 0; rot < 4; ++rot) {
      if (cur < best) best = cur;
      Board next = 0;  // rotate 90 degrees: (r, c) -> (c, N-1-r)
      for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
          next = setTile(next, c * N + (N - 1 - r), tileAt(cur, r * N + c));
      cur = next;
    }
    Board m = 0;  // mirror: (r, c) -> (r, N-1-c)
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < N; ++c) m = setTile(m, r * N + (N - 1 - c), tileAt(cur, r * N + c));
    cur = m;
  }
  return best;
}

std::unordered_map<Board, double> memoPlayer, memoChance;
size_t memoLimit = 200000000;
bool overflowed = false;

double valueChance(Board after);

double valuePlayer(Board s) {
  if (maxTile(s) >= TARGET) return 1.0;
  Board key = canonical(s);
  auto it = memoPlayer.find(key);
  if (it != memoPlayer.end()) return it->second;
  if (overflowed) return 0.0;

  double best = 0.0;
  bool any = false;
  for (int a = 0; a < 4; ++a) {
    Board after = applyMove(s, a);
    if (after == s) continue;
    any = true;
    best = std::max(best, valueChance(after));
  }
  double v = any ? best : 0.0;  // no legal move: the game is over, target missed

  if (memoPlayer.size() + memoChance.size() > memoLimit) {
    overflowed = true;
    return v;
  }
  memoPlayer.emplace(key, v);
  return v;
}

double valueChance(Board after) {
  if (maxTile(after) >= TARGET) return 1.0;
  Board key = canonical(after);
  auto it = memoChance.find(key);
  if (it != memoChance.end()) return it->second;
  if (overflowed) return 0.0;

  int empty = countEmpty(after);
  if (empty == 0) return 0.0;  // unreachable after a legal move, but be safe

  double acc = 0.0;
  const double p = 1.0 / static_cast<double>(empty);
  for (int i = 0; i < CELLS; ++i) {
    if (tileAt(after, i) != 0) continue;
    acc += p * 0.9 * valuePlayer(setTile(after, i, 1));
    acc += p * 0.1 * valuePlayer(setTile(after, i, 2));
  }
  double v = acc;

  if (memoPlayer.size() + memoChance.size() > memoLimit) {
    overflowed = true;
    return v;
  }
  memoChance.emplace(key, v);
  return v;
}

// The game opens with two independent spawns on an empty board.
double ceiling() {
  double total = 0.0;
  for (int i = 0; i < CELLS; ++i) {
    for (int vi = 1; vi <= 2; ++vi) {
      double pi = (vi == 1 ? 0.9 : 0.1) / CELLS;
      Board b1 = setTile(0, i, vi);
      for (int j = 0; j < CELLS; ++j) {
        if (j == i) continue;
        for (int vj = 1; vj <= 2; ++vj) {
          double pj = (vj == 1 ? 0.9 : 0.1) / (CELLS - 1);
          total += pi * pj * valuePlayer(setTile(b1, j, vj));
        }
      }
    }
  }
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  int maxTargetExp = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--size") N = std::atoi(next().c_str());
    else if (a == "--max-target") maxTargetExp = std::atoi(next().c_str());
    else if (a == "--memo-limit") memoLimit = std::strtoull(next().c_str(), nullptr, 10);
    else {
      std::printf("usage: solve_small [--size N] [--max-target EXP] [--memo-limit N]\n");
      return 1;
    }
  }
  CELLS = N * N;
  if (CELLS > 16) {
    std::printf("board too large for the 4-bit-per-cell encoding\n");
    return 1;
  }
  if (maxTargetExp == 0) maxTargetExp = (N == 2) ? 6 : 9;

  std::printf("Exact optimal-play ceilings on a %dx%d board\n", N, N);
  std::printf("(probability that PERFECT play ever produces the tile)\n\n");
  std::printf("%10s %14s %16s\n", "tile", "ceiling", "states solved");

  for (int t = 2; t <= maxTargetExp; ++t) {
    TARGET = t;
    memoPlayer.clear();
    memoChance.clear();
    overflowed = false;
    double p = ceiling();
    size_t states = memoPlayer.size() + memoChance.size();
    if (overflowed) {
      std::printf("%10d %14s %16zu  (memo limit hit — result not exact)\n", 1 << t, "?", states);
      break;
    }
    std::printf("%10d %13.6f%% %16zu\n", 1 << t, 100.0 * p, states);
    std::fflush(stdout);
    if (p == 0.0) break;
  }
  return 0;
}
