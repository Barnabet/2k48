// Bitboard 2048 engine.
//
// A board is a uint64_t holding 16 nibbles. Cell i (row-major, i = 4*row + col)
// lives at bits [4*i, 4*i+4). A nibble value v means an empty cell when v == 0
// and a tile of value 2^v otherwise, so v == 11 is the 2048 tile and v == 15 is
// 32768 (the largest representable tile; two 32768s are not allowed to merge).
#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

namespace g2048 {

using Board = uint64_t;

enum Action : int { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

inline int tileAt(Board b, int i) { return static_cast<int>((b >> (4 * i)) & 0xFULL); }

inline Board setTile(Board b, int i, int v) {
  return (b & ~(0xFULL << (4 * i))) | (static_cast<Board>(v) << (4 * i));
}

// Maps cell (r, c) -> (c, r). Self-inverse, so it is independent of which corner
// nibble 0 represents.
inline Board transpose(Board x) {
  Board a1 = x & 0xF0F00F0FF0F00F0FULL;
  Board a2 = x & 0x0000F0F00000F0F0ULL;
  Board a3 = x & 0x0F0F00000F0F0000ULL;
  Board a = a1 | (a2 << 12) | (a3 >> 12);
  Board b1 = a & 0xFF00FF0000FF00FFULL;
  Board b2 = a & 0x00FF00FF00000000ULL;
  Board b3 = a & 0x00000000FF00FF00ULL;
  return b1 | (b2 >> 24) | (b3 << 24);
}

// Row lookup tables. A row is a uint16 whose nibble c holds column c, so nibble
// 0 is the leftmost cell.
struct RowTables {
  uint16_t left[65536];
  uint16_t right[65536];
  uint32_t scoreLeft[65536];
  uint32_t scoreRight[65536];

  RowTables() {
    for (uint32_t row = 0; row < 65536; ++row) {
      int line[4];
      for (int i = 0; i < 4; ++i) line[i] = (row >> (4 * i)) & 0xF;

      int out[4];
      uint32_t sc = 0;
      slideMergeLeft(line, out, sc);
      left[row] = pack(out);
      scoreLeft[row] = sc;

      int rev[4] = {line[3], line[2], line[1], line[0]};
      int outRev[4];
      uint32_t scR = 0;
      slideMergeLeft(rev, outRev, scR);
      int outRight[4] = {outRev[3], outRev[2], outRev[1], outRev[0]};
      right[row] = pack(outRight);
      scoreRight[row] = scR;
    }
  }

 private:
  static uint16_t pack(const int* line) {
    uint16_t r = 0;
    for (int i = 0; i < 4; ++i) r |= static_cast<uint16_t>(line[i] & 0xF) << (4 * i);
    return r;
  }

  // Compact toward index 0, merge each pair at most once, compact again.
  static void slideMergeLeft(const int* in, int* out, uint32_t& score) {
    int tmp[4];
    int n = 0;
    for (int i = 0; i < 4; ++i)
      if (in[i] != 0) tmp[n++] = in[i];

    int m = 0;
    for (int i = 0; i < n;) {
      // 15 is the largest representable exponent, so two 32768 tiles cannot merge.
      if (i + 1 < n && tmp[i] == tmp[i + 1] && tmp[i] != 15) {
        int merged = tmp[i] + 1;
        out[m++] = merged;
        score += 1u << merged;
        i += 2;
      } else {
        out[m++] = tmp[i];
        i += 1;
      }
    }
    for (int i = m; i < 4; ++i) out[i] = 0;
  }
};

extern const RowTables kRows;

// Applies `action` and returns the resulting afterstate. `score` receives the
// merge reward. The board is unchanged when the move is illegal; callers detect
// that by comparing the result against the input.
inline Board moveLeft(Board b, uint32_t& score) {
  Board r = 0;
  score = 0;
  for (int i = 0; i < 4; ++i) {
    uint16_t row = static_cast<uint16_t>((b >> (16 * i)) & 0xFFFFULL);
    r |= static_cast<Board>(kRows.left[row]) << (16 * i);
    score += kRows.scoreLeft[row];
  }
  return r;
}

inline Board moveRight(Board b, uint32_t& score) {
  Board r = 0;
  score = 0;
  for (int i = 0; i < 4; ++i) {
    uint16_t row = static_cast<uint16_t>((b >> (16 * i)) & 0xFFFFULL);
    r |= static_cast<Board>(kRows.right[row]) << (16 * i);
    score += kRows.scoreRight[row];
  }
  return r;
}

inline Board moveUp(Board b, uint32_t& score) {
  return transpose(moveLeft(transpose(b), score));
}

inline Board moveDown(Board b, uint32_t& score) {
  return transpose(moveRight(transpose(b), score));
}

inline Board applyMove(Board b, int action, uint32_t& score) {
  switch (action) {
    case UP: return moveUp(b, score);
    case RIGHT: return moveRight(b, score);
    case DOWN: return moveDown(b, score);
    default: return moveLeft(b, score);
  }
}

inline int countEmpty(Board b) {
  int n = 0;
  for (int i = 0; i < 16; ++i)
    if (((b >> (4 * i)) & 0xFULL) == 0) ++n;
  return n;
}

inline int maxTile(Board b) {
  int m = 0;
  for (int i = 0; i < 16; ++i) {
    int v = tileAt(b, i);
    if (v > m) m = v;
  }
  return m;
}

inline bool hasMove(Board b) {
  uint32_t s;
  for (int a = 0; a < 4; ++a)
    if (applyMove(b, a, s) != b) return true;
  return false;
}

// xoshiro256++ — small, fast, and good enough for tile spawning and exploration.
struct Rng {
  uint64_t s[4];

  explicit Rng(uint64_t seed) {
    // SplitMix64 seeding.
    for (int i = 0; i < 4; ++i) {
      seed += 0x9E3779B97F4A7C15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      s[i] = z ^ (z >> 31);
    }
  }

  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  uint64_t next() {
    uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }

  // Unbiased value in [0, n).
  uint32_t below(uint32_t n) {
    uint64_t r = next();
    __uint128_t m = static_cast<__uint128_t>(r) * n;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < n) {
      uint64_t t = (-n) % n;
      while (l < t) {
        r = next();
        m = static_cast<__uint128_t>(r) * n;
        l = static_cast<uint64_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 64);
  }

  double uniform() { return (next() >> 11) * 0x1.0p-53; }
};

// Places a 2 (90%) or a 4 (10%) on a uniformly chosen empty cell.
inline Board spawnTile(Board b, Rng& rng) {
  int empty = countEmpty(b);
  if (empty == 0) return b;
  uint32_t k = rng.below(static_cast<uint32_t>(empty));
  for (int i = 0; i < 16; ++i) {
    if (tileAt(b, i) == 0) {
      if (k == 0) {
        int v = (rng.below(10) == 0) ? 2 : 1;
        return setTile(b, i, v);
      }
      --k;
    }
  }
  return b;
}

inline Board initialBoard(Rng& rng) { return spawnTile(spawnTile(0, rng), rng); }

}  // namespace g2048
