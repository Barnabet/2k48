// C ABI over the engine and the trained value function, so Python can drive the
// exact same code the trainer uses (via ctypes) instead of a re-implementation
// that could silently drift.

#include <cstdint>
#include <cstring>
#include <new>

#include "agent.hpp"
#include "board.hpp"
#include "ntuple.hpp"

using namespace g2048;

namespace {

// A loaded network plus a reusable search instance. Not thread safe: the
// transposition table is mutable, so one handle belongs to one thread.
struct AgentHandle {
  Network net;
  Expectimax* search = nullptr;
  SearchConfig cfg;
  ~AgentHandle() { delete search; }
};

}  // namespace

extern "C" {

// --- engine ---------------------------------------------------------------

uint64_t g2048_move(uint64_t board, int action, uint32_t* reward) {
  uint32_t r = 0;
  uint64_t out = applyMove(board, action, r);
  if (reward) *reward = r;
  return out;
}

int g2048_has_move(uint64_t board) { return hasMove(board) ? 1 : 0; }
int g2048_count_empty(uint64_t board) { return countEmpty(board); }
int g2048_max_tile(uint64_t board) { return maxTile(board); }

// Deterministic spawn helper: places `value` (1 for a 2, 2 for a 4) on the
// `nth` empty cell counting from index 0. Returns the board unchanged if there
// is no such cell. Keeping placement explicit lets Python own the RNG.
uint64_t g2048_place(uint64_t board, int nth, int value) {
  for (int i = 0; i < 16; ++i) {
    if (tileAt(board, i) == 0) {
      if (nth == 0) return setTile(board, i, value);
      --nth;
    }
  }
  return board;
}

// --- agent ----------------------------------------------------------------

void* g2048_agent_load(const char* path, int depth, double cutoff, int adaptive) {
  AgentHandle* h = new (std::nothrow) AgentHandle();
  if (!h) return nullptr;
  if (!h->net.load(path, false)) {
    delete h;
    return nullptr;
  }
  h->cfg.depth = depth;
  h->cfg.probCutoff = cutoff;
  h->cfg.adaptive = adaptive != 0;
  if (depth > 0) h->search = new (std::nothrow) Expectimax(h->net, h->cfg);
  return h;
}

void g2048_agent_free(void* handle) { delete static_cast<AgentHandle*>(handle); }

float g2048_agent_value(void* handle, uint64_t board) {
  return static_cast<AgentHandle*>(handle)->net.value(board);
}

int g2048_agent_stages(void* handle) {
  return static_cast<AgentHandle*>(handle)->net.nStages();
}

// Returns the chosen action, or -1 when no move is legal. `afterstate` and
// `reward` receive the resulting deterministic board and its merge reward.
int g2048_agent_act(void* handle, uint64_t board, uint64_t* afterstate, uint32_t* reward) {
  AgentHandle* h = static_cast<AgentHandle*>(handle);
  Move m = h->search ? h->search->best(board) : greedyMove(h->net, board);
  if (!m.legal) return -1;
  if (afterstate) *afterstate = m.afterstate;
  if (reward) *reward = m.reward;
  return m.action;
}

}  // extern "C"
