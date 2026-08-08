CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter -pthread -flto
LDFLAGS  ?= -pthread -flto

BIN := bin
SRC := cpp

BINARIES := $(BIN)/train $(BIN)/eval $(BIN)/test_engine $(BIN)/lib2048.so $(BIN)/solve_small

.PHONY: all clean test
all: $(BINARIES)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/train: $(SRC)/train.cpp $(SRC)/board.cpp $(SRC)/board.hpp $(SRC)/ntuple.hpp $(SRC)/agent.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)/train.cpp $(SRC)/board.cpp $(LDFLAGS)

$(BIN)/eval: $(SRC)/eval.cpp $(SRC)/board.cpp $(SRC)/board.hpp $(SRC)/ntuple.hpp $(SRC)/agent.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)/eval.cpp $(SRC)/board.cpp $(LDFLAGS)

$(BIN)/test_engine: $(SRC)/test_engine.cpp $(SRC)/board.cpp $(SRC)/board.hpp $(SRC)/ntuple.hpp $(SRC)/agent.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)/test_engine.cpp $(SRC)/board.cpp $(LDFLAGS)

# Exact solver: no -march=native or -flto needed, it is hash-map bound.
$(BIN)/solve_small: $(SRC)/solve_small.cpp | $(BIN)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -o $@ $(SRC)/solve_small.cpp

$(BIN)/lib2048.so: $(SRC)/capi.cpp $(SRC)/board.cpp $(SRC)/board.hpp $(SRC)/ntuple.hpp $(SRC)/agent.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $(SRC)/capi.cpp $(SRC)/board.cpp $(LDFLAGS)

test: $(BIN)/test_engine $(BIN)/lib2048.so
	./$(BIN)/test_engine
	python3 -m pytest python/test_env.py -q

clean:
	rm -rf $(BIN)
