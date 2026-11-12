CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude
HEADERS = $(wildcard include/mnn/*.hpp)

all: mnn_train mnn_bench

mnn_train: src/train.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

mnn_bench: src/bench.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

mnn_tests: tests/test_mnn.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

test: mnn_tests
	./mnn_tests

data: letter-recognition.data
letter-recognition.data:
	./scripts/get_letter_data.sh

clean:
	rm -f mnn_train mnn_bench mnn_tests

.PHONY: all test data clean
