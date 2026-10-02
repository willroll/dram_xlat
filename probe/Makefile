# Build the dram_probe acquisition harness.
#
# Build on the *target* machine so -march=native emits the right ISA
# (clflushopt, rdtscp). Override CC/CFLAGS as needed:
#     make CC=clang
#     make CFLAGS="-O2 -march=znver3"

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=gnu11 -march=native
LDFLAGS ?=

SRC := src/probe.c src/platform.c
HDR := src/timing.h src/platform.h
BIN := dram_probe

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN)
