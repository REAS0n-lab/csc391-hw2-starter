# Build the ping-pong benchmark.
#
#   make            build ./pingpong
#   make clean      remove the binary
#
# MPICC is overridable so that a site using a wrapper other than mpicc can
# build without editing this file.

MPICC  ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

pingpong: pingpong.c
	$(MPICC) $(CFLAGS) -o $@ $<

.PHONY: clean check
clean:
	rm -f pingpong

# Compile only. Useful for confirming a toolchain without a job allocation.
check: pingpong.c
	$(MPICC) $(CFLAGS) -fsyntax-only $<
	@echo "pingpong.c compiles"
