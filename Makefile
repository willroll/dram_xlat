# Convenience targets for the whole dram_xlat toolkit.
#   make build   build the C probe
#   make test    run the solver self-test on synthetic data (no hardware)
#   make clean   remove build artifacts

.PHONY: build test clean

build:
	$(MAKE) -C probe

test:
	python3 solver/selftest.py

clean:
	$(MAKE) -C probe clean
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
