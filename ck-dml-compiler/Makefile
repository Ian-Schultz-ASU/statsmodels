# dmlc build, test and benchmark targets.
#
# Run them inside the container:   ./dev make <target>
# On native Linux, override the paths, for example:
#   make test BUILD=$HOME/dmlc-build PYTHON=$HOME/venv/bin/python

# Where CMake puts its output. /build is a Docker volume inside the container.
BUILD   ?= /build
PYTHON  ?= python
MLIR_DIR ?= /usr/lib/llvm-18/lib/cmake/mlir
LLVM_DIR ?= /usr/lib/llvm-18/lib/cmake/llvm

# The compiler driver built from dmlc/. Python finds it through this variable.
DML_OPT := $(BUILD)/tools/dml-opt/dml-opt
export DML_OPT

.PHONY: build test test-mlir test-py bench-quick bench profile figures report results clean

## build: configure (first time) and compile dml-opt
build:
	cmake -S dmlc -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
	  -DMLIR_DIR=$(MLIR_DIR) -DLLVM_DIR=$(LLVM_DIR)
	ninja -C $(BUILD)

## test-mlir: FileCheck tests for the dialect and passes
test-mlir: build
	dmlc/test/run_tests.sh $(DML_OPT)

## test-py: pytest, reference vs EconML and compiled vs reference
test-py: build
	$(PYTHON) -m pytest -q -p no:cacheprovider python/tests

## test: everything
test: test-mlir test-py

## bench-quick: one-minute sanity run of the benchmark, output not kept
bench-quick: build
	$(PYTHON) bench/run.py --quick --out /tmp/dmlc-quick

## bench: full benchmark, writes results/*.csv (about 10 minutes)
bench: build
	$(PYTHON) bench/run.py --out results

## profile: where one EconML effect() call spends its time
profile:
	$(PYTHON) bench/profile_econml.py --out results

## figures: draw results/figures/*.png from the CSV files
figures:
	$(PYTHON) bench/plots.py --results results

## report: regenerate docs/RESULTS.md from the CSV files
report:
	$(PYTHON) bench/report.py --results results --out docs/RESULTS.md

## results: the full chain, benchmark to report
results: bench profile figures report

## clean: remove build output
clean:
	rm -rf $(BUILD)/* python/.pytest_cache
