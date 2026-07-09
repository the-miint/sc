CARGO ?= cargo
CC ?= cc
WASM_TARGET := wasm32-unknown-unknown
# The pinned oracle environment (numpy 1.26.4 / scipy 1.13.0 / sklearn 1.4.2 /
# q2-sample-classifier). Override to point at a different interpreter/env.
ORACLE_PYTHON ?= conda run -n qiime2-amplicon-2024.10 python

# C harness against the sc-capi staticlib (`-l:libsc_capi.a` forces the archive
# over the cdylib so the binary is self-contained for valgrind).
HARNESS := target/debug/sc_harness
HARNESS_SRC := crates/sc-capi/ctest/harness.c
VALGRIND_SUPP := crates/sc-capi/ctest/valgrind.supp
HARNESS_CFLAGS := -std=c11 -Wall -Iinclude
HARNESS_LDLIBS := -l:libsc_capi.a -lpthread -ldl -lm

.PHONY: all fmt lint test test-seq wasm deny capi ctest leaks clean regen-fixtures

all: fmt lint test test-seq wasm deny capi ctest

fmt:
	$(CARGO) fmt --all --check

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --all-features

# Scope the serde-free / rayon-free build to the crates that must stay minimal.
# A workspace-wide `--no-default-features` would let sc-arrow/sc-capi (which
# require persist+parallel) turn those features back on for sc-core via feature
# unification, silently stopping this gate from testing the minimal core.
test-seq:
	$(CARGO) test -p sc-core -p sc-oracle --no-default-features

wasm:
	$(CARGO) build -p sc-core --target $(WASM_TARGET) --no-default-features
	# Persistence must also be wasm-clean (serde + postcard, byte buffers only),
	# so the M9 C-API and M11 WASM packaging inherit a working save/load path.
	$(CARGO) build -p sc-core --target $(WASM_TARGET) --no-default-features --features persist

deny:
	$(CARGO) deny check

# Build the C-API staticlib/cdylib; build.rs regenerates include/sc.h.
capi:
	$(CARGO) build -p sc-capi

# Compile the C harness against the staticlib and run it (ABI / link proof).
$(HARNESS): capi $(HARNESS_SRC)
	$(CC) $(HARNESS_CFLAGS) $(HARNESS_SRC) -Ltarget/debug $(HARNESS_LDLIBS) -o $(HARNESS)

ctest: $(HARNESS)
	$(HARNESS)

# Leak-check the harness (exercises release callbacks + buffer/handle frees).
leaks: $(HARNESS)
	valgrind --error-exitcode=1 --leak-check=full --suppressions=$(VALGRIND_SUPP) $(HARNESS)

# Regenerate the committed oracle fixtures from the pinned q2/sklearn env.
# Deterministic: re-running produces byte-identical JSON.
regen-fixtures:
	$(ORACLE_PYTHON) oracle/generator/gen_metrics.py
	$(ORACLE_PYTHON) oracle/generator/gen_tree.py
	$(ORACLE_PYTHON) oracle/generator/gen_forest.py
	$(ORACLE_PYTHON) oracle/generator/gen_rfe.py
	$(ORACLE_PYTHON) oracle/generator/gen_ncv.py
	$(ORACLE_PYTHON) oracle/generator/gen_shap.py

clean:
	$(CARGO) clean
