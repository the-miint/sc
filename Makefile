CARGO ?= cargo
CC ?= cc
# The shipped wasm target (built, dep-audited); wasip1 is the sibling target the
# determinism parity suite executes on (adds only WASI syscalls for the test
# harness's stdout/exit — the numeric/PRNG codegen is identical).
WASM_TARGET := wasm32-unknown-unknown
WASM_WASI_TARGET := wasm32-wasip1
# Release version, read from the workspace manifest (single source of truth).
VERSION := $(shell awk -F'"' '/^version/ {print $$2; exit}' Cargo.toml)
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

# The Cargo.toml `rust-version` floor and the target duckdb-miint builds the
# C-API for (see `msrv`).
MSRV := 1.86.0
MSRV_TARGET := wasm32-unknown-emscripten

.PHONY: all fmt lint test test-seq wasm wasm-test wasm-audit msrv deny capi ctest leaks dist clean regen-fixtures

all: fmt lint test test-seq wasm wasm-test wasm-audit msrv deny capi ctest

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

# Native<->wasm determinism parity: run the frozen golden/digest suite on a wasm
# VM (wasm32-wasip1 under wasmtime; runner set in .cargo/config.toml). Green here
# means the wasm build reproduces the native golden vectors bit-for-bit, i.e.
# native == wasm. Scoped to the fs-free `determinism` test — the oracle/persist
# fixture tests read files, which wasm has no access to — across both shipped
# feature combos. Requires `rustup target add wasm32-wasip1` + a `wasmtime` on PATH.
wasm-test:
	$(CARGO) test -p sc-core --no-default-features --target $(WASM_WASI_TARGET) --test determinism
	$(CARGO) test -p sc-core --no-default-features --features persist --target $(WASM_WASI_TARGET) --test determinism

# Fail the build if any native-only dep (rayon/getrandom/crossbeam/arrow) reached
# the shipped wasm32-unknown-unknown tree.
wasm-audit:
	./ci/wasm-audit.sh

# Compile the C-API exactly as duckdb-miint's wasm build does: Rust $(MSRV) for
# $(MSRV_TARGET). clippy's `incompatible_msrv` (armed by `rust-version`) only
# catches std APIs newer than the floor; this also catches newer language
# features, a dependency bump that raises its own MSRV, and target-specific code.
# `check` links nothing, so no emsdk is needed. Requires
# `rustup toolchain install $(MSRV) --target $(MSRV_TARGET)`.
msrv:
	$(CARGO) +$(MSRV) check -p sc-capi --target $(MSRV_TARGET)

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

# Versioned C-header release artifact. `capi` regenerates include/sc.h (a
# gitignored build product) via cbindgen; copy it to dist/sc-$(VERSION).h. dist/
# is the (gitignored) release output directory.
dist: capi
	mkdir -p dist
	cp include/sc.h dist/sc-$(VERSION).h
	@echo "wrote dist/sc-$(VERSION).h"

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
