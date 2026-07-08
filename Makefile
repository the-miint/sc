CARGO ?= cargo
WASM_TARGET := wasm32-unknown-unknown
# The pinned oracle environment (numpy 1.26.4 / scipy 1.13.0 / sklearn 1.4.2 /
# q2-sample-classifier). Override to point at a different interpreter/env.
ORACLE_PYTHON ?= conda run -n qiime2-amplicon-2024.10 python

.PHONY: all fmt lint test test-seq wasm deny clean regen-fixtures

all: fmt lint test test-seq wasm deny

fmt:
	$(CARGO) fmt --all --check

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --all-features

test-seq:
	$(CARGO) test --no-default-features

wasm:
	$(CARGO) build -p sc-core --target $(WASM_TARGET) --no-default-features

deny:
	$(CARGO) deny check

# Regenerate the committed oracle fixtures from the pinned q2/sklearn env.
# Deterministic: re-running produces byte-identical JSON.
regen-fixtures:
	$(ORACLE_PYTHON) oracle/generator/gen_metrics.py
	$(ORACLE_PYTHON) oracle/generator/gen_tree.py

clean:
	$(CARGO) clean
