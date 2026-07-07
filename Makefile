CARGO ?= cargo
WASM_TARGET := wasm32-unknown-unknown

.PHONY: all fmt lint test test-seq wasm deny clean

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

clean:
	$(CARGO) clean
