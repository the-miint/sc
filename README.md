# sc

A pure-Rust reimplementation of the RandomForest classification and regression
workflows from [q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier),
usable as a Rust library, as a WebAssembly module, or through a reentrant C-API
centered on Apache Arrow for C/C++ consumers.

- **Scope (v1):** RandomForest classifier + regressor — fit/predict, class
  probabilities, feature importances, RFECV feature selection, randomized
  hyperparameter tuning, nested / out-of-fold cross-validation, and
  path-dependent TreeSHAP explanations.
- **Consistency:** statistically equivalent to q2-sample-classifier (not
  byte-exact against it), validated against committed oracle fixtures generated
  from the pinned q2/sklearn environment.
- **Reproducibility:** a fixed `(data, seed)` yields an identical model and
  predictions on every platform, including `wasm32` — proven in CI, not just
  claimed (see [WASM & determinism](#wasm--determinism)).
- **Pure Rust**, no C/C++/Fortran dependencies; the core is `#![forbid(unsafe_code)]`.

## Crate map

| Crate | Role | Ships to |
| --- | --- | --- |
| [`sc-core`](crates/sc-core) | The engine: data, RNG, trees, forest, CV, RFE, tuning, SHAP. Arrow-free, FFI-free, `#![forbid(unsafe_code)]`. | Rust + **wasm** |
| `sc-arrow` | Arrow C-Data-Interface marshaling (the `unsafe` boundary). | native only |
| `sc-capi` | The reentrant `sc_*` C-API: opaque handles, Arrow I/O, panic-safe ABI. | native only |
| `sc-oracle` | Dev-only fixture loader + tolerance assertions (not shipped). | tests only |

## Rust quickstart

```rust
use sc_core::{FeatureMatrix, MaxFeatures, RandomForest, RandomForestParams, Targets};

// COO triplets (rows = samples, cols = features), here a 2-feature, 2-class set.
let x = FeatureMatrix::from_coo(
    6, 2,
    &[1, 2, 3, 3, 4, 4, 5, 5],         // row (sample) indices
    &[1, 0, 0, 1, 0, 1, 0, 1],         // col (feature) indices
    &[1., 1., 3., 3., 3., 4., 4., 3.], // values
).unwrap();
let y = Targets::categorical(&["a", "a", "a", "b", "b", "b"]).unwrap();

let mut params = RandomForestParams::classifier_defaults();
params.n_estimators = 16;
params.tree.max_features = MaxFeatures::Sqrt;

// The seed is mandatory — there is no implicit entropy.
let forest = RandomForest::fit(&x, &y, &params, 42).unwrap();

let classes = forest.classes().unwrap();
let preds   = forest.predict_class_indices(&x).unwrap(); // hard labels (indices)
let proba   = forest.predict_proba(&x).unwrap();         // per-class probabilities
let mdi     = forest.feature_importances();              // mean-decrease-in-impurity
let shap    = forest.tree_shap(&x).unwrap();             // TreeSHAP; shap.mean_abs() for a summary
```

Runnable versions live in [`crates/sc-core/examples`](crates/sc-core/examples):

```sh
cargo run -p sc-core --example fit_predict_shap
cargo run -p sc-core --example persist_roundtrip   # needs the `persist` feature (default on)
```

## Feature flags

`sc-core` has a deliberately minimal, non-overlapping flag matrix; both defaults
are used and neither is redundant.

| Feature | Default | Enables | Adds deps | wasm |
| --- | --- | --- | --- | --- |
| `parallel` | on | rayon-parallel forest fit (`exec::indexed_map`) | `rayon` | **must be OFF** — rayon has no stock-wasm threads; enabling it is a compile error on `wasm32-unknown-unknown` |
| `persist` | on | model save/load (`ModelBundle::to_bytes`/`from_bytes`) | `serde`, `postcard` | ✅ wasm-clean (byte buffers only) |

Thread-count invariance is a guarantee, not a caveat: `fit` returns a
byte-identical forest whether run on 1 thread or many (the sequential path and
the rayon path are proven equal in `tests/determinism.rs`), so turning
`parallel` off on wasm changes performance, never results.

## WASM & determinism

The shipped wasm artifact is `wasm32-unknown-unknown`, built with
`--no-default-features` (no `parallel`) and optionally `--features persist`:

```sh
cargo build -p sc-core --target wasm32-unknown-unknown --no-default-features
cargo build -p sc-core --target wasm32-unknown-unknown --no-default-features --features persist
```

Determinism rests on two pillars, each enforced by CI:

1. **No excluded deps in the wasm build.** `ci/wasm-audit.sh` (`make wasm-audit`)
   fails if `rayon`, `getrandom`, `crossbeam`, or `arrow` ever appears in the
   `wasm32-unknown-unknown` dependency tree. The RNG is a value-stable
   `ChaCha8Rng` (`rand_chacha`, pinned) seeded only from the explicit, mandatory
   seed — never OS entropy — so there is no `getrandom` and no platform-dependent
   PRNG.

2. **Identical compute reproduces the frozen vectors on a wasm VM.**
   `crates/sc-core/tests/determinism.rs` freezes golden vectors (seed
   derivation, sampling, partitioning, end-to-end tree/forest/RFECV/tuning/NCV)
   plus a consolidated `cross_platform_model_digest` — one FNV-1a fingerprint
   over a full fit → predict → predict_proba → tree_shap run. `make wasm-test`
   runs that suite under [wasmtime](https://wasmtime.dev) on `wasm32-wasip1`:

   ```sh
   rustup target add wasm32-wasip1        # once
   # + a `wasmtime` binary on PATH (see .cargo/config.toml for the cargo runner)
   make wasm-test
   ```

   Green there means the wasm build reproduces the native constants bit-for-bit,
   hence **native == wasm**. Parity is proven on `wasm32-wasip1` rather than the
   shipped `wasm32-unknown-unknown` only because wasip1 adds the WASI syscalls a
   test harness needs for stdout/exit; the numeric and PRNG codegen is identical
   across the two wasm targets, so the proof transfers.

## C-API

The native C-API is generated into [`include/sc.h`](include/sc.h) by cbindgen on
every `cargo build -p sc-capi` (a build product; regenerate with `make capi`).
It exposes opaque `sc_context_t` / `sc_model_t` handles, Arrow C-Data-Interface
I/O, a `catch_unwind` panic barrier (no panic crosses FFI), and `sc_status_t`
error codes with per-context last-error strings.

- **Worked example:** [`crates/sc-capi/ctest/harness.c`](crates/sc-capi/ctest/harness.c)
  builds Arrow inputs, fits, predicts, serializes/reloads, runs TreeSHAP, and
  checks a forced error — end to end from C.
- **Versioning:** `sc_version()` returns the library version as a `'static`
  string, equal to the header's `#define SC_VERSION`.
- **Build / test:** `make capi ctest leaks` (compile the harness, run it, and
  leak-check under valgrind). `make dist` copies the header to
  `dist/sc-<version>.h` as the release artifact.

## Build & test

Everything is wired through the [`Makefile`](Makefile):

```sh
make            # fmt + clippy + tests (parallel & sequential) + wasm build + wasm parity + wasm audit + deny + C-API harness
make wasm-test  # determinism parity on wasm32-wasip1 (needs wasmtime)
make wasm-audit # forbid native-only deps in the wasm tree
make capi ctest leaks   # C-API harness + valgrind
make dist       # versioned dist/sc-<version>.h
make deny       # cargo-deny: licenses / advisories / bans / sources
```

Third-party dependency licenses are catalogued in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
