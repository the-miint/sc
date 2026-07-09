# sc-core

The pure-Rust core of [`sc`](../../README.md), a reimplementation of
[q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier)'s
RandomForest classification and regression workflows.

This crate is **Arrow-free and FFI-free** and `#![forbid(unsafe_code)]`, so it
compiles trivially for `wasm32` and is straightforward to unit-test. Arrow
marshaling (`sc-arrow`) and the C-API (`sc-capi`) live in sibling crates and are
native-only.

## What it does

- RandomForest **classifier** and **regressor**: `fit`, `predict_class_indices`,
  `predict_proba` / `predict_values`, `feature_importances` (MDI).
- **RFECV** recursive feature elimination, randomized hyperparameter **tuning**,
  and nested / out-of-fold **cross-validation**.
- Path-dependent **TreeSHAP** (`tree_shap`, with `ShapValues::mean_abs`).
- Optional model **persistence** (`ModelBundle::to_bytes` / `from_bytes`) behind
  the `persist` feature.

## Determinism

A fixed `(data, seed)` produces a bit-identical model and predictions on every
platform, native or `wasm32`. The RNG is a value-stable `ChaCha8Rng` seeded only
from the explicit, **mandatory** seed (never OS entropy), and the parallel fit is
proven equal to a single-threaded reference regardless of thread count. The
frozen golden vectors and the `cross_platform_model_digest` fingerprint in
`tests/determinism.rs` are executed on a wasm VM in CI to prove native == wasm.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `parallel` | on | rayon-parallel forest fit. **Must be off on wasm** (no stock-wasm threads). Results are identical with it on or off. |
| `persist` | on | serde + postcard model save/load. wasm-clean (byte buffers only). |

## Example

```sh
cargo run -p sc-core --example fit_predict_shap
```

See the crate root docs and the [workspace README](../../README.md) for the full
quickstart, the wasm/determinism story, and the C-API.

## License

BSD 3-Clause. See [LICENSE](../../LICENSE).
