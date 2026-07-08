//! The RandomForest bagging ensemble: import from sklearn, evaluate, and fit.
//!
//! Like the [tree module](crate::tree), building a forest is deliberately
//! separated from evaluating one (design §10). The evaluatable
//! [`RandomForest`] is a `Vec<Tree>` with two producers:
//!
//! - **Import (TIGHT tier)** — [`RandomForest::from_trees`] assembles a forest
//!   from per-tree sklearn internals (each imported via
//!   [`Tree::from_sklearn`](crate::Tree::from_sklearn)) and reproduces
//!   scikit-learn's forest `predict` / `predict_proba` / `feature_importances_`
//!   (MDI) to ~1e-9 — validating the aggregation math independently of any RNG.
//! - **Fit (AGGREGATE tier)** — [`RandomForest::fit`] grows a forest from data
//!   with a deterministic per-tree bootstrap + feature subsampling. It is NOT
//!   byte-matched to sklearn (we do not replicate numpy/sklearn RNG); it is
//!   validated statistically against sklearn's seed-to-seed band and for
//!   thread-count invariance.
//!
//! Semantics follow scikit-learn 1.4.2 `_forest.py`; see the per-fact notes in
//! [`model`] and [`params`].

mod fit;
mod model;
mod params;

pub use model::RandomForest;
pub use params::{MaxSamples, RandomForestParams};
