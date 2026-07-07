//! Pure-Rust metric implementations mirroring the values
//! q2-sample-classifier reports (`visuals.py`).
//!
//! Every function here is a faithful port of a specific q2/scikit-learn/scipy
//! computation and is validated against committed oracle fixtures (see
//! `crates/sc-core/tests/oracle_metrics.rs`). The module is Arrow-free and
//! RNG-free so it stays trivially wasm-buildable.
//!
//! Inputs use integer class *indices* (`0..k`) rather than label strings; the
//! caller owns the index↔label mapping. Confusion-matrix labels are therefore
//! the sorted unique union of the indices present, matching sklearn's default.

pub mod classification;
pub mod regression;
pub mod roc;

pub use classification::{
    accuracy, accuracy_ratio, baseline_accuracy, confusion_matrix, confusion_matrix_normalized,
};
pub use regression::{linear_regress, RegressionReport};
pub use roc::{auc, roc_auc_macro, roc_auc_micro, roc_auc_per_class, roc_curve, RocCurve};
