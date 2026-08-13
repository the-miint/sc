//! Pure-Rust metric implementations mirroring the values
//! q2-sample-classifier reports (`visuals.py`).
//!
//! Every function here is a faithful port of a specific q2/scikit-learn/scipy
//! computation and is validated against committed oracle fixtures (see
//! `crates/sc-core/tests/oracle_metrics.rs`). The module is Arrow-free and
//! RNG-free so it stays trivially wasm-buildable.
//!
//! **Attribution.** The behavior reproduced here is defined by
//! [q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier)
//! (BSD-3-Clause, © 2017-2026 QIIME 2 development team) and scikit-learn
//! (BSD-3-Clause). No upstream source is vendored — this is an independent
//! implementation validated numerically against them — but if you use these
//! metrics in published work, cite Bokulich et al. 2018
//! (doi:10.21105/joss.00934) and Pedregosa et al. 2011. See `CITATION.cff` and
//! `THIRD_PARTY_LICENSES.md` §(d) at the repository root.
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
pub use regression::{linear_regress, mean_squared_error, r2_score, RegressionReport};
pub use roc::{auc, roc_auc_macro, roc_auc_micro, roc_auc_per_class, roc_curve, RocCurve};
