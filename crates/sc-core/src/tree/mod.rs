//! A single CART decision tree: import from sklearn, evaluate, and grow.
//!
//! The module deliberately separates *building* a tree from *evaluating* one
//! (design §10). The evaluatable [`Tree`] is one struct-of-arrays shared by two
//! producers:
//!
//! - **Import (TIGHT tier)** — [`Tree::from_sklearn`] reconstructs a tree from
//!   scikit-learn's exported `tree_` internals and reproduces
//!   `predict`/`predict_proba`/`feature_importances_`(MDI)/per-node impurity to
//!   ~1e-9. This validates the fidelity math independently of any RNG.
//! - **Grow (property tier)** — [`grow`] builds a tree depth-first from data.
//!   It is unit-tested for its invariants, not byte-matched to sklearn (feature
//!   subsampling and tie-breaks are our own deterministic rules).
//!
//! The same `Tree` is what M4 (forest / bagging) and M10 (TreeSHAP) build on;
//! per-node `weighted_n_node_samples` ("cover") is retained so it is SHAP-ready.
//!
//! Semantics follow scikit-learn 1.4.2 `_tree.pyx` / `_splitter.pyx` /
//! `_criterion.pyx`; see the per-fact notes in [`model`], [`splitter`], and
//! [`criterion`].

pub mod criterion;
mod grow;
mod model;
mod params;
mod splitter;

pub use grow::grow;
pub use model::{Task, Tree};
pub use params::{Criterion, MaxFeatures, MinSamples, TreeParams};
