#![forbid(unsafe_code)]

//! `sc-core`: the pure-Rust core of `sc`, a reimplementation of
//! q2-sample-classifier's RandomForest workflows.
//!
//! This crate is Arrow-free and FFI-free so it stays trivial to compile for
//! `wasm32` and easy to unit-test. Arrow marshaling and the C-API live in
//! sibling crates added by later milestones.
//!
//! M0 establishes only the foundation: a shared error model, a deterministic
//! RNG boundary with per-tree seed derivation, and a parallel/sequential
//! execution shim whose output is invariant to thread count.

pub mod data;
mod error;
mod exec;
pub mod forest;
pub mod metrics;
pub mod partition;
#[cfg(feature = "persist")]
pub mod persist;
mod rng;
pub mod tree;

pub use data::{FeatureMatrix, Targets};
pub use error::{Error, Result};
pub use exec::indexed_map;
pub use forest::{MaxSamples, RandomForest, RandomForestParams};
pub use partition::{k_fold, stratified_k_fold, train_test_split, CvFolds, Split, TestSize};
#[cfg(feature = "persist")]
pub use persist::ModelBundle;
pub use rng::{tree_seed, DetRng};
pub use tree::criterion::{entropy, gini, variance};
pub use tree::{grow, Criterion, MaxFeatures, MinSamples, Task, Tree, TreeParams};
