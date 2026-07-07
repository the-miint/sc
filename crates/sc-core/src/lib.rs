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

mod error;
mod exec;
mod rng;

pub use error::{Error, Result};
pub use exec::indexed_map;
pub use rng::{tree_seed, DetRng};
