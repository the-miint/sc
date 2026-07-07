//! Data model: the internal feature matrix and prediction targets that the
//! partitioner and (later) the forest consume. Built from caller-aligned inputs
//! (targets in the feature matrix's sample order, no in-library ID join — see
//! `sc-design.html` §6).

pub mod matrix;
pub mod targets;

pub use matrix::FeatureMatrix;
pub use targets::Targets;
