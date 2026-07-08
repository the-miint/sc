//! The declarative search space and its deterministic, RNG-free decoding — the
//! `bookkeeping` analogue for tuning. A grid is a list of independent
//! [`ParamAxis`]es (one per tuned hyperparameter); the search space is their
//! Cartesian product, indexed `0..grid_size` in mixed-radix order (axis `0` is
//! the least-significant digit). [`decode`] turns an index into a concrete
//! [`RandomForestParams`]; the stochastic sampling of *which* indices to try
//! lives in [`super`].
//!
//! Grid values are stored as the typed enums they map to, so "ints stay ints,
//! fractions stay fractions" (design/M7) holds by construction — e.g.
//! `MaxFeatures::Fraction(0.1)` and `MinSamples::Fraction(0.001)` can never
//! decay to a float the way a stringly-typed grid would.

use crate::forest::RandomForestParams;
use crate::tree::{Criterion, MaxFeatures, MinSamples};
use crate::{Error, Result};

/// One tunable hyperparameter and its discrete list of candidate values. Each
/// variant targets exactly one field of [`RandomForestParams`]/`TreeParams`;
/// [`ParamAxis::apply`] writes the chosen value. Future estimators plug in by
/// adding axis variants (and, when continuous distributions are needed, a
/// separate distribution-backed axis kind) without touching the search loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamAxis {
    /// `tree.max_depth` (`None` = unbounded).
    MaxDepth(Vec<Option<usize>>),
    /// `tree.max_features`.
    MaxFeatures(Vec<MaxFeatures>),
    /// `tree.min_samples_split`.
    MinSamplesSplit(Vec<MinSamples>),
    /// `tree.min_weight_fraction_leaf`.
    MinWeightFractionLeaf(Vec<f64>),
    /// `bootstrap`.
    Bootstrap(Vec<bool>),
    /// `tree.criterion` (classification grids only).
    Criterion(Vec<Criterion>),
}

impl ParamAxis {
    /// Number of candidate values on this axis (the radix for this digit).
    pub(crate) fn len(&self) -> usize {
        match self {
            ParamAxis::MaxDepth(v) => v.len(),
            ParamAxis::MaxFeatures(v) => v.len(),
            ParamAxis::MinSamplesSplit(v) => v.len(),
            ParamAxis::MinWeightFractionLeaf(v) => v.len(),
            ParamAxis::Bootstrap(v) => v.len(),
            ParamAxis::Criterion(v) => v.len(),
        }
    }

    /// Write this axis's `choice`-th value into `p`. Precondition: `choice < len()`.
    pub(crate) fn apply(&self, choice: usize, p: &mut RandomForestParams) {
        match self {
            ParamAxis::MaxDepth(v) => p.tree.max_depth = v[choice],
            ParamAxis::MaxFeatures(v) => p.tree.max_features = v[choice],
            ParamAxis::MinSamplesSplit(v) => p.tree.min_samples_split = v[choice],
            ParamAxis::MinWeightFractionLeaf(v) => p.tree.min_weight_fraction_leaf = v[choice],
            ParamAxis::Bootstrap(v) => p.bootstrap = v[choice],
            ParamAxis::Criterion(v) => p.tree.criterion = v[choice],
        }
    }
}

/// q2's RandomForest**Classifier** search grid (`utilities.py` `ensemble +
/// bootstrap + criterion`).
pub fn rf_classifier_grid() -> Vec<ParamAxis> {
    let mut grid = ensemble_axes();
    grid.push(ParamAxis::Bootstrap(vec![true, false]));
    grid.push(ParamAxis::Criterion(vec![
        Criterion::Gini,
        Criterion::Entropy,
    ]));
    grid
}

/// q2's RandomForest**Regressor** search grid (`ensemble + bootstrap`, no
/// `criterion`).
pub fn rf_regressor_grid() -> Vec<ParamAxis> {
    let mut grid = ensemble_axes();
    grid.push(ParamAxis::Bootstrap(vec![true, false]));
    grid
}

/// The shared `parameters['ensemble']` axes (q2 `utilities.py:49-52`).
fn ensemble_axes() -> Vec<ParamAxis> {
    vec![
        ParamAxis::MaxDepth(vec![Some(4), Some(8), Some(16), None]),
        ParamAxis::MaxFeatures(vec![
            MaxFeatures::All,
            MaxFeatures::Sqrt,
            MaxFeatures::Log2,
            MaxFeatures::Fraction(0.1),
        ]),
        ParamAxis::MinSamplesSplit(vec![
            MinSamples::Fraction(0.001),
            MinSamples::Fraction(0.01),
            MinSamples::Fraction(0.1),
        ]),
        ParamAxis::MinWeightFractionLeaf(vec![0.0001, 0.001, 0.01]),
    ]
}

/// Size of the search space = product of the axis lengths. `Err(InvalidInput)`
/// if the product overflows `usize` (unreachable for realistic grids, but the
/// guard keeps the mixed-radix invariant honest). An empty grid yields `1` (the
/// empty product) and a zero-length axis yields `0`; both are rejected by the
/// caller before this is used to index.
pub(crate) fn grid_size(grid: &[ParamAxis]) -> Result<usize> {
    let mut size = 1usize;
    for axis in grid {
        size = size
            .checked_mul(axis.len())
            .ok_or_else(|| Error::InvalidInput("tuning grid size overflows usize".into()))?;
    }
    Ok(size)
}

/// Decode a product `index` into concrete params: clone `base`, then apply each
/// axis's digit (mixed-radix, axis `0` least-significant). Precondition:
/// `index < grid_size(grid)` and every axis is non-empty.
pub(crate) fn decode(
    grid: &[ParamAxis],
    base: &RandomForestParams,
    index: usize,
) -> RandomForestParams {
    let mut p = base.clone();
    let mut idx = index;
    for axis in grid {
        let len = axis.len();
        axis.apply(idx % len, &mut p);
        idx /= len;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_sizes_match_q2_combination_counts() {
        // 4·4·3·3·2·2 = 576 (classifier), 4·4·3·3·2 = 288 (regressor).
        assert_eq!(grid_size(&rf_classifier_grid()).unwrap(), 576);
        assert_eq!(grid_size(&rf_regressor_grid()).unwrap(), 288);
    }

    #[test]
    fn grid_size_edge_cases() {
        assert_eq!(grid_size(&[]).unwrap(), 1); // empty product
        assert_eq!(grid_size(&[ParamAxis::Bootstrap(vec![])]).unwrap(), 0); // zero-length axis
    }

    #[test]
    fn decode_index_zero_is_first_choice_of_each_axis() {
        let base = RandomForestParams::classifier_defaults();
        let grid = rf_classifier_grid();
        let p = decode(&grid, &base, 0);
        assert_eq!(p.tree.max_depth, Some(4));
        assert_eq!(p.tree.max_features, MaxFeatures::All);
        assert_eq!(p.tree.min_samples_split, MinSamples::Fraction(0.001));
        assert_eq!(p.tree.min_weight_fraction_leaf, 0.0001);
        assert!(p.bootstrap);
        assert_eq!(p.tree.criterion, Criterion::Gini);
        // Untuned fields are left at base.
        assert_eq!(p.n_estimators, base.n_estimators);
        assert_eq!(p.tree.min_samples_leaf, base.tree.min_samples_leaf);
    }

    #[test]
    fn decode_mixed_radix_digits() {
        // Axis 0 (max_depth, radix 4) is least-significant, so index 1 advances
        // only it; index 4 rolls into axis 1 (max_features).
        let base = RandomForestParams::classifier_defaults();
        let grid = rf_classifier_grid();
        let p1 = decode(&grid, &base, 1);
        assert_eq!(p1.tree.max_depth, Some(8));
        assert_eq!(p1.tree.max_features, MaxFeatures::All); // unchanged
        let p4 = decode(&grid, &base, 4);
        assert_eq!(p4.tree.max_depth, Some(4)); // wrapped
        assert_eq!(p4.tree.max_features, MaxFeatures::Sqrt); // carried
    }

    #[test]
    fn decode_is_injective_over_the_space() {
        // Every product index must map to a distinct params set.
        use std::collections::HashSet;
        let base = RandomForestParams::regressor_defaults();
        let grid = rf_regressor_grid();
        let n = grid_size(&grid).unwrap();
        let mut seen = HashSet::new();
        for i in 0..n {
            let p = decode(&grid, &base, i);
            // Key on the grid-varying fields (f64s via bit patterns).
            let key = (
                p.tree.max_depth,
                format!("{:?}", p.tree.max_features),
                format!("{:?}", p.tree.min_samples_split),
                p.tree.min_weight_fraction_leaf.to_bits(),
                p.bootstrap,
            );
            assert!(seen.insert(key), "index {i} collided");
        }
        assert_eq!(seen.len(), n);
    }
}
