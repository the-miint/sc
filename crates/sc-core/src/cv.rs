//! Shared cross-validation helpers: fold scoring and best-candidate selection.
//!
//! Single source of truth for the two pieces the model-selection milestones
//! (M6 RFECV, M7 tuning, later M8 nested CV) all need, so the "classifier →
//! accuracy, regressor → R²" policy and the `np.argmax` tie-break can't drift
//! between call sites. Pure and RNG-free.

use crate::data::{FeatureMatrix, Targets};
use crate::forest::RandomForest;
use crate::metrics::{accuracy, r2_score};

/// Score one CV fold with the fitted `forest`: accuracy (classification) or
/// coefficient-of-determination R² (regression) — matching sklearn's default
/// estimator `.score()`. The forest must have been fit on the same feature
/// columns as `x_test` presents, so the feature counts align.
pub(crate) fn score_fold(forest: &RandomForest, x_test: &FeatureMatrix, y_test: &Targets) -> f64 {
    match y_test {
        Targets::Categorical { codes, .. } => {
            accuracy(codes, &forest.predict_class_indices(x_test).unwrap())
        }
        Targets::Numeric(values) => r2_score(values, &forest.predict_values(x_test).unwrap()),
    }
}

/// Index of the maximum, taking the first (lowest-index) max on ties — matching
/// `np.argmax`, whose first-max rule is how both RFECV's optimal-count and a
/// randomized search's best-candidate tie-breaks are resolved.
///
/// NaN is treated as strictly worst: it never wins, and a NaN at index 0 never
/// shadows a later finite score (the naive `best = scores[0]` seed would). If
/// *every* score is NaN, returns 0 as a defined fallback. `scores` must be
/// non-empty (callers validate this upstream).
pub(crate) fn argmax_first(scores: &[f64]) -> usize {
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best_v {
            best_v = s;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_takes_fewest_on_ties() {
        // Ascending-by-count curve; tie between index 2 and 4 -> first (fewer).
        assert_eq!(argmax_first(&[0.5, 0.6, 0.9, 0.8, 0.9, 0.75]), 2);
        assert_eq!(argmax_first(&[0.99, 0.6, 0.7]), 0);
    }

    #[test]
    fn argmax_first_ignores_nan() {
        // A NaN never wins, even at index 0, when a finite score exists.
        assert_eq!(argmax_first(&[f64::NAN, 0.3, 0.9, 0.2]), 2);
        assert_eq!(argmax_first(&[0.4, f64::NAN, 0.1]), 0);
        // All-NaN falls back to index 0 (defined, not a panic).
        assert_eq!(argmax_first(&[f64::NAN, f64::NAN]), 0);
    }
}
