//! Random-forest hyperparameters: the per-tree [`TreeParams`] plus the ensemble
//! fields (`n_estimators` / `bootstrap` / `max_samples`), with the sklearn 1.4.2
//! `RandomForest*` defaults and bootstrap-count resolution.
//!
//! The per-tree defaults differ from a standalone tree only in `max_features`:
//! the classifier defaults to `sqrt` and the regressor to all features (fact 4),
//! matching sklearn's forest defaults rather than its single-tree defaults.

use crate::tree::{MaxFeatures, TreeParams};
use crate::{Error, Result};

/// How many samples to draw (with replacement) per tree when bootstrapping.
/// Mirrors sklearn's `max_samples`: `None` → all `n`, an int → that many, a
/// float → a fraction of `n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxSamples {
    /// Draw `n_samples` indices (sklearn `max_samples=None`).
    All,
    /// Draw an explicit count (validated `<= n_samples`).
    Count(usize),
    /// Draw `max(1, round(fraction * n_samples))` (fraction in `(0, 1]`).
    Fraction(f64),
}

/// Hyperparameters for [`RandomForest::fit`](crate::RandomForest::fit).
#[derive(Debug, Clone, PartialEq)]
pub struct RandomForestParams {
    /// Per-tree growth parameters (criterion / depth / min-samples /
    /// `max_features` / …).
    pub tree: TreeParams,
    /// Number of trees to grow (sklearn default 100; q2 sets it explicitly).
    pub n_estimators: usize,
    /// Whether each tree trains on a bootstrap sample (sklearn default `true`).
    pub bootstrap: bool,
    /// Bootstrap sample size when `bootstrap` is on (ignored otherwise).
    pub max_samples: MaxSamples,
}

impl RandomForestParams {
    /// sklearn `RandomForestClassifier` defaults: gini, `max_features='sqrt'`,
    /// 100 bootstrapped trees over all samples.
    pub fn classifier_defaults() -> Self {
        RandomForestParams {
            tree: TreeParams {
                max_features: MaxFeatures::Sqrt,
                ..TreeParams::classifier_defaults()
            },
            n_estimators: 100,
            bootstrap: true,
            max_samples: MaxSamples::All,
        }
    }

    /// sklearn `RandomForestRegressor` defaults: squared error,
    /// `max_features=1.0` (all features), 100 bootstrapped trees over all samples.
    pub fn regressor_defaults() -> Self {
        RandomForestParams {
            tree: TreeParams::regressor_defaults(), // max_features stays All
            n_estimators: 100,
            bootstrap: true,
            max_samples: MaxSamples::All,
        }
    }

    /// Validate the per-tree params for the task and the ensemble fields.
    ///
    /// Errors (`InvalidInput`): anything [`TreeParams::validate`] rejects;
    /// `n_estimators < 1`; a `MaxSamples::Count(0)`; or a `MaxSamples::Fraction`
    /// outside `(0, 1]`. Rejecting `Count(0)` here is load-bearing: an unchecked
    /// zero would resolve to a zero-sample bootstrap and panic deep inside
    /// `grow`'s `resolve` instead of returning this `Err`.
    pub fn validate(&self, is_classification: bool) -> Result<()> {
        self.tree.validate(is_classification)?;
        if self.n_estimators < 1 {
            return Err(Error::InvalidInput("n_estimators must be >= 1".into()));
        }
        match self.max_samples {
            MaxSamples::Count(0) => {
                return Err(Error::InvalidInput("max_samples count must be >= 1".into()));
            }
            MaxSamples::Fraction(f) if !(f.is_finite() && 0.0 < f && f <= 1.0) => {
                return Err(Error::InvalidInput(format!(
                    "max_samples fraction {f} must be in (0, 1]"
                )));
            }
            _ => {}
        }
        Ok(())
    }

    /// Resolve the per-tree bootstrap draw size for `n` training samples,
    /// applying sklearn's `_get_n_samples_bootstrap` rounding (fact 5):
    /// `All → n`; `Count(c) → c` (error if `c > n`); `Fraction(f) →
    /// max(1, round(f·n))` where `round` is round-half-to-**even** (Python's
    /// built-in `round`, which sklearn uses — not round-half-away-from-zero).
    pub fn resolve_n_bootstrap(&self, n: usize) -> Result<usize> {
        match self.max_samples {
            MaxSamples::All => Ok(n),
            MaxSamples::Count(c) => {
                if c > n {
                    return Err(Error::InvalidInput(format!(
                        "max_samples count {c} exceeds n_samples {n}"
                    )));
                }
                Ok(c)
            }
            MaxSamples::Fraction(f) => Ok(((f * n as f64).round_ties_even() as usize).max(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Criterion;

    #[test]
    fn defaults_carry_forest_max_features() {
        let c = RandomForestParams::classifier_defaults();
        assert_eq!(c.tree.criterion, Criterion::Gini);
        assert_eq!(c.tree.max_features, MaxFeatures::Sqrt);
        assert_eq!(c.n_estimators, 100);
        assert!(c.bootstrap);
        assert_eq!(c.max_samples, MaxSamples::All);

        let r = RandomForestParams::regressor_defaults();
        assert_eq!(r.tree.criterion, Criterion::SquaredError);
        assert_eq!(r.tree.max_features, MaxFeatures::All);
    }

    #[test]
    fn validate_rejects_bad_ensemble_fields() {
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 0;
        assert!(p.validate(true).is_err());

        let mut p = RandomForestParams::classifier_defaults();
        p.max_samples = MaxSamples::Fraction(1.5);
        assert!(p.validate(true).is_err());
        p.max_samples = MaxSamples::Fraction(0.0);
        assert!(p.validate(true).is_err());
        p.max_samples = MaxSamples::Fraction(0.5);
        assert!(p.validate(true).is_ok());
        // Count(0) must be rejected here (else fit panics inside grow's resolve).
        p.max_samples = MaxSamples::Count(0);
        assert!(p.validate(true).is_err());
        p.max_samples = MaxSamples::Count(1);
        assert!(p.validate(true).is_ok());

        // Criterion/task mismatch bubbles up from TreeParams::validate.
        assert!(RandomForestParams::regressor_defaults()
            .validate(true)
            .is_err());
    }

    #[test]
    fn resolve_n_bootstrap_rounding() {
        let mut p = RandomForestParams::classifier_defaults();
        assert_eq!(p.resolve_n_bootstrap(50).unwrap(), 50); // All

        p.max_samples = MaxSamples::Count(30);
        assert_eq!(p.resolve_n_bootstrap(50).unwrap(), 30);
        assert_eq!(p.resolve_n_bootstrap(30).unwrap(), 30); // exactly n
        assert!(p.resolve_n_bootstrap(20).is_err()); // 30 > 20

        p.max_samples = MaxSamples::Fraction(0.5);
        assert_eq!(p.resolve_n_bootstrap(50).unwrap(), 25); // 25.0, no tie
                                                            // Round-half-to-EVEN, matching Python/sklearn (not away-from-zero):
                                                            //   0.5*9 = 4.5 -> 4 (4 is even; away-from-zero would give 5),
                                                            //   0.5*7 = 3.5 -> 4 (4 is even),
                                                            //   0.5*3 = 1.5 -> 2 (2 is even).
        assert_eq!(p.resolve_n_bootstrap(9).unwrap(), 4);
        assert_eq!(p.resolve_n_bootstrap(7).unwrap(), 4);
        assert_eq!(p.resolve_n_bootstrap(3).unwrap(), 2);
        // Never drops below 1.
        p.max_samples = MaxSamples::Fraction(0.01);
        assert_eq!(p.resolve_n_bootstrap(10).unwrap(), 1);
    }
}
