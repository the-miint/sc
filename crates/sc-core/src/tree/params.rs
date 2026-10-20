//! Growth hyperparameters and their resolution to concrete integer thresholds.
//!
//! Mirrors the subset of scikit-learn's `DecisionTree*` parameters q2 exercises,
//! with the fact-6/fact-7 rounding rules from the M3 plan. `max_leaf_nodes`
//! (best-first builder) and `ccp_alpha` (post-pruning) are intentionally absent —
//! q2 never sets either, and adding them later is non-breaking.

use crate::{Error, Result};

/// Split criterion. Gini/Entropy are classification; SquaredError is regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// `1 − Σ p_c²`.
    Gini,
    /// `−Σ p_c·log2(p_c)` (log base 2).
    Entropy,
    /// Node variance (regression MSE criterion).
    SquaredError,
}

impl Criterion {
    /// Whether this criterion is for classification (Gini/Entropy).
    fn is_classification(self) -> bool {
        matches!(self, Criterion::Gini | Criterion::Entropy)
    }
}

/// Number of features to consider per split. Resolution rules (fact 6):
/// `Sqrt`→`max(1, ⌊√p⌋)`, `Log2`→`max(1, ⌊log2 p⌋)`, `Fraction(f)`→
/// `max(1, ⌊f·p⌋)` (floor, not round), `Count(c)` used as-is (validated `1..=p`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxFeatures {
    /// All features (sklearn `None`); the standalone-tree default. Consumes no RNG.
    All,
    /// `max(1, ⌊√n_features⌋)`.
    Sqrt,
    /// `max(1, ⌊log2 n_features⌋)`.
    Log2,
    /// `max(1, ⌊fraction·n_features⌋)`.
    Fraction(f64),
    /// An explicit count, validated to lie in `1..=n_features`.
    Count(usize),
}

/// A minimum-samples parameter expressed as an absolute count or a fraction of
/// `n_samples`. Fractions round per fact 6 (`ceil`), applied in [`TreeParams::resolve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MinSamples {
    /// An absolute sample count.
    Count(usize),
    /// A fraction of `n_samples`, in `(0, 1]`.
    Fraction(f64),
}

/// CART growth hyperparameters (depth-first builder only).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeParams {
    /// Split criterion (must match the task: classification vs regression).
    pub criterion: Criterion,
    /// Maximum depth; `None` is unbounded (root is depth 0).
    pub max_depth: Option<usize>,
    /// Minimum samples required to attempt a split at a node.
    pub min_samples_split: MinSamples,
    /// Minimum samples required in each resulting child.
    pub min_samples_leaf: MinSamples,
    /// Minimum weighted fraction of the total sample weight required in a leaf,
    /// in `[0, 0.5]` (with uniform weights this is a fraction of `n_samples`).
    pub min_weight_fraction_leaf: f64,
    /// Number of features considered per split.
    pub max_features: MaxFeatures,
    /// Minimum weighted impurity decrease for a split to be accepted.
    pub min_impurity_decrease: f64,
}

impl TreeParams {
    /// sklearn `DecisionTreeClassifier` defaults (Gini, everything else open).
    pub fn classifier_defaults() -> Self {
        TreeParams {
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: MinSamples::Count(2),
            min_samples_leaf: MinSamples::Count(1),
            min_weight_fraction_leaf: 0.0,
            max_features: MaxFeatures::All,
            min_impurity_decrease: 0.0,
        }
    }

    /// sklearn `DecisionTreeRegressor` defaults (SquaredError, everything else open).
    pub fn regressor_defaults() -> Self {
        TreeParams {
            criterion: Criterion::SquaredError,
            ..Self::classifier_defaults()
        }
    }

    /// Validate static invariants and that the criterion matches the task.
    ///
    /// Errors (`InvalidInput`): criterion/task mismatch; out-of-range fractions
    /// (`min_samples_*` fractions in `(0, 1]`, `min_weight_fraction_leaf` in
    /// `[0, 0.5]`, `MaxFeatures::Fraction` in `(0, 1]`); a non-finite or negative
    /// `min_impurity_decrease`; a `min_samples_split` count `< 2` or a
    /// `min_samples_leaf`/`MaxFeatures::Count` count `< 1`.
    pub fn validate(&self, is_classification: bool) -> Result<()> {
        if self.criterion.is_classification() != is_classification {
            return Err(Error::InvalidInput(format!(
                "criterion {:?} does not match the task ({})",
                self.criterion,
                if is_classification {
                    "classification"
                } else {
                    "regression"
                }
            )));
        }
        check_min_samples(self.min_samples_split, 2, "min_samples_split")?;
        check_min_samples(self.min_samples_leaf, 1, "min_samples_leaf")?;
        if !(0.0..=0.5).contains(&self.min_weight_fraction_leaf) {
            return Err(Error::InvalidInput(format!(
                "min_weight_fraction_leaf {} must be in [0, 0.5]",
                self.min_weight_fraction_leaf
            )));
        }
        if let MaxFeatures::Fraction(f) = self.max_features {
            if !(f.is_finite() && 0.0 < f && f <= 1.0) {
                return Err(Error::InvalidInput(format!(
                    "max_features fraction {f} must be in (0, 1]"
                )));
            }
        }
        if let MaxFeatures::Count(0) = self.max_features {
            return Err(Error::InvalidInput(
                "max_features count must be >= 1".into(),
            ));
        }
        if !self.min_impurity_decrease.is_finite() || self.min_impurity_decrease < 0.0 {
            return Err(Error::InvalidInput(format!(
                "min_impurity_decrease {} must be finite and >= 0",
                self.min_impurity_decrease
            )));
        }
        Ok(())
    }

    /// Resolve to concrete integer thresholds for a dataset of `n_samples` rows
    /// and `n_features` columns, applying the fact-6 rounding rules and the
    /// unconditional `min_samples_split = max(min_samples_split, 2·min_samples_leaf)`
    /// coupling.
    ///
    /// Precondition: [`Self::validate`] has passed for the matching task; this
    /// asserts the post-resolution invariants. Errors (`InvalidInput`) only on a
    /// `MaxFeatures::Count` that exceeds `n_features`.
    pub(crate) fn resolve(&self, n_samples: usize, n_features: usize) -> Result<Resolved> {
        assert!(n_samples > 0 && n_features > 0, "resolve: empty dataset");

        let min_samples_leaf = match self.min_samples_leaf {
            MinSamples::Count(c) => c,
            // sklearn: ceil(fraction * n_samples).
            MinSamples::Fraction(f) => ((f * n_samples as f64).ceil() as usize).max(1),
        };
        let mut min_samples_split = match self.min_samples_split {
            MinSamples::Count(c) => c,
            // sklearn: max(2, ceil(fraction * n_samples)).
            MinSamples::Fraction(f) => (f * n_samples as f64).ceil() as usize,
        };
        min_samples_split = min_samples_split.max(2);
        // The unconditional coupling sklearn applies after both are resolved.
        // Saturating: a leaf count past usize::MAX / 2 (2^31 on wasm32) already
        // forbids every split, and so does usize::MAX.
        min_samples_split = min_samples_split.max(min_samples_leaf.saturating_mul(2));

        let max_features = match self.max_features {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => ((n_features as f64).sqrt().floor() as usize).max(1),
            MaxFeatures::Log2 => ((n_features as f64).log2().floor() as usize).max(1),
            MaxFeatures::Fraction(f) => ((f * n_features as f64).floor() as usize).max(1),
            MaxFeatures::Count(c) => {
                if c > n_features {
                    return Err(Error::InvalidInput(format!(
                        "max_features count {c} exceeds n_features {n_features}"
                    )));
                }
                c
            }
        };

        // Uniform weights: total weight == n_samples, so min_weight_leaf is a
        // plain sample count threshold.
        let min_weight_leaf = self.min_weight_fraction_leaf * n_samples as f64;
        let max_depth = self.max_depth.unwrap_or(usize::MAX);

        let resolved = Resolved {
            criterion: self.criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_features,
            min_impurity_decrease: self.min_impurity_decrease,
        };
        assert!(resolved.min_samples_leaf >= 1);
        assert!(resolved.min_samples_split >= resolved.min_samples_leaf.saturating_mul(2));
        assert!((1..=n_features).contains(&resolved.max_features));
        Ok(resolved)
    }
}

fn check_min_samples(m: MinSamples, min_count: usize, name: &str) -> Result<()> {
    match m {
        MinSamples::Count(c) if c < min_count => Err(Error::InvalidInput(format!(
            "{name} count {c} must be >= {min_count}"
        ))),
        MinSamples::Count(_) => Ok(()),
        MinSamples::Fraction(f) if !(f.is_finite() && 0.0 < f && f <= 1.0) => Err(
            Error::InvalidInput(format!("{name} fraction {f} must be in (0, 1]")),
        ),
        MinSamples::Fraction(_) => Ok(()),
    }
}

/// Concrete thresholds after [`TreeParams::resolve`]; consumed by the builder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Resolved {
    pub criterion: Criterion,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_weight_leaf: f64,
    pub max_features: usize,
    pub min_impurity_decrease: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_carry_the_right_criterion() {
        assert_eq!(TreeParams::classifier_defaults().criterion, Criterion::Gini);
        assert_eq!(
            TreeParams::regressor_defaults().criterion,
            Criterion::SquaredError
        );
    }

    #[test]
    fn criterion_must_match_task() {
        assert!(TreeParams::classifier_defaults().validate(true).is_ok());
        assert!(TreeParams::classifier_defaults().validate(false).is_err());
        assert!(TreeParams::regressor_defaults().validate(false).is_ok());
        assert!(TreeParams::regressor_defaults().validate(true).is_err());
    }

    #[test]
    fn resolve_applies_fact6_rounding() {
        // sqrt(10) = 3.16 -> floor 3; log2(10) = 3.32 -> floor 3.
        let mut p = TreeParams::classifier_defaults();
        p.max_features = MaxFeatures::Sqrt;
        assert_eq!(p.resolve(100, 10).unwrap().max_features, 3);
        p.max_features = MaxFeatures::Log2;
        assert_eq!(p.resolve(100, 10).unwrap().max_features, 3);
        // fraction 0.35 * 10 = 3.5 -> floor 3.
        p.max_features = MaxFeatures::Fraction(0.35);
        assert_eq!(p.resolve(100, 10).unwrap().max_features, 3);
        // fraction that floors to 0 clamps up to 1.
        p.max_features = MaxFeatures::Fraction(0.01);
        assert_eq!(p.resolve(100, 10).unwrap().max_features, 1);
    }

    #[test]
    fn resolve_couples_split_to_twice_leaf() {
        let mut p = TreeParams::classifier_defaults();
        p.min_samples_leaf = MinSamples::Count(5);
        p.min_samples_split = MinSamples::Count(3); // < 2*5
        let r = p.resolve(100, 4).unwrap();
        assert_eq!(r.min_samples_leaf, 5);
        assert_eq!(r.min_samples_split, 10, "coupled up to 2*min_samples_leaf");
    }

    #[test]
    fn resolve_saturates_the_coupling_for_huge_leaf_counts() {
        // 2 * min_samples_leaf overflowed for a count past usize::MAX / 2: a
        // panic in a debug build, a wrap in release. On wasm32 that is any count
        // from 2^31, and the C API saturates larger ones to usize::MAX. No sample
        // count reaches the threshold either way, so it saturates too.
        let mut p = TreeParams::classifier_defaults();
        p.min_samples_leaf = MinSamples::Count(usize::MAX);
        let r = p.resolve(100, 4).unwrap();
        assert_eq!(r.min_samples_leaf, usize::MAX);
        assert_eq!(r.min_samples_split, usize::MAX);
    }

    #[test]
    fn resolve_min_samples_fractions() {
        let mut p = TreeParams::classifier_defaults();
        // leaf ceil(0.1*25)=3; split ceil(0.1*25)=3 then max(2, .) then max(., 2*3)=6.
        p.min_samples_leaf = MinSamples::Fraction(0.1);
        p.min_samples_split = MinSamples::Fraction(0.1);
        let r = p.resolve(25, 4).unwrap();
        assert_eq!(r.min_samples_leaf, 3);
        assert_eq!(r.min_samples_split, 6);
    }

    #[test]
    fn resolve_rejects_oversized_feature_count() {
        let mut p = TreeParams::classifier_defaults();
        p.max_features = MaxFeatures::Count(5);
        assert!(p.resolve(100, 4).is_err());
        assert!(p.resolve(100, 5).is_ok());
    }

    #[test]
    fn validate_rejects_bad_fractions() {
        let mut p = TreeParams::classifier_defaults();
        p.min_weight_fraction_leaf = 0.6;
        assert!(p.validate(true).is_err());
        let mut p = TreeParams::classifier_defaults();
        p.min_samples_leaf = MinSamples::Fraction(1.5);
        assert!(p.validate(true).is_err());
        let mut p = TreeParams::classifier_defaults();
        p.min_impurity_decrease = -1.0;
        assert!(p.validate(true).is_err());
    }
}
