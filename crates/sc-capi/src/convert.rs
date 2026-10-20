//! Map the C-ABI `sc_rf_params_t` POD struct onto sc-core's typed parameter
//! objects. Pure, total functions — every field is translated; validation is
//! left to sc-core's own `validate()`/`resolve()` (surfaced as
//! `SC_ERR_INVALID_INPUT` by the caller), so this layer never needs to fail.
//!
//! Counts arrive as 64-bit integers and become `usize` through [`count`], never
//! `as`: on wasm32 `usize` is 32 bits, and `as` wraps.

use sc_core::{
    rf_classifier_grid, rf_regressor_grid, Criterion, MaxFeatures, MaxSamples, MinSamples,
    NcvParams, RandomForestParams, RfeStep, RfecvParams, TreeParams, TuningParams,
};

use crate::{
    sc_criterion_t, sc_max_features_kind_t, sc_max_features_t, sc_max_samples_kind_t,
    sc_max_samples_t, sc_min_samples_kind_t, sc_min_samples_t, sc_rf_params_t,
};

/// A non-negative count from the C ABI as a `T` (`usize` in use), saturating at
/// `max` (`usize::MAX`). On wasm32 `as usize` wraps 2^32 + 1 to 1, which would
/// turn a `min_samples_leaf` that forbids every split into one that allows all
/// of them. Saturating keeps the meaning instead: no wasm32 table has 2^32
/// samples or features, so `usize::MAX` gives the same result there that the
/// original count gives on a 64-bit target, where it converts losslessly. sc-core
/// then accepts or rejects it exactly as it does natively (a `max_samples` past
/// n is an error on both; a huge `min_samples_leaf` grows a stump on both).
fn count<T: TryFrom<u64>>(v: u64, max: T) -> T {
    T::try_from(v).unwrap_or(max)
}

fn criterion(c: sc_criterion_t) -> Criterion {
    match c {
        sc_criterion_t::SC_CRITERION_GINI => Criterion::Gini,
        sc_criterion_t::SC_CRITERION_ENTROPY => Criterion::Entropy,
        sc_criterion_t::SC_CRITERION_SQUARED_ERROR => Criterion::SquaredError,
    }
}

fn max_features(m: &sc_max_features_t) -> MaxFeatures {
    match m.kind {
        sc_max_features_kind_t::SC_MAX_FEATURES_ALL => MaxFeatures::All,
        sc_max_features_kind_t::SC_MAX_FEATURES_SQRT => MaxFeatures::Sqrt,
        sc_max_features_kind_t::SC_MAX_FEATURES_LOG2 => MaxFeatures::Log2,
        sc_max_features_kind_t::SC_MAX_FEATURES_FRACTION => MaxFeatures::Fraction(m.fraction),
        sc_max_features_kind_t::SC_MAX_FEATURES_COUNT => {
            MaxFeatures::Count(count(m.count, usize::MAX))
        }
    }
}

fn min_samples(m: &sc_min_samples_t) -> MinSamples {
    match m.kind {
        sc_min_samples_kind_t::SC_MIN_SAMPLES_COUNT => {
            MinSamples::Count(count(m.count, usize::MAX))
        }
        sc_min_samples_kind_t::SC_MIN_SAMPLES_FRACTION => MinSamples::Fraction(m.fraction),
    }
}

fn max_samples(m: &sc_max_samples_t) -> MaxSamples {
    match m.kind {
        sc_max_samples_kind_t::SC_MAX_SAMPLES_ALL => MaxSamples::All,
        sc_max_samples_kind_t::SC_MAX_SAMPLES_COUNT => {
            MaxSamples::Count(count(m.count, usize::MAX))
        }
        sc_max_samples_kind_t::SC_MAX_SAMPLES_FRACTION => MaxSamples::Fraction(m.fraction),
    }
}

/// sklearn's `step`: an integer count (`>= 1`) or a fraction in `(0, 1)`. The C
/// side carries one `double`; `>= 1.0` reads as `Count`, otherwise `Fraction`.
fn rfe_step(step: f64) -> RfeStep {
    if step >= 1.0 {
        RfeStep::Count(step as usize)
    } else {
        RfeStep::Fraction(step)
    }
}

/// CV fold count for the tuning / cross-validation workflows. `cv <= 1` (unset /
/// nonsensical) falls back to sklearn's default of 5; a real value passes
/// through for sc-core to accept or reject.
fn cv_splits(p: &sc_rf_params_t) -> usize {
    if p.cv >= 2 {
        count(p.cv as u64, usize::MAX)
    } else {
        5
    }
}

pub(crate) fn tree_params(p: &sc_rf_params_t) -> TreeParams {
    TreeParams {
        criterion: criterion(p.criterion),
        // sklearn convention mirrored on the ABI: max_depth <= 0 means unbounded.
        max_depth: if p.max_depth <= 0 {
            None
        } else {
            Some(count(p.max_depth as u64, usize::MAX))
        },
        min_samples_split: min_samples(&p.min_samples_split),
        min_samples_leaf: min_samples(&p.min_samples_leaf),
        min_weight_fraction_leaf: p.min_weight_fraction_leaf,
        max_features: max_features(&p.max_features),
        min_impurity_decrease: p.min_impurity_decrease,
    }
}

pub(crate) fn rf_params(p: &sc_rf_params_t) -> RandomForestParams {
    RandomForestParams {
        tree: tree_params(p),
        // A negative/zero count maps to 0, which `validate()` rejects cleanly; a
        // count past usize::MAX fails fit's allocation, as it does natively.
        n_estimators: u64::try_from(p.n_estimators).map_or(0, |n| count(n, usize::MAX)),
        bootstrap: p.bootstrap,
        max_samples: max_samples(&p.max_samples),
    }
}

pub(crate) fn rfecv_params(p: &sc_rf_params_t) -> RfecvParams {
    RfecvParams {
        forest: rf_params(p),
        step: rfe_step(p.rfe_step),
        min_features_to_select: 1,
        n_splits: cv_splits(p),
    }
}

pub(crate) fn tuning_params(p: &sc_rf_params_t, is_classification: bool) -> TuningParams {
    TuningParams {
        base: rf_params(p),
        grid: if is_classification {
            rf_classifier_grid()
        } else {
            rf_regressor_grid()
        },
        n_iter: 20,
        n_splits: cv_splits(p),
    }
}

pub(crate) fn ncv_params(p: &sc_rf_params_t, is_classification: bool) -> NcvParams {
    NcvParams {
        forest: rf_params(p),
        n_splits: cv_splits(p),
        tuning: if p.parameter_tuning {
            Some(tuning_params(p, is_classification))
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::count;

    #[test]
    fn count_saturates_where_as_wraps() {
        // The u32 instance is what `count(_, usize::MAX)` is on wasm32.
        let past = u64::from(u32::MAX) + 2;
        assert_eq!(past as u32, 1, "`as` wraps");
        assert_eq!(count(past, u32::MAX), u32::MAX);
        assert_eq!(count(u64::from(u32::MAX), u32::MAX), u32::MAX);
        assert_eq!(count(7, u32::MAX), 7);
        // On a 64-bit target the conversion is lossless.
        assert_eq!(count(past, usize::MAX), past as usize);
    }
}
