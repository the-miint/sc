//! Map the C-ABI `sc_rf_params_t` POD struct onto sc-core's typed parameter
//! objects. Pure, total functions — every field is translated; validation is
//! left to sc-core's own `validate()`/`resolve()` (surfaced as
//! `SC_ERR_INVALID_INPUT` by the caller), so this layer never needs to fail.

use sc_core::{
    rf_classifier_grid, rf_regressor_grid, Criterion, MaxFeatures, MaxSamples, MinSamples,
    NcvParams, RandomForestParams, RfeStep, RfecvParams, TreeParams, TuningParams,
};

use crate::{
    sc_criterion_t, sc_max_features_kind_t, sc_max_features_t, sc_max_samples_kind_t,
    sc_max_samples_t, sc_min_samples_kind_t, sc_min_samples_t, sc_rf_params_t,
};

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
        sc_max_features_kind_t::SC_MAX_FEATURES_COUNT => MaxFeatures::Count(m.count as usize),
    }
}

fn min_samples(m: &sc_min_samples_t) -> MinSamples {
    match m.kind {
        sc_min_samples_kind_t::SC_MIN_SAMPLES_COUNT => MinSamples::Count(m.count as usize),
        sc_min_samples_kind_t::SC_MIN_SAMPLES_FRACTION => MinSamples::Fraction(m.fraction),
    }
}

fn max_samples(m: &sc_max_samples_t) -> MaxSamples {
    match m.kind {
        sc_max_samples_kind_t::SC_MAX_SAMPLES_ALL => MaxSamples::All,
        sc_max_samples_kind_t::SC_MAX_SAMPLES_COUNT => MaxSamples::Count(m.count as usize),
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
        p.cv as usize
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
            Some(p.max_depth as usize)
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
        // A negative/zero count maps to 0, which `validate()` rejects cleanly.
        n_estimators: usize::try_from(p.n_estimators).unwrap_or(0),
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
