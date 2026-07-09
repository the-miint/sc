//! Nested / out-of-fold cross-validation, equivalent to q2's
//! `classify_samples_ncv` / `regress_samples_ncv` (`classify.py:350,368` →
//! `utilities.py:nested_cross_validation` → `_fit_and_predict_cv`).
//!
//! Given `x`/`y`, [`ncv`] produces an **out-of-fold (OOF) prediction for every
//! sample**: it partitions the data into `n_splits` folds (StratifiedKFold for
//! classification, KFold for regression — q2's policy), fits a fresh
//! [`RandomForest`] on each fold's training complement, and predicts that fold's
//! held-out rows. Every sample is therefore predicted exactly once, by a model
//! that never saw it. Alongside the predictions it returns fold-averaged feature
//! importances and the per-fold scores with their mean±std (q2's printed
//! "Estimator Accuracy: μ ± σ" line).
//!
//! Design tier is **AGGREGATE** (`sc-implementation-plan.html` §M8, mirroring M4's
//! `ForestAggregate`): with q2's exact outer folds injected via
//! [`ncv_with_folds`] + [`CvFolds::from_fold_ids`], our OOF score lands inside an
//! empirically-calibrated band vs q2 (our forest RNG ≠ sklearn's; the band
//! absorbs it). [`ncv_with_folds`] is the first-class injection hook the oracle
//! drives.
//!
//! Deviations from q2 (also noted at each site):
//! 1. OOF predictions are returned in **original sample order** (q2 concatenates
//!    them in fold order). Scores are order-invariant; original order makes
//!    "every sample predicted once" trivially checkable and is more useful.
//! 2. The per-fold **regression** score is MSE (matching `regress_samples_ncv`'s
//!    `scoring=mean_squared_error`), deliberately unlike RFECV/tuning which use
//!    sklearn's default `.score()` (R²).
//! 3. `mean_score`/`std_score` use **population** std (ddof=0), matching `np.std`.
//! 4. Inner-loop tuning selection is **not** oracle-matched (per M7); the
//!    `tuning = Some` path is property-tested only, and the oracle uses
//!    `tuning = None` (q2's default).
//! 5. Probabilities are **full-width / no-NaN** even if an injected fold's train
//!    split omits a class ([`Targets::select`] preserves the full class set), so
//!    every OOF proba row has one column per class and sums to 1. On the real
//!    StratifiedKFold path no class is ever absent, so q2 emits no NaN either —
//!    this is invariant-cleanliness under arbitrary injected folds, not a win.

use crate::data::{FeatureMatrix, Targets};
use crate::forest::{RandomForest, RandomForestParams};
use crate::metrics::{accuracy, mean_squared_error};
use crate::partition::{k_fold, stratified_k_fold, CvFolds};
use crate::rng::tree_seed;
use crate::tuning::{tune, TuningParams};
use crate::{Error, Result};

/// Parameters for [`ncv`]. Defaults mirror q2: `n_splits = 5`, `tuning = None`
/// (`parameter_tuning=False`), and the RF defaults for the task.
#[derive(Debug, Clone, PartialEq)]
pub struct NcvParams {
    /// The forest fit on each outer-training fold **when `tuning` is `None`**.
    ///
    /// ⚠️ When `tuning` is `Some`, this field is **ignored** — the per-fold params
    /// come entirely from the inner search, whose fixed (non-searched) fields are
    /// supplied by `tuning`'s own `base`, a separate [`RandomForestParams`]. Set
    /// per-fold estimator options on `tuning.base`, not here, whenever tuning is
    /// on. (This mirrors q2, where `parameter_tuning=True` replaces the estimator
    /// with the tuned `best_estimator_`.)
    pub forest: RandomForestParams,
    /// Number of outer CV folds (q2's `cv`).
    pub n_splits: usize,
    /// Optional inner-loop hyperparameter tuning (q2's `parameter_tuning=True`):
    /// per outer fold, run [`tune`] on the training split and refit with the
    /// winning params. `None` skips tuning (q2's default).
    pub tuning: Option<TuningParams>,
}

impl NcvParams {
    /// q2 `classify_samples_ncv` defaults: classifier forest, 5 stratified folds,
    /// no inner tuning.
    pub fn classifier_defaults() -> Self {
        NcvParams {
            forest: RandomForestParams::classifier_defaults(),
            n_splits: 5,
            tuning: None,
        }
    }

    /// q2 `regress_samples_ncv` defaults: regressor forest, 5 plain folds, no
    /// inner tuning.
    pub fn regressor_defaults() -> Self {
        NcvParams {
            forest: RandomForestParams::regressor_defaults(),
            n_splits: 5,
            tuning: None,
        }
    }
}

/// The assembled OOF predictions, one entry per sample in **original order**.
#[derive(Debug, Clone, PartialEq)]
pub enum OofPredictions {
    /// Classification: `classes` are the sorted-unique labels (sklearn
    /// `classes_` order); `codes[i]` is sample `i`'s predicted class index into
    /// `classes`; `proba[i]` is its full-width class-probability row (sums to 1).
    Classification {
        classes: Vec<String>,
        codes: Vec<usize>,
        proba: Vec<Vec<f64>>,
    },
    /// Regression: `values[i]` is sample `i`'s predicted value.
    Regression { values: Vec<f64> },
}

/// The outcome of [`ncv`] / [`ncv_with_folds`].
#[derive(Debug, Clone, PartialEq)]
pub struct NcvResult {
    /// OOF prediction per sample (original order).
    pub predictions: OofPredictions,
    /// Fold-averaged feature importances (mean over folds of each fold-forest's
    /// MDI). Each fold's MDI sums to 1, so the mean does too — q2's
    /// `_mean_feature_importance`.
    pub importances: Vec<f64>,
    /// Per-fold score in fold order: accuracy (classification) /
    /// mean-squared-error (regression).
    pub fold_scores: Vec<f64>,
    /// Mean of `fold_scores` (q2's printed `np.mean`).
    pub mean_score: f64,
    /// Population std (ddof=0) of `fold_scores` (q2's printed `np.std`).
    pub std_score: f64,
}

/// Run OOF cross-validation on `x`/`y` under `params`, building the outer folds
/// from `master_seed` and seeding every forest from it.
///
/// Folds are stratified for classification (StratifiedKFold) and plain for
/// regression (KFold), matching q2. Deterministic under a fixed seed. Delegates
/// to [`ncv_with_folds`]; see it for the OOF loop and the full error list.
pub fn ncv(
    x: &FeatureMatrix,
    y: &Targets,
    params: &NcvParams,
    master_seed: u64,
) -> Result<NcvResult> {
    let n = x.n_samples();
    if y.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "ncv: x has {n} samples but y has {}",
            y.n_samples()
        )));
    }
    // Build the outer folds by task (q2 policy). Seeded raw from master_seed, the
    // rfecv/tuning convention; the per-fold forests use tree_seed(master_seed, ·)
    // at indices >= 1, so they never collide with this fold seed.
    let folds = match y.strata() {
        Some((codes, k)) => stratified_k_fold(codes, k, params.n_splits, master_seed)?,
        None => k_fold(n, params.n_splits, master_seed)?,
    };
    ncv_with_folds(x, y, params, &folds, master_seed)
}

/// Run OOF cross-validation over caller-supplied `folds` — the injection hook the
/// AGGREGATE oracle drives (via [`CvFolds::from_fold_ids`]) to feed q2 and our
/// forest the identical partition.
///
/// For each fold `f`, fit a forest on the training complement and predict the
/// held-out rows, scattering the predictions (and, for classification, the
/// probability rows) into their original sample positions. [`CvFolds`] guarantees
/// disjoint full coverage, so every sample is written exactly once.
///
/// Seeding is keyed off `folds.n_splits()` (call it `k`), **not**
/// `params.n_splits` — an injected fold count may differ. Fold `f`'s forest fit
/// is seeded `tree_seed(master_seed, 1 + f)`; its inner tuning (if any) is seeded
/// `tree_seed(master_seed, 1 + k + f)`. The ranges `1..=k` and `k+1..=2k` are
/// disjoint and `tree_seed` is injective per master seed, so fits and tunings
/// never share a seed. (Using `params.n_splits` for the offset while looping over
/// `k` would overlap the ranges whenever they differ — a real collision.)
///
/// Errors (`InvalidInput`): x/y sample mismatch; a `folds` sample count that
/// disagrees with the data (a caller-supplied [`CvFolds`] is not otherwise tied
/// to it); invalid forest params for the task; and anything the per-fold
/// [`RandomForest::fit`] / [`tune`] surfaces.
pub fn ncv_with_folds(
    x: &FeatureMatrix,
    y: &Targets,
    params: &NcvParams,
    folds: &CvFolds,
    master_seed: u64,
) -> Result<NcvResult> {
    let n = x.n_samples();
    if y.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "ncv: x has {n} samples but y has {}",
            y.n_samples()
        )));
    }
    if folds.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "ncv: folds cover {} samples but x has {n}",
            folds.n_samples()
        )));
    }
    let is_classification = matches!(y, Targets::Categorical { .. });
    // Validate the estimator config the run actually uses. On the no-tuning path
    // that is `params.forest`; on the tuning path `params.forest` is ignored (the
    // per-fold params come from the inner search), so validation is deferred to
    // `tune`, which checks its own `base` + grid. Validating `params.forest` here
    // regardless would reject a config that never runs while skipping the one that
    // does.
    if params.tuning.is_none() {
        params.forest.validate(is_classification)?;
    }

    let k = folds.n_splits();
    let n_features = x.n_features();
    let mut fold_scores = Vec::with_capacity(k);
    let mut importance_sum = vec![0.0f64; n_features];

    // OOF buffers; only the buffer matching the task is populated. CvFolds
    // guarantees full coverage, so every slot is overwritten (initial values are
    // never observed).
    let mut oof_codes = vec![0usize; n];
    let mut oof_proba: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut oof_values = vec![0.0f64; n];

    for f in 0..k {
        let train = folds.train(f);
        let test = folds.test(f);
        let x_tr = x.select_rows(&train);
        let y_tr = y.select(&train);
        let x_te = x.select_rows(test);
        let forest = fit_fold(&x_tr, &y_tr, params, master_seed, f as u64, k as u64)?;

        let score = match y {
            Targets::Categorical { codes: y_codes, .. } => {
                let proba = forest.predict_proba(&x_te)?;
                let pred = forest.predict_class_indices(&x_te)?;
                // Scatter into original positions (deviation 1).
                for (local, &orig) in test.iter().enumerate() {
                    oof_codes[orig] = pred[local];
                    oof_proba[orig] = proba[local].clone();
                }
                let truth: Vec<usize> = test.iter().map(|&i| y_codes[i]).collect();
                accuracy(&truth, &pred) // per-fold accuracy (q2's accuracy_score)
            }
            Targets::Numeric(values) => {
                let pred = forest.predict_values(&x_te)?;
                for (local, &orig) in test.iter().enumerate() {
                    oof_values[orig] = pred[local];
                }
                let truth: Vec<f64> = test.iter().map(|&i| values[i]).collect();
                mean_squared_error(&truth, &pred) // per-fold MSE (deviation 2)
            }
        };
        fold_scores.push(score);

        // Accumulate fold-averaged importances (each fold's MDI sums to 1). NCV
        // keeps the same column set across folds, so widths always match; the
        // debug assert fails fast if a future column-subsetting composition (e.g.
        // RFE-then-NCV) ever breaks that instead of silently truncating the zip.
        let fi = forest.feature_importances();
        debug_assert_eq!(fi.len(), n_features, "fold MDI width != n_features");
        for (acc, c) in importance_sum.iter_mut().zip(fi) {
            *acc += c;
        }
    }

    let predictions = match y {
        Targets::Categorical { classes, .. } => OofPredictions::Classification {
            classes: classes.clone(),
            codes: oof_codes,
            proba: oof_proba,
        },
        Targets::Numeric(_) => OofPredictions::Regression { values: oof_values },
    };

    let kf = k as f64;
    let importances: Vec<f64> = importance_sum.iter().map(|s| s / kf).collect();
    let mean_score = fold_scores.iter().sum::<f64>() / kf;
    // Population std (ddof=0), matching np.std (deviation 3).
    let variance = fold_scores
        .iter()
        .map(|s| (s - mean_score) * (s - mean_score))
        .sum::<f64>()
        / kf;
    let std_score = variance.sqrt();

    Ok(NcvResult {
        predictions,
        importances,
        fold_scores,
        mean_score,
        std_score,
    })
}

/// Fit fold `f`'s forest on its training split. Without tuning, fits
/// `params.forest`; with tuning, runs [`tune`] on the split first and refits with
/// the winning params. The fit is always seeded `tree_seed(master_seed, 1 + f)`;
/// the inner search (when present) is seeded `tree_seed(master_seed, 1 + k + f)`
/// — see [`ncv_with_folds`] for why the ranges are disjoint.
fn fit_fold(
    x_tr: &FeatureMatrix,
    y_tr: &Targets,
    params: &NcvParams,
    master_seed: u64,
    f: u64,
    k: u64,
) -> Result<RandomForest> {
    let fit_seed = tree_seed(master_seed, 1 + f);
    let forest_params = match &params.tuning {
        None => params.forest.clone(),
        Some(tp) => {
            let tune_seed = tree_seed(master_seed, 1 + k + f);
            tune(x_tr, y_tr, tp, tune_seed)?.best_params
        }
    };
    RandomForest::fit(x_tr, y_tr, &forest_params, fit_seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaxFeatures, ParamAxis};

    fn dense(rows: &[&[f64]]) -> FeatureMatrix {
        let (n, m) = (rows.len(), rows[0].len());
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    r.push(i as i64);
                    c.push(j as i64);
                    v.push(val);
                }
            }
        }
        FeatureMatrix::from_coo(n, m, &r, &c, &v).unwrap()
    }

    // Balanced 8-sample, 4-feature, 2-class problem (classes split on features 0/1).
    fn cls_data() -> (FeatureMatrix, Targets) {
        let x = dense(&[
            &[0.0, 0.0, 1.0, 2.0],
            &[0.0, 1.0, 3.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 1.0, 2.0, 2.0],
            &[5.0, 5.0, 2.0, 0.0],
            &[5.0, 6.0, 0.0, 3.0],
            &[6.0, 5.0, 1.0, 1.0],
            &[6.0, 6.0, 0.0, 0.0],
        ]);
        let y = Targets::categorical(&["a", "a", "a", "a", "b", "b", "b", "b"]).unwrap();
        (x, y)
    }

    // 8-sample regression problem (target ~ feature 0).
    fn reg_data() -> (FeatureMatrix, Targets) {
        let x = dense(&[
            &[1.0, 2.0, 0.0, 1.0],
            &[2.0, 0.0, 1.0, 0.0],
            &[3.0, 1.0, 0.0, 2.0],
            &[4.0, 0.0, 2.0, 1.0],
            &[5.0, 3.0, 0.0, 0.0],
            &[6.0, 0.0, 1.0, 3.0],
            &[7.0, 2.0, 0.0, 1.0],
            &[8.0, 1.0, 2.0, 0.0],
        ]);
        let y = Targets::numeric(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        (x, y)
    }

    fn fast_cls() -> NcvParams {
        let mut p = NcvParams::classifier_defaults();
        p.forest.n_estimators = 8;
        p.forest.tree.max_features = MaxFeatures::Sqrt;
        p.n_splits = 4;
        p
    }

    fn fast_reg() -> NcvParams {
        let mut p = NcvParams::regressor_defaults();
        p.forest.n_estimators = 8;
        p.n_splits = 4;
        p
    }

    #[test]
    fn classification_predicts_every_sample_once_with_valid_proba() {
        let (x, y) = cls_data();
        let p = fast_cls();
        let r = ncv(&x, &y, &p, 20260708).unwrap();
        let OofPredictions::Classification {
            classes,
            codes,
            proba,
        } = &r.predictions
        else {
            panic!("expected classification predictions");
        };
        assert_eq!(classes, &["a".to_string(), "b".into()]);
        // One prediction and one proba row per sample.
        assert_eq!(codes.len(), 8);
        assert_eq!(proba.len(), 8);
        for row in proba {
            assert_eq!(row.len(), 2, "proba row width == n_classes");
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "proba row sums to {s}");
            assert!(row.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        // Every predicted code is a valid class index.
        assert!(codes.iter().all(|&c| c < 2));
        // The argmax of each proba row equals the reported code (predict ==
        // argmax(proba)); confirms the scatter kept codes/proba aligned per sample.
        for (i, row) in proba.iter().enumerate() {
            let am = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();
            assert_eq!(am, codes[i], "proba argmax != code at {i}");
        }
    }

    #[test]
    fn regression_predicts_every_sample_with_finite_values() {
        let (x, y) = reg_data();
        let p = fast_reg();
        let r = ncv(&x, &y, &p, 20260708).unwrap();
        let OofPredictions::Regression { values } = &r.predictions else {
            panic!("expected regression predictions");
        };
        assert_eq!(values.len(), 8);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fold_scores_and_summary_match_hand_recompute() {
        let (x, y) = cls_data();
        let p = fast_cls();
        let r = ncv(&x, &y, &p, 42).unwrap();
        assert_eq!(r.fold_scores.len(), p.n_splits);
        let k = r.fold_scores.len() as f64;
        let mean = r.fold_scores.iter().sum::<f64>() / k;
        let var = r
            .fold_scores
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / k;
        assert!((r.mean_score - mean).abs() < 1e-12);
        assert!((r.std_score - var.sqrt()).abs() < 1e-12);
        // Importances are a genuine distribution (fold-mean of MDIs → sums to 1).
        assert_eq!(r.importances.len(), 4);
        assert!((r.importances.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn same_seed_reproducible_different_seed_differs() {
        let (x, y) = cls_data();
        let p = fast_cls();
        let a = ncv(&x, &y, &p, 7).unwrap();
        let b = ncv(&x, &y, &p, 7).unwrap();
        assert_eq!(a, b, "same seed ⇒ identical NcvResult");
        let c = ncv(&x, &y, &p, 8).unwrap();
        // A different seed reshuffles the folds, so the OOF assignment changes.
        assert_ne!(
            a.predictions, c.predictions,
            "different seed ⇒ different OOF"
        );
    }

    #[test]
    fn tuning_path_is_deterministic_and_reproducible() {
        let (x, y) = cls_data();
        let mut p = fast_cls();
        let mut tp = TuningParams::classifier_defaults();
        tp.base.n_estimators = 4;
        tp.base.tree.max_features = MaxFeatures::Sqrt;
        tp.grid = vec![ParamAxis::MaxDepth(vec![Some(2), None])];
        tp.n_iter = 2;
        tp.n_splits = 2;
        p.tuning = Some(tp);
        let a = ncv(&x, &y, &p, 3).unwrap();
        let b = ncv(&x, &y, &p, 3).unwrap();
        assert_eq!(a, b, "tuning path must be reproducible under a fixed seed");
        assert_eq!(a.fold_scores.len(), p.n_splits);
    }

    #[test]
    fn tuning_with_injected_fold_count_differing_from_params() {
        // The scenario the seeding doc defends: an injected fold count (here 2)
        // that differs from params.n_splits (5) WITH tuning on. Seeds must key off
        // folds.n_splits() = 2, so fits use tree_seed(·, {1,2}) and inner tunings
        // use tree_seed(·, {3,4}) — disjoint. If the offset wrongly used
        // params.n_splits = 5, the ranges {1,2} and {6,7} would still be disjoint
        // here, so we assert the load-bearing observable directly: the loop runs
        // over the INJECTED fold count, not params.n_splits, and is reproducible.
        let (x, y) = cls_data();
        // 2 outer folds, each training complement balanced 2 'a' / 2 'b' so the
        // inner 2-fold stratified tuning CV is valid.
        let folds = CvFolds::from_test_folds(8, vec![vec![0, 1, 4, 5], vec![2, 3, 6, 7]]).unwrap();
        let mut p = NcvParams::classifier_defaults();
        p.n_splits = 5; // deliberately != folds.n_splits() == 2
        p.forest.n_estimators = 4;
        p.forest.tree.max_features = MaxFeatures::Sqrt;
        let mut tp = TuningParams::classifier_defaults();
        tp.base.n_estimators = 4;
        tp.base.tree.max_features = MaxFeatures::Sqrt;
        tp.grid = vec![ParamAxis::MaxDepth(vec![Some(2), None])];
        tp.n_iter = 2;
        tp.n_splits = 2;
        p.tuning = Some(tp);

        let a = ncv_with_folds(&x, &y, &p, &folds, 5).unwrap();
        let b = ncv_with_folds(&x, &y, &p, &folds, 5).unwrap();
        assert_eq!(a, b, "tuning + injected folds must be reproducible");
        assert_eq!(
            a.fold_scores.len(),
            folds.n_splits(),
            "loop must run over folds.n_splits(), not params.n_splits"
        );
    }

    #[test]
    fn injected_folds_round_trip_matches_generated() {
        // ncv() builds stratified folds from the seed; rebuilding those exact folds
        // via from_fold_ids and calling ncv_with_folds must reproduce it identically
        // — the property the oracle relies on.
        let (x, y) = cls_data();
        let p = fast_cls();
        let (codes, kc) = y.strata().unwrap();
        let folds = stratified_k_fold(codes, kc, p.n_splits, 99).unwrap();
        let mut fold_ids = vec![0usize; 8];
        for f in 0..folds.n_splits() {
            for &i in folds.test(f) {
                fold_ids[i] = f;
            }
        }
        let rebuilt = CvFolds::from_fold_ids(&fold_ids, p.n_splits).unwrap();
        let via_folds = ncv_with_folds(&x, &y, &p, &rebuilt, 99).unwrap();
        let via_ncv = ncv(&x, &y, &p, 99).unwrap();
        assert_eq!(via_folds, via_ncv);
    }

    #[test]
    fn rejects_bad_inputs() {
        let (x, y) = cls_data();

        // x/y sample mismatch.
        let y_short = Targets::categorical(&["a", "b"]).unwrap();
        assert!(ncv(&x, &y_short, &fast_cls(), 0).is_err());

        // n_splits < 2 (surfaced by the partitioner).
        let mut p = fast_cls();
        p.n_splits = 1;
        assert!(ncv(&x, &y, &p, 0).is_err());

        // Regressor forest params on a categorical target (task mismatch).
        let mut pr = NcvParams::regressor_defaults();
        pr.n_splits = 4;
        assert!(ncv(&x, &y, &pr, 0).is_err());

        // Injected folds whose sample count disagrees with the (matched) data:
        // x and y both have 8 samples, but these folds cover only 6.
        let folds6 = CvFolds::from_test_folds(6, vec![vec![0, 1, 2], vec![3, 4, 5]]).unwrap();
        assert!(ncv_with_folds(&x, &y, &fast_cls(), &folds6, 0).is_err());
    }
}
