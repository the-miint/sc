//! Hyperparameter tuning — a `sklearn.model_selection.RandomizedSearchCV`
//! equivalent over the RandomForest parameter grids, matching q2's
//! `_tune_parameters` (`utilities.py:614`).
//!
//! Design tier is **PROPERTY** (`sc-design.html:229`): the grids and search
//! *semantics* mirror q2, but candidate selection is our own deterministic
//! scheme, not byte-matched to sklearn (whose `ParameterSampler` RNG is
//! impl-specific). Structure follows M6/RFECV: a stochastic loop here + a pure
//! RNG-free submodule ([`grid`]); a `Params`/`Result` pair; a persist-gated
//! refit-and-bundle helper.
//!
//! Algorithm: resolve the task; build `n_splits` CV folds (stratified for
//! classification, plain for regression); sample `min(n_iter, grid_size)`
//! distinct candidates *without replacement* from the grid's product space; for
//! each, mean-CV-score a forest refit per fold (accuracy / R², sklearn's default
//! `.score()`); pick the best mean score (ties → earliest sampled, `np.argmax`).

mod grid;

pub use grid::{rf_classifier_grid, rf_regressor_grid, ParamAxis};

use crate::cv::{argmax_first, score_fold};
use crate::data::{FeatureMatrix, Targets};
use crate::forest::{RandomForest, RandomForestParams};
use crate::partition::{k_fold, stratified_k_fold};
use crate::rng::{tree_seed, DetRng};
use crate::{Error, Result};

/// Parameters for [`tune`]. Defaults mirror q2/sklearn: `n_iter = 20`,
/// `n_splits = 5`, and the RF grid for the task.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningParams {
    /// The forest whose untuned fields (e.g. `n_estimators`, `min_samples_leaf`,
    /// `max_samples`) are held fixed; the grid overrides only its tuned fields.
    pub base: RandomForestParams,
    /// The declarative search space (see [`ParamAxis`]).
    pub grid: Vec<ParamAxis>,
    /// Number of candidates to sample (clamped to the grid size).
    pub n_iter: usize,
    /// Number of CV folds used to score each candidate.
    pub n_splits: usize,
}

impl TuningParams {
    /// q2/sklearn RandomForest**Classifier** tuning defaults.
    pub fn classifier_defaults() -> Self {
        TuningParams {
            base: RandomForestParams::classifier_defaults(),
            grid: rf_classifier_grid(),
            n_iter: 20,
            n_splits: 5,
        }
    }

    /// q2/sklearn RandomForest**Regressor** tuning defaults (grid omits `criterion`).
    pub fn regressor_defaults() -> Self {
        TuningParams {
            base: RandomForestParams::regressor_defaults(),
            grid: rf_regressor_grid(),
            n_iter: 20,
            n_splits: 5,
        }
    }
}

/// One evaluated candidate: its concrete params and mean CV score.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub params: RandomForestParams,
    pub mean_cv_score: f64,
}

/// The outcome of [`tune`]: the winner plus the full evaluated-candidate table
/// (in sampled order), so callers can inspect the search. No fitted model is
/// returned — refit with [`fit_tuned`] (or by the caller, e.g. M8's nested CV).
#[derive(Debug, Clone, PartialEq)]
pub struct TuningResult {
    /// The best candidate's params (sklearn `best_params_`).
    pub best_params: RandomForestParams,
    /// The best candidate's mean CV score (sklearn `best_score_`).
    pub best_score: f64,
    /// Index into `candidates` of the winner.
    pub best_index: usize,
    /// Every sampled candidate with its mean CV score, in sampled order.
    pub candidates: Vec<Candidate>,
}

/// Run a randomized hyperparameter search on `x`/`y` under `params`, seeding all
/// randomness from `master_seed`.
///
/// Deterministic under a fixed seed: the CV folds, the candidate sampling, and
/// every per-fold forest are independently seeded, so the whole search is
/// reproducible regardless of thread count. Search-only — see [`fit_tuned`] for
/// the refit.
///
/// Errors (`InvalidInput`): x/y sample mismatch; invalid base params for the
/// task; an empty grid, a zero-length axis, or a grid value incompatible with
/// the task (e.g. a `criterion` axis on a regressor); `n_iter == 0`; a
/// regression CV fold with `< 2` test samples (R² undefined); or an invalid CV
/// configuration surfaced by the partitioner.
pub fn tune(
    x: &FeatureMatrix,
    y: &Targets,
    params: &TuningParams,
    master_seed: u64,
) -> Result<TuningResult> {
    let n = x.n_samples();
    if y.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "tune: x has {n} samples but y has {}",
            y.n_samples()
        )));
    }
    let is_classification = matches!(y, Targets::Categorical { .. });
    params.base.validate(is_classification)?;
    if params.n_iter == 0 {
        return Err(Error::InvalidInput("tune: n_iter must be >= 1".into()));
    }
    validate_grid(
        &params.grid,
        &params.base,
        is_classification,
        n,
        x.n_features(),
    )?;
    let size = grid::grid_size(&params.grid)?;

    // CV folds: stratified for classification, plain for regression (q2 policy).
    let folds = match y.strata() {
        Some((codes, k)) => stratified_k_fold(codes, k, params.n_splits, master_seed)?,
        None => k_fold(n, params.n_splits, master_seed)?,
    };
    // R² is undefined on a <2-sample fold; reject deterministically rather than
    // letting a NaN mean silently decide the winner.
    if !is_classification {
        for f in 0..folds.n_splits() {
            if folds.test(f).len() < 2 {
                return Err(Error::InvalidInput(format!(
                    "tune: regression CV fold {f} has < 2 test samples; R² is undefined"
                )));
            }
        }
    }

    // Sample distinct candidate indices without replacement: shuffle the product
    // space and take the first min(n_iter, grid_size). Seeded disjointly from the
    // folds (u64::MAX tag ≠ any per-fit index, and ≠ the fold seed for every
    // master_seed — incl. 0, where tree_seed(0,0)==0 would collide).
    let effective = params.n_iter.min(size);
    let mut indices: Vec<usize> = (0..size).collect();
    DetRng::from_seed(tree_seed(master_seed, u64::MAX)).shuffle(&mut indices);
    indices.truncate(effective);

    // Fold assignment is fixed, so materialize each fold's train/test slices once
    // (select_rows rebuilds CSC — not a cheap slice) rather than per candidate.
    // Mirrors rfecv, which hoists this out of its inner loop.
    let n_splits = folds.n_splits();
    let fold_data: Vec<(FeatureMatrix, Targets, FeatureMatrix, Targets)> = (0..n_splits)
        .map(|f| {
            let train = folds.train(f);
            let test = folds.test(f);
            (
                x.select_rows(&train),
                y.select(&train),
                x.select_rows(test),
                y.select(test),
            )
        })
        .collect();

    let mut candidates = Vec::with_capacity(effective);
    let mut means = Vec::with_capacity(effective);
    for (i, &index) in indices.iter().enumerate() {
        let cand = grid::decode(&params.grid, &params.base, index);
        let mut acc = 0.0;
        for (f, (x_tr, y_tr, x_te, y_te)) in fold_data.iter().enumerate() {
            // Positional per-fit seed: deterministic regardless of loop/thread
            // order, and disjoint from the sampler (u64::MAX) and refit (0) tags.
            let seed = tree_seed(master_seed, 1 + (i as u64) * (n_splits as u64) + f as u64);
            let forest = RandomForest::fit(x_tr, y_tr, &cand, seed)?;
            acc += score_fold(&forest, x_te, y_te);
        }
        let mean = acc / n_splits as f64;
        means.push(mean);
        candidates.push(Candidate {
            params: cand,
            mean_cv_score: mean,
        });
    }

    let best_index = argmax_first(&means);
    Ok(TuningResult {
        best_params: candidates[best_index].params.clone(),
        best_score: means[best_index],
        best_index,
        candidates,
    })
}

/// Reject a grid that can't produce a valid candidate for the task, *before*
/// sampling, so the error never depends on which combos happen to be drawn.
/// Each axis value must keep `base` both `validate`- and `resolve`-valid; since
/// every check in `validate`/`resolve` depends on a single field and every axis
/// writes a single field, a valid single-axis override implies every product is
/// valid. The `resolve` probe (against `n_features`) catches the one otherwise-
/// sampling-dependent failure the public API allows: a `MaxFeatures::Count`
/// exceeding the feature count.
fn validate_grid(
    grid: &[ParamAxis],
    base: &RandomForestParams,
    is_classification: bool,
    n_samples: usize,
    n_features: usize,
) -> Result<()> {
    if grid.is_empty() {
        return Err(Error::InvalidInput("tune: empty grid".into()));
    }
    for axis in grid {
        if axis.len() == 0 {
            return Err(Error::InvalidInput(
                "tune: grid axis has no candidate values".into(),
            ));
        }
        for choice in 0..axis.len() {
            let mut probe = base.clone();
            axis.apply(choice, &mut probe);
            probe.validate(is_classification)?;
            // resolve() asserts a non-empty dataset; skip if n_samples == 0
            // (the partitioner will reject that below regardless).
            if n_samples > 0 {
                probe.tree.resolve(n_samples, n_features)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "persist")]
pub use with_persist::fit_tuned;

#[cfg(feature = "persist")]
mod with_persist {
    use super::*;
    use crate::persist::ModelBundle;

    /// Run [`tune`], then refit a forest with the winning params on all of
    /// `x`/`y` and bundle it for persistence (`params` = the tuned params). The
    /// refit is seeded `tree_seed(master_seed, 0)` — disjoint from the search's
    /// seeds. Returns the bundle and the full [`TuningResult`].
    pub fn fit_tuned(
        x: &FeatureMatrix,
        y: &Targets,
        params: &TuningParams,
        master_seed: u64,
    ) -> Result<(ModelBundle, TuningResult)> {
        let result = tune(x, y, params, master_seed)?;
        let forest = RandomForest::fit(x, y, &result.best_params, tree_seed(master_seed, 0))?;
        let bundle = ModelBundle {
            forest,
            params: Some(result.best_params.clone()),
            feature_ids: None,
            rfe_selection: None,
        };
        Ok((bundle, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Criterion, MaxFeatures, MaxSamples, MinSamples};

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

    fn fast(mut p: TuningParams, n_iter: usize) -> TuningParams {
        p.base.n_estimators = 4;
        p.base.tree.max_features = MaxFeatures::Sqrt;
        p.n_iter = n_iter;
        p.n_splits = 2;
        p
    }

    #[test]
    fn samples_exactly_n_iter_distinct_candidates() {
        let (x, y) = cls_data();
        let p = fast(TuningParams::classifier_defaults(), 8);
        let r = tune(&x, &y, &p, 7).unwrap();
        assert_eq!(r.candidates.len(), 8); // n_iter < grid_size (576)
        for i in 0..r.candidates.len() {
            for j in (i + 1)..r.candidates.len() {
                assert_ne!(
                    r.candidates[i].params, r.candidates[j].params,
                    "candidates {i},{j} not distinct"
                );
            }
        }
    }

    #[test]
    fn clamps_n_iter_to_grid_size() {
        // A tiny 2-value grid; asking for more candidates than exist clamps.
        let (x, y) = cls_data();
        let mut p = fast(TuningParams::classifier_defaults(), 100);
        p.grid = vec![ParamAxis::Bootstrap(vec![true, false])];
        let r = tune(&x, &y, &p, 1).unwrap();
        assert_eq!(r.candidates.len(), 2);
    }

    #[test]
    fn candidates_lie_within_declared_grid_and_leave_untuned_fields() {
        let (x, y) = cls_data();
        let base = RandomForestParams::classifier_defaults();
        let p = fast(TuningParams::classifier_defaults(), 12);
        let r = tune(&x, &y, &p, 3).unwrap();
        let depths = [Some(4), Some(8), Some(16), None];
        let mfs = [
            MaxFeatures::All,
            MaxFeatures::Sqrt,
            MaxFeatures::Log2,
            MaxFeatures::Fraction(0.1),
        ];
        for c in &r.candidates {
            assert!(depths.contains(&c.params.tree.max_depth));
            assert!(mfs.contains(&c.params.tree.max_features));
            // ints stay ints, fractions stay fractions.
            assert!(matches!(
                c.params.tree.min_samples_split,
                MinSamples::Fraction(f) if [0.001, 0.01, 0.1].contains(&f)
            ));
            assert!([0.0001, 0.001, 0.01].contains(&c.params.tree.min_weight_fraction_leaf));
            assert!(matches!(
                c.params.tree.criterion,
                Criterion::Gini | Criterion::Entropy
            ));
            // Untuned fields untouched.
            assert_eq!(c.params.n_estimators, 4);
            assert_eq!(c.params.tree.min_samples_leaf, base.tree.min_samples_leaf);
            assert_eq!(c.params.max_samples, MaxSamples::All);
        }
    }

    #[test]
    fn same_seed_reproducible_different_seed_differs() {
        let (x, y) = cls_data();
        let p = fast(TuningParams::classifier_defaults(), 10);
        let a = tune(&x, &y, &p, 42).unwrap();
        let b = tune(&x, &y, &p, 42).unwrap();
        assert_eq!(a, b, "same seed ⇒ identical search");
        let c = tune(&x, &y, &p, 43).unwrap();
        assert_ne!(
            a.candidates, c.candidates,
            "different seed ⇒ different sequence"
        );
    }

    #[test]
    fn best_is_argmax_of_mean_scores() {
        let (x, y) = cls_data();
        let p = fast(TuningParams::classifier_defaults(), 10);
        let r = tune(&x, &y, &p, 5).unwrap();
        // best_index is the first index attaining the max mean score.
        let max = r
            .candidates
            .iter()
            .map(|c| c.mean_cv_score)
            .fold(f64::NEG_INFINITY, f64::max);
        let expect = r
            .candidates
            .iter()
            .position(|c| c.mean_cv_score == max)
            .unwrap();
        assert_eq!(r.best_index, expect);
        assert_eq!(r.best_params, r.candidates[r.best_index].params);
        assert_eq!(r.best_score, r.candidates[r.best_index].mean_cv_score);
    }

    #[test]
    fn regression_task_uses_plain_folds_and_runs() {
        let (x, y) = reg_data();
        let p = fast(TuningParams::regressor_defaults(), 6);
        let r = tune(&x, &y, &p, 9).unwrap();
        assert_eq!(r.candidates.len(), 6);
        // Regressor grid has no criterion axis, so criterion stays the base value.
        for c in &r.candidates {
            assert_eq!(c.params.tree.criterion, Criterion::SquaredError);
        }
    }

    #[test]
    fn rejects_bad_configurations() {
        let (x, y) = cls_data();
        let base = fast(TuningParams::classifier_defaults(), 5);

        // empty grid
        let mut p = base.clone();
        p.grid = vec![];
        assert!(tune(&x, &y, &p, 0).is_err());

        // zero-length axis
        let mut p = base.clone();
        p.grid = vec![ParamAxis::Bootstrap(vec![])];
        assert!(tune(&x, &y, &p, 0).is_err());

        // n_iter = 0
        let mut p = base.clone();
        p.n_iter = 0;
        assert!(tune(&x, &y, &p, 0).is_err());

        // x/y sample mismatch
        let y_short = Targets::categorical(&["a", "b"]).unwrap();
        assert!(tune(&x, &y_short, &base, 0).is_err());

        // criterion axis on a regression task (grid incompatible with task)
        let (xr, yr) = reg_data();
        let mut pr = fast(TuningParams::regressor_defaults(), 5);
        pr.grid.push(ParamAxis::Criterion(vec![Criterion::Gini]));
        assert!(tune(&xr, &yr, &pr, 0).is_err());

        // n_splits < 2 (surfaced by the partitioner)
        let mut p = base.clone();
        p.n_splits = 1;
        assert!(tune(&x, &y, &p, 0).is_err());
    }

    #[test]
    fn rejects_resolve_incompatible_grid_deterministically() {
        // MaxFeatures::Count(999) can't resolve on 4-feature data; it's rejected
        // before sampling, so the error is seed-independent (not combo-dependent).
        let (x, y) = cls_data();
        let mut p = fast(TuningParams::classifier_defaults(), 5);
        p.grid = vec![ParamAxis::MaxFeatures(vec![
            MaxFeatures::Sqrt,
            MaxFeatures::Count(999),
        ])];
        for seed in [0u64, 1, 2, 999] {
            assert!(tune(&x, &y, &p, seed).is_err(), "seed {seed} should error");
        }
    }

    #[test]
    fn rejects_regression_fold_below_two_samples() {
        // 3 samples, 3 folds → 1-sample test folds → R² undefined → reject.
        let x = dense(&[&[1.0, 0.0], &[2.0, 1.0], &[3.0, 0.0]]);
        let y = Targets::numeric(&[1.0, 2.0, 3.0]).unwrap();
        let mut p = TuningParams::regressor_defaults();
        p.base.n_estimators = 2;
        p.n_iter = 2;
        p.n_splits = 3;
        assert!(tune(&x, &y, &p, 0).is_err());
    }
}
