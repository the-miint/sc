//! Cross-validated recursive feature elimination (RFECV), equivalent to q2's
//! `_rfecv_feature_selection` (`utilities.py:198`) wrapping
//! `sklearn.feature_selection.RFECV`.
//!
//! Building follows the design's decomposition (design §"RFE-selection"):
//!
//! - **[`bookkeeping`] (deterministic, TIGHT tier)** — step resolution, the
//!   elimination schedule, ranking, and the score-curve x-axis are matched
//!   byte-exactly to scikit-learn 1.4.2 given fixed per-round importances.
//! - **[`rfecv`] (stochastic, AGGREGATE tier)** — the loop that *produces* those
//!   importances/scores by refitting a [`RandomForest`] each round is validated
//!   statistically (selected-feature Jaccard overlap vs q2 within a band), since
//!   our forest and CV folds are not sklearn-identical.
//!
//! Algorithm (matching sklearn 1.4.2): resolve `step`; build `n_splits` CV folds
//! (StratifiedKFold for classification, KFold for regression); for each fold, run
//! progressive RFE — at each surviving feature-count, refit the forest on the
//! training rows/surviving columns and score the held-out rows (accuracy /
//! coefficient-of-determination R²), then drop the least-important `step`
//! features; average scores across folds per feature-count; pick the count with
//! the best mean score (ties → fewest features); finally re-run RFE on all data
//! down to that count to get the selected set.

mod bookkeeping;

use crate::cv::{argmax_first, score_fold};
use crate::data::{FeatureMatrix, Targets};
use crate::forest::{RandomForest, RandomForestParams};
use crate::partition::{k_fold, stratified_k_fold};
use crate::rng::tree_seed;
use crate::{Error, Result};

/// How many features to eliminate per round — sklearn's `step` (int or fraction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RfeStep {
    /// Remove an explicit count each round (validated `>= 1`).
    Count(usize),
    /// Remove `max(1, ⌊fraction·n_features⌋)` each round (fraction in `(0, 1)`).
    Fraction(f64),
}

/// Parameters for [`rfecv`]. Defaults mirror q2/sklearn: `step = Count(1)`,
/// `min_features_to_select = 1`, `n_splits = 5`.
#[derive(Debug, Clone, PartialEq)]
pub struct RfecvParams {
    /// The estimator refit on the surviving features each round.
    pub forest: RandomForestParams,
    /// Features eliminated per round.
    pub step: RfeStep,
    /// The fewest features the elimination is allowed to reach.
    pub min_features_to_select: usize,
    /// Number of CV folds.
    pub n_splits: usize,
}

impl RfecvParams {
    /// q2/sklearn RFECV defaults over a classifier forest (`max_features='sqrt'`).
    pub fn classifier_defaults() -> Self {
        RfecvParams {
            forest: RandomForestParams::classifier_defaults(),
            step: RfeStep::Count(1),
            min_features_to_select: 1,
            n_splits: 5,
        }
    }

    /// q2/sklearn RFECV defaults over a regressor forest (`max_features=all`).
    pub fn regressor_defaults() -> Self {
        RfecvParams {
            forest: RandomForestParams::regressor_defaults(),
            ..Self::classifier_defaults()
        }
    }
}

/// The outcome of [`rfecv`].
#[derive(Debug, Clone, PartialEq)]
pub struct RfecvResult {
    /// Selected original-feature indices (sklearn `support_`), ascending.
    pub selected: Vec<usize>,
    /// Per original feature, its RFE rank (`1` == selected); sklearn `ranking_`.
    pub ranking: Vec<usize>,
    /// The chosen number of features (sklearn `n_features_`).
    pub n_optimal: usize,
    /// Score-curve x-axis: feature counts, ascending (fewest first).
    pub grid_n_features: Vec<usize>,
    /// Mean CV score at each grid point, aligned to `grid_n_features`.
    pub grid_scores: Vec<f64>,
}

/// Run RFECV on `x`/`y` under `params`, seeding all randomness from `master_seed`.
///
/// Deterministic under a fixed seed (CV folds and every per-round forest are
/// seeded), so the whole selection is reproducible. Errors (`InvalidInput`): x/y
/// sample mismatch; a criterion/task mismatch or bad forest params;
/// `min_features_to_select` outside `1..=n_features`; a bad `step`; or an invalid
/// CV configuration (surfaced by the partitioner).
pub fn rfecv(
    x: &FeatureMatrix,
    y: &Targets,
    params: &RfecvParams,
    master_seed: u64,
) -> Result<RfecvResult> {
    let n = x.n_samples();
    if y.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "rfecv: x has {n} samples but y has {}",
            y.n_samples()
        )));
    }
    let n_features = x.n_features();
    let is_classification = matches!(y, Targets::Categorical { .. });
    params.forest.validate(is_classification)?;
    if params.min_features_to_select < 1 || params.min_features_to_select > n_features {
        return Err(Error::InvalidInput(format!(
            "rfecv: min_features_to_select {} must be in 1..={n_features}",
            params.min_features_to_select
        )));
    }
    let step = bookkeeping::resolve_step(params.step, n_features)?;

    // CV folds: stratified for classification, plain for regression (q2 policy).
    let folds = match y.strata() {
        Some((codes, k)) => stratified_k_fold(codes, k, params.n_splits, master_seed)?,
        None => k_fold(n, params.n_splits, master_seed)?,
    };

    let grid = bookkeeping::grid_counts(n_features, step, params.min_features_to_select); // descending
    let mut score_sum = vec![0.0f64; grid.len()];
    let mut fit_seq: u64 = 1; // per-forest seed counter (sequential ⇒ deterministic)

    for f in 0..folds.n_splits() {
        let train = folds.train(f);
        let test = folds.test(f);
        let x_tr = x.select_rows(&train);
        let y_tr = y.select(&train);
        let x_te = x.select_rows(test);
        let y_te = y.select(test);

        let mut surviving: Vec<usize> = (0..n_features).collect();
        for (gi, &count) in grid.iter().enumerate() {
            let xs_tr = x_tr.select_columns(&surviving);
            let xs_te = x_te.select_columns(&surviving);
            let forest = RandomForest::fit(
                &xs_tr,
                &y_tr,
                &params.forest,
                tree_seed(master_seed, fit_seq),
            )?;
            fit_seq += 1;
            score_sum[gi] += score_fold(&forest, &xs_te, &y_te);

            if gi + 1 < grid.len() {
                let importances = forest.feature_importances();
                // Read the removal from the precomputed grid, but keep it tied to
                // the one shared definition so the loop can't silently diverge
                // from grid_counts / run_rfe (all three must agree — TIGHT tier).
                let remove = count - grid[gi + 1];
                debug_assert_eq!(
                    remove,
                    bookkeeping::clamped_step(step, count, params.min_features_to_select),
                    "rfecv grid step diverged from bookkeeping::clamped_step"
                );
                let drop_local = bookkeeping::lowest_indices(&importances, remove);
                let mut keep = vec![true; surviving.len()];
                for &local in &drop_local {
                    keep[local] = false;
                }
                surviving = surviving
                    .iter()
                    .zip(keep)
                    .filter_map(|(&s, k)| k.then_some(s))
                    .collect();
            }
        }
    }

    // Mean per grid point, then reverse to ascending feature count (sklearn's
    // cv_results_ order) so argmax's first-max tie-break favors fewest features.
    let inv = 1.0 / folds.n_splits() as f64;
    let grid_scores: Vec<f64> = score_sum.iter().rev().map(|s| s * inv).collect();
    let grid_n_features: Vec<usize> = grid.iter().rev().copied().collect();
    let n_optimal = grid_n_features[argmax_first(&grid_scores)];

    // Final RFE on all data down to n_optimal → support_/ranking_.
    let final_rfe = bookkeeping::run_rfe(n_features, step, n_optimal, |surviving| {
        let xs = x.select_columns(surviving);
        let forest = RandomForest::fit(&xs, y, &params.forest, tree_seed(master_seed, fit_seq))?;
        fit_seq += 1;
        Ok(forest.feature_importances())
    })?;
    let selected: Vec<usize> = (0..n_features).filter(|&i| final_rfe.support[i]).collect();

    Ok(RfecvResult {
        selected,
        ranking: final_rfe.ranking,
        n_optimal,
        grid_n_features,
        grid_scores,
    })
}

#[cfg(feature = "persist")]
pub use with_persist::fit_selected;

#[cfg(feature = "persist")]
mod with_persist {
    use super::*;
    use crate::persist::ModelBundle;

    /// Run [`rfecv`], then refit a forest on just the selected columns and bundle
    /// it for persistence with `rfe_selection` populated (the original-feature
    /// indices to keep at predict time). Feeds the M5 envelope.
    pub fn fit_selected(
        x: &FeatureMatrix,
        y: &Targets,
        params: &RfecvParams,
        master_seed: u64,
    ) -> Result<(ModelBundle, RfecvResult)> {
        let result = rfecv(x, y, params, master_seed)?;
        let xs = x.select_columns(&result.selected);
        let forest = RandomForest::fit(&xs, y, &params.forest, tree_seed(master_seed, 0))?;
        let bundle = ModelBundle {
            forest,
            params: Some(params.forest.clone()),
            feature_ids: None,
            rfe_selection: Some(result.selected.clone()),
        };
        Ok((bundle, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MaxFeatures;

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

    // Two informative features (0,1) + two noise features (2,3); classes split on
    // feature 0/1. RFECV should keep the informative ones and shrink the count.
    fn toy() -> (FeatureMatrix, Targets) {
        let x = dense(&[
            &[0.0, 0.0, 1.0, 2.0],
            &[0.0, 1.0, 3.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[5.0, 5.0, 2.0, 0.0],
            &[5.0, 6.0, 0.0, 3.0],
            &[6.0, 5.0, 1.0, 1.0],
            &[0.0, 1.0, 2.0, 2.0],
            &[6.0, 6.0, 0.0, 0.0],
        ]);
        let y = Targets::categorical(&["a", "a", "a", "b", "b", "b", "a", "b"]).unwrap();
        (x, y)
    }

    #[test]
    fn rfecv_runs_and_is_reproducible() {
        let (x, y) = toy();
        let mut p = RfecvParams::classifier_defaults();
        p.forest.n_estimators = 8;
        p.forest.tree.max_features = MaxFeatures::Sqrt;
        p.n_splits = 3;
        let a = rfecv(&x, &y, &p, 0).unwrap();
        let b = rfecv(&x, &y, &p, 0).unwrap();
        assert_eq!(a, b, "same seed ⇒ identical selection");
        // Shape invariants: selection non-empty, within range, ranking rank-1 count
        // equals the chosen count, curve ascending covering all counts.
        assert!(!a.selected.is_empty() && a.selected.len() == a.n_optimal);
        assert_eq!(a.ranking.iter().filter(|&&r| r == 1).count(), a.n_optimal);
        assert_eq!(a.grid_n_features, vec![1, 2, 3, 4]);
        assert_eq!(a.grid_scores.len(), 4);
        assert!(a.selected.iter().all(|&i| i < 4));
    }

    #[test]
    fn rfecv_rejects_bad_params() {
        let (x, y) = toy();
        let mut p = RfecvParams::classifier_defaults();
        p.n_splits = 3;
        // min_features_to_select out of range.
        p.min_features_to_select = 5; // > 4 features
        assert!(rfecv(&x, &y, &p, 0).is_err());
        // x/y mismatch.
        let mut p2 = RfecvParams::classifier_defaults();
        p2.n_splits = 3;
        let y_short = Targets::categorical(&["a", "b"]).unwrap();
        assert!(rfecv(&x, &y_short, &p2, 0).is_err());
        // Regressor params on categorical target.
        let mut p3 = RfecvParams::regressor_defaults();
        p3.n_splits = 3;
        assert!(rfecv(&x, &y, &p3, 0).is_err());
    }
}
