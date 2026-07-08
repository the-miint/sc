//! Deterministic RFE elimination bookkeeping — the part matched byte-exactly to
//! scikit-learn 1.4.2 `_rfe.py` given fixed per-round importances (design's TIGHT
//! tier). Pure: no RNG, no estimator. The stochastic forest fitting that
//! *produces* the importances and scores lives in [`super`].

use super::RfeStep;
use crate::{Error, Result};

/// Resolve sklearn's `step` to an absolute per-round removal count, computed once
/// from the initial `n_features` (`_rfe.py`): `Count(c) → c`; `Fraction(f) →
/// max(1, ⌊f·n_features⌋)`. Errors on `Count(0)` or a fraction outside `(0, 1)`.
pub(crate) fn resolve_step(step: RfeStep, n_features: usize) -> Result<usize> {
    match step {
        RfeStep::Count(0) => Err(Error::InvalidInput("rfe step count must be >= 1".into())),
        RfeStep::Count(c) => Ok(c),
        RfeStep::Fraction(f) => {
            if !(f.is_finite() && 0.0 < f && f < 1.0) {
                return Err(Error::InvalidInput(format!(
                    "rfe step fraction {f} must be in (0, 1)"
                )));
            }
            Ok(((f * n_features as f64).floor() as usize).max(1))
        }
    }
}

/// The **descending** sequence of surviving feature counts at which a fit/score
/// happens, from `n_features` down to `min_features` (the last entry is exactly
/// `min_features`), removing `step` per round (clamped on the final round).
/// E.g. `n=6, step=2, min=1 → [6, 4, 2, 1]`. Reversed, this is the score-curve
/// x-axis. Precondition: `1 <= min_features <= n_features`.
pub(crate) fn grid_counts(n_features: usize, step: usize, min_features: usize) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut c = n_features;
    while c > min_features {
        counts.push(c);
        c -= step.min(c - min_features);
    }
    counts.push(min_features);
    counts
}

/// The `k` lowest-importance positions of `importances`, ascending by importance
/// with ties broken by ascending index — the features eliminated this round.
/// (sklearn uses an unstable `argsort`; on the **distinct** importances the TIGHT
/// fixtures use, both agree, and the stable rule keeps our own output
/// deterministic when real-data ties occur.)
pub(crate) fn lowest_indices(importances: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..importances.len()).collect();
    order.sort_by(|&a, &b| {
        importances[a]
            .partial_cmp(&importances[b])
            .expect("finite importances")
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// q2's sklearn-<1.5 score-curve x-axis reconstruction
/// (`utilities.py:_extract_rfe_scores`), reproduced verbatim: `x[i] = n_features −
/// (n_grid−1−i)·step`, ascending, with only the first entry clamped up to 1.
/// Values are `i64` because the pre-clamp arithmetic can be `<= 0` exactly as in
/// q2. For the q2 config (`min_features_to_select == 1`) this equals
/// `grid_counts` reversed. Test-only: `rfecv` derives its (always-correct) x-axis
/// from `grid_counts`; this reproduces q2's formula to prove the two agree.
#[cfg(test)]
pub(crate) fn curve_x_axis(n_features: usize, step: usize, n_grid: usize) -> Vec<i64> {
    let (nf, s) = (n_features as i64, step as i64);
    let mut x: Vec<i64> = (0..n_grid as i64).rev().map(|k| nf - k * s).collect();
    if let Some(first) = x.first_mut() {
        if *first < 1 {
            *first = 1;
        }
    }
    x
}

/// Index of the maximum, taking the first (lowest-index) max on ties — matching
/// `np.argmax`. On the ascending-by-feature-count score curve, the first max is
/// the **fewest** features, which is sklearn RFECV's optimal-count tie-break.
pub(crate) fn argmax_first(scores: &[f64]) -> usize {
    let mut best = 0;
    let mut best_v = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_v {
            best_v = s;
            best = i;
        }
    }
    best
}

/// The outcome of an RFE elimination: sklearn's `support_` (kept mask) and
/// `ranking_` (1 == selected; higher == eliminated earlier).
pub(crate) struct Rfe {
    pub support: Vec<bool>,
    pub ranking: Vec<usize>,
}

/// Recursive feature elimination down to `target` surviving features. Each round
/// asks `importances_of(surviving)` for the current survivors' importances
/// (aligned to the surviving order), drops the lowest `step` (clamped so it never
/// overshoots `target`), and — matching sklearn's `ranking_[~support] += 1` —
/// bumps the rank of every already-eliminated feature. Precondition:
/// `1 <= target <= n_features`.
pub(crate) fn run_rfe<F>(
    n_features: usize,
    step: usize,
    target: usize,
    mut importances_of: F,
) -> Result<Rfe>
where
    F: FnMut(&[usize]) -> Result<Vec<f64>>,
{
    let mut support = vec![true; n_features];
    let mut ranking = vec![1usize; n_features];
    let mut surviving: Vec<usize> = (0..n_features).collect();
    while surviving.len() > target {
        let importances = importances_of(&surviving)?;
        debug_assert_eq!(importances.len(), surviving.len());
        let remove = step.min(surviving.len() - target);
        for &local in &lowest_indices(&importances, remove) {
            support[surviving[local]] = false;
        }
        for (feat, kept) in support.iter().enumerate() {
            if !kept {
                ranking[feat] += 1;
            }
        }
        surviving = (0..n_features).filter(|&i| support[i]).collect();
    }
    Ok(Rfe { support, ranking })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_step_int_and_fraction() {
        assert_eq!(resolve_step(RfeStep::Count(3), 10).unwrap(), 3);
        // 0.34 * 6 = 2.04 -> floor 2 (== sklearn int(max(1, 0.34*6))).
        assert_eq!(resolve_step(RfeStep::Fraction(0.34), 6).unwrap(), 2);
        // 0.5 * 6 = 3.
        assert_eq!(resolve_step(RfeStep::Fraction(0.5), 6).unwrap(), 3);
        // Tiny fraction floors to 0 -> clamped to 1.
        assert_eq!(resolve_step(RfeStep::Fraction(0.01), 10).unwrap(), 1);
        assert!(resolve_step(RfeStep::Count(0), 10).is_err());
        assert!(resolve_step(RfeStep::Fraction(1.0), 10).is_err());
        assert!(resolve_step(RfeStep::Fraction(0.0), 10).is_err());
    }

    #[test]
    fn grid_counts_matches_sklearn_shapes() {
        assert_eq!(grid_counts(6, 1, 1), vec![6, 5, 4, 3, 2, 1]);
        assert_eq!(grid_counts(6, 2, 1), vec![6, 4, 2, 1]);
        assert_eq!(grid_counts(6, 3, 1), vec![6, 3, 1]);
        assert_eq!(grid_counts(6, 1, 2), vec![6, 5, 4, 3, 2]);
        assert_eq!(grid_counts(5, 5, 1), vec![5, 1]); // one big step then final
        assert_eq!(grid_counts(4, 1, 4), vec![4]); // already at target
    }

    #[test]
    fn curve_x_axis_reproduces_q2() {
        // step=1: [1..6]; step=2 (grid len 4): pre-clamp [0,2,4,6] -> [1,2,4,6].
        assert_eq!(curve_x_axis(6, 1, 6), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(curve_x_axis(6, 2, 4), vec![1, 2, 4, 6]);
        // For the q2 config (min=1) it equals grid_counts reversed.
        let g = grid_counts(6, 2, 1);
        let asc: Vec<i64> = g.iter().rev().map(|&c| c as i64).collect();
        assert_eq!(curve_x_axis(6, 2, g.len()), asc);
    }

    // With fixed distinct importances, run_rfe must reproduce sklearn's ranking_.
    #[test]
    fn run_rfe_reproduces_sklearn_ranking() {
        // Importances per ORIGINAL feature (stable across rounds, like a mock that
        // returns column-mean importances). Ascending importance order defines the
        // elimination order.
        let imp = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2];
        let importances_of =
            |surv: &[usize]| -> Result<Vec<f64>> { Ok(surv.iter().map(|&i| imp[i]).collect()) };

        // step=1 -> sklearn ranking_ = [3, 6, 1, 4, 2, 5] (probed).
        let r = run_rfe(6, 1, 1, importances_of).unwrap();
        assert_eq!(r.ranking, vec![3, 6, 1, 4, 2, 5]);
        assert_eq!(r.support, vec![false, false, true, false, false, false]);

        // step=2 -> sklearn ranking_ = [3, 4, 1, 3, 2, 4] (probed).
        let r2 = run_rfe(6, 2, 1, importances_of).unwrap();
        assert_eq!(r2.ranking, vec![3, 4, 1, 3, 2, 4]);
    }

    #[test]
    fn argmax_first_takes_fewest_on_ties() {
        // Ascending-by-count curve; tie between index 2 and 4 -> first (fewer).
        assert_eq!(argmax_first(&[0.5, 0.6, 0.9, 0.8, 0.9, 0.75]), 2);
        assert_eq!(argmax_first(&[0.99, 0.6, 0.7]), 0);
    }
}

/// TIGHT-tier oracle: the deterministic bookkeeping reproduces scikit-learn 1.4.2
/// exactly, given the fixed (distinct) per-feature importances a scripted mock
/// estimator produced. Hits the pure fns directly (design: "RFE-selection is a
/// separate fn so tight-tier tests hit it directly"); hermetic — just the
/// committed fixtures from `oracle/generator/gen_rfe.py`.
#[cfg(test)]
mod oracle {
    use super::*;
    use sc_oracle::{load_dir, OracleCase, Payload};

    fn step_of(step: f64) -> RfeStep {
        // sklearn treats a float >= 1 as int(step); a fraction < 1 as a fraction.
        if step >= 1.0 {
            RfeStep::Count(step as usize)
        } else {
            RfeStep::Fraction(step)
        }
    }

    #[test]
    fn rfe_bookkeeping_matches_oracle() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../oracle/fixtures/rfe");
        let cases: Vec<OracleCase> = load_dir(dir);
        let mut n = 0;
        for case in &cases {
            let Payload::RfeTight(c) = &case.payload else {
                continue;
            };
            n += 1;
            let id = &case.id;
            let step = step_of(c.step);

            // Step resolution.
            let resolved = resolve_step(step, c.n_features).unwrap();
            assert_eq!(resolved, c.expected.resolved_step, "{id}: resolved_step");

            // Elimination bookkeeping: RFE down to 1 with the stable importances.
            let importances_of = |surv: &[usize]| -> Result<Vec<f64>> {
                Ok(surv.iter().map(|&i| c.importances[i]).collect())
            };
            let rfe = run_rfe(
                c.n_features,
                resolved,
                c.min_features_to_select,
                importances_of,
            )
            .unwrap();
            assert_eq!(rfe.ranking, c.expected.rfe_ranking, "{id}: ranking_");
            assert_eq!(rfe.support, c.expected.rfe_support, "{id}: support_");

            // Score-curve x-axis (q2's reconstruction) and the optimal-count rule.
            let x = curve_x_axis(c.n_features, resolved, c.expected.grid_scores.len());
            assert_eq!(x, c.expected.grid_n_features, "{id}: curve x-axis");
            let best = argmax_first(&c.expected.grid_scores);
            assert_eq!(
                x[best], c.expected.n_optimal as i64,
                "{id}: n_optimal from argmax"
            );
        }
        assert!(n >= 4, "expected >= 4 rfe_tight fixtures, saw {n}");
    }
}
