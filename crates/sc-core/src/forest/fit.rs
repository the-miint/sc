//! The grow path (AGGREGATE tier): [`RandomForest::fit`] and its private
//! per-tree build seam.
//!
//! This path is NOT byte-matched to sklearn — the bootstrap draw and per-node
//! feature subsampling are our own deterministic [`DetRng`] rules, not numpy's
//! MT19937 or sklearn's splitter RNG. It is instead validated statistically
//! (test scores land inside sklearn's own seed-to-seed band; see the aggregate
//! oracle) and for **thread-count invariance**: because each tree's RNG is
//! seeded by the pure `tree_seed(master_seed, i)` (independent of build order)
//! and the per-tree build is a pure function, a forest grown with 1 vs N threads
//! is byte-identical (`tests/determinism.rs`).

use crate::data::{FeatureMatrix, Targets};
use crate::exec::try_indexed_map;
use crate::rng::DetRng;
use crate::tree::{grow, Task, Tree};
use crate::{Error, Result};

use super::model::RandomForest;
use super::params::RandomForestParams;

impl RandomForest {
    /// Fit a forest to `x`/`y` under `params`, seeding all randomness from
    /// `master_seed`.
    ///
    /// Each tree `i` trains on its own bootstrap sample (drawn with replacement
    /// when `params.bootstrap`) and grows with per-node feature subsampling, both
    /// driven by a single [`DetRng`] seeded from `tree_seed(master_seed, i)`.
    /// The build fans out over the [`try_indexed_map`] shim, so the result is
    /// identical regardless of thread count.
    ///
    /// Errors (`InvalidInput`): `x` and `y` disagree on sample count; the
    /// criterion does not match the task; [`RandomForestParams::validate`] or
    /// [`RandomForestParams::resolve_n_bootstrap`] reject the parameters;
    /// `n_estimators` trees cannot even be allocated; or a per-tree [`grow`]
    /// fails.
    pub fn fit(
        x: &FeatureMatrix,
        y: &Targets,
        params: &RandomForestParams,
        master_seed: u64,
    ) -> Result<RandomForest> {
        let n = x.n_samples();
        if y.n_samples() != n {
            return Err(Error::InvalidInput(format!(
                "fit: x has {n} samples but y has {}",
                y.n_samples()
            )));
        }
        let (task, classes) = match y {
            Targets::Categorical { classes, .. } => (
                Task::Classification {
                    n_classes: classes.len(),
                },
                Some(classes.clone()),
            ),
            Targets::Numeric(_) => (Task::Regression, None),
        };
        let is_classification = matches!(task, Task::Classification { .. });
        params.validate(is_classification)?;
        let n_boot = params.resolve_n_bootstrap(n)?;
        let n_features = x.n_features();

        // The closure is pure (captures only shared refs + Copy scalars), so it
        // is Send+Sync and `try_indexed_map` returns trees in 0..n_estimators
        // order regardless of the thread count. Its fallible reservation turns
        // an unallocatable `n_estimators` into this error instead of an abort.
        let built: Vec<Result<Tree>> = try_indexed_map(params.n_estimators, |i| {
            build_tree(x, y, params, master_seed, i, n_boot)
        })
        .map_err(|e| {
            Error::InvalidInput(format!(
                "fit: n_estimators {} is too large to allocate ({e})",
                params.n_estimators
            ))
        })?;
        let trees = built.into_iter().collect::<Result<Vec<Tree>>>()?;
        Ok(RandomForest::from_parts(trees, task, n_features, classes))
    }
}

/// Grow tree `i`: draw its bootstrap sample and grow it, both from one per-tree
/// [`DetRng`]. Pure in its inputs (no shared mutable state), which is what makes
/// [`RandomForest::fit`] thread-count invariant.
fn build_tree(
    x: &FeatureMatrix,
    y: &Targets,
    params: &RandomForestParams,
    master_seed: u64,
    i: usize,
    n_boot: usize,
) -> Result<Tree> {
    let n = x.n_samples();
    let mut rng = DetRng::for_tree(master_seed, i as u64);
    if params.bootstrap {
        // Draw n_boot row indices with replacement, then grow on that resample.
        // The same rng subsequently drives grow's per-node feature subsampling.
        let sample: Vec<usize> = (0..n_boot)
            .map(|_| rng.uniform_u64(n as u64) as usize)
            .collect();
        let xb = x.select_rows(&sample);
        let yb = y.select(&sample);
        grow(&xb, &yb, &params.tree, &mut rng)
    } else {
        // No resampling: grow on all rows in order (no RNG consumed for sampling).
        grow(x, y, &params.tree, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::params::MaxSamples;
    use crate::tree::MaxFeatures;

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

    // A cleanly separable 2-feature, 2-class problem.
    fn separable() -> (FeatureMatrix, Targets) {
        let x = dense(&[
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[3.0, 3.0],
            &[3.0, 4.0],
            &[4.0, 3.0],
        ]);
        let y = Targets::categorical(&["a", "a", "a", "b", "b", "b"]).unwrap();
        (x, y)
    }

    #[test]
    fn fit_classifier_learns_separable_data() {
        let (x, y) = separable();
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 8;
        let forest = RandomForest::fit(&x, &y, &p, 0).unwrap();
        assert_eq!(forest.n_trees(), 8);
        assert_eq!(forest.task(), Task::Classification { n_classes: 2 });
        assert_eq!(
            forest.classes(),
            Some(["a".to_string(), "b".into()].as_slice())
        );
        // Predicts the training data perfectly, proba rows sum to 1.
        let pred = forest.predict_class_indices(&x).unwrap();
        assert_eq!(pred, y.codes().unwrap());
        for row in forest.predict_proba(&x).unwrap() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn fit_is_reproducible_for_a_fixed_seed() {
        let (x, y) = separable();
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 5;
        let a = RandomForest::fit(&x, &y, &p, 42).unwrap();
        let b = RandomForest::fit(&x, &y, &p, 42).unwrap();
        assert_eq!(a, b, "same seed must give an identical forest");
    }

    #[test]
    fn fit_regressor_predicts_in_range() {
        let x = dense(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let y = Targets::numeric(&[1.0, 1.5, 2.0, 8.0, 8.5, 9.0]).unwrap();
        let mut p = RandomForestParams::regressor_defaults();
        p.n_estimators = 10;
        let forest = RandomForest::fit(&x, &y, &p, 0).unwrap();
        assert_eq!(forest.task(), Task::Regression);
        assert!(forest.classes().is_none());
        for v in forest.predict_values(&x).unwrap() {
            assert!(
                (1.0..=9.0).contains(&v),
                "prediction {v} out of target range"
            );
        }
    }

    #[test]
    fn bootstrap_off_uses_all_rows() {
        // Without bootstrap every tree sees identical data + identical features
        // (max_features=All consumes no RNG), so all trees are equal.
        let (x, y) = separable();
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 4;
        p.bootstrap = false;
        p.tree.max_features = MaxFeatures::All;
        let forest = RandomForest::fit(&x, &y, &p, 7).unwrap();
        for i in 1..forest.n_trees() {
            assert_eq!(forest.tree(0), forest.tree(i), "trees should be identical");
        }
    }

    #[test]
    fn fit_rejects_bad_inputs() {
        let (x, _) = separable();
        // x/y sample mismatch.
        let y_short = Targets::categorical(&["a", "b"]).unwrap();
        assert!(
            RandomForest::fit(&x, &y_short, &RandomForestParams::classifier_defaults(), 0).is_err()
        );
        // Regressor params on categorical target (criterion/task mismatch).
        let (_, y) = separable();
        assert!(RandomForest::fit(&x, &y, &RandomForestParams::regressor_defaults(), 0).is_err());
        // max_samples count exceeding n.
        let mut p = RandomForestParams::classifier_defaults();
        p.max_samples = MaxSamples::Count(999);
        assert!(RandomForest::fit(&x, &y, &p, 0).is_err());
        // max_samples count of zero must be a clean Err, not a panic inside grow.
        p.max_samples = MaxSamples::Count(0);
        assert!(RandomForest::fit(&x, &y, &p, 0).is_err());
    }

    /// `n_estimators` sizes the per-tree output buffer before any tree is grown.
    /// Reserved infallibly, an unallocatable count reaches the alloc-error
    /// handler, which aborts the process: no `Result`, and no `catch_unwind` at
    /// the C API, can stop it (in duckdb-miint, `n_estimators := 4000000000000`
    /// killed the host this way). It must be a clean `InvalidInput` instead.
    #[test]
    fn unallocatable_n_estimators_is_an_error_not_an_abort() {
        let (x, y) = separable();
        let mut p = RandomForestParams::classifier_defaults();
        // The largest count whose byte size is still a valid layout, so the
        // request reaches the allocator (whose failure is the abort) instead of
        // tripping the capacity check; ~2^63 bytes on the 64-bit hosts these
        // tests run on. Then a count whose byte size overflows outright, as any
        // large count does on a 32-bit target such as wasm.
        let at_allocator = isize::MAX as usize / std::mem::size_of::<Result<Tree>>();
        for n in [at_allocator, usize::MAX] {
            p.n_estimators = n;
            match RandomForest::fit(&x, &y, &p, 0) {
                Err(Error::InvalidInput(msg)) => {
                    assert!(msg.contains("n_estimators"), "n_estimators={n}: {msg}")
                }
                other => panic!("n_estimators={n}: expected InvalidInput, got {other:?}"),
            }
        }
    }
}
