//! M0 determinism-harness tests: seed derivation and the thread-count-invariance
//! guarantee of the execution shim.
//!
//! # Cross-platform tripwire (M11)
//!
//! Every `*_golden*` test here freezes an output to a hardcoded constant. Those
//! constants are the load-bearing evidence for the project's native↔wasm
//! determinism guarantee: this fs-free, oracle-free suite is *also* run on a wasm
//! VM (`wasm32-wasip1` under wasmtime; see `.cargo/config.toml` and the Makefile
//! `wasm-test` target), and green there proves the wasm build reproduces the
//! identical constants — hence native == wasm, transitively. The consolidated
//! [`cross_platform_model_digest`] test folds a whole fit→predict→proba→shap run
//! into one hex fingerprint as the single value a reviewer can eyeball.

use std::collections::HashSet;

use sc_core::{
    grow, indexed_map, k_fold, ncv, rfecv, stratified_k_fold, train_test_split, tree_seed, tune,
    Criterion, DetRng, Error, FeatureMatrix, MaxFeatures, NcvParams, OofPredictions, ParamAxis,
    RandomForest, RandomForestParams, RfeStep, RfecvParams, Targets, Task, TestSize, TreeParams,
    TuningParams,
};
// Only the `parallel`-gated invariance test names `Tree` (via `sequential_forest`).
#[cfg(feature = "parallel")]
use sc_core::Tree;

#[test]
fn tree_seed_is_deterministic() {
    for &(m, i) in &[(0u64, 0u64), (1, 0), (42, 7), (u64::MAX, 123)] {
        assert_eq!(tree_seed(m, i), tree_seed(m, i));
    }
}

#[test]
fn tree_seeds_are_distinct_across_indices() {
    let master = 0xDEAD_BEEF_u64;
    let n = 100_000_u64;
    let seeds: HashSet<u64> = (0..n).map(|i| tree_seed(master, i)).collect();
    assert_eq!(seeds.len() as u64, n, "per-tree seeds collided");
}

/// The core M0 guarantee: the shim's output equals a straightforward sequential
/// reference, so the parallel build produces identical results to running on a
/// single thread.
#[test]
fn indexed_map_matches_sequential_reference() {
    let seed = 0x0BAD_C0DE_u64;
    let n = 1024;
    let got = indexed_map(n, |i| DetRng::for_tree(seed, i as u64).next_u64());
    let reference: Vec<u64> = (0..n)
        .map(|i| DetRng::for_tree(seed, i as u64).next_u64())
        .collect();
    assert_eq!(
        got, reference,
        "shim diverged from sequential reference order"
    );
}

/// Guards the cross-version reproducibility invariant: these frozen outputs
/// must never change silently. A diff here means the RNG algorithm or seed
/// derivation drifted, which would break equivalence with previously trained
/// models and committed oracle fixtures.
#[test]
fn rng_golden_vectors() {
    assert_eq!(tree_seed(0, 0), 0x0000_0000_0000_0000);
    assert_eq!(tree_seed(42, 7), 0x37E9_671C_4537_6D5D);
    assert_eq!(tree_seed(u64::MAX, 1), 0xE4D9_7177_1B65_2C20);
    assert_eq!(DetRng::for_tree(42, 7).next_u64(), 0x97A8_3994_AFF1_EC83);
    assert_eq!(DetRng::from_seed(0).next_u64(), 0xB585_F767_A79A_3B6C);
}

/// Freezes the M2 sampling primitives. `uniform_u64` and `shuffle` are the basis
/// of split/CV reproducibility; a diff here means the sampling algorithm drifted
/// and would silently change every generated partition (and break equivalence
/// with any partition recorded elsewhere).
#[test]
fn sampling_golden_vectors() {
    let draws = |seed: u64, bound: u64| {
        let mut r = DetRng::from_seed(seed);
        (0..5).map(|_| r.uniform_u64(bound)).collect::<Vec<_>>()
    };
    assert_eq!(draws(0, 6), [0, 3, 4, 0, 5]);
    assert_eq!(draws(0, 10), [2, 3, 8, 6, 1]);
    assert_eq!(draws(42, 6), [3, 4, 4, 4, 2]);
    assert_eq!(draws(42, 10), [7, 2, 0, 6, 4]);

    let perm = |seed: u64| {
        let mut v: Vec<usize> = (0..10).collect();
        DetRng::from_seed(seed).shuffle(&mut v);
        v
    };
    assert_eq!(perm(0), [7, 8, 9, 0, 1, 5, 4, 6, 3, 2]);
    assert_eq!(perm(42), [1, 5, 9, 6, 3, 2, 0, 8, 4, 7]);
}

/// Freezes the whole partitioner (build-groups → shuffle → slice) end to end.
/// A diff here means the split/CV assembly drifted and would silently change
/// every generated partition — the cross-version reproducibility tripwire for
/// M2. Values are for the fixed 10-sample, 3-class input below.
#[test]
fn partition_golden_vectors() {
    let codes = [0usize, 0, 0, 0, 1, 1, 1, 2, 2, 2];

    let split = train_test_split(10, Some((&codes, 3)), TestSize::Fraction(0.3), 0).unwrap();
    assert_eq!(split.train(), [0, 2, 3, 4, 5, 7, 9]);
    assert_eq!(split.test(), [1, 6, 8]);

    let skf = stratified_k_fold(&codes, 3, 3, 0).unwrap();
    let skf_folds: Vec<&[usize]> = (0..skf.n_splits()).map(|j| skf.test(j)).collect();
    assert_eq!(skf_folds, vec![&[1, 2, 6, 8][..], &[3, 5, 9], &[0, 4, 7]]);

    let kf = k_fold(10, 5, 0).unwrap();
    let kf_folds: Vec<&[usize]> = (0..kf.n_splits()).map(|j| kf.test(j)).collect();
    assert_eq!(
        kf_folds,
        vec![&[7, 8][..], &[0, 9], &[1, 5], &[4, 6], &[2, 3]]
    );
}

#[test]
fn error_is_display_and_std_error() {
    let e = Error::InvalidInput("boom".to_string());
    assert!(e.to_string().contains("boom"));
    let _dyn: &dyn std::error::Error = &e;
}

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

/// Freezes the M3 grow builder end to end (per-node stats → stopping →
/// per-node feature subsampling via `DetRng` → CSC split search → DFS node
/// numbering). `max_features = 2 < 4` so the tree genuinely consumes the RNG.
/// A diff here means the grow algorithm drifted and would silently change every
/// tree (and, in M4, every forest) built from a given seed. Values are for the
/// fixed 12-sample, 4-feature, 3-class input below.
#[test]
fn tree_grow_golden() {
    let x = dense(&[
        &[0.0, 1.0, 0.0, 2.0],
        &[1.0, 0.0, 3.0, 0.0],
        &[2.0, 2.0, 0.0, 1.0],
        &[0.0, 3.0, 1.0, 0.0],
        &[3.0, 0.0, 2.0, 2.0],
        &[1.0, 1.0, 0.0, 3.0],
        &[2.0, 0.0, 3.0, 0.0],
        &[0.0, 2.0, 1.0, 1.0],
        &[3.0, 3.0, 0.0, 0.0],
        &[1.0, 0.0, 2.0, 2.0],
        &[2.0, 1.0, 0.0, 3.0],
        &[0.0, 0.0, 3.0, 1.0],
    ]);
    let y = Targets::categorical(&["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"])
        .unwrap();
    let mut p = TreeParams::classifier_defaults();
    p.max_features = MaxFeatures::Count(2);
    p.max_depth = Some(3);
    let t = grow(&x, &y, &p, &mut DetRng::from_seed(20260707)).unwrap();

    let n = t.n_nodes();
    assert_eq!(n, 9);
    let children_left: Vec<i32> = (0..n).map(|i| t.children_left(i)).collect();
    let children_right: Vec<i32> = (0..n).map(|i| t.children_right(i)).collect();
    let feature: Vec<i32> = (0..n).map(|i| t.feature(i)).collect();
    let threshold: Vec<f64> = (0..n).map(|i| t.threshold(i)).collect();
    let value: Vec<Vec<f64>> = (0..n).map(|i| t.value(i).to_vec()).collect();
    let impurity: Vec<f64> = (0..n).map(|i| t.impurity(i)).collect();
    let n_node_samples: Vec<u64> = (0..n).map(|i| t.n_node_samples(i)).collect();
    let weighted: Vec<f64> = (0..n).map(|i| t.weighted_n_node_samples(i)).collect();

    assert_eq!(children_left, [1, 2, 3, -1, -1, -1, 7, -1, -1]);
    assert_eq!(children_right, [6, 5, 4, -1, -1, -1, 8, -1, -1]);
    assert_eq!(feature, [0, 1, 3, -2, -2, -2, 2, -2, -2]);
    assert_eq!(
        threshold,
        [2.5, 2.5, 0.5, -2.0, -2.0, -2.0, 1.0, -2.0, -2.0]
    );
    assert_eq!(
        value,
        vec![
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.4, 0.3, 0.3],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.5, 0.5, 0.0],
            vec![2.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.5],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ]
    );
    assert_eq!(
        impurity,
        [
            0.6666666666666667,
            0.66,
            0.6666666666666667,
            0.5,
            0.653061224489796,
            0.0,
            0.5,
            0.0,
            0.0
        ]
    );
    assert_eq!(n_node_samples, [12, 10, 9, 2, 7, 1, 2, 1, 1]);
    assert_eq!(weighted, [12.0, 10.0, 9.0, 2.0, 7.0, 1.0, 2.0, 1.0, 1.0]);
}

/// The fixed 12-sample, 4-feature, 3-class problem the forest golden/invariance
/// tests share.
fn forest_golden_data() -> (FeatureMatrix, Targets) {
    let x = dense(&[
        &[0.0, 1.0, 0.0, 2.0],
        &[1.0, 0.0, 3.0, 0.0],
        &[2.0, 2.0, 0.0, 1.0],
        &[0.0, 3.0, 1.0, 0.0],
        &[3.0, 0.0, 2.0, 2.0],
        &[1.0, 1.0, 0.0, 3.0],
        &[2.0, 0.0, 3.0, 0.0],
        &[0.0, 2.0, 1.0, 1.0],
        &[3.0, 3.0, 0.0, 0.0],
        &[1.0, 0.0, 2.0, 2.0],
        &[2.0, 1.0, 0.0, 3.0],
        &[0.0, 0.0, 3.0, 1.0],
    ]);
    let y = Targets::categorical(&["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"])
        .unwrap();
    (x, y)
}

/// A random forest built by an independent, plainly-sequential per-tree loop
/// (no `indexed_map`, no rayon) — the reference the parallel `fit` must match
/// byte-for-byte. Uses only public API: the same per-tree seed derivation,
/// bootstrap draw, and `grow` that `fit` uses internally. Only the
/// `parallel`-gated invariance test uses it, so it is gated to match (else it is
/// dead code under `--no-default-features`).
#[cfg(feature = "parallel")]
fn sequential_forest(
    x: &FeatureMatrix,
    y: &Targets,
    p: &RandomForestParams,
    master_seed: u64,
) -> RandomForest {
    let n = x.n_samples();
    let n_boot = p.resolve_n_bootstrap(n).unwrap();
    let trees: Vec<Tree> = (0..p.n_estimators)
        .map(|i| {
            let mut rng = DetRng::for_tree(master_seed, i as u64);
            let sample: Vec<usize> = if p.bootstrap {
                (0..n_boot)
                    .map(|_| rng.uniform_u64(n as u64) as usize)
                    .collect()
            } else {
                (0..n).collect()
            };
            let xb = x.select_rows(&sample);
            let yb = y.select(&sample);
            grow(&xb, &yb, &p.tree, &mut rng).unwrap()
        })
        .collect();
    let classes = y.classes().map(|c| c.to_vec());
    RandomForest::from_trees(trees, classes).unwrap()
}

/// Thread-count invariance for the forest build: `fit` must return a
/// byte-identical [`RandomForest`] whether run under a 1- or 8-thread rayon pool,
/// and it must equal the plainly-sequential reference. This is the forest-level
/// analogue of `indexed_map_matches_sequential_reference`, and the guarantee the
/// design leans on so a parallel training run reproduces a single-threaded one.
#[cfg(feature = "parallel")]
#[test]
fn forest_matches_sequential_reference() {
    let (x, y) = forest_golden_data();
    let mut p = RandomForestParams::classifier_defaults();
    p.n_estimators = 12;
    p.tree.max_features = MaxFeatures::Sqrt; // sqrt(4)=2 → genuinely subsamples per node
    let master_seed = 20260707;

    let reference = sequential_forest(&x, &y, &p, master_seed);
    for k in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(k)
            .build()
            .unwrap();
        let got = pool.install(|| RandomForest::fit(&x, &y, &p, master_seed).unwrap());
        assert_eq!(got, reference, "forest changed with {k} threads");
    }
}

/// Freezes the whole forest grow+aggregate pipeline end to end (per-tree
/// bootstrap draw → grow → soft-vote / summed-normalized MDI). A diff here means
/// the fit or aggregation drifted and would silently change every forest built
/// from a given seed. Values are for the fixed input above with `max_features =
/// sqrt` (so the RNG genuinely drives both bootstrap and feature subsampling).
#[test]
fn forest_grow_golden() {
    let (x, y) = forest_golden_data();
    let mut p = RandomForestParams::classifier_defaults();
    p.n_estimators = 5;
    p.tree.max_features = MaxFeatures::Sqrt;
    let forest = RandomForest::fit(&x, &y, &p, 20260707).unwrap();

    let pred = forest.predict_class_indices(&x).unwrap();
    let proba = forest.predict_proba(&x).unwrap();
    let mdi = forest.feature_importances();

    assert_eq!(pred, [0, 1, 2, 0, 1, 2, 1, 1, 2, 0, 2, 2]);

    let expected_proba = [
        [0.8, 0.0, 0.2],
        [0.18, 0.82, 0.0],
        [0.0, 0.2, 0.8],
        [0.6, 0.3, 0.1],
        [0.13333333333333333, 0.6666666666666666, 0.2],
        [0.2, 0.0, 0.8],
        [0.18, 0.82, 0.0],
        [0.2, 0.5, 0.3],
        [0.2, 0.0, 0.8],
        [0.5333333333333333, 0.4666666666666666, 0.0],
        [0.2, 0.2, 0.6],
        [0.2, 0.3, 0.5],
    ];
    for (i, (got, exp)) in proba.iter().zip(&expected_proba).enumerate() {
        for (j, (g, e)) in got.iter().zip(exp).enumerate() {
            assert!((g - e).abs() < 1e-12, "proba[{i}][{j}] = {g} != {e}");
        }
    }

    let expected_mdi = [
        0.37086951169878,
        0.2291603670422669,
        0.33930164790755296,
        0.06066847335140017,
    ];
    for (i, (g, e)) in mdi.iter().zip(&expected_mdi).enumerate() {
        assert!((g - e).abs() < 1e-12, "mdi[{i}] = {g} != {e}");
    }
}

/// Freezes the whole RFECV pipeline end to end (seeded CV folds → per-round
/// forest refits → importance-ranked elimination → mean-score curve → optimal
/// count → final selection). A diff means the RFE mechanics or the forest drifted
/// and would silently change every selection from a given seed. Values are for
/// the fixed 12-sample, 4-feature, 3-class input above.
#[test]
fn rfecv_golden() {
    let (x, y) = forest_golden_data();
    let mut p = RfecvParams::classifier_defaults();
    p.forest.n_estimators = 8;
    p.forest.tree.max_features = MaxFeatures::Sqrt;
    p.step = RfeStep::Count(1);
    p.n_splits = 3;
    let r = rfecv(&x, &y, &p, 20260707).unwrap();

    assert_eq!(r.grid_n_features, [1, 2, 3, 4]);
    assert_eq!(r.selected, [0, 2]);
    assert_eq!(r.n_optimal, 2);
    assert_eq!(r.ranking, [1, 2, 1, 3]);
    let expected_scores = [
        0.05555555555555555,
        0.16666666666666666,
        0.05555555555555555,
        0.0,
    ];
    for (i, (g, e)) in r.grid_scores.iter().zip(&expected_scores).enumerate() {
        assert!((g - e).abs() < 1e-12, "grid_scores[{i}] = {g} != {e}");
    }
}

/// The fixed reduced tuning grid the golden test searches (size 8: 2·2·2), with
/// `n_iter < grid_size` so sampling-without-replacement is genuinely exercised.
fn tuning_golden_grid() -> Vec<ParamAxis> {
    vec![
        ParamAxis::MaxDepth(vec![Some(2), None]),
        ParamAxis::MaxFeatures(vec![MaxFeatures::All, MaxFeatures::Sqrt]),
        ParamAxis::Criterion(vec![Criterion::Gini, Criterion::Entropy]),
    ]
}

/// Freezes the whole tuning pipeline end to end (candidate sampling without
/// replacement → mixed-radix decode → seeded CV folds → per-fold forest refits →
/// mean-score → argmax). A diff means the sampler, the decode radix order, the
/// fold/fit seeding, the CV scoring, or the tie-break drifted, which would
/// silently change every tuned model from a given seed. Values are for the fixed
/// 12-sample, 4-feature, 3-class input above with a size-8 grid and n_iter=4 (so
/// sampling-without-replacement is genuinely exercised).
#[test]
fn tuning_golden() {
    let (x, y) = forest_golden_data();
    let mut base = RandomForestParams::classifier_defaults();
    base.n_estimators = 8;
    base.tree.max_features = MaxFeatures::Sqrt;
    let p = TuningParams {
        base,
        grid: tuning_golden_grid(),
        n_iter: 4,
        n_splits: 3,
    };
    let r = tune(&x, &y, &p, 20260707).unwrap();

    // The sampled candidate sequence (pins the sampler + decode radix order).
    let seq: Vec<(Option<usize>, MaxFeatures, Criterion)> = r
        .candidates
        .iter()
        .map(|c| {
            (
                c.params.tree.max_depth,
                c.params.tree.max_features,
                c.params.tree.criterion,
            )
        })
        .collect();
    assert_eq!(
        seq,
        vec![
            (None, MaxFeatures::Sqrt, Criterion::Gini),
            (Some(2), MaxFeatures::All, Criterion::Gini),
            (None, MaxFeatures::All, Criterion::Gini),
            (Some(2), MaxFeatures::All, Criterion::Entropy),
        ]
    );

    // Mean CV scores (pins fold seeding + per-fit forests + scoring).
    let expected_scores = [
        0.0,
        0.1111111111111111,
        0.16666666666666666,
        0.05555555555555555,
    ];
    for (i, (c, e)) in r.candidates.iter().zip(&expected_scores).enumerate() {
        assert!(
            (c.mean_cv_score - e).abs() < 1e-12,
            "candidate {i} score {} != {e}",
            c.mean_cv_score
        );
    }

    // The winner (pins the argmax tie-break).
    assert_eq!(r.best_index, 2);
    assert!((r.best_score - 0.16666666666666666).abs() < 1e-12);
    assert_eq!(r.best_params.tree.max_depth, None);
    assert_eq!(r.best_params.tree.max_features, MaxFeatures::All);
    assert_eq!(r.best_params.tree.criterion, Criterion::Gini);
}

/// Freezes the whole classification OOF-CV pipeline end to end (seeded stratified
/// folds → per-fold forest fits at `tree_seed(seed, 1+f)` → held-out predict →
/// OOF scatter into original positions → soft-vote proba → fold-averaged MDI →
/// per-fold accuracy + population mean/std). A diff means the fold seeding, the
/// fit, the scatter, or the aggregation drifted and would silently change every
/// OOF run from a given seed. Values are for the fixed 12-sample, 4-feature,
/// 3-class input above with `n_splits=3`, `n_estimators=8`, `max_features=sqrt`.
/// The cyclic a/b/c target is independent of the features, so every held-out fold
/// scores 0 accuracy — the discriminating frozen values are the OOF codes/proba
/// (which pin the scatter) and the importances; the MSE/mean/std score path is
/// pinned by `ncv_regression_golden` below.
#[test]
fn ncv_classification_golden() {
    let (x, y) = forest_golden_data();
    let mut p = NcvParams::classifier_defaults();
    p.forest.n_estimators = 8;
    p.forest.tree.max_features = MaxFeatures::Sqrt;
    p.n_splits = 3;
    let r = ncv(&x, &y, &p, 20260707).unwrap();

    let OofPredictions::Classification { codes, proba, .. } = &r.predictions else {
        panic!("expected classification predictions");
    };
    assert_eq!(codes, &[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0]);
    // Two full OOF proba rows (each the fold-forest's soft vote for that sample).
    let expected_proba0 = [0.125, 0.5, 0.375];
    let expected_proba1 = [0.75, 0.25, 0.0];
    for (g, e) in proba[0].iter().zip(&expected_proba0) {
        assert!((g - e).abs() < 1e-12, "proba[0] {g} != {e}");
    }
    for (g, e) in proba[1].iter().zip(&expected_proba1) {
        assert!((g - e).abs() < 1e-12, "proba[1] {g} != {e}");
    }
    assert_eq!(r.fold_scores, [0.0, 0.0, 0.0]);
    assert_eq!(r.mean_score, 0.0);
    assert_eq!(r.std_score, 0.0);

    let expected_mdi = [
        0.34110616235616237,
        0.21520438473563472,
        0.2819593427405928,
        0.16173011016761016,
    ];
    for (i, (g, e)) in r.importances.iter().zip(&expected_mdi).enumerate() {
        assert!((g - e).abs() < 1e-12, "importances[{i}] = {g} != {e}");
    }
}

/// Freezes the **regression** OOF-CV pipeline, and in particular the new
/// per-fold `mean_squared_error` fold-score path with its population (ddof=0)
/// mean/std — the piece the classification golden's all-zero-accuracy data can't
/// exercise. Same fixed 12×4 design as above but with a numeric target, `KFold`
/// splits (regression), `n_splits=3`, `n_estimators=8`.
#[test]
fn ncv_regression_golden() {
    let x = dense(&[
        &[0.0, 1.0, 0.0, 2.0],
        &[1.0, 0.0, 3.0, 0.0],
        &[2.0, 2.0, 0.0, 1.0],
        &[0.0, 3.0, 1.0, 0.0],
        &[3.0, 0.0, 2.0, 2.0],
        &[1.0, 1.0, 0.0, 3.0],
        &[2.0, 0.0, 3.0, 0.0],
        &[0.0, 2.0, 1.0, 1.0],
        &[3.0, 3.0, 0.0, 0.0],
        &[1.0, 0.0, 2.0, 2.0],
        &[2.0, 1.0, 0.0, 3.0],
        &[0.0, 0.0, 3.0, 1.0],
    ]);
    let y = Targets::numeric(&[
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ])
    .unwrap();
    let mut p = NcvParams::regressor_defaults();
    p.forest.n_estimators = 8;
    p.n_splits = 3;
    let r = ncv(&x, &y, &p, 20260707).unwrap();

    let OofPredictions::Regression { values } = &r.predictions else {
        panic!("expected regression predictions");
    };
    let expected_values = [
        10.125, 6.0, 8.875, 7.75, 4.625, 2.5, 4.875, 6.625, 5.375, 2.5, 8.25, 4.0,
    ];
    for (i, (g, e)) in values.iter().zip(&expected_values).enumerate() {
        assert!((g - e).abs() < 1e-12, "values[{i}] = {g} != {e}");
    }

    // Per-fold MSE, then population mean/std over the folds.
    let expected_scores = [23.36328125, 8.2109375, 37.69921875];
    for (i, (g, e)) in r.fold_scores.iter().zip(&expected_scores).enumerate() {
        assert!((g - e).abs() < 1e-12, "fold_scores[{i}] = {g} != {e}");
    }
    assert!((r.mean_score - 23.091145833333332).abs() < 1e-12);
    assert!((r.std_score - 12.040078239876701).abs() < 1e-12);

    let expected_mdi = [
        0.29150836535771124,
        0.1416519659938166,
        0.263166597555021,
        0.3036730710934512,
    ];
    for (i, (g, e)) in r.importances.iter().zip(&expected_mdi).enumerate() {
        assert!((g - e).abs() < 1e-12, "importances[{i}] = {g} != {e}");
    }
}

/// A 64-bit FNV-1a accumulator over the raw bits of model outputs, hand-rolled so
/// the digest test below pulls in no dependency and runs under
/// `--no-default-features` on wasm. Lengths/indices are folded as `u64` via a
/// zero-extending `as` cast: every count here fits in 32 bits, so the byte stream
/// is identical on 32-bit wasm and 64-bit native (the property this digest exists
/// to prove); `f64`s are folded by their IEEE-754 `to_bits()` (bit-exact, not
/// tolerance-based).
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Fnv1a(Self::OFFSET_BASIS)
    }
    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ b as u64).wrapping_mul(Self::PRIME);
    }
    fn u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.byte(b);
        }
    }
    fn len(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    /// Fold a length-prefixed slice of `f64`s. Length-prefixing keeps the digest
    /// sensitive to shape (a differently-shaped output can't alias the same byte
    /// stream) — the collision resistance a change-detection tripwire needs; a
    /// 64-bit hash of an unbounded stream is of course not literally injective.
    fn f64s(&mut self, vs: &[f64]) {
        self.len(vs.len());
        for &v in vs {
            self.f64(v);
        }
    }
    fn finish(self) -> u64 {
        self.0
    }
}

/// Fold a fit model's full public prediction surface (indices/proba OR values,
/// then feature importances, then the whole TreeSHAP tensor) into the running
/// digest.
fn fold_model(h: &mut Fnv1a, forest: &RandomForest, x: &FeatureMatrix) {
    match forest.task() {
        Task::Classification { n_classes } => {
            h.len(n_classes);
            let idx = forest.predict_class_indices(x).unwrap();
            h.len(idx.len());
            for &i in &idx {
                h.len(i);
            }
            let proba = forest.predict_proba(x).unwrap();
            h.len(proba.len());
            for row in &proba {
                h.f64s(row);
            }
        }
        Task::Regression => {
            let vals = forest.predict_values(x).unwrap();
            h.f64s(&vals);
        }
    }
    h.f64s(&forest.feature_importances());

    let shap = forest.tree_shap(x).unwrap();
    h.len(shap.n_outputs);
    h.len(shap.n_features);
    h.f64s(&shap.base_values);
    h.len(shap.values.len());
    for sample in &shap.values {
        for per_output in sample {
            h.f64s(per_output);
        }
    }
}

/// The M11 cross-platform red criterion, distilled to one number. Over a couple
/// of fixed `(data, seed)` cases — a classifier and a regressor — fit a forest
/// and fold `fit → predict → predict_proba/predict_values → feature_importances
/// → tree_shap` into a single FNV-1a fingerprint, asserted equal to a frozen
/// constant. Running this same test on a wasm VM (see the module doc) and getting
/// green is the direct proof that the model + predictions are bit-identical on
/// native and wasm. A diff here means the compute path drifted on *some*
/// platform — exactly the regression the guarantee forbids.
#[test]
fn cross_platform_model_digest() {
    let mut h = Fnv1a::new();

    // Case 1: classifier on the shared 12×4, 3-class golden design, two seeds.
    // `max_features = sqrt` (=2 of 4) so the RNG genuinely drives both bootstrap
    // and per-node subsampling.
    let (xc, yc) = forest_golden_data();
    for &seed in &[20260707u64, 777] {
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 6;
        p.tree.max_features = MaxFeatures::Sqrt;
        let forest = RandomForest::fit(&xc, &yc, &p, seed).unwrap();
        fold_model(&mut h, &forest, &xc);
    }

    // Case 2: regressor on the same design with a numeric target, two seeds.
    let xr = dense(&[
        &[0.0, 1.0, 0.0, 2.0],
        &[1.0, 0.0, 3.0, 0.0],
        &[2.0, 2.0, 0.0, 1.0],
        &[0.0, 3.0, 1.0, 0.0],
        &[3.0, 0.0, 2.0, 2.0],
        &[1.0, 1.0, 0.0, 3.0],
        &[2.0, 0.0, 3.0, 0.0],
        &[0.0, 2.0, 1.0, 1.0],
        &[3.0, 3.0, 0.0, 0.0],
        &[1.0, 0.0, 2.0, 2.0],
        &[2.0, 1.0, 0.0, 3.0],
        &[0.0, 0.0, 3.0, 1.0],
    ]);
    let yr = Targets::numeric(&[
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ])
    .unwrap();
    for &seed in &[20260707u64, 777] {
        let mut p = RandomForestParams::regressor_defaults();
        p.n_estimators = 6;
        let forest = RandomForest::fit(&xr, &yr, &p, seed).unwrap();
        fold_model(&mut h, &forest, &xr);
    }

    let digest = h.finish();
    assert_eq!(
        digest, 0x95a3_1ff9_410a_938e,
        "cross-platform model digest changed: {digest:#018x}"
    );
}
