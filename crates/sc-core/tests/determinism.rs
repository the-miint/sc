//! M0 determinism-harness tests: seed derivation and the thread-count-invariance
//! guarantee of the execution shim.

use std::collections::HashSet;

use sc_core::{
    grow, indexed_map, k_fold, stratified_k_fold, train_test_split, tree_seed, DetRng, Error,
    FeatureMatrix, MaxFeatures, Targets, TestSize, TreeParams,
};

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
