//! M0 determinism-harness tests: seed derivation and the thread-count-invariance
//! guarantee of the execution shim.

use std::collections::HashSet;

use sc_core::{
    indexed_map, k_fold, stratified_k_fold, train_test_split, tree_seed, DetRng, Error, TestSize,
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
