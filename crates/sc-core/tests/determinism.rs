//! M0 determinism-harness tests: seed derivation and the thread-count-invariance
//! guarantee of the execution shim.

use std::collections::HashSet;

use sc_core::{indexed_map, tree_seed, DetRng, Error};

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

#[test]
fn error_is_display_and_std_error() {
    let e = Error::InvalidInput("boom".to_string());
    assert!(e.to_string().contains("boom"));
    let _dyn: &dyn std::error::Error = &e;
}
