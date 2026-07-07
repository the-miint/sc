//! M2 property tests for the partitioner (PROPERTY tier — invariants only, no
//! oracle value-match, hermetic). Each test is annotated with the milestone
//! invariant it pins:
//!   I1 sizes honor test_size · I2 stratified proportions within rounding ·
//!   I3 train ∩ test = ∅ · I4 every sample in exactly one CV test fold ·
//!   I5 identical seed → identical partition · I6 singleton-class errors.

use sc_core::{k_fold, stratified_k_fold, train_test_split, CvFolds, Split, TestSize};

// ---- helpers ----

/// Assert a Split is disjoint, fully covers 0..n, and each side is sorted.
fn assert_split_partition(s: &Split, n: usize) {
    let mut seen = vec![0u8; n];
    for &i in s.train() {
        assert!(i < n, "train index {i} out of range");
        assert_eq!(seen[i], 0, "index {i} appears twice");
        seen[i] = 1;
    }
    for &i in s.test() {
        assert!(i < n, "test index {i} out of range");
        assert_eq!(seen[i], 0, "index {i} in both train and test"); // I3
        seen[i] = 2;
    }
    assert!(
        seen.iter().all(|&x| x != 0),
        "train ∪ test must cover 0..{n}"
    );
    assert!(
        s.train().windows(2).all(|w| w[0] < w[1]),
        "train not sorted"
    );
    assert!(s.test().windows(2).all(|w| w[0] < w[1]), "test not sorted");
}

/// Assert folds partition 0..n (every sample in exactly one test fold).
fn assert_folds_partition(cv: &CvFolds, n: usize) {
    let mut count = vec![0u32; n];
    for j in 0..cv.n_splits() {
        for &i in cv.test(j) {
            assert!(i < n);
            count[i] += 1;
        }
    }
    assert!(
        count.iter().all(|&c| c == 1),
        "every sample in exactly one fold"
    ); // I4
}

fn class_test_counts(codes: &[usize], k: usize, test: &[usize]) -> Vec<usize> {
    let mut c = vec![0usize; k];
    for &i in test {
        c[codes[i]] += 1;
    }
    c
}

// ---- train_test_split ----

#[test]
fn split_fraction_size_within_one() {
    // I1: |test| ≈ round(f*n), and the partition is valid.
    for &n in &[10usize, 37, 100, 250] {
        for &f in &[0.1, 0.2, 0.33, 0.5] {
            let s = train_test_split(n, None, TestSize::Fraction(f), 20260707).unwrap();
            let expected = (f * n as f64).round() as usize;
            assert_eq!(s.test().len(), expected, "n={n} f={f}");
            assert_eq!(s.train().len(), n - expected);
            assert_split_partition(&s, n);
        }
    }
}

#[test]
fn split_count_is_exact() {
    // I1: Count(k) → exactly k test samples.
    for &k in &[0usize, 1, 5, 49] {
        let s = train_test_split(50, None, TestSize::Count(k), 1).unwrap();
        assert_eq!(s.test().len(), k);
        assert_split_partition(&s, 50);
    }
}

#[test]
fn split_fraction_zero_is_empty_test() {
    // Deliberate deviation from q2's degenerate test_size==0 (train==test==full):
    // we keep disjointness (I3) — empty test, all train.
    let s = train_test_split(10, None, TestSize::Fraction(0.0), 0).unwrap();
    assert!(s.test().is_empty());
    assert_eq!(s.train().len(), 10);
    assert_split_partition(&s, 10);
}

#[test]
fn stratified_split_preserves_proportions() {
    // I2: per-class test counts within 1 of f*m_c, and every class stays in train.
    // n=40, classes of size 20/12/8, f=0.2.
    let mut codes = Vec::new();
    for (c, m) in [(0usize, 20usize), (1, 12), (2, 8)] {
        codes.extend(std::iter::repeat_n(c, m));
    }
    let f = 0.2;
    let s = train_test_split(40, Some((&codes, 3)), TestSize::Fraction(f), 99).unwrap();
    assert_split_partition(&s, 40);
    let test_c = class_test_counts(&codes, 3, s.test());
    for (c, &m) in [20usize, 12, 8].iter().enumerate() {
        let ideal = f * m as f64;
        assert!(
            (test_c[c] as f64 - ideal).abs() <= 1.0,
            "class {c}: test count {} not within 1 of {ideal}",
            test_c[c]
        );
        // each class present in both sides
        assert!(test_c[c] >= 1, "class {c} absent from test");
        assert!(m - test_c[c] >= 1, "class {c} absent from train");
    }
}

#[test]
fn split_is_seed_reproducible() {
    // I5: same seed → identical; a different seed → different (n large enough
    // that a collision is astronomically unlikely).
    let a = train_test_split(200, None, TestSize::Fraction(0.25), 7).unwrap();
    let b = train_test_split(200, None, TestSize::Fraction(0.25), 7).unwrap();
    assert_eq!(a, b);
    let c = train_test_split(200, None, TestSize::Fraction(0.25), 8).unwrap();
    assert_ne!(a, c);
}

#[test]
fn split_rejects_bad_params() {
    assert!(train_test_split(0, None, TestSize::Fraction(0.2), 0).is_err());
    assert!(train_test_split(10, None, TestSize::Fraction(1.0), 0).is_err());
    assert!(train_test_split(10, None, TestSize::Fraction(-0.1), 0).is_err());
    assert!(train_test_split(10, None, TestSize::Fraction(f64::NAN), 0).is_err());
    assert!(train_test_split(10, None, TestSize::Count(10), 0).is_err()); // k >= n
                                                                          // A fraction that rounds to n leaves no training data → reject (symmetric with Count).
    assert!(train_test_split(2, None, TestSize::Fraction(0.75), 0).is_err()); // 1.5 → 2 == n
                                                                              // strata length != n
    let codes = [0usize, 1, 2];
    assert!(train_test_split(4, Some((&codes, 3)), TestSize::Fraction(0.2), 0).is_err());
}

#[test]
fn stratified_split_singleton_class_errors() {
    // I6: a class with a single member cannot be stratified.
    let codes = [0usize, 0, 0, 1]; // class 1 has one member
    assert!(train_test_split(4, Some((&codes, 2)), TestSize::Fraction(0.25), 0).is_err());
}

// ---- k_fold / stratified_k_fold ----

#[test]
fn kfold_partitions_and_balances() {
    // I4 + balanced fold sizes.
    for &(n, k) in &[(10usize, 5usize), (23, 4), (100, 7), (9, 9)] {
        let cv = k_fold(n, k, 3).unwrap();
        assert_eq!(cv.n_splits(), k);
        assert_folds_partition(&cv, n);
        let sizes: Vec<usize> = (0..k).map(|j| cv.test(j).len()).collect();
        let (mn, mx) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
        assert!(mx - mn <= 1, "fold sizes {sizes:?} differ by > 1");
        // train is the complement, disjoint and covering.
        for j in 0..k {
            let (train, test) = (cv.train(j), cv.test(j));
            assert_eq!(train.len() + test.len(), n);
            let mut seen = vec![false; n];
            for &i in &train {
                assert!(!seen[i]);
                seen[i] = true;
            }
            for &i in test {
                assert!(!seen[i], "train/test overlap in fold {j}");
                seen[i] = true;
            }
            assert!(seen.iter().all(|&x| x));
        }
    }
}

#[test]
fn stratified_kfold_preserves_proportions() {
    // I2 + I4: each fold's per-class count within 1 of m_c/k.
    let mut codes = Vec::new();
    for (c, m) in [(0usize, 20usize), (1, 15), (2, 10)] {
        codes.extend(std::iter::repeat_n(c, m));
    }
    let k = 5;
    let cv = stratified_k_fold(&codes, 3, k, 2024).unwrap();
    assert_folds_partition(&cv, 45);
    for j in 0..k {
        let cnt = class_test_counts(&codes, 3, cv.test(j));
        for (c, &m) in [20usize, 15, 10].iter().enumerate() {
            let ideal = m as f64 / k as f64;
            assert!(
                (cnt[c] as f64 - ideal).abs() <= 1.0,
                "fold {j} class {c}: {} not within 1 of {ideal}",
                cnt[c]
            );
        }
    }
}

#[test]
fn kfold_is_seed_reproducible() {
    // I5
    assert_eq!(k_fold(60, 5, 11).unwrap(), k_fold(60, 5, 11).unwrap());
    assert_ne!(k_fold(60, 5, 11).unwrap(), k_fold(60, 5, 12).unwrap());
    let codes: Vec<usize> = (0..60).map(|i| i % 3).collect();
    assert_eq!(
        stratified_k_fold(&codes, 3, 5, 11).unwrap(),
        stratified_k_fold(&codes, 3, 5, 11).unwrap()
    );
}

#[test]
fn kfold_rejects_bad_params() {
    assert!(k_fold(10, 1, 0).is_err()); // n_splits < 2
    assert!(k_fold(3, 5, 0).is_err()); // n_splits > n
    let codes = [0usize, 0, 1, 1, 1]; // class 0 has 2 members
    assert!(stratified_k_fold(&codes, 2, 3, 0).is_err()); // class < n_splits (I6)
    assert!(stratified_k_fold(&codes, 2, 1, 0).is_err()); // n_splits < 2
    assert!(stratified_k_fold(&[], 0, 2, 0).is_err()); // empty / no classes → no empty folds
}

// ---- injection hook ----

#[test]
fn split_injection_roundtrips_and_validates() {
    let s = Split::from_indices(5, vec![0, 2, 4], vec![1, 3]).unwrap();
    assert_eq!(s.train(), [0, 2, 4]);
    assert_eq!(s.test(), [1, 3]);
    assert_split_partition(&s, 5);
    // rejections
    assert!(Split::from_indices(5, vec![0, 5], vec![1]).is_err()); // out of range
    assert!(Split::from_indices(5, vec![0, 0, 2], vec![1, 3, 4]).is_err()); // dup within
    assert!(Split::from_indices(5, vec![0, 1, 2], vec![1, 3, 4]).is_err()); // overlap
    assert!(Split::from_indices(5, vec![0, 2], vec![1, 3]).is_err()); // gap (4 uncovered)
}

#[test]
fn cvfolds_injection_validates() {
    let cv = CvFolds::from_test_folds(6, vec![vec![0, 3], vec![1, 4], vec![2, 5]]).unwrap();
    assert_eq!(cv.n_splits(), 3);
    assert_folds_partition(&cv, 6);
    // rejections
    assert!(CvFolds::from_test_folds(6, vec![vec![0, 1, 2, 3, 4, 5]]).is_err()); // < 2 folds
    assert!(CvFolds::from_test_folds(6, vec![vec![0, 6], vec![1, 2, 3, 4, 5]]).is_err()); // OOB
    assert!(CvFolds::from_test_folds(6, vec![vec![0, 1], vec![1, 2, 3, 4, 5]]).is_err()); // cross-fold dup
    assert!(CvFolds::from_test_folds(6, vec![vec![0, 1], vec![2, 3]]).is_err()); // gap (4,5)
    assert!(CvFolds::from_test_folds(6, vec![vec![], vec![0, 1, 2, 3, 4, 5]]).is_err());
    // empty fold
}

#[test]
fn cvfolds_from_fold_ids_matches_test_folds() {
    let fold_ids = [0usize, 1, 2, 0, 1, 2];
    let a = CvFolds::from_fold_ids(&fold_ids, 3).unwrap();
    let b = CvFolds::from_test_folds(6, vec![vec![0, 3], vec![1, 4], vec![2, 5]]).unwrap();
    assert_eq!(a, b);
    assert!(CvFolds::from_fold_ids(&[0, 1, 3], 3).is_err()); // id out of range
    assert!(CvFolds::from_fold_ids(&[0, 0, 0], 1).is_err()); // n_splits < 2
}

#[test]
fn generated_and_injected_folds_yield_identical_iter() {
    // Proves the injection hook is first-class: a generated CvFolds and an
    // equivalent injected one produce identical (train, test) pairs.
    let cv = k_fold(12, 4, 55).unwrap();
    let test_folds: Vec<Vec<usize>> = (0..cv.n_splits()).map(|j| cv.test(j).to_vec()).collect();
    let injected = CvFolds::from_test_folds(12, test_folds).unwrap();
    let gen: Vec<(Vec<usize>, Vec<usize>)> = cv.iter().map(|(tr, te)| (tr, te.to_vec())).collect();
    let inj: Vec<(Vec<usize>, Vec<usize>)> =
        injected.iter().map(|(tr, te)| (tr, te.to_vec())).collect();
    assert_eq!(gen, inj);
}
