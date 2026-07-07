//! Seeded partitioning: train/test split, K-fold / stratified K-fold, and a
//! first-class injection hook for externally supplied partitions.
//!
//! This is the PROPERTY tier (see `sc-implementation-plan.html` M2): we do NOT
//! replicate numpy/sklearn's RNG-driven partitions byte-for-byte. We use our own
//! seeded [`DetRng`](crate::DetRng)-based algorithm and guarantee only the
//! invariants — correct sizes, stratified proportions within rounding,
//! disjointness, full coverage, reproducibility under our seed, and
//! singleton-class errors like q2.
//!
//! [`Split`] and [`CvFolds`] are the *same* types whether produced by
//! [`train_test_split`]/[`k_fold`]/[`stratified_k_fold`] or built from
//! caller-supplied indices via [`Split::from_indices`] /
//! [`CvFolds::from_test_folds`] / [`CvFolds::from_fold_ids`]. That shared type is
//! the injection hook M4 (single split) and M8 (OOF folds) rely on to feed q2
//! and our forest the identical partition.

mod core;

use crate::rng::DetRng;
use crate::{Error, Result};

/// How large the test set should be. q2's default is `Fraction(0.2)`; the API
/// takes it explicitly rather than baking in a default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestSize {
    /// A fraction of `n`, valid in `[0.0, 1.0)`. `0.0` yields an empty test set
    /// (all train); the rounded count is `(fraction * n).round()`.
    Fraction(f64),
    /// An absolute test count, valid in `0..n`.
    Count(usize),
}

impl TestSize {
    /// Resolve to an absolute test count for `n` samples, validating the input.
    fn resolve(self, n: usize) -> Result<usize> {
        match self {
            TestSize::Count(k) => {
                if k >= n {
                    return Err(Error::InvalidInput(format!(
                        "test_size count {k} must be < n_samples {n}"
                    )));
                }
                Ok(k)
            }
            TestSize::Fraction(f) => {
                if !f.is_finite() || !(0.0..1.0).contains(&f) {
                    return Err(Error::InvalidInput(format!(
                        "test_size fraction {f} must be in [0.0, 1.0)"
                    )));
                }
                // Symmetric with Count (which rejects k >= n): the rounded test
                // count must leave at least one training sample.
                let n_test = (f * n as f64).round() as usize;
                if n_test >= n {
                    return Err(Error::InvalidInput(format!(
                        "test_size fraction {f} rounds to {n_test} of {n} samples, leaving no training data"
                    )));
                }
                Ok(n_test)
            }
        }
    }
}

/// A train/test partition of `0..n`. Train and test are disjoint, sorted
/// ascending, and together cover every sample (`test` may be empty).
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    train: Vec<usize>,
    test: Vec<usize>,
    n: usize,
}

impl Split {
    /// Injection hook: build a `Split` from caller-supplied index lists.
    ///
    /// Validates (`InvalidInput`): every index `< n`; no duplicate within or
    /// across the two sides (disjointness); and full coverage
    /// (`train ∪ test == 0..n`; `test` may be empty). Sides are stored ascending
    /// to match generated splits.
    pub fn from_indices(n: usize, train: Vec<usize>, test: Vec<usize>) -> Result<Self> {
        validate_partition(n, &[train.as_slice(), test.as_slice()])?;
        let (mut train, mut test) = (train, test);
        train.sort_unstable();
        test.sort_unstable();
        Ok(Split { train, test, n })
    }

    /// Number of samples across both sides.
    pub fn n_samples(&self) -> usize {
        self.n
    }

    /// Training sample indices, ascending.
    pub fn train(&self) -> &[usize] {
        &self.train
    }

    /// Test sample indices, ascending.
    pub fn test(&self) -> &[usize] {
        &self.test
    }
}

/// A K-fold partition: `n_splits` disjoint test folds covering every sample
/// exactly once. The training set for a fold is the complement of its test fold.
#[derive(Debug, Clone, PartialEq)]
pub struct CvFolds {
    folds: Vec<Vec<usize>>,
    n: usize,
}

impl CvFolds {
    /// Injection hook (fold-list form): each `folds[j]` is the test indices of
    /// fold `j`. Validates (`InvalidInput`): at least 2 folds; no empty fold;
    /// every index `< n`; and that the folds partition `0..n` (no duplicate
    /// within/across folds, every sample covered). Folds are stored ascending.
    pub fn from_test_folds(n: usize, folds: Vec<Vec<usize>>) -> Result<Self> {
        if folds.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "cv folds: need >= 2 folds, got {}",
                folds.len()
            )));
        }
        for f in &folds {
            if f.is_empty() {
                return Err(Error::InvalidInput("cv folds: a fold is empty".into()));
            }
        }
        let slices: Vec<&[usize]> = folds.iter().map(Vec::as_slice).collect();
        validate_partition(n, &slices)?;
        let mut folds = folds;
        for f in folds.iter_mut() {
            f.sort_unstable();
        }
        Ok(CvFolds { folds, n })
    }

    /// Injection hook (per-sample fold-id form; the natural OOF shape for M8):
    /// `fold_ids[i]` is the test fold sample `i` belongs to. Validates as
    /// [`Self::from_test_folds`] after bucketing (`fold_ids[i] < n_splits`).
    pub fn from_fold_ids(fold_ids: &[usize], n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::InvalidInput(format!(
                "cv folds: need >= 2 folds, got {n_splits}"
            )));
        }
        let mut folds = vec![Vec::new(); n_splits];
        for (i, &fid) in fold_ids.iter().enumerate() {
            if fid >= n_splits {
                return Err(Error::InvalidInput(format!(
                    "fold id {fid} out of range [0, {n_splits})"
                )));
            }
            folds[fid].push(i);
        }
        Self::from_test_folds(fold_ids.len(), folds)
    }

    /// Number of folds.
    pub fn n_splits(&self) -> usize {
        self.folds.len()
    }

    /// Number of samples.
    pub fn n_samples(&self) -> usize {
        self.n
    }

    /// Test indices of `fold`, ascending. Panics if `fold >= n_splits`.
    pub fn test(&self, fold: usize) -> &[usize] {
        &self.folds[fold]
    }

    /// Training indices of `fold` (complement of its test fold), ascending.
    /// Panics if `fold >= n_splits`.
    pub fn train(&self, fold: usize) -> Vec<usize> {
        let mut train = Vec::with_capacity(self.n - self.folds[fold].len());
        for (j, f) in self.folds.iter().enumerate() {
            if j != fold {
                train.extend_from_slice(f);
            }
        }
        train.sort_unstable();
        train
    }

    /// Iterate `(train, test)` index pairs in fold order — the shape M8's OOF
    /// loop consumes.
    pub fn iter(&self) -> impl Iterator<Item = (Vec<usize>, &[usize])> + '_ {
        (0..self.folds.len()).map(move |j| (self.train(j), self.folds[j].as_slice()))
    }
}

/// Seeded train/test split. `strata = Some((codes, k))` stratifies by class
/// (classification); `None` is a plain shuffled split (regression).
///
/// Errors (`InvalidInput`): `n == 0`; strata length mismatch or an out-of-range
/// code; invalid `test_size`; and — when stratified — any class with fewer than
/// 2 members (the singleton-class case q2 rejects).
pub fn train_test_split(
    n: usize,
    strata: Option<(&[usize], usize)>,
    test_size: TestSize,
    random_state: u64,
) -> Result<Split> {
    if n == 0 {
        return Err(Error::InvalidInput(
            "train_test_split: n_samples must be > 0".into(),
        ));
    }
    let n_test = test_size.resolve(n)?;
    let mut groups = core::build_groups(n, strata)?;
    if strata.is_some() {
        require_min_class_size(&groups, 2, "stratified split")?;
    }
    let mut rng = DetRng::from_seed(random_state);
    core::shuffle_groups(&mut groups, &mut rng);
    let (train, test) = core::split_indices(&groups, n, n_test);
    Ok(Split { train, test, n })
}

/// Seeded, shuffled K-fold over `0..n` (the regression CV path).
///
/// Errors (`InvalidInput`): `n_splits < 2` or `n_splits > n`.
pub fn k_fold(n: usize, n_splits: usize, random_state: u64) -> Result<CvFolds> {
    validate_n_splits(n_splits)?;
    if n_splits > n {
        return Err(Error::InvalidInput(format!(
            "k_fold: n_splits {n_splits} must be <= n_samples {n}"
        )));
    }
    let mut groups = core::build_groups(n, None)?;
    let mut rng = DetRng::from_seed(random_state);
    core::shuffle_groups(&mut groups, &mut rng);
    Ok(CvFolds {
        folds: core::fold_indices(&groups, n_splits),
        n,
    })
}

/// Seeded, shuffled stratified K-fold (the classification CV path).
///
/// Errors (`InvalidInput`): `n_splits < 2`; strata inconsistency; or any class
/// with fewer than `n_splits` members (sklearn's StratifiedKFold requirement).
pub fn stratified_k_fold(
    codes: &[usize],
    n_classes: usize,
    n_splits: usize,
    random_state: u64,
) -> Result<CvFolds> {
    validate_n_splits(n_splits)?;
    let n = codes.len();
    if n == 0 {
        return Err(Error::InvalidInput(
            "stratified_k_fold: n_samples must be > 0".into(),
        ));
    }
    let mut groups = core::build_groups(n, Some((codes, n_classes)))?;
    // Each class must supply at least one sample per fold (sklearn's rule); this
    // also guarantees no fold is empty.
    require_min_class_size(&groups, n_splits, "stratified k-fold")?;
    let mut rng = DetRng::from_seed(random_state);
    core::shuffle_groups(&mut groups, &mut rng);
    Ok(CvFolds {
        folds: core::fold_indices(&groups, n_splits),
        n,
    })
}

fn validate_n_splits(n_splits: usize) -> Result<()> {
    if n_splits < 2 {
        return Err(Error::InvalidInput(format!(
            "n_splits must be >= 2, got {n_splits}"
        )));
    }
    Ok(())
}

/// Validate that the given index lists form a partition of `0..n`: every index
/// in range, no duplicate within or across lists, and full coverage. Shared by
/// the [`Split`] and [`CvFolds`] injection constructors.
fn validate_partition(n: usize, lists: &[&[usize]]) -> Result<()> {
    let mut seen = vec![false; n];
    for list in lists {
        for &i in *list {
            if i >= n {
                return Err(Error::InvalidInput(format!(
                    "partition index {i} out of range [0, {n})"
                )));
            }
            if seen[i] {
                return Err(Error::InvalidInput(format!(
                    "partition index {i} appears more than once"
                )));
            }
            seen[i] = true;
        }
    }
    if let Some(missing) = seen.iter().position(|&s| !s) {
        return Err(Error::InvalidInput(format!(
            "partition does not cover sample {missing} of 0..{n}"
        )));
    }
    Ok(())
}

/// Error if any class group has fewer than `min` members, labeling the failure
/// with `context`. Shared by the stratified split and stratified K-fold checks.
fn require_min_class_size(groups: &[Vec<usize>], min: usize, context: &str) -> Result<()> {
    for (c, g) in groups.iter().enumerate() {
        if g.len() < min {
            return Err(Error::InvalidInput(format!(
                "{context}: class {c} has {} member(s); each class needs >= {min}",
                g.len()
            )));
        }
    }
    Ok(())
}
