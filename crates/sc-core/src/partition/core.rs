//! Private partitioner engine shared by `train_test_split`, `k_fold`, and
//! `stratified_k_fold`.
//!
//! One pipeline — build class groups → shuffle each group → slice — serves both
//! split and CV. Stratified vs. plain differs only in `build_groups`; everything
//! downstream is shared, which is what lets a single seeded algorithm back all
//! three entry points.
//!
//! **Frozen for reproducibility** (a change breaks the determinism golden
//! vectors, by design): the ascending-sample gather order in `build_groups`, the
//! ascending-group processing order in `shuffle_groups`, the largest-remainder
//! apportionment with an ascending-class-index tie-break in `split_indices`, and
//! the contiguous-chunk fold assignment (first `m % n_splits` chunks one larger)
//! in `fold_indices`.

use crate::rng::DetRng;
use crate::{Error, Result};

/// Partition sample indices `0..n` into groups. `Some((codes, k))` groups by
/// class (ascending sample order within each class); `None` yields a single
/// group `0..n`. Validates `codes.len() == n` and every `code < k`.
pub(super) fn build_groups(n: usize, strata: Option<(&[usize], usize)>) -> Result<Vec<Vec<usize>>> {
    match strata {
        Some((codes, k)) => {
            if codes.len() != n {
                return Err(Error::InvalidInput(format!(
                    "strata length {} does not match n_samples {n}",
                    codes.len()
                )));
            }
            let mut groups = vec![Vec::new(); k];
            for (i, &c) in codes.iter().enumerate() {
                if c >= k {
                    return Err(Error::InvalidInput(format!(
                        "class code {c} out of range [0, {k})"
                    )));
                }
                groups[c].push(i);
            }
            Ok(groups)
        }
        None => Ok(vec![(0..n).collect()]),
    }
}

/// Shuffle each group in place, in ascending group index (fixed order ⇒
/// reproducible given `(codes, seed)`).
pub(super) fn shuffle_groups(groups: &mut [Vec<usize>], rng: &mut DetRng) {
    for g in groups.iter_mut() {
        rng.shuffle(g);
    }
}

/// Slice shuffled `groups` into `(train, test)` with a global `n_test`
/// apportioned across groups by the largest-remainder (Hamilton) method, so
/// per-class test proportions are preserved within rounding. Both sides are
/// returned sorted ascending (a split is a set; ordering carries no meaning and
/// sorting makes downstream checks stable).
pub(super) fn split_indices(
    groups: &[Vec<usize>],
    n: usize,
    n_test: usize,
) -> (Vec<usize>, Vec<usize>) {
    let k = groups.len();
    let mut t = vec![0usize; k];
    let mut frac = vec![0.0f64; k];
    let mut assigned = 0usize;
    for (c, g) in groups.iter().enumerate() {
        let ideal = n_test as f64 * g.len() as f64 / n as f64;
        let floor = ideal.floor();
        t[c] = floor as usize;
        frac[c] = ideal - floor;
        assigned += t[c];
    }
    // Distribute the leftover to the largest fractional remainders, tie-broken
    // by ascending class index. `leftover < k`, and any group whose floor
    // already equals its size has remainder 0 (sorts last), so incrementing the
    // top `leftover` groups never exceeds a group's size — the clamp is defensive.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        frac[b]
            .partial_cmp(&frac[a])
            .expect("finite remainders")
            .then(a.cmp(&b))
    });
    let mut leftover = n_test - assigned;
    for &c in &order {
        if leftover == 0 {
            break;
        }
        if t[c] < groups[c].len() {
            t[c] += 1;
            leftover -= 1;
        }
    }

    let mut train = Vec::with_capacity(n - n_test);
    let mut test = Vec::with_capacity(n_test);
    for (c, g) in groups.iter().enumerate() {
        let (te, tr) = g.split_at(t[c]);
        test.extend_from_slice(te);
        train.extend_from_slice(tr);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Slice shuffled `groups` into `n_splits` disjoint test folds. Within each
/// group of size `m`, chunk sizes are `m/n_splits`, with the first `m %
/// n_splits` chunks one larger (sklearn's fold-size rule); chunk `j` of every
/// class joins test fold `j`. Each fold is returned sorted ascending.
pub(super) fn fold_indices(groups: &[Vec<usize>], n_splits: usize) -> Vec<Vec<usize>> {
    let mut folds = vec![Vec::new(); n_splits];
    for g in groups {
        let (base, rem) = (g.len() / n_splits, g.len() % n_splits);
        let mut start = 0;
        for (j, fold) in folds.iter_mut().enumerate() {
            let sz = base + usize::from(j < rem);
            fold.extend_from_slice(&g[start..start + sz]);
            start += sz;
        }
    }
    for f in folds.iter_mut() {
        f.sort_unstable();
    }
    folds
}
