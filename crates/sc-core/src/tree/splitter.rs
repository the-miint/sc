//! Best-split search over a CSC [`FeatureMatrix`] with implicit zeros.
//!
//! This is the crux of the grow path (design §10, `_splitter.pyx`). For a node's
//! sorted sample list and a candidate feature it gathers the feature values with
//! a merge-join against the column's stored entries — every node sample not
//! stored in the column reads `0.0`, exactly like an implicit zero. Because
//! stored values may legitimately be `0.0` (a summed-to-zero explicit entry), we
//! never assume "stored ⇒ nonzero"; the zero block therefore lands wherever
//! `0.0` sorts, which is **mid-scan when negatives are present**.
//!
//! The grow path is the property tier: it is unit-tested for its invariants, not
//! matched byte-for-byte to sklearn (feature sampling and tie-breaks are our own
//! deterministic rules). Split selection maximizes the weighted impurity
//! decrease, breaking ties by lowest feature index then lowest threshold.

use crate::data::FeatureMatrix;

use super::criterion::{impurity_from_counts, variance};
use super::params::Criterion;

/// Per-sample target access plus the node's precomputed totals, so the scan can
/// derive each child's impurity as `total − left` without a second pass.
pub(crate) enum Totals<'a> {
    /// Classification: `codes[sample]` is the class; `counts` are node totals.
    Class {
        codes: &'a [usize],
        counts: &'a [f64],
    },
    /// Regression: `y[sample]` is the target; `sum`/`sumsq` are node totals.
    Reg { y: &'a [f64], sum: f64, sumsq: f64 },
}

/// The winning split at a node.
pub(crate) struct BestSplit {
    pub feature: usize,
    pub threshold: f64,
    /// Fully-scaled weighted impurity decrease `(N_t/N)·(imp − …)`, ready to
    /// compare against `min_impurity_decrease`.
    pub weighted_decrease: f64,
}

/// Gather feature `feature`'s values for `samples` (sorted ascending, unique)
/// into a vector aligned to `samples`, casting each to f32. Unmatched samples
/// read `0.0`. `O(|samples| + nnz(column))` via a merge-join — no dense pass.
pub(crate) fn gather_column_f32(x: &FeatureMatrix, feature: usize, samples: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; samples.len()];
    let mut col = x.column(feature).peekable();
    for (i, &s) in samples.iter().enumerate() {
        // Column rows are strictly ascending and `samples` is too, so any row
        // below the current sample is below every later sample: consume it.
        while col.peek().is_some_and(|&(r, _)| r < s) {
            col.next();
        }
        if let Some(&(r, v)) = col.peek() {
            if r == s {
                // A matched entry is left in place, not consumed here: since
                // `samples` is strictly ascending, this row is < every later
                // sample, so the `< s` guard above discards it next iteration.
                out[i] = v as f32;
            }
        }
    }
    out
}

/// Find the best split for a node over the given `candidate_features`.
///
/// Returns `None` if no feature yields a valid split (every candidate is
/// constant, or every partition violates the leaf-size / leaf-weight minimums).
/// The returned `weighted_decrease` is *not* yet checked against
/// `min_impurity_decrease` — the caller does that.
#[allow(clippy::too_many_arguments)]
pub(crate) fn best_split(
    x: &FeatureMatrix,
    samples: &[usize],
    totals: &Totals,
    node_impurity: f64,
    criterion: Criterion,
    candidate_features: &[usize],
    min_samples_leaf: usize,
    min_weight_leaf: f64,
    n_total: f64,
) -> Option<BestSplit> {
    let n_t = samples.len() as f64;
    // Evaluate features in ascending index order so ties resolve to the lowest
    // feature index (then, within a feature, to the lowest threshold).
    let mut features = candidate_features.to_vec();
    features.sort_unstable();

    let mut best: Option<BestSplit> = None;
    for &f in &features {
        let vals = gather_column_f32(x, f, samples);
        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by(|&a, &b| {
            vals[a]
                .partial_cmp(&vals[b])
                .expect("finite feature values")
        });

        // Running left-side accumulators; the right side is derived as total − left.
        let mut acc = Accumulator::new(totals);
        let mut i = 0;
        while i < order.len() {
            let cur = vals[order[i]];
            if i > 0 {
                // Candidate boundary between the previous block and this one.
                let prev = vals[order[i - 1]];
                let n_left = i as f64;
                let n_right = n_t - n_left;
                if n_left >= min_samples_leaf as f64
                    && n_right >= min_samples_leaf as f64
                    && n_left >= min_weight_leaf
                    && n_right >= min_weight_leaf
                {
                    let imp_left = acc.left_impurity(n_left, criterion);
                    let imp_right = acc.right_impurity(n_right, criterion);
                    let decrease = (n_t / n_total)
                        * (node_impurity - (n_right / n_t) * imp_right - (n_left / n_t) * imp_left);
                    let threshold = midpoint(prev, cur);
                    // Strict `>` keeps the earliest candidate on ties.
                    if best.as_ref().is_none_or(|b| decrease > b.weighted_decrease) {
                        best = Some(BestSplit {
                            feature: f,
                            threshold,
                            weighted_decrease: decrease,
                        });
                    }
                }
            }
            // Absorb the whole equal-value block into the left side.
            let block_val = cur;
            while i < order.len() && vals[order[i]] == block_val {
                acc.push(samples[order[i]]);
                i += 1;
            }
        }
    }
    best
}

/// Threshold between two distinct sorted f32 values, computed as sklearn does
/// (`prev/2 + cur/2` in f64), clamped down to `prev` if rounding lands it on
/// `cur` or produces a non-finite value.
fn midpoint(prev: f32, cur: f32) -> f64 {
    let (prev, cur) = (prev as f64, cur as f64);
    let mid = prev / 2.0 + cur / 2.0;
    if mid == cur || !mid.is_finite() {
        prev
    } else {
        mid
    }
}

/// Incremental left-side accumulator; the right side is `total − left`.
enum Accumulator<'a> {
    Class {
        codes: &'a [usize],
        total: &'a [f64],
        left: Vec<f64>,
    },
    Reg {
        y: &'a [f64],
        total_sum: f64,
        total_sumsq: f64,
        left_sum: f64,
        left_sumsq: f64,
    },
}

impl<'a> Accumulator<'a> {
    fn new(totals: &Totals<'a>) -> Self {
        match *totals {
            Totals::Class { codes, counts } => Accumulator::Class {
                codes,
                total: counts,
                left: vec![0.0; counts.len()],
            },
            Totals::Reg { y, sum, sumsq } => Accumulator::Reg {
                y,
                total_sum: sum,
                total_sumsq: sumsq,
                left_sum: 0.0,
                left_sumsq: 0.0,
            },
        }
    }

    /// Move one sample (weight 1) from the right side to the left side.
    fn push(&mut self, sample: usize) {
        match self {
            Accumulator::Class { codes, left, .. } => left[codes[sample]] += 1.0,
            Accumulator::Reg {
                y,
                left_sum,
                left_sumsq,
                ..
            } => {
                let v = y[sample];
                *left_sum += v;
                *left_sumsq += v * v;
            }
        }
    }

    fn left_impurity(&self, n_left: f64, criterion: Criterion) -> f64 {
        match self {
            Accumulator::Class { left, .. } => impurity_from_counts(left, n_left, criterion),
            Accumulator::Reg {
                left_sum,
                left_sumsq,
                ..
            } => variance(*left_sum, *left_sumsq, n_left),
        }
    }

    fn right_impurity(&self, n_right: f64, criterion: Criterion) -> f64 {
        match self {
            Accumulator::Class { total, left, .. } => {
                let right: Vec<f64> = total.iter().zip(left).map(|(t, l)| t - l).collect();
                impurity_from_counts(&right, n_right, criterion)
            }
            Accumulator::Reg {
                total_sum,
                total_sumsq,
                left_sum,
                left_sumsq,
                ..
            } => variance(total_sum - left_sum, total_sumsq - left_sumsq, n_right),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_merges_implicit_and_explicit_zeros() {
        // 4 samples x 1 feature; rows 0 and 3 stored (one an explicit 0.0),
        // rows 1 and 2 implicit zeros.
        let x = FeatureMatrix::from_coo(4, 1, &[0, 3], &[0, 0], &[-1.0, 0.0]).unwrap();
        let got = gather_column_f32(&x, 0, &[0, 1, 2, 3]);
        assert_eq!(got, vec![-1.0, 0.0, 0.0, 0.0]);
        // A subset of samples still gathers correctly.
        assert_eq!(gather_column_f32(&x, 0, &[1, 3]), vec![0.0, 0.0]);
    }

    #[test]
    fn split_picks_clean_binary_boundary() {
        // Feature 0 = [0, 0, 1, 1]; classes = [0, 0, 1, 1]. Best split thr 0.5.
        let x = FeatureMatrix::from_coo(4, 1, &[2, 3], &[0, 0], &[1.0, 1.0]).unwrap();
        let codes = [0usize, 0, 1, 1];
        let counts = [2.0, 2.0];
        let totals = Totals::Class {
            codes: &codes,
            counts: &counts,
        };
        let node_imp = 0.5; // gini of [.5,.5]
        let s = best_split(
            &x,
            &[0, 1, 2, 3],
            &totals,
            node_imp,
            Criterion::Gini,
            &[0],
            1,
            0.0,
            4.0,
        )
        .unwrap();
        assert_eq!(s.feature, 0);
        assert_eq!(s.threshold, 0.5);
        // Perfect split: both children pure, decrease = full node impurity.
        assert!((s.weighted_decrease - 0.5).abs() < 1e-12);
    }

    #[test]
    fn constant_feature_yields_no_split() {
        let x = FeatureMatrix::from_coo(3, 1, &[], &[], &[]).unwrap(); // all-zero column
        let codes = [0usize, 1, 0];
        let counts = [2.0, 1.0];
        let totals = Totals::Class {
            codes: &codes,
            counts: &counts,
        };
        assert!(best_split(
            &x,
            &[0, 1, 2],
            &totals,
            0.44,
            Criterion::Gini,
            &[0],
            1,
            0.0,
            3.0
        )
        .is_none());
    }

    #[test]
    fn zero_block_lands_mid_scan_with_negatives() {
        // Feature = [-1, implicit 0, 2] for samples [0,1,2]. Sorted: -1, 0, 2.
        // The single best split separates {-1} from {0, 2}. classes 0,1,1.
        let x = FeatureMatrix::from_coo(3, 1, &[0, 2], &[0, 0], &[-1.0, 2.0]).unwrap();
        let codes = [0usize, 1, 1];
        let counts = [1.0, 2.0];
        let totals = Totals::Class {
            codes: &codes,
            counts: &counts,
        };
        let node_imp = 1.0 - ((1.0f64 / 3.0).powi(2) + (2.0f64 / 3.0).powi(2));
        let s = best_split(
            &x,
            &[0, 1, 2],
            &totals,
            node_imp,
            Criterion::Gini,
            &[0],
            1,
            0.0,
            3.0,
        )
        .unwrap();
        // Threshold between -1 and 0 => midpoint -0.5.
        assert_eq!(s.threshold, -0.5);
    }
}
