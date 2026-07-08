//! Depth-first CART growth from data (the property tier).
//!
//! Builds a [`Tree`] with an explicit LIFO stack, pushing the right child before
//! the left so node ids follow sklearn's depth-first preorder numbering. Every
//! node's statistics (class counts or sum/sumsq → value + impurity + cover) are
//! computed as it is popped; the fact-7 stopping rules and the [`super::splitter`]
//! split search decide whether it becomes internal or a leaf.
//!
//! This path is NOT byte-matched to sklearn — feature subsampling and tie-breaks
//! are our own deterministic [`DetRng`] rules — but it is reproducible under a
//! seed (see the `tree_grow_golden` tripwire) and unit-tested against
//! hand-computed values. Weights are uniform in M3, so `weighted_n_node_samples`
//! equals the unweighted count; the weight-free signature is exactly what M4's
//! bootstrap (select-rows-with-replacement + uniform grow) needs.

use crate::data::{FeatureMatrix, Targets};
use crate::rng::DetRng;
use crate::{Error, Result};

use super::criterion::{entropy, gini, variance};
use super::model::{Task, Tree};
use super::params::{Criterion, Resolved, TreeParams};
use super::splitter::{best_split, gather_column_f32, BestSplit, Totals};

/// sklearn's `TREE_UNDEFINED` — the threshold stored at a leaf.
const TREE_UNDEFINED: f64 = -2.0;

/// Node impurity at or below this is treated as pure (sklearn's `EPSILON`).
const IMPURITY_TOL: f64 = f64::EPSILON;

/// Grow a single CART tree from `x`/`y` under `params`, drawing per-node feature
/// subsets from `rng`.
///
/// Errors (`InvalidInput`): `x` and `y` disagree on sample count; the criterion
/// does not match the task; or [`TreeParams::validate`]/[`TreeParams::resolve`]
/// reject the parameters.
///
/// Preconditions (caller's responsibility, not re-checked here): feature values
/// are finite (NaN/inf routing/thresholds are undefined).
pub fn grow(x: &FeatureMatrix, y: &Targets, params: &TreeParams, rng: &mut DetRng) -> Result<Tree> {
    let n = x.n_samples();
    if y.n_samples() != n {
        return Err(Error::InvalidInput(format!(
            "grow: x has {n} samples but y has {}",
            y.n_samples()
        )));
    }
    let target = match y {
        Targets::Categorical { codes, classes } => TargetData::Class {
            codes,
            n_classes: classes.len(),
        },
        Targets::Numeric(v) => TargetData::Reg { y: v },
    };
    let is_classification = matches!(target, TargetData::Class { .. });
    params.validate(is_classification)?;
    let n_features = x.n_features();
    let resolved = params.resolve(n, n_features)?;

    let (task, stride) = match target {
        TargetData::Class { n_classes, .. } => (Task::Classification { n_classes }, n_classes),
        TargetData::Reg { .. } => (Task::Regression, 1),
    };

    let mut b = Builder::default();
    let mut stack = vec![Frame {
        samples: (0..n).collect(),
        depth: 0,
        parent: None,
    }];

    while let Some(frame) = stack.pop() {
        let node_id = b.children_left.len();
        let n_node = frame.samples.len();
        let n_node_f = n_node as f64;

        if let Some((pid, is_left)) = frame.parent {
            if is_left {
                b.children_left[pid] = node_id as i32;
            } else {
                b.children_right[pid] = node_id as i32;
            }
        }

        // Structural placeholders (default: leaf). A split overwrites feature +
        // threshold; the child pointers are filled when the children are popped.
        b.children_left.push(-1);
        b.children_right.push(-1);
        b.feature.push(-2);
        b.threshold.push(TREE_UNDEFINED);
        b.n_node_samples.push(n_node as u64);
        b.weighted_n_node_samples.push(n_node_f); // uniform weights

        // Node value + impurity, plus the split totals for the splitter.
        let stop = should_stop(frame.depth, n_node, &resolved);
        match target {
            TargetData::Class { codes, n_classes } => {
                let mut counts = vec![0.0f64; n_classes];
                for &s in &frame.samples {
                    counts[codes[s]] += 1.0;
                }
                let node_value: Vec<f64> = counts.iter().map(|&c| c / n_node_f).collect();
                let imp = match resolved.criterion {
                    Criterion::Gini => gini(&node_value),
                    Criterion::Entropy => entropy(&node_value),
                    Criterion::SquaredError => unreachable!("classification with SquaredError"),
                };
                b.value.extend_from_slice(&node_value);
                b.impurity.push(imp);
                if !stop && imp > IMPURITY_TOL {
                    let feats = sample_features(n_features, resolved.max_features, rng);
                    let totals = Totals::Class {
                        codes,
                        counts: &counts,
                    };
                    try_split(
                        node_id, &frame, &resolved, x, &totals, imp, &feats, &mut b, &mut stack,
                    );
                }
            }
            TargetData::Reg { y } => {
                let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
                for &s in &frame.samples {
                    let v = y[s];
                    sum += v;
                    sumsq += v * v;
                }
                let imp = variance(sum, sumsq, n_node_f);
                b.value.push(sum / n_node_f);
                b.impurity.push(imp);
                if !stop && imp > IMPURITY_TOL {
                    let feats = sample_features(n_features, resolved.max_features, rng);
                    let totals = Totals::Reg { y, sum, sumsq };
                    try_split(
                        node_id, &frame, &resolved, x, &totals, imp, &feats, &mut b, &mut stack,
                    );
                }
            }
        }
    }

    Ok(Tree::from_parts(
        task,
        stride,
        n_features,
        b.children_left,
        b.children_right,
        b.feature,
        b.threshold,
        b.value,
        b.impurity,
        b.n_node_samples,
        b.weighted_n_node_samples,
    ))
}

/// A pending node: its (sorted, unique) sample rows, its depth, and how to link
/// it into its already-emitted parent.
struct Frame {
    samples: Vec<usize>,
    depth: usize,
    parent: Option<(usize, bool)>, // (parent node id, is_left_child)
}

/// The growing struct-of-arrays.
#[derive(Default)]
struct Builder {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    value: Vec<f64>,
    impurity: Vec<f64>,
    n_node_samples: Vec<u64>,
    weighted_n_node_samples: Vec<f64>,
}

/// Target views bound to slices, chosen once per grow.
#[derive(Clone, Copy)]
enum TargetData<'a> {
    Class {
        codes: &'a [usize],
        n_classes: usize,
    },
    Reg {
        y: &'a [f64],
    },
}

/// The pre-split leaf conditions (fact 7): depth cap, too few samples to split,
/// or too few to leave two valid leaves (by count and by weight). Purity is
/// checked separately by the caller (it already has the node impurity).
fn should_stop(depth: usize, n_node: usize, r: &Resolved) -> bool {
    depth >= r.max_depth
        || n_node < r.min_samples_split
        || n_node < 2 * r.min_samples_leaf
        || (n_node as f64) < 2.0 * r.min_weight_leaf
}

/// Search for a split and, if one clears `min_impurity_decrease`, mark `node_id`
/// internal and push its two children (right first, so left gets the next id).
#[allow(clippy::too_many_arguments)]
fn try_split(
    node_id: usize,
    frame: &Frame,
    r: &Resolved,
    x: &FeatureMatrix,
    totals: &Totals,
    node_impurity: f64,
    features: &[usize],
    b: &mut Builder,
    stack: &mut Vec<Frame>,
) {
    let Some(split) = best_split(
        x,
        &frame.samples,
        totals,
        node_impurity,
        r.criterion,
        features,
        r.min_samples_leaf,
        r.min_weight_leaf,
        x.n_samples() as f64,
    ) else {
        return;
    };
    if split.weighted_decrease < r.min_impurity_decrease {
        return;
    }
    apply_split(node_id, &split, &frame.samples, frame.depth, x, b, stack);
}

/// Record the chosen split on `node_id` and enqueue its children.
fn apply_split(
    node_id: usize,
    split: &BestSplit,
    samples: &[usize],
    depth: usize,
    x: &FeatureMatrix,
    b: &mut Builder,
    stack: &mut Vec<Frame>,
) {
    b.feature[node_id] = split.feature as i32;
    b.threshold[node_id] = split.threshold;
    let (left, right) = partition(x, split.feature, split.threshold, samples);
    debug_assert!(
        is_sorted_unique(&left) && is_sorted_unique(&right),
        "children unsorted"
    );
    debug_assert!(
        !left.is_empty() && !right.is_empty(),
        "empty child from a split"
    );
    // Push right then left: the left child is popped next and gets the smaller
    // node id, reproducing sklearn's depth-first numbering.
    stack.push(Frame {
        samples: right,
        depth: depth + 1,
        parent: Some((node_id, false)),
    });
    stack.push(Frame {
        samples: left,
        depth: depth + 1,
        parent: Some((node_id, true)),
    });
}

/// Split `samples` (sorted) by `feature (as f32) <= threshold`, preserving order
/// so both children stay sorted (required by the splitter's merge-join).
fn partition(
    x: &FeatureMatrix,
    feature: usize,
    threshold: f64,
    samples: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let vals = gather_column_f32(x, feature, samples);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (i, &s) in samples.iter().enumerate() {
        if vals[i] as f64 <= threshold {
            left.push(s);
        } else {
            right.push(s);
        }
    }
    (left, right)
}

/// Choose the per-node candidate features. With `max_features >= n_features`,
/// every feature is evaluated in ascending order and **no RNG is consumed**
/// (deterministic and self-consistent). Otherwise a partial Fisher–Yates draws
/// `max_features` distinct indices via [`DetRng`].
fn sample_features(n_features: usize, max_features: usize, rng: &mut DetRng) -> Vec<usize> {
    if max_features >= n_features {
        return (0..n_features).collect();
    }
    let mut idx: Vec<usize> = (0..n_features).collect();
    for i in 0..max_features {
        let j = i + rng.uniform_u64((n_features - i) as u64) as usize;
        idx.swap(i, j);
    }
    idx.truncate(max_features);
    idx
}

fn is_sorted_unique(v: &[usize]) -> bool {
    v.windows(2).all(|w| w[0] < w[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::params::{MaxFeatures, MinSamples};

    fn dense_matrix(rows: &[&[f64]]) -> FeatureMatrix {
        let n = rows.len();
        let m = rows[0].len();
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

    #[test]
    fn max_depth_zero_makes_a_stump() {
        let x = dense_matrix(&[&[0.0], &[1.0], &[2.0]]);
        let y = Targets::categorical(&["a", "b", "a"]).unwrap();
        let mut p = TreeParams::classifier_defaults();
        p.max_depth = Some(0);
        let t = grow(&x, &y, &p, &mut DetRng::from_seed(0)).unwrap();
        assert_eq!(t.n_nodes(), 1);
        assert!(t.is_leaf(0));
    }

    #[test]
    fn pure_node_is_a_leaf() {
        let x = dense_matrix(&[&[0.0], &[1.0], &[2.0]]);
        let y = Targets::categorical(&["a", "a", "a"]).unwrap();
        let t = grow(
            &x,
            &y,
            &TreeParams::classifier_defaults(),
            &mut DetRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(t.n_nodes(), 1);
        assert_eq!(t.value(0), &[1.0]);
    }

    #[test]
    fn clean_split_produces_three_nodes() {
        // Feature 0 perfectly separates the two classes at 0.5.
        let x = dense_matrix(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
        let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
        let t = grow(
            &x,
            &y,
            &TreeParams::classifier_defaults(),
            &mut DetRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(t.n_nodes(), 3);
        assert!(!t.is_leaf(0));
        assert_eq!(t.feature(0), 0);
        assert_eq!(t.threshold(0), 0.5);
        // Left child (node 1) is class a, right (node 2) is class b.
        assert_eq!(t.value(1), &[1.0, 0.0]);
        assert_eq!(t.value(2), &[0.0, 1.0]);
    }

    #[test]
    fn min_samples_split_blocks_splitting() {
        let x = dense_matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
        let mut p = TreeParams::classifier_defaults();
        p.min_samples_split = MinSamples::Count(5); // > 4 samples
        let t = grow(&x, &y, &p, &mut DetRng::from_seed(0)).unwrap();
        assert_eq!(t.n_nodes(), 1);
    }

    #[test]
    fn regression_recovers_group_means() {
        // Feature separates {1,2} from {10,12}; leaf means 1.5 and 11.
        let x = dense_matrix(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
        let y = Targets::numeric(&[1.0, 2.0, 10.0, 12.0]).unwrap();
        let t = grow(
            &x,
            &y,
            &TreeParams::regressor_defaults(),
            &mut DetRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(t.n_nodes(), 3);
        assert_eq!(t.value(1), &[1.5]);
        assert_eq!(t.value(2), &[11.0]);
        // Root impurity is the variance of all four targets.
        assert!((t.impurity(0) - variance(25.0, 249.0, 4.0)).abs() < 1e-12);
    }

    #[test]
    fn min_impurity_decrease_blocks_marginal_split() {
        // A clean split gives decrease 0.5; a threshold just above it forbids it.
        let x = dense_matrix(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
        let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
        let mut p = TreeParams::classifier_defaults();
        p.min_impurity_decrease = 0.6;
        let t = grow(&x, &y, &p, &mut DetRng::from_seed(0)).unwrap();
        assert_eq!(t.n_nodes(), 1);
    }

    #[test]
    fn tie_break_prefers_lowest_feature_index() {
        // Two identical, equally-good features. The split must use feature 0.
        let x = dense_matrix(&[&[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], &[1.0, 1.0]]);
        let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
        let t = grow(
            &x,
            &y,
            &TreeParams::classifier_defaults(),
            &mut DetRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(t.feature(0), 0);
    }

    #[test]
    fn all_zero_feature_cannot_split() {
        // Only feature is all zeros → constant → no split possible → stump.
        let x = FeatureMatrix::from_coo(3, 1, &[], &[], &[]).unwrap();
        let y = Targets::categorical(&["a", "b", "a"]).unwrap();
        let t = grow(
            &x,
            &y,
            &TreeParams::classifier_defaults(),
            &mut DetRng::from_seed(0),
        )
        .unwrap();
        assert_eq!(t.n_nodes(), 1);
    }

    #[test]
    fn max_features_one_on_constant_draw_makes_leaf() {
        // Feature 0 is constant (all zero), feature 1 separates the classes.
        // With max_features=1, some seeds draw only the constant feature 0 and
        // must yield a stump rather than splitting.
        let x = dense_matrix(&[&[0.0, 0.0], &[0.0, 0.0], &[0.0, 1.0], &[0.0, 1.0]]);
        let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
        let mut p = TreeParams::classifier_defaults();
        p.max_features = MaxFeatures::Count(1);
        // Try several seeds; at least one draws the constant feature → stump.
        let stumped = (0..20u64).any(|seed| {
            let t = grow(&x, &y, &p, &mut DetRng::from_seed(seed)).unwrap();
            t.n_nodes() == 1
        });
        assert!(
            stumped,
            "expected some seed to draw only the constant feature"
        );
    }

    #[test]
    fn rejects_x_y_mismatch_and_bad_criterion() {
        let x = dense_matrix(&[&[0.0], &[1.0]]);
        let y = Targets::categorical(&["a", "b", "c"]).unwrap(); // 3 != 2
        assert!(grow(
            &x,
            &y,
            &TreeParams::classifier_defaults(),
            &mut DetRng::from_seed(0)
        )
        .is_err());

        let y2 = Targets::categorical(&["a", "b"]).unwrap();
        // Regressor criterion on categorical target → task mismatch.
        assert!(grow(
            &x,
            &y2,
            &TreeParams::regressor_defaults(),
            &mut DetRng::from_seed(0)
        )
        .is_err());
    }
}
