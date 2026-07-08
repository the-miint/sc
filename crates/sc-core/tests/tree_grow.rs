//! Property/unit tests for the M3 tree **grow** path and the **import**
//! validation surface — pure Rust, no fixtures.
//!
//! The grow path is the property tier: these assert hand-computable structure
//! and invariants (stopping rules, the CSC zero-block placement, tie-breaks,
//! error handling), NOT byte-equality with sklearn. Determinism/reproducibility
//! is frozen separately by `tree_grow_golden` in `tests/determinism.rs`; the
//! per-module `#[cfg(test)]` blocks cover the finer internal cases.

use sc_core::{
    grow, DetRng, FeatureMatrix, MaxFeatures, MinSamples, Targets, Task, Tree, TreeParams,
};

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

/// The CSC implicit-zero crux: a feature `[-1, implicit 0, 2]` must sort its
/// zero (row 1) *between* the negative and the positive, so the winning
/// threshold is the midpoint of the f32 values 0 and 2 = 1.0 — proving the
/// merge-join gathers the implicit zero into the mid-scan position.
#[test]
fn zero_block_lands_mid_scan_when_growing() {
    // Column: row0 = -1 (stored), row1 = 0 (implicit), row2 = 2 (stored).
    let x = FeatureMatrix::from_coo(3, 1, &[0, 2], &[0, 0], &[-1.0, 2.0]).unwrap();
    let y = Targets::categorical(&["a", "a", "b"]).unwrap(); // codes [0, 0, 1]
    let t = grow(
        &x,
        &y,
        &TreeParams::classifier_defaults(),
        &mut DetRng::from_seed(0),
    )
    .unwrap();
    assert_eq!(t.n_nodes(), 3);
    assert_eq!(t.feature(0), 0);
    assert_eq!(t.threshold(0), 1.0, "midpoint of f32 0 and 2");
    // Left leaf holds {-1, 0} (both class a); right leaf holds {2} (class b).
    assert_eq!(t.value(1), &[1.0, 0.0]);
    assert_eq!(t.value(2), &[0.0, 1.0]);
}

/// `min_weight_fraction_leaf` (uniform weights) rejects a split that would leave
/// too little weight in a child even when the sample *count* would allow it.
#[test]
fn min_weight_fraction_leaf_blocks_small_child() {
    // 10 samples; the only informative split is 8 (class a) vs 2 (class b).
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0]; 8];
    rows.extend(vec![vec![1.0]; 2]);
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let x = dense(&refs);
    let mut labels = vec!["a"; 8];
    labels.extend(["b", "b"]);
    let y = Targets::categorical(&labels).unwrap();

    let mut p = TreeParams::classifier_defaults();
    // min_weight_leaf = 0.3 * 10 = 3.0, so the 2-sample child is invalid.
    p.min_weight_fraction_leaf = 0.3;
    let t = grow(&x, &y, &p, &mut DetRng::from_seed(0)).unwrap();
    assert_eq!(t.n_nodes(), 1, "the only split leaves an underweight child");
}

#[test]
fn single_sample_is_a_leaf() {
    let x = dense(&[&[3.0, 1.0]]);
    let y = Targets::numeric(&[42.0]).unwrap();
    let t = grow(
        &x,
        &y,
        &TreeParams::regressor_defaults(),
        &mut DetRng::from_seed(0),
    )
    .unwrap();
    assert_eq!(t.n_nodes(), 1);
    assert_eq!(t.value(0), &[42.0]);
    assert_eq!(t.impurity(0), 0.0);
}

/// A grown tree evaluates correctly on its own training data: a separable set is
/// classified perfectly, and traversal reproduces the training labels.
#[test]
fn grow_then_evaluate_is_consistent() {
    let x = dense(&[&[0.0], &[0.0], &[1.0], &[1.0]]);
    let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
    let t = grow(
        &x,
        &y,
        &TreeParams::classifier_defaults(),
        &mut DetRng::from_seed(7),
    )
    .unwrap();
    assert_eq!(t.task(), Task::Classification { n_classes: 2 });
    assert_eq!(t.predict_class_indices(&x).unwrap(), vec![0, 0, 1, 1]);
    // Probabilities are one-hot on this perfectly separable set.
    assert_eq!(
        t.predict_proba(&x).unwrap(),
        vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0]
        ]
    );
}

/// Tie-break rule: when several features (and thresholds) give the same decrease,
/// the lowest feature index and then the lowest threshold win.
#[test]
fn tie_break_is_feature_then_threshold() {
    // Features 0 and 1 are identical and equally good → feature 0 must win.
    let x = dense(&[&[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], &[1.0, 1.0]]);
    let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
    let t = grow(
        &x,
        &y,
        &TreeParams::classifier_defaults(),
        &mut DetRng::from_seed(0),
    )
    .unwrap();
    assert_eq!(t.feature(0), 0);
    assert_eq!(t.threshold(0), 0.5);
}

#[test]
fn grow_rejects_bad_params() {
    let x = dense(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let y = Targets::categorical(&["a", "a", "b", "b"]).unwrap();
    // A max_features count exceeding n_features is rejected by resolve().
    let mut p = TreeParams::classifier_defaults();
    p.max_features = MaxFeatures::Count(5);
    assert!(grow(&x, &y, &p, &mut DetRng::from_seed(0)).is_err());
    // An out-of-range min_samples_leaf fraction is rejected by validate().
    let mut p = TreeParams::classifier_defaults();
    p.min_samples_leaf = MinSamples::Fraction(2.0);
    assert!(grow(&x, &y, &p, &mut DetRng::from_seed(0)).is_err());
}

/// `from_sklearn` rejects structurally malformed arrays (public-API surface;
/// the inline module tests cover more shapes).
#[test]
fn from_sklearn_rejects_structural_errors() {
    let cls = Task::Classification { n_classes: 2 };
    // Internal node whose feature index is out of range.
    assert!(Tree::from_sklearn(
        cls,
        &[1, -1, -1],
        &[2, -1, -1],
        &[9, -2, -2], // feature 9 >= n_features 1
        &[0.5, -2.0, -2.0],
        &[0.5, 0.5, 1.0, 0.0, 0.0, 1.0],
        &[0.5, 0.0, 0.0],
        &[4, 2, 2],
        &[4.0, 2.0, 2.0],
        1,
    )
    .is_err());
    // A leaf carrying a real feature instead of the -2 sentinel.
    assert!(Tree::from_sklearn(
        cls,
        &[-1],
        &[-1],
        &[0], // should be -2 for a leaf
        &[-2.0],
        &[1.0, 0.0],
        &[0.0],
        &[1],
        &[1.0],
        1,
    )
    .is_err());
    // A negative sample count.
    assert!(Tree::from_sklearn(
        cls,
        &[-1],
        &[-1],
        &[-2],
        &[-2.0],
        &[1.0, 0.0],
        &[0.0],
        &[-3],
        &[1.0],
        1,
    )
    .is_err());
}
