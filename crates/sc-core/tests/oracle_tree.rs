//! TIGHT-tier oracle tests for the M3 tree **import** path.
//!
//! Loads the committed JSON fixtures (generated against `qiime2-amplicon-2024.10`
//! by `oracle/generator/gen_tree.py`), rebuilds each tree via
//! [`Tree::from_sklearn`], and asserts that `predict` / `predict_proba` /
//! `feature_importances_` (MDI) and per-node impurity reproduce scikit-learn to
//! ~1e-9. Hermetic: no Python, no network — just the checked-in fixtures.
//!
//! The grow path is validated separately by pure-Rust property/unit tests
//! (`tests/tree_grow.rs`, `tests/determinism.rs`, and inline module tests); it is
//! deliberately not compared byte-for-byte against sklearn.

use sc_core::{Criterion, FeatureMatrix, Task, Tree};
use sc_oracle::{assert_close, assert_close_slice, load_dir, CooTable, OracleCase, Payload};

fn fixtures() -> Vec<OracleCase> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../oracle/fixtures/tree");
    let cases = load_dir(dir);
    assert!(!cases.is_empty(), "no fixtures loaded from {dir}");
    cases
}

fn query_matrix(coo: &CooTable) -> FeatureMatrix {
    FeatureMatrix::from_coo(
        coo.n_samples,
        coo.n_features,
        &coo.rows,
        &coo.cols,
        &coo.vals,
    )
    .expect("valid query COO")
}

#[test]
fn classifier_import_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::TreeClassifier(c) = &case.payload else {
            continue;
        };
        n += 1;
        let (id, t) = (&case.id, &case.tolerance);
        let n_features = c.test_coo.n_features;

        // sklearn stores `value` as one proportion vector per node; flatten it.
        let value: Vec<f64> = c.value.iter().flatten().copied().collect();
        let tree = Tree::from_sklearn(
            Task::Classification {
                n_classes: c.n_classes,
            },
            &c.children_left,
            &c.children_right,
            &c.feature,
            &c.threshold,
            &value,
            &c.impurity,
            &c.n_node_samples,
            &c.weighted_n_node_samples,
            n_features,
        )
        .unwrap_or_else(|e| panic!("{id}: from_sklearn failed: {e}"));

        let x = query_matrix(&c.test_coo);

        // predict = argmax(value(leaf)), first-max on ties.
        assert_eq!(
            tree.predict_class_indices(&x).unwrap(),
            c.expected.predict,
            "{id}: predict"
        );

        // predict_proba = value(leaf).
        let proba = tree.predict_proba(&x).unwrap();
        assert_eq!(proba.len(), c.expected.proba.len(), "{id}: proba row count");
        for (i, (g, e)) in proba.iter().zip(&c.expected.proba).enumerate() {
            assert_close_slice(g, e, t, &format!("{id}: proba[{i}]"));
        }

        // feature_importances_ (MDI).
        assert_close_slice(
            &tree.feature_importances(),
            &c.expected.mdi,
            t,
            &format!("{id}: mdi"),
        );

        // Per-node impurity recomputed from stored `value` proportions must equal
        // sklearn's stored impurity — this nails the criterion (esp. entropy=log2).
        let crit = match c.criterion.as_str() {
            "gini" => Criterion::Gini,
            "entropy" => Criterion::Entropy,
            other => panic!("{id}: unexpected criterion {other}"),
        };
        for node in 0..tree.n_nodes() {
            assert_close(
                tree.class_impurity_from_value(node, crit),
                tree.impurity(node),
                t,
                &format!("{id}: impurity[{node}]"),
            );
        }
    }
    assert!(n >= 6, "expected >=6 classifier fixtures, saw {n}");
}

#[test]
fn regressor_import_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::TreeRegressor(c) = &case.payload else {
            continue;
        };
        n += 1;
        let (id, t) = (&case.id, &case.tolerance);
        let n_features = c.test_coo.n_features;

        let tree = Tree::from_sklearn(
            Task::Regression,
            &c.children_left,
            &c.children_right,
            &c.feature,
            &c.threshold,
            &c.value, // regressor value is already one scalar per node (stride 1)
            &c.impurity,
            &c.n_node_samples,
            &c.weighted_n_node_samples,
            n_features,
        )
        .unwrap_or_else(|e| panic!("{id}: from_sklearn failed: {e}"));

        let x = query_matrix(&c.test_coo);

        assert_close_slice(
            &tree.predict_values(&x).unwrap(),
            &c.expected.predict,
            t,
            &format!("{id}: predict"),
        );
        assert_close_slice(
            &tree.feature_importances(),
            &c.expected.mdi,
            t,
            &format!("{id}: mdi"),
        );
    }
    assert!(n >= 1, "expected >=1 regressor fixture, saw {n}");
}
