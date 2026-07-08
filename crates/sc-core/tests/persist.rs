//! M5 model-persistence tests: round-trip fidelity + a golden byte fixture.
//!
//! The whole file is gated on the `persist` feature (default-on); under
//! `--no-default-features` it compiles to nothing, so the minimal serde-free
//! core still builds/tests.

#![cfg(feature = "persist")]

use sc_core::{
    fit_tuned, FeatureMatrix, MaxFeatures, ModelBundle, RandomForest, RandomForestParams, Targets,
    Task, TuningParams,
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

/// The fixed 12-sample, 4-feature, 3-class problem (shared with
/// `determinism.rs::forest_grow_golden`), and the classifier fit from it.
fn fixed_classifier() -> (FeatureMatrix, RandomForest, RandomForestParams) {
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
    let mut p = RandomForestParams::classifier_defaults();
    p.n_estimators = 5;
    p.tree.max_features = MaxFeatures::Sqrt;
    let forest = RandomForest::fit(&x, &y, &p, 20260707).unwrap();
    (x, forest, p)
}

/// The fixed model as the full envelope (forest + params + feature names) — the
/// subject of the golden byte fixture, so the golden exercises the whole format.
fn fixed_bundle() -> ModelBundle {
    let (_, forest, params) = fixed_classifier();
    ModelBundle {
        forest,
        params: Some(params),
        feature_ids: Some(vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()]),
        rfe_selection: None,
    }
}

#[test]
fn round_trip_classifier() {
    let (x, forest, _) = fixed_classifier();
    let bytes = forest.to_bytes();
    let back = RandomForest::from_bytes(&bytes).unwrap();
    // A reloaded model is byte-identical to the original...
    assert_eq!(back, forest, "reloaded classifier differs");
    // ...and predicts / reports importances identically.
    assert_eq!(
        back.predict_class_indices(&x).unwrap(),
        forest.predict_class_indices(&x).unwrap()
    );
    assert_eq!(
        back.predict_proba(&x).unwrap(),
        forest.predict_proba(&x).unwrap()
    );
    assert_eq!(back.feature_importances(), forest.feature_importances());
    assert!(matches!(back.task(), Task::Classification { n_classes: 3 }));
    assert_eq!(back.classes(), forest.classes());
}

#[test]
fn round_trip_regressor() {
    let x = dense(&[
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[2.0, 2.0],
        &[3.0, 1.0],
        &[1.0, 3.0],
        &[2.0, 0.0],
        &[0.0, 2.0],
        &[3.0, 3.0],
    ]);
    let y = Targets::numeric(&[1.0, 1.5, 8.0, 9.0, 2.0, 7.5, 1.2, 9.5]).unwrap();
    let mut p = RandomForestParams::regressor_defaults();
    p.n_estimators = 6;
    let forest = RandomForest::fit(&x, &y, &p, 7).unwrap();

    let back = RandomForest::from_bytes(&forest.to_bytes()).unwrap();
    assert_eq!(back, forest, "reloaded regressor differs");
    assert_eq!(
        back.predict_values(&x).unwrap(),
        forest.predict_values(&x).unwrap()
    );
    assert_eq!(back.feature_importances(), forest.feature_importances());
    assert!(matches!(back.task(), Task::Regression));
    assert!(back.classes().is_none());
}

#[test]
fn round_trip_preserves_metadata() {
    let bundle = fixed_bundle();
    let back = ModelBundle::from_bytes(&bundle.to_bytes()).unwrap();
    assert_eq!(back, bundle, "envelope metadata did not round-trip");
    // The RFE slot is reserved (M6); it stays None for an M5 model.
    assert!(back.rfe_selection.is_none());
    assert_eq!(
        back.feature_ids.as_deref(),
        Some(["f0", "f1", "f2", "f3"].map(String::from).as_slice())
    );
}

#[test]
fn serialization_is_byte_deterministic() {
    // Two serializations of the same fixed model are identical.
    assert_eq!(fixed_bundle().to_bytes(), fixed_bundle().to_bytes());
}

#[test]
fn fit_tuned_round_trip() {
    // Tune on the fixed 12-sample, 3-class problem, then round-trip the bundle.
    let (x, _, _) = fixed_classifier();
    let y = Targets::categorical(&["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"])
        .unwrap();
    let mut p = TuningParams::classifier_defaults();
    p.base.n_estimators = 6;
    p.base.tree.max_features = MaxFeatures::Sqrt;
    p.n_iter = 5;
    p.n_splits = 3;
    let (bundle, result) = fit_tuned(&x, &y, &p, 20260707).unwrap();

    // The bundle carries the *tuned* params (not the defaults), and no RFE slot.
    assert_eq!(bundle.params.as_ref(), Some(&result.best_params));
    assert!(bundle.rfe_selection.is_none());

    // It round-trips byte-identically and predicts the same after reload.
    let back = ModelBundle::from_bytes(&bundle.to_bytes()).unwrap();
    assert_eq!(back, bundle, "tuned bundle did not round-trip");
    assert_eq!(
        back.forest.predict_class_indices(&x).unwrap(),
        bundle.forest.predict_class_indices(&x).unwrap()
    );
}

// The byte-exact golden-fixture stability test lives inline in
// `persist::tests` (it pins a version-independent lib_semver via a crate-internal
// seam, so a release version bump does not spuriously trip the tripwire).
