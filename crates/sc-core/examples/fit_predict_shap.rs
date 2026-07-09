//! End-to-end `sc-core` walkthrough: fit a RandomForest classifier on a tiny
//! in-memory dataset, then exercise the full prediction + explanation surface —
//! hard class predictions, class probabilities, MDI feature importances, and
//! path-dependent TreeSHAP with its global mean-|SHAP| summary.
//!
//! Doubles as a documented smoke test (built by `clippy --all-targets`).
//!
//! Run with:
//!
//! ```text
//! cargo run -p sc-core --example fit_predict_shap
//! ```

use sc_core::{FeatureMatrix, MaxFeatures, RandomForest, RandomForestParams, Targets};

/// Dense rows → COO triplets (skipping structural zeros) → a [`FeatureMatrix`]
/// (rows = samples, columns = features). Small models are usually easier to
/// write out densely; the engine stores them sparsely (CSC) internally.
fn matrix_from_rows(rows: &[&[f64]]) -> FeatureMatrix {
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
    FeatureMatrix::from_coo(n, m, &r, &c, &v).expect("valid COO triplets")
}

fn main() {
    // A cleanly separable 2-feature, 2-class problem: the first three samples
    // sit near the origin (class "a"), the last three near (3–4, 3–4) ("b").
    let x = matrix_from_rows(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[3.0, 3.0],
        &[3.0, 4.0],
        &[4.0, 3.0],
    ]);
    let y = Targets::categorical(&["a", "a", "a", "b", "b", "b"]).expect("valid labels");

    // Classifier defaults with a small, fixed ensemble so the run is fast and
    // fully reproducible. The seed is mandatory — there is no implicit entropy.
    let mut params = RandomForestParams::classifier_defaults();
    params.n_estimators = 16;
    params.tree.max_features = MaxFeatures::Sqrt;
    let seed = 42;

    let forest = RandomForest::fit(&x, &y, &params, seed).expect("fit succeeds");
    let classes = forest.classes().expect("a classifier has classes");

    // Hard class predictions (indices into `classes`).
    let idx = forest.predict_class_indices(&x).expect("predict");
    let labels: Vec<&str> = idx.iter().map(|&i| classes[i].as_str()).collect();
    println!("classes:     {classes:?}");
    println!("predictions: {labels:?}");

    // Class probabilities — one row per sample, each summing to 1.
    let proba = forest.predict_proba(&x).expect("predict_proba");
    println!("\nprobabilities:");
    for (i, row) in proba.iter().enumerate() {
        println!("  sample {i}: {row:?}");
    }

    // MDI (mean decrease in impurity) feature importances, one per feature.
    println!(
        "\nfeature importances (MDI): {:?}",
        forest.feature_importances()
    );

    // Path-dependent TreeSHAP: per-sample [output][feature] attributions that
    // satisfy base + Σ_feature shap == predict_proba, plus the global mean-|SHAP|.
    let shap = forest.tree_shap(&x).expect("tree_shap");
    println!(
        "\nTreeSHAP base values (E[f] per class): {:?}",
        shap.base_values
    );
    for (o, per_feature) in shap.mean_abs().iter().enumerate() {
        println!("  mean|SHAP| for class {:?}: {per_feature:?}", classes[o]);
    }
}
