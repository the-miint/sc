//! Persist round-trip: fit a model, serialize it to bytes with
//! [`ModelBundle::to_bytes`], reload with [`ModelBundle::from_bytes`], and assert
//! the reloaded model predicts identically. Gated on the `persist` feature
//! (declared via `required-features` in `Cargo.toml`, so it is skipped under
//! `--no-default-features`).
//!
//! Run with:
//!
//! ```text
//! cargo run -p sc-core --example persist_roundtrip
//! ```

use sc_core::{FeatureMatrix, MaxFeatures, ModelBundle, RandomForest, RandomForestParams, Targets};

/// Dense rows → COO triplets (skipping structural zeros) → a [`FeatureMatrix`].
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
    let x = matrix_from_rows(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[3.0, 3.0],
        &[3.0, 4.0],
        &[4.0, 3.0],
    ]);
    let y = Targets::categorical(&["a", "a", "a", "b", "b", "b"]).expect("valid labels");

    let mut params = RandomForestParams::classifier_defaults();
    params.n_estimators = 16;
    params.tree.max_features = MaxFeatures::Sqrt;
    let forest = RandomForest::fit(&x, &y, &params, 7).expect("fit succeeds");
    let before = forest.predict_class_indices(&x).expect("predict");

    // Wrap the trained forest in a ModelBundle (forest + params + optional
    // feature ids / RFE selection) and round-trip it through the postcard wire
    // envelope.
    let bundle = ModelBundle {
        forest,
        params: Some(params),
        feature_ids: None,
        rfe_selection: None,
    };
    let bytes = bundle.to_bytes();
    println!("serialized model: {} bytes", bytes.len());

    let reloaded = ModelBundle::from_bytes(&bytes).expect("from_bytes round-trips");
    let after = reloaded.forest.predict_class_indices(&x).expect("predict");

    assert_eq!(
        before, after,
        "reloaded predictions differ from the original"
    );
    println!(
        "round-trip OK: predictions identical across serialize/deserialize ({} samples)",
        after.len()
    );
}
