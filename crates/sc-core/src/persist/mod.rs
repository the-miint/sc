//! Durable, versioned model persistence (design §9).
//!
//! A trained [`RandomForest`] (plus optional metadata) serializes to a
//! self-describing byte buffer and reloads in another process or a future build
//! with **no training data**. The buffer is:
//!
//! ```text
//! MAGIC (4 bytes) ++ FORMAT_VERSION (u32 LE) ++ postcard(Payload)
//! ```
//!
//! The magic + version sit as raw bytes *outside* postcard so the version guard
//! is robust even against a foreign or newer file that isn't valid postcard.
//! **Loadability is keyed to our own [`FORMAT_VERSION`] only** — never to an
//! external library version — which is the deliberate fix for q2's
//! joblib/sklearn-version pickle lock. A bumped/foreign version fails with a
//! clear [`Error::Deserialize`], never a panic.
//!
//! The on-disk layout lives entirely in [`wire`]; the core model/param types
//! carry no serde derives, so format evolution is a localized change.

mod wire;

use wire::{KindWire, Payload, TreeWire};

use crate::{Error, RandomForest, RandomForestParams, Result, Task, Tree};

/// Magic prefix identifying an `sc` model buffer ("SC Random Forest").
const MAGIC: [u8; 4] = *b"SCRF";
/// On-disk format version. Bump only on an intentional wire-format change; the
/// golden fixture in `tests/persist.rs` is the tripwire for accidental drift.
const FORMAT_VERSION: u32 = 1;

/// A persistable model: a trained [`RandomForest`] plus the optional metadata the
/// envelope records (hyperparameters, feature names, and — from M6 — the
/// RFECV-selected feature set).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelBundle {
    /// The trained forest (required).
    pub forest: RandomForest,
    /// Hyperparameters the forest was fit with.
    pub params: Option<RandomForestParams>,
    /// Feature names aligned to the model's columns.
    pub feature_ids: Option<Vec<String>>,
    /// RFECV-selected feature indices (reserved for M6; `None` for an M5 model).
    pub rfe_selection: Option<Vec<usize>>,
}

impl ModelBundle {
    /// Bundle a forest with no extra metadata.
    pub fn new(forest: RandomForest) -> Self {
        ModelBundle {
            forest,
            params: None,
            feature_ids: None,
            rfe_selection: None,
        }
    }

    /// Serialize to a versioned byte buffer. Infallible for a valid in-memory
    /// model (postcard encoding of our own types cannot fail).
    pub fn to_bytes(&self) -> Vec<u8> {
        serialize(
            &self.forest,
            self.params.as_ref(),
            self.feature_ids.as_deref(),
            self.rfe_selection.as_deref(),
        )
    }

    /// Load a bundle from a buffer produced by [`Self::to_bytes`].
    ///
    /// Errors ([`Error::Deserialize`], never a panic): buffer too short, bad
    /// magic, unsupported format version, corrupt/garbage payload, or an
    /// internally inconsistent one (e.g. a classifier without class labels or a
    /// tree that fails structural validation).
    pub fn from_bytes(bytes: &[u8]) -> Result<ModelBundle> {
        deserialize(bytes)
    }
}

impl RandomForest {
    /// Serialize just the forest (no metadata) — the common save path. See
    /// [`ModelBundle::to_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        serialize(self, None, None, None)
    }

    /// Load a forest, discarding any envelope metadata. See
    /// [`ModelBundle::from_bytes`] for the error contract.
    pub fn from_bytes(bytes: &[u8]) -> Result<RandomForest> {
        Ok(ModelBundle::from_bytes(bytes)?.forest)
    }
}

/// Build the framed byte buffer from a forest + optional metadata, stamping the
/// current library version.
fn serialize(
    forest: &RandomForest,
    params: Option<&RandomForestParams>,
    feature_ids: Option<&[String]>,
    rfe_selection: Option<&[usize]>,
) -> Vec<u8> {
    serialize_with_semver(
        forest,
        params,
        feature_ids,
        rfe_selection,
        env!("CARGO_PKG_VERSION"),
    )
}

/// As [`serialize`], but with an explicit `lib_semver`. Exists so the golden-byte
/// test can pin a fixed, version-independent stamp — `lib_semver` is informational
/// data embedded in the payload, so letting the live crate version leak into the
/// golden bytes would make the format-drift tripwire fire on every release bump.
fn serialize_with_semver(
    forest: &RandomForest,
    params: Option<&RandomForestParams>,
    feature_ids: Option<&[String]>,
    rfe_selection: Option<&[usize]>,
    lib_semver: &str,
) -> Vec<u8> {
    let kind = match forest.task() {
        Task::Classification { .. } => KindWire::Classifier,
        Task::Regression => KindWire::Regressor,
    };
    let trees = (0..forest.n_trees())
        .map(|i| tree_to_wire(forest.tree(i)))
        .collect();
    let payload = Payload {
        lib_semver: lib_semver.to_string(),
        kind,
        n_features: forest.n_features(),
        classes: forest.classes().map(<[String]>::to_vec),
        params: params.map(Into::into),
        feature_ids: feature_ids.map(<[String]>::to_vec),
        rfe_selection: rfe_selection.map(<[usize]>::to_vec),
        trees,
    };
    let body = postcard::to_allocvec(&payload)
        .expect("postcard serialization of a valid model is infallible");
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Read a tree's struct-of-arrays into its wire form.
fn tree_to_wire(t: &Tree) -> TreeWire {
    TreeWire {
        children_left: t.children_left_slice().to_vec(),
        children_right: t.children_right_slice().to_vec(),
        feature: t.feature_slice().to_vec(),
        threshold: t.threshold_slice().to_vec(),
        value: t.value_slice().to_vec(),
        impurity: t.impurity_slice().to_vec(),
        n_node_samples: t.n_node_samples_slice().to_vec(),
        weighted_n_node_samples: t.weighted_n_node_samples_slice().to_vec(),
    }
}

/// Parse the framed buffer: header + version guard, then decode the payload.
fn deserialize(bytes: &[u8]) -> Result<ModelBundle> {
    if bytes.len() < 8 {
        return Err(Error::Deserialize(format!(
            "buffer too short: {} bytes (need >= 8 for the header)",
            bytes.len()
        )));
    }
    if bytes[..4] != MAGIC {
        return Err(Error::Deserialize(
            "bad magic (not an sc model buffer)".into(),
        ));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != FORMAT_VERSION {
        return Err(Error::Deserialize(format!(
            "unsupported format version {version} (this build reads version {FORMAT_VERSION})"
        )));
    }
    // `take_from_bytes` (not `from_bytes`) so trailing garbage after a valid
    // payload is rejected rather than silently ignored — a buffer with
    // unaccounted bytes is as corrupt as a truncated one.
    let (payload, rest): (Payload, &[u8]) = postcard::take_from_bytes(&bytes[8..])
        .map_err(|e| Error::Deserialize(format!("corrupt payload: {e}")))?;
    if !rest.is_empty() {
        return Err(Error::Deserialize(format!(
            "{} unexpected trailing byte(s) after payload",
            rest.len()
        )));
    }
    payload_to_bundle(payload)
}

/// Rebuild a [`ModelBundle`] from a decoded payload, validating every tree
/// through the [`Tree::from_sklearn`] hardening (the untrusted-input boundary).
fn payload_to_bundle(p: Payload) -> Result<ModelBundle> {
    // Derive the task from kind + class labels, rejecting an inconsistent pair.
    let task = match (&p.kind, &p.classes) {
        (KindWire::Classifier, Some(classes)) => Task::Classification {
            n_classes: classes.len(),
        },
        (KindWire::Classifier, None) => {
            return Err(Error::Deserialize(
                "classifier model without class labels".into(),
            ));
        }
        (KindWire::Regressor, None) => Task::Regression,
        (KindWire::Regressor, Some(_)) => {
            return Err(Error::Deserialize(
                "regressor model carries class labels".into(),
            ));
        }
    };

    let n_features = p.n_features;
    let mut trees = Vec::with_capacity(p.trees.len());
    for (i, tw) in p.trees.into_iter().enumerate() {
        let tree = wire_to_tree(task, n_features, tw)
            .map_err(|e| Error::Deserialize(format!("tree {i}: {e}")))?;
        trees.push(tree);
    }
    // `from_trees` re-derives task/n_features from tree 0 and re-validates
    // cross-tree consistency and the class count.
    let forest = RandomForest::from_trees(trees, p.classes)
        .map_err(|e| Error::Deserialize(e.to_string()))?;

    // If feature names are carried, they must align to the model's columns, so a
    // downstream consumer that zips them with importances can't silently truncate.
    if let Some(ids) = &p.feature_ids {
        if ids.len() != forest.n_features() {
            return Err(Error::Deserialize(format!(
                "feature_ids length {} != n_features {}",
                ids.len(),
                forest.n_features()
            )));
        }
    }

    Ok(ModelBundle {
        forest,
        params: p.params.map(Into::into),
        feature_ids: p.feature_ids,
        rfe_selection: p.rfe_selection,
    })
}

/// Reconstruct one [`Tree`] from its wire arrays via the validating
/// `from_sklearn` path (widening the compact `i32`/`u64` arrays to the `i64` it
/// expects). Any structural defect surfaces as an `Err` here.
fn wire_to_tree(task: Task, n_features: usize, tw: TreeWire) -> Result<Tree> {
    // i32 -> i64 is lossless; u64 -> i64 is checked so an out-of-range count
    // reports itself clearly instead of wrapping negative and tripping
    // from_sklearn's "is negative" check with a misleading value.
    let children_left: Vec<i64> = tw.children_left.iter().map(|&x| x as i64).collect();
    let children_right: Vec<i64> = tw.children_right.iter().map(|&x| x as i64).collect();
    let feature: Vec<i64> = tw.feature.iter().map(|&x| x as i64).collect();
    let n_node_samples: Vec<i64> = tw
        .n_node_samples
        .iter()
        .map(|&x| {
            i64::try_from(x)
                .map_err(|_| Error::Deserialize(format!("n_node_samples {x} exceeds i64::MAX")))
        })
        .collect::<Result<_>>()?;
    Tree::from_sklearn(
        task,
        &children_left,
        &children_right,
        &feature,
        &tw.threshold,
        &tw.value,
        &tw.impurity,
        &n_node_samples,
        &tw.weighted_n_node_samples,
        n_features,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureMatrix, MaxFeatures, Targets};
    use wire::*;

    // Frame a hand-built payload exactly as `serialize` does, so tests can craft
    // malformed buffers that still pass the magic + version guard.
    fn frame(payload: &Payload) -> Vec<u8> {
        let body = postcard::to_allocvec(payload).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    // A minimal valid single-stump classifier payload (root leaf, 2 classes).
    fn stump_payload() -> Payload {
        Payload {
            lib_semver: "test".into(),
            kind: KindWire::Classifier,
            n_features: 1,
            classes: Some(vec!["a".into(), "b".into()]),
            params: None,
            feature_ids: None,
            rfe_selection: None,
            trees: vec![TreeWire {
                children_left: vec![-1],
                children_right: vec![-1],
                feature: vec![-2],
                threshold: vec![-2.0],
                value: vec![0.5, 0.5],
                impurity: vec![0.5],
                n_node_samples: vec![2],
                weighted_n_node_samples: vec![2.0],
            }],
        }
    }

    #[test]
    fn valid_payload_round_trips() {
        let bytes = frame(&stump_payload());
        let bundle = ModelBundle::from_bytes(&bytes).unwrap();
        assert_eq!(bundle.forest.n_trees(), 1);
        assert_eq!(bundle.forest.task(), Task::Classification { n_classes: 2 });
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(matches!(
            ModelBundle::from_bytes(&[1, 2, 3]),
            Err(Error::Deserialize(_))
        ));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = frame(&stump_payload());
        bytes[0] = b'X';
        let err = ModelBundle::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Deserialize(_)));
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = frame(&stump_payload());
        // Bump the u32 version to a value this build does not read.
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        let err = ModelBundle::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported format version 999"));
    }

    #[test]
    fn rejects_garbage_body() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&[0xFF; 32]); // not a valid postcard Payload
        assert!(matches!(
            ModelBundle::from_bytes(&bytes),
            Err(Error::Deserialize(_))
        ));
    }

    #[test]
    fn rejects_corrupt_tree_topology() {
        // A two-node tree whose root's left child points back at itself — invalid
        // topology that `from_sklearn` must reject rather than let through.
        let mut p = stump_payload();
        p.trees[0] = TreeWire {
            children_left: vec![0, -1],
            children_right: vec![1, -1],
            feature: vec![0, -2],
            threshold: vec![0.5, -2.0],
            value: vec![0.5, 0.5, 1.0, 0.0],
            impurity: vec![0.5, 0.0],
            n_node_samples: vec![2, 1],
            weighted_n_node_samples: vec![2.0, 1.0],
        };
        let err = ModelBundle::from_bytes(&frame(&p)).unwrap_err();
        assert!(matches!(err, Error::Deserialize(_)));
        assert!(err.to_string().contains("tree 0"));
    }

    #[test]
    fn rejects_classifier_without_classes() {
        let mut p = stump_payload();
        p.classes = None; // classifier kind but no labels
        assert!(matches!(
            ModelBundle::from_bytes(&frame(&p)),
            Err(Error::Deserialize(_))
        ));
    }

    #[test]
    fn rejects_regressor_with_classes() {
        // Regressor kind but class labels present — an inconsistent pair.
        let mut p = stump_payload();
        p.kind = KindWire::Regressor;
        assert!(matches!(
            ModelBundle::from_bytes(&frame(&p)),
            Err(Error::Deserialize(_))
        ));
    }

    #[test]
    fn rejects_trailing_bytes() {
        // A valid payload with junk appended must fail, not silently ignore it.
        let mut bytes = frame(&stump_payload());
        bytes.push(0x00);
        let err = ModelBundle::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Deserialize(_)));
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_feature_ids_length_mismatch() {
        let mut p = stump_payload(); // n_features == 1
        p.feature_ids = Some(vec!["a".into(), "b".into()]); // length 2
        let err = ModelBundle::from_bytes(&frame(&p)).unwrap_err();
        assert!(matches!(err, Error::Deserialize(_)));
        assert!(err.to_string().contains("feature_ids"));
    }

    #[test]
    fn params_wire_round_trips() {
        // Every param mirror survives a core -> wire -> core round trip.
        let p = RandomForestParams::classifier_defaults();
        let w: ParamsWire = (&p).into();
        let back: RandomForestParams = w.into();
        assert_eq!(back, p);

        let mut r = RandomForestParams::regressor_defaults();
        r.max_samples = crate::MaxSamples::Fraction(0.5);
        r.n_estimators = 7;
        r.bootstrap = false;
        let w: ParamsWire = (&r).into();
        let back: RandomForestParams = w.into();
        assert_eq!(back, r);
    }

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

    /// The fixed model behind the golden byte fixture (small + self-contained).
    fn golden_model() -> ModelBundle {
        let x = dense(&[
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[5.0, 5.0],
            &[5.0, 6.0],
            &[6.0, 5.0],
        ]);
        let y = Targets::categorical(&["a", "a", "a", "b", "b", "b"]).unwrap();
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 3;
        p.tree.max_features = MaxFeatures::Sqrt;
        let forest = RandomForest::fit(&x, &y, &p, 20260707).unwrap();
        ModelBundle {
            forest,
            params: Some(p),
            feature_ids: Some(vec!["f0".into(), "f1".into()]),
            rfe_selection: None,
        }
    }

    // A fixed, version-independent stamp for the golden bytes: the live crate
    // version must NOT leak into the fixture, or a routine release bump would
    // trip `golden_bytes_stable` with no actual format change.
    const GOLDEN_SEMVER: &str = "golden";

    fn golden_bytes() -> Vec<u8> {
        let b = golden_model();
        serialize_with_semver(
            &b.forest,
            b.params.as_ref(),
            b.feature_ids.as_deref(),
            b.rfe_selection.as_deref(),
            GOLDEN_SEMVER,
        )
    }

    /// The committed golden bytes for the fixed model. A diff here means the wire
    /// format drifted — the cross-version / cross-platform stability tripwire
    /// (design §9 exit). If the change is intentional, regenerate the fixture
    /// (see `regenerate_golden_fixture`) and bump `FORMAT_VERSION` if the layout
    /// changed.
    #[test]
    fn golden_bytes_stable() {
        let golden: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/persist/forest_v1.postcard"
        ));
        let got = golden_bytes();
        assert_eq!(
            got,
            golden,
            "serialized bytes drifted from the committed golden fixture (got {} bytes, \
             golden {}); if intentional, rerun regenerate_golden_fixture",
            got.len(),
            golden.len()
        );
        // The committed golden still loads and round-trips (lib_semver is dropped).
        assert_eq!(ModelBundle::from_bytes(golden).unwrap(), golden_model());
    }

    /// Not a test: rewrite the committed golden fixture. Run by hand only on an
    /// intentional wire-format change:
    /// `cargo test -p sc-core persist::tests::regenerate_golden_fixture -- --ignored`.
    #[test]
    #[ignore = "writes the committed golden fixture; run manually on an intentional format change"]
    fn regenerate_golden_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/persist/forest_v1.postcard"
        );
        let bytes = golden_bytes();
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, &bytes).unwrap();
        eprintln!("wrote {} bytes to {path}", bytes.len());
    }
}
