//! Oracle fixture loader + tolerance assertions.
//!
//! Test-support only: this crate is a dev-dependency of `sc-core`, never shipped
//! and never in the wasm build. It deserializes the JSON fixtures committed under
//! `oracle/fixtures/` (produced by the Python generator against a pinned
//! q2/sklearn env) into typed cases, and offers abs+rel tolerance checks so the
//! Rust metric implementations can be asserted against q2's ground truth.
//!
//! The [`Payload`] enum is the format seam: M1 carries inline JSON arrays; later
//! milestones can add array-backed variants without disturbing the loader or the
//! tolerance helpers.

use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Absolute + relative tolerance for a fixture's comparisons.
///
/// A value passes when `|got - expected| <= abs + rel * |expected|`, the same
/// two-sided form `numpy.isclose` uses, so tiny expected values lean on `abs`
/// while large ones lean on `rel`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
}

impl Tolerance {
    /// Whether `got` is within tolerance of `expected`. Exact equality (incl.
    /// matching infinities) short-circuits so `inf == inf` never fails.
    pub fn close(&self, got: f64, expected: f64) -> bool {
        if got == expected {
            return true;
        }
        (got - expected).abs() <= self.abs + self.rel * expected.abs()
    }
}

/// Provenance recorded in every fixture: the exact oracle library versions and
/// the generator seed. Asserting on these would make tests brittle; they exist
/// so a fixture's origin is self-describing and re-generation is reproducible.
#[derive(Debug, Clone, Deserialize)]
pub struct Generator {
    pub sklearn: String,
    pub scipy: String,
    pub numpy: String,
    pub qiime2_amplicon: String,
    pub seed: u64,
}

/// One fixture file = one case: common metadata plus a kind-tagged [`Payload`].
#[derive(Debug, Clone, Deserialize)]
pub struct OracleCase {
    pub id: String,
    pub generator: Generator,
    pub tolerance: Tolerance,
    #[serde(flatten)]
    pub payload: Payload,
}

/// The case body, internally tagged by the fixture's `kind` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Payload {
    Classification(ClassificationCase),
    Regression(RegressionCase),
    RocCurve(RocCurveCase),
    TreeClassifier(TreeClassifierCase),
    TreeRegressor(TreeRegressorCase),
}

/// Accuracy / confusion / baseline, and — when `proba` is present — multiclass
/// ROC AUCs. `y_true`/`y_pred` are integer class indices into `classes`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClassificationCase {
    pub classes: Vec<String>,
    pub y_true: Vec<usize>,
    pub y_pred: Vec<usize>,
    #[serde(default)]
    pub proba: Option<Vec<Vec<f64>>>,
    pub expected: ClassificationExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassificationExpected {
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub accuracy_ratio: f64,
    /// Confusion-matrix labels: the sorted unique union of `y_true ∪ y_pred`.
    pub labels: Vec<i64>,
    pub confusion_matrix: Vec<Vec<i64>>,
    pub confusion_matrix_normalized: Vec<Vec<f64>>,
    /// Per-class AUC aligned to `classes` (proba column order). Present iff `proba` is.
    #[serde(default)]
    pub auc_per_class: Option<Vec<f64>>,
    #[serde(default)]
    pub auc_micro: Option<f64>,
    #[serde(default)]
    pub auc_macro: Option<f64>,
}

/// Regression against `scipy.stats.linregress` (x=actual, y=pred) + sklearn MSE.
#[derive(Debug, Clone, Deserialize)]
pub struct RegressionCase {
    pub actual: Vec<f64>,
    pub pred: Vec<f64>,
    pub expected: RegressionExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegressionExpected {
    pub mse: f64,
    pub r_value: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub std_err: f64,
    pub slope: f64,
    pub intercept: f64,
}

/// A single `sklearn.metrics.roc_curve` call: the highest-risk piece
/// (`drop_intermediate`) validated by direct array comparison.
#[derive(Debug, Clone, Deserialize)]
pub struct RocCurveCase {
    pub y_true_binary: Vec<u8>,
    pub y_score: Vec<f64>,
    pub expected: RocCurveExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RocCurveExpected {
    pub fpr: Vec<f64>,
    pub tpr: Vec<f64>,
    /// Informational; used only for a point-count/finite-value sanity check. The
    /// first threshold is `inf` in sklearn 1.4.2 and is stored as JSON `null`.
    pub thresholds: Vec<Option<f64>>,
    pub auc: f64,
}

/// A query table in COO form (implicit zeros omitted; explicit zeros allowed) —
/// the input to a tree's `apply`/`predict_*`, fed to `FeatureMatrix::from_coo`.
#[derive(Debug, Clone, Deserialize)]
pub struct CooTable {
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub vals: Vec<f64>,
    pub n_samples: usize,
    pub n_features: usize,
}

/// A fitted `DecisionTreeClassifier`: its exported `tree_` arrays (per-node
/// `value` is the class-proportion vector), a query table, and sklearn's
/// `predict`/`predict_proba`/`feature_importances_` on that table. `criterion`
/// is `"gini"` or `"entropy"` so the per-node impurity round-trip can pick the
/// matching measure.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeClassifierCase {
    pub children_left: Vec<i64>,
    pub children_right: Vec<i64>,
    pub feature: Vec<i64>,
    pub threshold: Vec<f64>,
    pub value: Vec<Vec<f64>>,
    pub impurity: Vec<f64>,
    pub n_node_samples: Vec<i64>,
    pub weighted_n_node_samples: Vec<f64>,
    pub n_classes: usize,
    pub criterion: String,
    pub test_coo: CooTable,
    pub expected: TreeClassifierExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TreeClassifierExpected {
    /// Predicted class index per query row (labels are `0..n_classes`).
    pub predict: Vec<usize>,
    /// `predict_proba` rows (each equals the reached leaf's `value`).
    pub proba: Vec<Vec<f64>>,
    /// `feature_importances_` (MDI), one per feature.
    pub mdi: Vec<f64>,
}

/// A fitted `DecisionTreeRegressor`: like [`TreeClassifierCase`] but per-node
/// `value` is a scalar (the node mean) and `expected.predict` is real-valued.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeRegressorCase {
    pub children_left: Vec<i64>,
    pub children_right: Vec<i64>,
    pub feature: Vec<i64>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
    pub impurity: Vec<f64>,
    pub n_node_samples: Vec<i64>,
    pub weighted_n_node_samples: Vec<f64>,
    pub criterion: String,
    pub test_coo: CooTable,
    pub expected: TreeRegressorExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TreeRegressorExpected {
    pub predict: Vec<f64>,
    pub mdi: Vec<f64>,
}

/// Load every `*.json` fixture in `dir`, sorted by path for stable test order.
///
/// Panics with the offending path on any I/O or parse error — this is
/// test-support, so a bad fixture should fail loudly and immediately.
pub fn load_dir(dir: impl AsRef<Path>) -> Vec<OracleCase> {
    let dir = dir.as_ref();
    let mut paths: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading fixture dir {}: {e}", dir.display()))
        .map(|e| e.expect("reading fixture dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let text =
                fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", p.display()))
        })
        .collect()
}

/// Assert `got` is within `tol` of `expected`, labeling failures with `what`.
#[track_caller]
pub fn assert_close(got: f64, expected: f64, tol: &Tolerance, what: &str) {
    assert!(
        tol.close(got, expected),
        "{what}: got {got:.17e}, expected {expected:.17e} \
         (|diff| {:.3e} > abs {:.1e} + rel {:.1e})",
        (got - expected).abs(),
        tol.abs,
        tol.rel
    );
}

/// Elementwise [`assert_close`] over two equal-length slices.
#[track_caller]
pub fn assert_close_slice(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch (got {}, expected {})",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_close(*g, *e, tol, &format!("{what}[{i}]"));
    }
}
