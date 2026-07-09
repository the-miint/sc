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
    ForestClassifier(ForestClassifierCase),
    ForestRegressor(ForestRegressorCase),
    ForestAggregate(ForestAggregateCase),
    RfeTight(RfeTightCase),
    RfecvAggregate(RfecvAggregateCase),
    NcvAggregate(NcvAggregateCase),
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

/// One tree of a **classifier** forest: the eight exported `tree_` arrays, with
/// per-node `value` a class-proportion vector. Imported via `Tree::from_sklearn`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClsTree {
    pub children_left: Vec<i64>,
    pub children_right: Vec<i64>,
    pub feature: Vec<i64>,
    pub threshold: Vec<f64>,
    pub value: Vec<Vec<f64>>,
    pub impurity: Vec<f64>,
    pub n_node_samples: Vec<i64>,
    pub weighted_n_node_samples: Vec<f64>,
}

/// One tree of a **regressor** forest: like [`ClsTree`] but per-node `value` is a
/// scalar (the node mean, stride 1).
#[derive(Debug, Clone, Deserialize)]
pub struct RegTree {
    pub children_left: Vec<i64>,
    pub children_right: Vec<i64>,
    pub feature: Vec<i64>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
    pub impurity: Vec<f64>,
    pub n_node_samples: Vec<i64>,
    pub weighted_n_node_samples: Vec<f64>,
}

/// A fitted `RandomForestClassifier`: every tree's exported internals, the class
/// labels, a query table, and sklearn's **forest-level** `predict` /
/// `predict_proba` / `feature_importances_`. The TIGHT arbiter for the
/// soft-vote + summed-normalized-MDI aggregation (facts 1 and 3).
#[derive(Debug, Clone, Deserialize)]
pub struct ForestClassifierCase {
    pub trees: Vec<ClsTree>,
    pub n_classes: usize,
    pub classes: Vec<String>,
    pub n_features: usize,
    pub criterion: String,
    pub test_coo: CooTable,
    pub expected: ForestClassifierExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ForestClassifierExpected {
    pub predict: Vec<usize>,
    pub proba: Vec<Vec<f64>>,
    pub mdi: Vec<f64>,
}

/// A fitted `RandomForestRegressor`: like [`ForestClassifierCase`] but with
/// scalar-`value` trees and real-valued `predict` (the TIGHT arbiter for fact 2).
#[derive(Debug, Clone, Deserialize)]
pub struct ForestRegressorCase {
    pub trees: Vec<RegTree>,
    pub n_features: usize,
    pub test_coo: CooTable,
    pub expected: ForestRegressorExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ForestRegressorExpected {
    pub predict: Vec<f64>,
    pub mdi: Vec<f64>,
}

/// One empirically-calibrated score band for the AGGREGATE tier. `metric` is
/// `"accuracy"` / `"r2"` / `"mse"` for a scored band, or `"invariant"` for a
/// robustness case that asserts only shape invariants (no score check).
/// `min`/`max`/`mean`/`std` summarize sklearn's score over K seeds on the fixed
/// split; a result passes when it lands in `[min - margin, max + margin]` where
/// `margin = margin_abs + margin_rel * |band-edge|`.
#[derive(Debug, Clone, Deserialize)]
pub struct AggregateBand {
    pub metric: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub margin_abs: f64,
    pub margin_rel: f64,
}

impl AggregateBand {
    /// Whether this band is a scored one (vs an `"invariant"` robustness case).
    pub fn is_invariant(&self) -> bool {
        self.metric == "invariant"
    }

    /// Whether `got` lands within the margin-widened band.
    pub fn contains(&self, got: f64) -> bool {
        let lo = self.min - (self.margin_abs + self.margin_rel * self.min.abs());
        let hi = self.max + (self.margin_abs + self.margin_rel * self.max.abs());
        (lo..=hi).contains(&got)
    }
}

/// An AGGREGATE-tier case: fit our forest on a fixed injected split of a whole
/// dataset and assert the test score lands inside a band calibrated from
/// sklearn's own seed-to-seed spread, and that importance ranking is
/// rank-correlated with sklearn's mean importances. Classification carries an
/// accuracy `band`; regression carries an r² `band` plus an `mse_band`.
///
/// Targets arrive as raw per-sample `labels` (classification) or `values`
/// (regression) in dataset-sample order, so the Rust side rebuilds them through
/// the same sorted-unique encoding sklearn used.
#[derive(Debug, Clone, Deserialize)]
pub struct ForestAggregateCase {
    pub task: String,
    pub data_coo: CooTable,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub values: Option<Vec<f64>>,
    pub train_idx: Vec<usize>,
    pub test_idx: Vec<usize>,
    pub n_estimators: usize,
    pub band: AggregateBand,
    #[serde(default)]
    pub mse_band: Option<AggregateBand>,
    pub ref_importances: Vec<f64>,
    pub rank_corr_floor: f64,
    pub rank_corr_margin: f64,
}

/// TIGHT-tier RFE bookkeeping: a mock estimator returning `importances` (distinct,
/// so elimination is unambiguous) drives sklearn 1.4.2 `RFE`/`RFECV`, and we
/// assert our deterministic bookkeeping reproduces the results exactly. `step` is
/// sklearn's raw step (an int `>= 1`, or a fraction `< 1`); `min_features_to_select`
/// is 1 (the q2 config). Elimination runs to 1 feature (`rfe_ranking`/`rfe_support`
/// from `RFE(n_features_to_select=1)`); the mock's scripted scores yield the curve
/// (`grid_scores` ascending, aligned to `grid_n_features`) and `n_optimal`. A case
/// may also carry `rfe_target > 1` (with matching `rfe_*_at_target` expectations)
/// to check `run_rfe` below the full count — the path `rfecv` drives its final
/// elimination through, and the one that actually ships the selection.
#[derive(Debug, Clone, Deserialize)]
pub struct RfeTightCase {
    pub importances: Vec<f64>,
    pub step: f64,
    pub min_features_to_select: usize,
    pub n_features: usize,
    pub expected: RfeTightExpected,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RfeTightExpected {
    /// sklearn's resolved per-round removal count.
    pub resolved_step: usize,
    /// `RFE(n_features_to_select=1).ranking_` (1 == selected).
    pub rfe_ranking: Vec<usize>,
    /// `RFE(n_features_to_select=1).support_`.
    pub rfe_support: Vec<bool>,
    /// The score-curve feature counts, ascending (q2's reconstruction).
    pub grid_n_features: Vec<i64>,
    /// `RFECV.cv_results_['mean_test_score']`, ascending by feature count.
    pub grid_scores: Vec<f64>,
    /// `RFECV.n_features_` (chosen count).
    pub n_optimal: usize,
    /// Optional second elimination target `> 1` for a `RFE(n_features_to_select=
    /// rfe_target)` run — exercises `run_rfe` below the full count. Present only in
    /// fixtures that opt in; when set, `rfe_ranking_at_target`/`rfe_support_at_target`
    /// carry that run's `ranking_`/`support_`.
    #[serde(default)]
    pub rfe_target: Option<usize>,
    #[serde(default)]
    pub rfe_ranking_at_target: Option<Vec<usize>>,
    #[serde(default)]
    pub rfe_support_at_target: Option<Vec<bool>>,
}

/// AGGREGATE-tier: end-to-end RFECV selected-feature overlap vs q2. `reference_sets`
/// are q2's `RFECV` selected feature-index sets over K seeds (same config as ours);
/// `jaccard_floor` is the min pairwise Jaccard among them (a self-consistency
/// floor). Our selection must overlap q2's at least `jaccard_floor - jaccard_margin`
/// (mean Jaccard vs the reference sets), and the chosen count must land in
/// `[n_optimal_min, n_optimal_max]` widened by `n_optimal_margin`.
#[derive(Debug, Clone, Deserialize)]
pub struct RfecvAggregateCase {
    pub task: String,
    pub data_coo: CooTable,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub values: Option<Vec<f64>>,
    pub step: f64,
    pub n_splits: usize,
    pub n_estimators: usize,
    pub min_features_to_select: usize,
    pub reference_sets: Vec<Vec<usize>>,
    pub jaccard_floor: f64,
    pub jaccard_margin: f64,
    pub n_optimal_min: usize,
    pub n_optimal_max: usize,
    pub n_optimal_margin: usize,
}

/// An AGGREGATE-tier nested/OOF-CV case (M8), mirroring [`ForestAggregateCase`]
/// but with **injected folds** instead of a single split: `fold_ids[i]` is the
/// test fold sample `i` belongs to (rebuilt into `CvFolds` via
/// `CvFolds::from_fold_ids(&fold_ids, n_splits)`). Over K forest seeds (folds
/// fixed), the generator ran the OOF loop and scored the **overall** concatenated
/// prediction — `accuracy_score(y, oof)` (classification) or squared-Pearson r² +
/// MSE via the `linear_regress` formula (regression), matching M4's
/// `ForestAggregate`, not sklearn's `r2_score`. `band` is the accuracy (cls) or
/// r² (reg) band; `mse_band` accompanies regression. `ref_importances` is the
/// seed-mean of the fold-averaged importances; the run's importances must be
/// Spearman-correlated with it at least `rank_corr_floor - rank_corr_margin`.
///
/// Targets arrive as raw per-sample `labels` (classification) or `values`
/// (regression) in dataset order, rebuilt Rust-side through the same sorted-unique
/// encoding sklearn used.
#[derive(Debug, Clone, Deserialize)]
pub struct NcvAggregateCase {
    pub task: String,
    pub data_coo: CooTable,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub values: Option<Vec<f64>>,
    pub fold_ids: Vec<usize>,
    pub n_splits: usize,
    pub n_estimators: usize,
    pub band: AggregateBand,
    #[serde(default)]
    pub mse_band: Option<AggregateBand>,
    pub ref_importances: Vec<f64>,
    pub rank_corr_floor: f64,
    pub rank_corr_margin: f64,
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
