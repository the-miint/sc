//! The shared, evaluatable [`Tree`] (struct-of-arrays) plus its import
//! constructor [`Tree::from_sklearn`] and evaluation API.
//!
//! This is the TIGHT-tier surface: a `Tree` built from sklearn's exported
//! `tree_` internals reproduces `predict` / `predict_proba` /
//! `feature_importances_` (MDI) / per-node impurity to ~1e-9. The same `Tree` is
//! what the grow path produces and what M4 (forest) and M10 (TreeSHAP) reuse —
//! building a tree and evaluating one are deliberately decoupled (design §10).
//!
//! Layout and semantics follow sklearn 1.4.2 `_tree.pyx`:
//! - node `value` for a classifier is the vector of **class proportions**
//!   (summing to 1), stored verbatim so `predict_proba(leaf) == value(leaf)`;
//!   for a regressor it is the node mean (stride 1).
//! - traversal goes **left when `X[feature] <= threshold`**, and the comparison
//!   casts the feature value to **f32** first (sklearn splits on a float32 view
//!   of `X`); this cast is load-bearing for import exactness.
//! - leaf sentinels: `children_left == children_right == -1`, `feature == -2`.

use crate::data::FeatureMatrix;
use crate::{Error, Result};

use super::criterion;
use super::params::Criterion;

/// The learning task a [`Tree`] was built for. Governs the meaning of each
/// node's `value` and which prediction methods are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Classification with `n_classes` classes; `value` is a proportion vector.
    Classification {
        /// Number of classes (the `value` stride).
        n_classes: usize,
    },
    /// Regression; `value` is the node mean (stride 1).
    Regression,
}

/// A single CART decision tree in struct-of-arrays form. Every per-node array
/// has length `n_nodes`; node 0 is the root. Internal nodes reference children
/// by index (always `> self`, since sklearn emits nodes in DFS order); leaves
/// carry the `-1`/`-2` sentinels.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    n_nodes: usize,
    n_features: usize,
    task: Task,
    value_stride: usize,
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    /// Flattened `n_nodes * value_stride`: class proportions or node mean.
    value: Vec<f64>,
    impurity: Vec<f64>,
    n_node_samples: Vec<u64>,
    /// Per-node "cover" (weighted sample count); stored so the tree is SHAP-ready.
    weighted_n_node_samples: Vec<f64>,
}

impl Tree {
    /// Build a `Tree` from sklearn's exported `tree_` arrays (plain slices — this
    /// crate never depends on serde). `value` is flattened row-major
    /// `[n_nodes, stride]` where `stride` is `n_classes` (classification) or `1`
    /// (regression). All other slices are length `n_nodes`.
    ///
    /// Errors (`InvalidInput`): unequal per-node lengths; `value.len() !=
    /// n_nodes * stride`; a zero-node tree; a feature index outside
    /// `[0, n_features)`; a leaf/internal sentinel inconsistency; a child
    /// index that is out of range or not strictly greater than its parent
    /// (which would imply a cycle or non-DFS ordering); or an internal node
    /// whose cover is not its children's covers summed (within
    /// [`COVER_RTOL`]).
    #[allow(clippy::too_many_arguments)]
    pub fn from_sklearn(
        task: Task,
        children_left: &[i64],
        children_right: &[i64],
        feature: &[i64],
        threshold: &[f64],
        value: &[f64],
        impurity: &[f64],
        n_node_samples: &[i64],
        weighted_n_node_samples: &[f64],
        n_features: usize,
    ) -> Result<Tree> {
        let n_nodes = children_left.len();
        if n_nodes == 0 {
            return Err(Error::InvalidInput("from_sklearn: zero-node tree".into()));
        }
        if n_nodes > i32::MAX as usize {
            return Err(Error::InvalidInput(
                "from_sklearn: node count exceeds i32::MAX".into(),
            ));
        }
        let stride = match task {
            Task::Classification { n_classes } => {
                if n_classes == 0 {
                    return Err(Error::InvalidInput(
                        "from_sklearn: classification needs n_classes >= 1".into(),
                    ));
                }
                n_classes
            }
            Task::Regression => 1,
        };
        // Every per-node array must have the same length.
        for (name, len) in [
            ("children_right", children_right.len()),
            ("feature", feature.len()),
            ("threshold", threshold.len()),
            ("impurity", impurity.len()),
            ("n_node_samples", n_node_samples.len()),
            ("weighted_n_node_samples", weighted_n_node_samples.len()),
        ] {
            if len != n_nodes {
                return Err(Error::InvalidInput(format!(
                    "from_sklearn: {name} length {len} != n_nodes {n_nodes}"
                )));
            }
        }
        // `checked_mul` so a 32-bit/wasm32 target can't wrap `n_nodes*stride`
        // into a value that spuriously matches a wrongly-sized `value` slice.
        let expected_value_len = n_nodes.checked_mul(stride).ok_or_else(|| {
            Error::InvalidInput("from_sklearn: n_nodes*stride overflows usize".into())
        })?;
        if value.len() != expected_value_len {
            return Err(Error::InvalidInput(format!(
                "from_sklearn: value length {} != n_nodes*stride {expected_value_len}",
                value.len(),
            )));
        }
        // The float arrays feed predictions and MDI at ~1e-9, so reject non-finite
        // inputs at the boundary rather than letting a NaN/Inf silently degrade a
        // result. Impurity is only required to be finite — sklearn's variance
        // criterion legitimately emits tiny negative values (e.g. -1.8e-15) for
        // near-pure regression nodes via `sumsq/n − mean²` cancellation, and the
        // import path must reproduce those verbatim. Weighted counts, being
        // sample weights, are genuinely non-negative.
        check_all(threshold, "threshold", |v| v.is_finite())?;
        check_all(value, "value", |v| v.is_finite())?;
        check_all(impurity, "impurity", |v| v.is_finite())?;
        check_all(weighted_n_node_samples, "weighted_n_node_samples", |v| {
            v.is_finite() && v >= 0.0
        })?;
        // The root's cover divides the whole MDI vector, so it must be positive.
        if weighted_n_node_samples[0] <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "from_sklearn: root weighted_n_node_samples {} must be > 0",
                weighted_n_node_samples[0]
            )));
        }

        let mut cl = Vec::with_capacity(n_nodes);
        let mut cr = Vec::with_capacity(n_nodes);
        let mut feat = Vec::with_capacity(n_nodes);
        let mut nns = Vec::with_capacity(n_nodes);
        // In-degree per node: a proper rooted tree has the root referenced zero
        // times and every other node referenced exactly once. This rejects
        // orphaned/phantom subtrees that the per-node checks alone would admit.
        let mut referenced = vec![0u32; n_nodes];
        for i in 0..n_nodes {
            let (l, r, f) = (children_left[i], children_right[i], feature[i]);
            let is_leaf = l == -1;
            if is_leaf {
                if r != -1 || f != -2 {
                    return Err(Error::InvalidInput(format!(
                        "from_sklearn: node {i} has leaf sentinel children_left=-1 but \
                         children_right={r}, feature={f} (expected -1, -2)"
                    )));
                }
            } else {
                // Internal node: both children in (i, n_nodes), distinct; feature valid.
                for (side, child) in [("left", l), ("right", r)] {
                    if child <= i as i64 || child >= n_nodes as i64 {
                        return Err(Error::InvalidInput(format!(
                            "from_sklearn: node {i} {side} child {child} out of range \
                             ({i}, {n_nodes})"
                        )));
                    }
                    referenced[child as usize] += 1;
                }
                if l == r {
                    return Err(Error::InvalidInput(format!(
                        "from_sklearn: node {i} has identical children {l}"
                    )));
                }
                if f < 0 || f >= n_features as i64 {
                    return Err(Error::InvalidInput(format!(
                        "from_sklearn: node {i} feature {f} out of range [0, {n_features})"
                    )));
                }
                // TreeSHAP divides each internal node's cover into its children's
                // hot/cold fractions, so an internal node's cover must be strictly
                // positive (a genuine split always has samples). Only the root was
                // guaranteed `> 0` above; require it for every internal node so a
                // malformed import can't silently produce NaN/Inf attributions.
                if weighted_n_node_samples[i] <= 0.0 {
                    return Err(Error::InvalidInput(format!(
                        "from_sklearn: internal node {i} weighted_n_node_samples {} must be > 0",
                        weighted_n_node_samples[i]
                    )));
                }
                // Each child's hot/cold fraction is its cover over this one, so
                // the two must add up to it, or finite covers give non-finite
                // attributions (a child of 1e300 under a parent of 1e-300).
                let (wp, wl, wr) = (
                    weighted_n_node_samples[i],
                    weighted_n_node_samples[l as usize],
                    weighted_n_node_samples[r as usize],
                );
                if (wl + wr - wp).abs() > COVER_RTOL * wp {
                    return Err(Error::InvalidInput(format!(
                        "from_sklearn: internal node {i} cover {wp} is not its children's \
                         covers summed ({wl} + {wr})"
                    )));
                }
            }
            if n_node_samples[i] < 0 {
                return Err(Error::InvalidInput(format!(
                    "from_sklearn: node {i} n_node_samples {} is negative",
                    n_node_samples[i]
                )));
            }
            cl.push(l as i32);
            cr.push(r as i32);
            feat.push(f as i32);
            nns.push(n_node_samples[i] as u64);
        }
        // Node 0 is the root (no parent); every other node has exactly one parent.
        for (i, &deg) in referenced.iter().enumerate() {
            let expected = u32::from(i != 0);
            if deg != expected {
                return Err(Error::InvalidInput(format!(
                    "from_sklearn: node {i} referenced {deg} time(s) as a child (expected {expected}) \
                     — not a proper rooted tree"
                )));
            }
        }

        Ok(Tree {
            n_nodes,
            n_features,
            task,
            value_stride: stride,
            children_left: cl,
            children_right: cr,
            feature: feat,
            threshold: threshold.to_vec(),
            value: value.to_vec(),
            impurity: impurity.to_vec(),
            n_node_samples: nns,
            weighted_n_node_samples: weighted_n_node_samples.to_vec(),
        })
    }

    /// Assemble a `Tree` from arrays the grow builder produced (already in this
    /// crate's representation). Internal to the tree module; skips the
    /// import-time revalidation since the builder maintains the invariants.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        task: Task,
        value_stride: usize,
        n_features: usize,
        children_left: Vec<i32>,
        children_right: Vec<i32>,
        feature: Vec<i32>,
        threshold: Vec<f64>,
        value: Vec<f64>,
        impurity: Vec<f64>,
        n_node_samples: Vec<u64>,
        weighted_n_node_samples: Vec<f64>,
    ) -> Tree {
        let n_nodes = children_left.len();
        debug_assert_eq!(value.len(), n_nodes * value_stride);
        Tree {
            n_nodes,
            n_features,
            task,
            value_stride,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            impurity,
            n_node_samples,
            weighted_n_node_samples,
        }
    }

    /// The learning task.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of input features the tree expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Whether `node` is a leaf. Panics if `node >= n_nodes`.
    pub fn is_leaf(&self, node: usize) -> bool {
        self.children_left[node] == -1
    }

    /// Split feature of an internal `node` (`-2` for a leaf). Panics if OOB.
    pub fn feature(&self, node: usize) -> i32 {
        self.feature[node]
    }

    /// Split threshold of an internal `node`. Panics if OOB.
    pub fn threshold(&self, node: usize) -> f64 {
        self.threshold[node]
    }

    /// Left child index of `node` (`-1` for a leaf). Panics if OOB.
    pub fn children_left(&self, node: usize) -> i32 {
        self.children_left[node]
    }

    /// Right child index of `node` (`-1` for a leaf). Panics if OOB.
    pub fn children_right(&self, node: usize) -> i32 {
        self.children_right[node]
    }

    /// The `value` slice of `node`: class proportions or `[mean]`. Panics if OOB.
    pub fn value(&self, node: usize) -> &[f64] {
        &self.value[node * self.value_stride..(node + 1) * self.value_stride]
    }

    /// Impurity stored at `node`. Panics if OOB.
    pub fn impurity(&self, node: usize) -> f64 {
        self.impurity[node]
    }

    /// Unweighted sample count reaching `node`. Panics if OOB.
    pub fn n_node_samples(&self, node: usize) -> u64 {
        self.n_node_samples[node]
    }

    /// Weighted sample count ("cover") at `node`. Panics if OOB.
    pub fn weighted_n_node_samples(&self, node: usize) -> f64 {
        self.weighted_n_node_samples[node]
    }

    // Whole-array getters used by the `persist` module to read a tree's
    // struct-of-arrays in one shot (no per-node reassembly). Kept `pub(crate)`
    // and `persist`-gated: the public surface stays the per-node accessors +
    // `from_sklearn`, and they'd be dead code without the feature.
    #[cfg(feature = "persist")]
    pub(crate) fn children_left_slice(&self) -> &[i32] {
        &self.children_left
    }
    #[cfg(feature = "persist")]
    pub(crate) fn children_right_slice(&self) -> &[i32] {
        &self.children_right
    }
    #[cfg(feature = "persist")]
    pub(crate) fn feature_slice(&self) -> &[i32] {
        &self.feature
    }
    #[cfg(feature = "persist")]
    pub(crate) fn threshold_slice(&self) -> &[f64] {
        &self.threshold
    }
    /// The flattened `n_nodes * value_stride` value array.
    #[cfg(feature = "persist")]
    pub(crate) fn value_slice(&self) -> &[f64] {
        &self.value
    }
    #[cfg(feature = "persist")]
    pub(crate) fn impurity_slice(&self) -> &[f64] {
        &self.impurity
    }
    #[cfg(feature = "persist")]
    pub(crate) fn n_node_samples_slice(&self) -> &[u64] {
        &self.n_node_samples
    }
    #[cfg(feature = "persist")]
    pub(crate) fn weighted_n_node_samples_slice(&self) -> &[f64] {
        &self.weighted_n_node_samples
    }

    /// Whether a sample routes LEFT at internal `node`, given that feature's
    /// value `x_val`: `(x_val as f32 as f64) <= threshold`. The `as f32` cast
    /// reproduces sklearn's float32 view of `X` and is load-bearing for import
    /// exactness. `pub(crate)` so the in-crate `shap` module reuses the exact
    /// hot-child rule instead of duplicating the cast. Panics if `node` is OOB.
    pub(crate) fn goes_left(&self, node: usize, x_val: f64) -> bool {
        (x_val as f32 as f64) <= self.threshold[node]
    }

    /// Route one already-densified feature row to its leaf node id. The shared
    /// traversal: left when [`goes_left`](Self::goes_left) holds. `row` must have
    /// length `n_features`.
    fn descend(&self, row: &[f64]) -> usize {
        let mut node = 0usize;
        while self.children_left[node] != -1 {
            let f = self.feature[node] as usize;
            node = if self.goes_left(node, row[f]) {
                self.children_left[node] as usize
            } else {
                self.children_right[node] as usize
            };
        }
        node
    }

    /// Leaf node id for each sample of `x`, in sample order.
    ///
    /// Errors (`InvalidInput`) if `x.n_features() != self.n_features()`.
    pub fn apply(&self, x: &FeatureMatrix) -> Result<Vec<usize>> {
        self.check_features(x)?;
        let mut view = RowView::new(x);
        Ok((0..x.n_samples()).map(|s| view.leaf_of(self, s)).collect())
    }

    /// Class-probability rows (`= value(leaf)`) for each sample of `x`.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// tree is a regressor (API misuse).
    pub fn predict_proba(&self, x: &FeatureMatrix) -> Result<Vec<Vec<f64>>> {
        assert!(
            matches!(self.task, Task::Classification { .. }),
            "predict_proba called on a regression tree"
        );
        Ok(self
            .apply(x)?
            .into_iter()
            .map(|leaf| self.value(leaf).to_vec())
            .collect())
    }

    /// Predicted class **index** (`argmax(value(leaf))`, first-max on ties) for
    /// each sample of `x`.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// tree is a regressor (API misuse).
    pub fn predict_class_indices(&self, x: &FeatureMatrix) -> Result<Vec<usize>> {
        assert!(
            matches!(self.task, Task::Classification { .. }),
            "predict_class_indices called on a regression tree"
        );
        Ok(self
            .apply(x)?
            .into_iter()
            .map(|leaf| argmax_first(self.value(leaf)))
            .collect())
    }

    /// Predicted value (`value(leaf)[0]`, the node mean) for each sample of `x`.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// tree is a classifier (API misuse).
    pub fn predict_values(&self, x: &FeatureMatrix) -> Result<Vec<f64>> {
        assert!(
            matches!(self.task, Task::Regression),
            "predict_values called on a classification tree"
        );
        Ok(self
            .apply(x)?
            .into_iter()
            .map(|leaf| self.value(leaf)[0])
            .collect())
    }

    /// Mean-decrease-in-impurity feature importances (fact 4): for each internal
    /// node accumulate `wn·imp − wn_L·imp_L − wn_R·imp_R` into its split feature,
    /// divide the vector by the root cover, then normalize to sum 1 (unless the
    /// sum is 0, e.g. a single-node tree).
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut imp = vec![0.0f64; self.n_features];
        for node in 0..self.n_nodes {
            if self.is_leaf(node) {
                continue;
            }
            let l = self.children_left[node] as usize;
            let r = self.children_right[node] as usize;
            let wn = &self.weighted_n_node_samples;
            imp[self.feature[node] as usize] += wn[node] * self.impurity[node]
                - wn[l] * self.impurity[l]
                - wn[r] * self.impurity[r];
        }
        let root = self.weighted_n_node_samples[0];
        for v in imp.iter_mut() {
            *v /= root;
        }
        let total: f64 = imp.iter().sum();
        if total > 0.0 {
            for v in imp.iter_mut() {
                *v /= total;
            }
        }
        imp
    }

    /// Recompute `node`'s classification impurity from its stored `value`
    /// proportions under `crit` — used by the import oracle test to prove the
    /// stored `impurity` came from the same criterion (nailing entropy's base).
    /// Panics on a regression tree or OOB node.
    pub fn class_impurity_from_value(&self, node: usize, crit: Criterion) -> f64 {
        assert!(
            matches!(self.task, Task::Classification { .. }),
            "class_impurity_from_value on a regression tree"
        );
        // Proportions have denominator 1, so this is gini/entropy of `value`.
        criterion::impurity_from_counts(self.value(node), 1.0, crit)
    }

    fn check_features(&self, x: &FeatureMatrix) -> Result<()> {
        if x.n_features() != self.n_features {
            return Err(Error::InvalidInput(format!(
                "tree expects {} features, got {}",
                self.n_features,
                x.n_features()
            )));
        }
        Ok(())
    }
}

/// How far, relative to a parent's cover, its children's covers may sum from it.
/// sklearn sums the same sample weights for a node and for each child, so only
/// float rounding separates them: nothing with integer weights (sc's own
/// bootstrap counts, every oracle fixture), and at most 1.1e-12 measured with
/// fractional ones (sklearn 1.4.2 forests on 200,000 samples with
/// `class_weight="balanced"`, `"balanced_subsample"`, or a `sample_weight`). The
/// error grows about linearly with n, so 1e-6 leaves room for far larger data
/// and still rejects covers that do not describe the same samples.
const COVER_RTOL: f64 = 1e-6;

/// Error (`InvalidInput`) if any entry of `xs` fails `ok`, naming `what` and the
/// offending index/value — the shared shape-agnostic check for `from_sklearn`'s
/// float arrays.
fn check_all(xs: &[f64], what: &str, ok: impl Fn(f64) -> bool) -> Result<()> {
    if let Some((i, v)) = xs.iter().copied().enumerate().find(|&(_, v)| !ok(v)) {
        return Err(Error::InvalidInput(format!(
            "from_sklearn: {what}[{i}] = {v} is invalid"
        )));
    }
    Ok(())
}

/// First-argmax (lowest index wins ties), matching sklearn's `predict`.
fn argmax_first(v: &[f64]) -> usize {
    let mut best = 0;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Row-major view over a CSC [`FeatureMatrix`] for traversal.
///
/// `FeatureMatrix` has no row accessor (it is column-compressed), so we bucket
/// the stored entries by sample once (`O(nnz)`), then descend each sample by
/// scattering its entries into a reusable dense scratch (unwritten slots read
/// `0.0`, the implicit-zero value) and clearing them afterward.
///
/// M4 optimization opportunity: this buckets *every* column, but traversal only
/// reads features that appear in an internal node. For the wide, sparse tables
/// this crate targets, restricting the bucketing to the tree's referenced
/// features would drop `apply` from `O(nnz(table))` to `O(nnz(used columns))`.
struct RowView {
    by_row: Vec<Vec<(usize, f64)>>,
    scratch: Vec<f64>,
}

impl RowView {
    fn new(x: &FeatureMatrix) -> Self {
        let mut by_row = vec![Vec::new(); x.n_samples()];
        for f in 0..x.n_features() {
            for (r, v) in x.column(f) {
                by_row[r].push((f, v));
            }
        }
        RowView {
            by_row,
            scratch: vec![0.0; x.n_features()],
        }
    }

    fn leaf_of(&mut self, tree: &Tree, sample: usize) -> usize {
        // Disjoint field borrows: `by_row` (shared) + `scratch` (mut).
        let row = &self.by_row[sample];
        for &(f, v) in row {
            self.scratch[f] = v;
        }
        let leaf = tree.descend(&self.scratch);
        for &(f, _) in row {
            self.scratch[f] = 0.0;
        }
        leaf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A hand-built binary classifier: root splits feature 0 at 0.5.
    //   node 0: internal, feat 0, thr 0.5, children (1, 2)
    //   node 1: leaf, value [1, 0]   (class 0)
    //   node 2: leaf, value [0, 1]   (class 1)
    fn tiny_classifier() -> Tree {
        Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[1, -1, -1],
            &[2, -1, -1],
            &[0, -2, -2],
            &[0.5, -2.0, -2.0],
            &[0.5, 0.5, 1.0, 0.0, 0.0, 1.0], // node0 [.5,.5], node1 [1,0], node2 [0,1]
            &[0.5, 0.0, 0.0],
            &[4, 2, 2],
            &[4.0, 2.0, 2.0],
            1,
        )
        .unwrap()
    }

    #[test]
    fn evaluate_hand_built_tree() {
        let t = tiny_classifier();
        // rows 0.0 -> left (class 0), 1.0 -> right (class 1).
        let x = FeatureMatrix::from_coo(2, 1, &[1], &[0], &[1.0]).unwrap();
        assert_eq!(t.apply(&x).unwrap(), vec![1, 2]);
        assert_eq!(t.predict_class_indices(&x).unwrap(), vec![0, 1]);
        assert_eq!(
            t.predict_proba(&x).unwrap(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
    }

    #[test]
    fn feature_importances_single_feature() {
        // Only feature 0 splits, so all importance is on it.
        let fi = tiny_classifier().feature_importances();
        assert_eq!(fi, vec![1.0]);
    }

    #[test]
    fn argmax_takes_first_on_ties() {
        assert_eq!(argmax_first(&[0.5, 0.5]), 0);
        assert_eq!(argmax_first(&[0.1, 0.4, 0.4, 0.1]), 1);
    }

    #[test]
    fn from_sklearn_rejects_malformed() {
        let cls = Task::Classification { n_classes: 2 };
        // length mismatch (feature too short).
        assert!(Tree::from_sklearn(
            cls,
            &[-1],
            &[-1],
            &[],
            &[-2.0],
            &[1.0, 0.0],
            &[0.0],
            &[1],
            &[1.0],
            1
        )
        .is_err());
        // child index not greater than parent (cycle).
        assert!(Tree::from_sklearn(
            cls,
            &[0, -1],
            &[1, -1],
            &[0, -2],
            &[0.5, -2.0],
            &[0.5, 0.5, 1.0, 0.0],
            &[0.5, 0.0],
            &[2, 1],
            &[2.0, 1.0],
            1
        )
        .is_err());
        // wrong value length.
        assert!(Tree::from_sklearn(
            cls,
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[1.0],
            &[0.0],
            &[1],
            &[1.0],
            1
        )
        .is_err());
    }

    #[test]
    fn from_sklearn_rejects_semantic_errors() {
        let cls = Task::Classification { n_classes: 2 };
        let leaf = |value: &[f64], impurity: &[f64], weighted: &[f64]| {
            Tree::from_sklearn(
                cls,
                &[-1],
                &[-1],
                &[-2],
                &[-2.0],
                value,
                impurity,
                &[1],
                weighted,
                1,
            )
        };
        // A one-node tree with each float defect in turn.
        assert!(
            leaf(&[f64::NAN, 0.0], &[0.0], &[1.0]).is_err(),
            "non-finite value"
        );
        assert!(
            leaf(&[1.0, 0.0], &[f64::INFINITY], &[1.0]).is_err(),
            "non-finite impurity"
        );
        assert!(
            leaf(&[1.0, 0.0], &[0.0], &[0.0]).is_err(),
            "zero root cover"
        );
        // A tiny negative impurity (sklearn's variance cancellation) is accepted.
        assert!(
            leaf(&[1.0, 0.0], &[-1.8e-15], &[1.0]).is_ok(),
            "sklearn variance noise"
        );
        // A structurally-valid-looking input with an orphaned internal subtree
        // (nodes 3..5 are never referenced as a child) is not a proper tree.
        assert!(Tree::from_sklearn(
            cls,
            &[1, -1, -1, 4, -1, -1],
            &[2, -1, -1, 5, -1, -1],
            &[0, -2, -2, 0, -2, -2],
            &[0.5, -2.0, -2.0, 0.5, -2.0, -2.0],
            &[0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0],
            &[0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            &[4, 2, 2, 4, 2, 2],
            &[4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            1,
        )
        .is_err());
        // A non-root *internal* node (node 1) with zero cover is rejected: it
        // would make TreeSHAP divide by zero. Root (node 0) cover is positive, so
        // this exercises the per-internal-node check, not the root check.
        assert!(
            Tree::from_sklearn(
                cls,
                &[1, 2, -1, -1, -1],
                &[4, 3, -1, -1, -1],
                &[0, 0, -2, -2, -2],
                &[0.5, 0.5, -2.0, -2.0, -2.0],
                &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
                &[0.0, 0.0, 0.0, 0.0, 0.0],
                &[4, 0, 0, 0, 4],
                &[4.0, 0.0, 0.0, 0.0, 4.0],
                1,
            )
            .is_err(),
            "zero-cover internal node"
        );
    }

    /// TreeSHAP weights each child by its cover over its parent's, so covers that
    /// do not add up drive finite inputs to inf/NaN attributions: a one-split
    /// model with root cover 1e-300 and a left child of 1e300 explained a sample
    /// with a NaN SHAP value. A parent's cover must equal its children's sum, up
    /// to float rounding in sklearn's weighted sums.
    #[test]
    fn from_sklearn_rejects_inconsistent_covers() {
        let stump = |weighted: &[f64]| {
            Tree::from_sklearn(
                Task::Regression,
                &[1, -1, -1],
                &[2, -1, -1],
                &[0, -2, -2],
                &[0.5, -2.0, -2.0],
                &[1.0, 0.0, 2.0],
                &[1.0, 0.0, 0.0],
                &[2, 1, 1],
                weighted,
                1,
            )
        };
        for (weighted, what) in [
            (&[1e-300, 1e300, 1.0][..], "child cover above the parent's"),
            (&[2.0, 1.0, 0.5][..], "children short of the parent"),
            (&[2.0, 1.5, 1.5][..], "children over the parent"),
        ] {
            let err = stump(weighted).unwrap_err();
            assert!(
                matches!(&err, Error::InvalidInput(m) if m.contains("cover")),
                "{what}: got {err:?}"
            );
        }
        // Rounding in a weighted sum is not an inconsistency.
        assert!(stump(&[0.3, 0.1, 0.2]).is_ok(), "0.1 + 0.2 != 0.3 in f64");
        assert!(stump(&[2.0, 1.0, 1.0]).is_ok());
    }

    #[test]
    #[should_panic(expected = "regression tree")]
    fn predict_proba_panics_on_regressor() {
        let reg = Tree::from_sklearn(
            Task::Regression,
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[3.5],
            &[0.0],
            &[1],
            &[1.0],
            1,
        )
        .unwrap();
        let x = FeatureMatrix::from_coo(1, 1, &[0], &[0], &[1.0]).unwrap();
        let _ = reg.predict_proba(&x);
    }
}
