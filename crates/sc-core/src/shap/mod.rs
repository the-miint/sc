//! Path-dependent TreeSHAP: exact per-prediction Shapley feature attributions
//! over a [`RandomForest`] (design §8/§10, M10).
//!
//! Given a *fixed* forest, Shapley values are a deterministic function of the
//! tree structure + per-node cover (`weighted_n_node_samples`) — there is no RNG
//! and no background dataset. This is why M4/M5 retained per-node cover: the tree
//! is "SHAP-ready". The algorithm is Lundberg et al. 2018 ("Consistent
//! Individualized Feature Attribution for Tree Ensembles", arXiv:1802.03888,
//! Algorithm 2 — cite this if you publish `tree_shap` results): the
//! O(TLD²) EXTEND/UNWIND recursion over each tree's unique-feature decision path,
//! with the reference distribution given by node cover
//! (`feature_perturbation="tree_path_dependent"` in the Python `shap` library).
//!
//! **Forest = mean of trees.** `predict_proba` / `predict_values` are the
//! arithmetic mean over trees; SHAP is linear in the model output, so forest SHAP
//! is the mean over trees of per-tree SHAP, and each output's base value is the
//! mean over trees of that tree's root `value` (sklearn's root value already *is*
//! the cover-weighted mean over descendant leaves, i.e. the path-dependent
//! expectation E[f]).
//!
//! **Additivity identity** (the exact self-check): for every sample `s` and
//! output `o`, `base_values[o] + Σ_f values[s][o][f]` equals `predict_proba`
//! (classifier) / `predict_values` (regressor). Holds to ~1e-12 for our own
//! computation.
//!
//! Pure safe Rust, no new dependencies — `sc-core` stays
//! `#![forbid(unsafe_code)]`, Arrow-free, and wasm-clean.

use crate::data::FeatureMatrix;
use crate::exec::fill_chunks;
use crate::forest::RandomForest;
use crate::tree::{Task, Tree};
use crate::{Error, Result};

/// Per-prediction Shapley attributions for a forest, as returned by
/// [`RandomForest::tree_shap`].
///
/// For a classifier, `n_outputs == n_classes` and each output explains that
/// class's predicted probability; for a regressor, `n_outputs == 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapValues {
    /// Number of explained outputs: `n_classes` (classifier) or `1` (regressor).
    pub n_outputs: usize,
    /// Number of input features.
    pub n_features: usize,
    /// Expected model value E[f] per output (`len == n_outputs`): the mean over
    /// trees of each tree's root `value`.
    pub base_values: Vec<f64>,
    /// Shapley values, indexed `[sample][output][feature]`.
    pub values: Vec<Vec<Vec<f64>>>,
}

impl ShapValues {
    /// Global feature importance: the mean over samples of `|shap|` per
    /// `(output, feature)`, indexed `[output][feature]` (plan §M10, "global
    /// importance via mean|SHAP|"). Zero-length (no samples) yields a zero matrix.
    pub fn mean_abs(&self) -> Vec<Vec<f64>> {
        let mut acc = vec![vec![0.0f64; self.n_features]; self.n_outputs];
        for sample in &self.values {
            for (o, row) in sample.iter().enumerate() {
                for (f, &v) in row.iter().enumerate() {
                    acc[o][f] += v.abs();
                }
            }
        }
        if !self.values.is_empty() {
            let inv = 1.0 / self.values.len() as f64;
            for row in acc.iter_mut() {
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        acc
    }
}

/// [`ShapValues`] as one flat buffer, as returned by
/// [`RandomForest::tree_shap_flat`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShapValuesFlat {
    /// Number of explained samples.
    pub n_samples: usize,
    /// Number of explained outputs: `n_classes` (classifier) or `1` (regressor).
    pub n_outputs: usize,
    /// Number of input features.
    pub n_features: usize,
    /// Expected model value E[f] per output (`len == n_outputs`).
    pub base_values: Vec<f64>,
    /// Shapley values, `len == n_samples * n_outputs * n_features`, indexed
    /// `[(sample * n_outputs + output) * n_features + feature]` — output-major
    /// within a sample.
    pub values: Vec<f64>,
}

impl RandomForest {
    /// Path-dependent TreeSHAP over the forest. A classifier explains
    /// `predict_proba` (one attribution set per class); a regressor explains
    /// `predict_values`.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch between `x` and the
    /// forest.
    pub fn tree_shap(&self, x: &FeatureMatrix) -> Result<ShapValues> {
        let flat = self.tree_shap_flat(x)?;
        let (n_outputs, n_features) = (flat.n_outputs, flat.n_features);
        let values: Vec<Vec<Vec<f64>>> = (0..flat.n_samples)
            .map(|s| {
                (0..n_outputs)
                    .map(|o| {
                        let at = (s * n_outputs + o) * n_features;
                        flat.values[at..at + n_features].to_vec()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect();
        Ok(ShapValues {
            n_outputs,
            n_features,
            base_values: flat.base_values,
            values,
        })
    }

    /// [`tree_shap`](Self::tree_shap) into one flat buffer, indexed
    /// `values[(sample * n_outputs + output) * n_features + feature]`.
    ///
    /// Each sample's attributions are written straight into its own slice, so the
    /// matrix exists exactly once — no per-sample allocation and no copy to join
    /// them. This is the layout the C API exports; at the sizes TreeSHAP reaches
    /// (samples × outputs × features doubles) the nested form's extra copies are
    /// what run out of memory first, notably on wasm32.
    pub fn tree_shap_flat(&self, x: &FeatureMatrix) -> Result<ShapValuesFlat> {
        if x.n_features() != self.n_features() {
            return Err(Error::InvalidInput(format!(
                "tree_shap: forest expects {} features, got {}",
                self.n_features(),
                x.n_features()
            )));
        }
        let n_outputs = match self.task() {
            Task::Classification { n_classes } => n_classes,
            Task::Regression => 1,
        };
        let n_features = self.n_features();
        let inv_trees = 1.0 / self.n_trees() as f64;

        // Base value per output = mean over trees of the tree's root value. The
        // additivity identity (base + Σ shap == prediction) holds because each
        // tree's `value(0)` is the cover-weighted mean over its leaves — an
        // invariant of a genuine sklearn export (and of this crate's own grow
        // path), assumed rather than re-verified at the `from_sklearn` boundary.
        let mut base_values = vec![0.0f64; n_outputs];
        for t in 0..self.n_trees() {
            for (b, &v) in base_values.iter_mut().zip(self.tree(t).value(0)) {
                *b += v;
            }
        }
        for b in base_values.iter_mut() {
            *b *= inv_trees;
        }

        // Bucket the sparse table's stored entries by sample once (O(nnz)); each
        // sample then scatters its entries into a dense scratch row. Fanning out
        // over samples via `fill_chunks` keeps the result thread-count-invariant
        // (each closure owns its scratch), matching `fit`/`predict`.
        let mut by_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); x.n_samples()];
        for f in 0..n_features {
            for (r, v) in x.column(f) {
                by_row[r].push((f, v));
            }
        }

        let width = n_outputs * n_features;
        let mut values = vec![0.0f64; x.n_samples() * width];
        fill_chunks(&mut values, width, |s, row| {
            let mut scratch = vec![0.0f64; n_features];
            for &(f, v) in &by_row[s] {
                scratch[f] = v;
            }
            // Accumulate per-tree phi (feature-major: phi[f * n_outputs + o]).
            let mut phi = vec![0.0f64; n_features * n_outputs];
            for t in 0..self.n_trees() {
                tree_shap_recurse(
                    self.tree(t),
                    &scratch,
                    &mut phi,
                    n_outputs,
                    0,
                    Vec::new(),
                    1.0,
                    1.0,
                    ROOT_SENTINEL,
                );
            }
            // Transpose into this sample's output-major row, scaled by 1/n_trees.
            for f in 0..n_features {
                for o in 0..n_outputs {
                    row[o * n_features + f] = phi[f * n_outputs + o] * inv_trees;
                }
            }
        });

        Ok(ShapValuesFlat {
            n_samples: x.n_samples(),
            n_outputs,
            n_features,
            base_values,
            values,
        })
    }
}

/// Feature-index sentinel for the root element of a decision path (matches the
/// `shap` C++ reference, which seeds the recursion with feature `-1`).
const ROOT_SENTINEL: i32 = -1;

/// One element of a unique decision path (Algorithm 2). `zero_fraction` is the
/// fraction of "zero" paths (subsets *not* containing this feature) that flow
/// through, `one_fraction` the fraction of "one" paths (subsets containing it),
/// and `pweight` the proportion of feature-permutation weight carried here.
#[derive(Debug, Clone, Copy)]
struct PathElement {
    feature_index: i32,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// Grow the decision path by one element carrying `zero_fraction`/`one_fraction`
/// for `feature_index`, updating every prior element's permutation weight
/// (Algorithm 2, EXTEND). On entry `path.len()` is the depth *before* extension.
fn extend_path(
    path: &mut Vec<PathElement>,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i32,
) {
    let unique_depth = path.len();
    path.push(PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    });
    let denom = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        let w_i = path[i].pweight;
        path[i + 1].pweight += one_fraction * w_i * (i as f64 + 1.0) / denom;
        path[i].pweight = zero_fraction * w_i * (unique_depth as f64 - i as f64) / denom;
    }
}

/// Undo a previous [`extend_path`] of the element at `path_index`, restoring the
/// permutation weights and removing that element (Algorithm 2, UNWIND). Leaves
/// `path.len()` one shorter.
fn unwind_path(path: &mut Vec<PathElement>, path_index: usize) {
    let unique_depth = path.len() - 1;
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let denom = (unique_depth + 1) as f64;
    let mut next_one_portion = path[unique_depth].pweight;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * denom / ((i as f64 + 1.0) * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (unique_depth as f64 - i as f64) / denom;
        } else {
            path[i].pweight =
                path[i].pweight * denom / (zero_fraction * (unique_depth as f64 - i as f64));
        }
    }
    // Shift feature/zero/one down over the removed slot (pweights stay as
    // recomputed above), then drop the now-stale tail element.
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
    path.truncate(unique_depth);
}

/// The total permutation weight the path would carry if the element at
/// `path_index` were unwound — computed without mutating `path` (Algorithm 2,
/// used at leaves to weight each feature's marginal contribution).
fn unwound_path_sum(path: &[PathElement], path_index: usize) -> f64 {
    let unique_depth = path.len() - 1;
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let denom = (unique_depth + 1) as f64;
    let mut next_one_portion = path[unique_depth].pweight;
    let mut total = 0.0;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * denom / ((i as f64 + 1.0) * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction * (unique_depth as f64 - i as f64) / denom;
        } else {
            total += path[i].pweight * denom / (zero_fraction * (unique_depth as f64 - i as f64));
        }
    }
    total
}

/// TreeSHAP recursion for one tree + one sample (Algorithm 2). Accumulates each
/// feature's Shapley contribution into `phi` (feature-major:
/// `phi[feature * n_outputs + output]`). `path` is the parent's decision path
/// (owned per call so sibling recursions don't alias); `zero_fraction`,
/// `one_fraction`, and `feature_index` describe the edge just taken.
#[allow(clippy::too_many_arguments)]
fn tree_shap_recurse(
    tree: &Tree,
    x: &[f64],
    phi: &mut [f64],
    n_outputs: usize,
    node: usize,
    mut path: Vec<PathElement>,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i32,
) {
    extend_path(&mut path, zero_fraction, one_fraction, feature_index);

    if tree.is_leaf(node) {
        let leaf_value = tree.value(node);
        // Skip index 0 (the root sentinel, feature `-1`).
        for (i, el) in path.iter().enumerate().skip(1) {
            let w = unwound_path_sum(&path, i);
            let scale = w * (el.one_fraction - el.zero_fraction);
            let f = el.feature_index as usize;
            for o in 0..n_outputs {
                phi[f * n_outputs + o] += scale * leaf_value[o];
            }
        }
        return;
    }

    // Internal node: pick the hot (observed) child vs the cold one, and weight
    // both by cover fraction — the path-dependent reference distribution.
    let split = tree.feature(node) as usize;
    let node_cover = tree.weighted_n_node_samples(node);
    let (hot, cold) = if tree.goes_left(node, x[split]) {
        (
            tree.children_left(node) as usize,
            tree.children_right(node) as usize,
        )
    } else {
        (
            tree.children_right(node) as usize,
            tree.children_left(node) as usize,
        )
    };
    let hot_zero = tree.weighted_n_node_samples(hot) / node_cover;
    let cold_zero = tree.weighted_n_node_samples(cold) / node_cover;

    // If an ancestor already split on this feature, unwind that extension so it
    // can be re-applied for this (tighter) split.
    let mut incoming_zero = 1.0;
    let mut incoming_one = 1.0;
    let split_i32 = split as i32;
    if let Some(pi) = path.iter().position(|e| e.feature_index == split_i32) {
        incoming_zero = path[pi].zero_fraction;
        incoming_one = path[pi].one_fraction;
        unwind_path(&mut path, pi);
    }

    tree_shap_recurse(
        tree,
        x,
        phi,
        n_outputs,
        hot,
        path.clone(),
        incoming_zero * hot_zero,
        incoming_one,
        split_i32,
    );
    tree_shap_recurse(
        tree,
        x,
        phi,
        n_outputs,
        cold,
        path,
        incoming_zero * cold_zero,
        0.0,
        split_i32,
    );
}

#[cfg(test)]
mod tests {
    // These tests are index-driven numeric checks that cross-index parallel
    // arrays (shap values, oracle predictions, brute-force output) by the same
    // subscript — range loops read more clearly than zipped iterators here.
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::tree::Task;

    // --- hand-built forests --------------------------------------------------

    // 2-class, 3-feature classifier tree. Splits feats 0/1/2; internal `value`s
    // are the cover-weighted means of descendant leaves (as sklearn stores them),
    // so `value(root)` equals the path-dependent E[f].
    //   0: feat0<=0.5 -> (1,4)
    //     1: feat1<=0.5 -> (2,3)
    //       2: leaf [1,0]      cover 4
    //       3: leaf [0,1]      cover 2
    //     4: feat2<=0.5 -> (5,6)
    //       5: leaf [.5,.5]    cover 1
    //       6: leaf [.25,.75]  cover 3
    fn cls_tree_a() -> Tree {
        Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[1, 2, -1, -1, 5, -1, -1],
            &[4, 3, -1, -1, 6, -1, -1],
            &[0, 1, -2, -2, 2, -2, -2],
            &[0.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0],
            &[
                0.525,
                0.475, // node0
                2.0 / 3.0,
                1.0 / 3.0, // node1
                1.0,
                0.0, // node2
                0.0,
                1.0, // node3
                0.3125,
                0.6875, // node4
                0.5,
                0.5, // node5
                0.25,
                0.75, // node6
            ],
            &[0.4, 0.44, 0.0, 0.0, 0.375, 0.5, 0.375],
            &[10, 6, 4, 2, 4, 1, 3],
            &[10.0, 6.0, 4.0, 2.0, 4.0, 1.0, 3.0],
            3,
        )
        .unwrap()
    }

    // 2-class, 3-feature classifier tree that splits feature 0 TWICE on one path
    // — exercises the UNWIND branch.
    //   0: feat0<=0.0 -> (1,2)
    //     1: leaf [.9,.1]     cover 3
    //     2: feat0<=1.0 -> (3,4)
    //       3: leaf [.4,.6]   cover 2
    //       4: leaf [.2,.8]   cover 3
    fn cls_tree_b() -> Tree {
        Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[1, -1, 3, -1, -1],
            &[2, -1, 4, -1, -1],
            &[0, -2, 0, -2, -2],
            &[0.0, -2.0, 1.0, -2.0, -2.0],
            &[
                0.5125, 0.4875, // node0
                0.9, 0.1, // node1
                0.28, 0.72, // node2
                0.4, 0.6, // node3
                0.2, 0.8, // node4
            ],
            &[0.4, 0.0, 0.4, 0.0, 0.0],
            &[8, 3, 5, 2, 3],
            &[8.0, 3.0, 5.0, 2.0, 3.0],
            3,
        )
        .unwrap()
    }

    fn cls_forest() -> RandomForest {
        RandomForest::from_trees(
            vec![cls_tree_a(), cls_tree_b()],
            Some(vec!["x".into(), "y".into()]),
        )
        .unwrap()
    }

    // 3-feature regressor tree splitting feats 0/1.
    //   0: feat0<=0.5 -> (1,4)
    //     1: feat1<=0.5 -> (2,3)
    //       2: leaf 1.0   cover 2
    //       3: leaf 2.0   cover 2
    //     4: leaf 5.0     cover 4
    fn reg_tree_1() -> Tree {
        Tree::from_sklearn(
            Task::Regression,
            &[1, 2, -1, -1, -1],
            &[4, 3, -1, -1, -1],
            &[0, 1, -2, -2, -2],
            &[0.5, 0.5, -2.0, -2.0, -2.0],
            &[3.25, 1.5, 1.0, 2.0, 5.0],
            &[0.0, 0.25, 0.0, 0.0, 0.0],
            &[8, 4, 2, 2, 4],
            &[8.0, 4.0, 2.0, 2.0, 4.0],
            3,
        )
        .unwrap()
    }

    // 3-feature regressor tree splitting feat 2.
    fn reg_tree_2() -> Tree {
        Tree::from_sklearn(
            Task::Regression,
            &[1, -1, -1],
            &[2, -1, -1],
            &[2, -2, -2],
            &[1.0, -2.0, -2.0],
            &[2.0, 0.0, 4.0],
            &[0.0, 0.0, 0.0],
            &[10, 5, 5],
            &[10.0, 5.0, 5.0],
            3,
        )
        .unwrap()
    }

    fn reg_forest() -> RandomForest {
        RandomForest::from_trees(vec![reg_tree_1(), reg_tree_2()], None).unwrap()
    }

    // --- brute-force reference over 2^M subsets -------------------------------

    /// Path-dependent conditional expectation v(S) for one tree: descend on
    /// features in `subset` (bit `f` set), cover-weight the average over the rest.
    fn cond_expect(tree: &Tree, node: usize, x: &[f64], subset: u32, n_outputs: usize) -> Vec<f64> {
        if tree.is_leaf(node) {
            return tree.value(node).to_vec();
        }
        let f = tree.feature(node) as usize;
        let l = tree.children_left(node) as usize;
        let r = tree.children_right(node) as usize;
        if subset & (1 << f) != 0 {
            let child = if tree.goes_left(node, x[f]) { l } else { r };
            cond_expect(tree, child, x, subset, n_outputs)
        } else {
            let cover = tree.weighted_n_node_samples(node);
            let wl = tree.weighted_n_node_samples(l) / cover;
            let wr = tree.weighted_n_node_samples(r) / cover;
            let vl = cond_expect(tree, l, x, subset, n_outputs);
            let vr = cond_expect(tree, r, x, subset, n_outputs);
            (0..n_outputs).map(|o| wl * vl[o] + wr * vr[o]).collect()
        }
    }

    /// Forest v(S) = mean over trees of per-tree v(S).
    fn forest_cond_expect(f: &RandomForest, x: &[f64], subset: u32, n_outputs: usize) -> Vec<f64> {
        let mut acc = vec![0.0f64; n_outputs];
        for t in 0..f.n_trees() {
            let v = cond_expect(f.tree(t), 0, x, subset, n_outputs);
            for (a, vo) in acc.iter_mut().zip(v) {
                *a += vo;
            }
        }
        let inv = 1.0 / f.n_trees() as f64;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        acc
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
    }

    /// Brute-force Shapley values by the definition (sum over all 2^M subsets),
    /// indexed [output][feature]. O(M·2^M) — small M only.
    fn brute_force_shap(f: &RandomForest, x: &[f64], m: usize, n_outputs: usize) -> Vec<Vec<f64>> {
        let mut phi = vec![vec![0.0f64; m]; n_outputs];
        for i in 0..m {
            for subset in 0u32..(1 << m) {
                if subset & (1 << i) != 0 {
                    continue; // enumerate S ⊆ F\{i}
                }
                let s = (subset.count_ones() as usize).min(m);
                let weight = factorial(s) * factorial(m - s - 1) / factorial(m);
                let v_without = forest_cond_expect(f, x, subset, n_outputs);
                let v_with = forest_cond_expect(f, x, subset | (1 << i), n_outputs);
                for o in 0..n_outputs {
                    phi[o][i] += weight * (v_with[o] - v_without[o]);
                }
            }
        }
        phi
    }

    fn matrix(rows: &[&[f64]]) -> FeatureMatrix {
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

    fn assert_close_2d(got: &[Vec<f64>], want: &[Vec<f64>], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: outer len");
        for (o, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.len(), w.len(), "{what}[{o}]: inner len");
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{what}[{o}][{i}]: got {a:.15e} want {b:.15e} (|diff| {:.2e} > {tol:.1e})",
                    (a - b).abs()
                );
            }
        }
    }

    #[test]
    fn classifier_matches_brute_force_and_is_additive() {
        let f = cls_forest();
        // Query rows spanning every branch, incl. the doubly-split feature 0.
        let x = matrix(&[
            &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 0.0, 0.0],
            &[2.0, 1.0, 1.0],
            &[0.6, 0.2, 0.9],
        ]);
        let shap = f.tree_shap(&x).unwrap();
        assert_eq!(shap.n_outputs, 2);
        assert_eq!(shap.n_features, 3);
        // base = mean over trees of root value.
        assert_close_2d(
            std::slice::from_ref(&shap.base_values),
            &[vec![(0.525 + 0.5125) / 2.0, (0.475 + 0.4875) / 2.0]],
            1e-12,
            "base",
        );

        let proba = f.predict_proba(&x).unwrap();
        for s in 0..x.n_samples() {
            let bf = brute_force_shap(&f, dense_row(&x, s).as_slice(), 3, 2);
            assert_close_2d(&shap.values[s], &bf, 1e-12, &format!("shap[{s}]"));
            // Additivity: base + Σ_f shap == predict_proba, per class.
            for o in 0..2 {
                let recon: f64 = shap.base_values[o] + shap.values[s][o].iter().sum::<f64>();
                assert!(
                    (recon - proba[s][o]).abs() <= 1e-12,
                    "additivity[{s}][{o}]: {recon} vs {}",
                    proba[s][o]
                );
            }
        }
    }

    #[test]
    fn regressor_matches_brute_force_and_is_additive() {
        let f = reg_forest();
        let x = matrix(&[
            &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 2.0],
            &[1.0, 0.0, 0.5],
            &[0.7, 0.7, 1.5],
        ]);
        let shap = f.tree_shap(&x).unwrap();
        assert_eq!(shap.n_outputs, 1);
        let pred = f.predict_values(&x).unwrap();
        for s in 0..x.n_samples() {
            let bf = brute_force_shap(&f, dense_row(&x, s).as_slice(), 3, 1);
            assert_close_2d(&shap.values[s], &bf, 1e-12, &format!("shap[{s}]"));
            let recon: f64 = shap.base_values[0] + shap.values[s][0].iter().sum::<f64>();
            assert!(
                (recon - pred[s]).abs() <= 1e-12,
                "additivity[{s}]: {recon} vs {}",
                pred[s]
            );
        }
    }

    #[test]
    fn mean_abs_averages_absolute_values() {
        let f = cls_forest();
        let x = matrix(&[&[0.0, 0.0, 0.0], &[2.0, 1.0, 1.0]]);
        let shap = f.tree_shap(&x).unwrap();
        let ma = shap.mean_abs();
        assert_eq!(ma.len(), 2);
        assert_eq!(ma[0].len(), 3);
        for o in 0..2 {
            for feat in 0..3 {
                let want = (shap.values[0][o][feat].abs() + shap.values[1][o][feat].abs()) / 2.0;
                assert!((ma[o][feat] - want).abs() <= 1e-15);
            }
        }
    }

    #[test]
    fn feature_count_mismatch_errors() {
        let f = cls_forest(); // expects 3 features
        let x = matrix(&[&[1.0, 2.0]]);
        assert!(f.tree_shap(&x).is_err());
    }

    /// Dense feature row for sample `s` (for feeding the brute-force reference).
    fn dense_row(x: &FeatureMatrix, s: usize) -> Vec<f64> {
        let mut row = vec![0.0f64; x.n_features()];
        for f in 0..x.n_features() {
            for (r, v) in x.column(f) {
                if r == s {
                    row[f] = v;
                }
            }
        }
        row
    }
}
