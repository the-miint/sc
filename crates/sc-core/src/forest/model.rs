//! The evaluatable [`RandomForest`]: a bagging ensemble of [`Tree`]s plus its
//! import constructor [`RandomForest::from_trees`] and aggregating evaluation
//! API.
//!
//! This is the TIGHT-tier surface. A forest assembled from imported per-tree
//! sklearn internals reproduces scikit-learn 1.4.2's `RandomForest*` aggregation
//! exactly (~1e-9), independent of any RNG — which is what decouples *building* a
//! forest from *evaluating* one (design §10). Semantics follow `_forest.py`:
//!
//! - **classifier `predict_proba`** = elementwise mean over trees of each tree's
//!   `predict_proba` (soft vote); each forest tree's `predict_proba(leaf)` is its
//!   stored `value` proportions.
//! - **classifier `predict`** = `classes_[argmax(mean proba)]`, argmax taking the
//!   first (lowest-index) max on ties.
//! - **regressor `predict`** = mean over trees of each tree's leaf mean.
//! - **`feature_importances_`** = sum every tree's normalized MDI, then normalize
//!   to sum 1. sklearn means the per-tree normalized MDI over the non-stump trees;
//!   because each non-stump tree's MDI already sums to 1 (and a stump's is all
//!   zero), summing-then-normalizing is algebraically identical (see the fact-3
//!   note in the M4 plan) — the import fixtures confirm it to 1e-9.

use crate::data::FeatureMatrix;
use crate::tree::{Task, Tree};
use crate::{Error, Result};

/// A trained random forest: a non-empty ensemble of [`Tree`]s that all share the
/// same [`Task`] and feature count. Built either by importing sklearn's per-tree
/// internals ([`Self::from_trees`]) or by growing from data
/// ([`Self::fit`](RandomForest::fit)).
#[derive(Debug, Clone, PartialEq)]
pub struct RandomForest {
    trees: Vec<Tree>,
    task: Task,
    n_features: usize,
    /// `Some(labels)` for classification (the probability-column labels, in
    /// sklearn `classes_` order); `None` for regression.
    classes: Option<Vec<String>>,
}

impl RandomForest {
    /// Assemble a forest from already-built [`Tree`]s (e.g. imported via
    /// [`Tree::from_sklearn`]). `classes` must be `Some` for a classification
    /// ensemble and `None` for a regression one.
    ///
    /// Errors (`InvalidInput`): an empty ensemble; trees that disagree on
    /// [`Task`] or feature count; `classes` presence not matching the task; or a
    /// `classes` length that differs from the classifier's `n_classes`.
    pub fn from_trees(trees: Vec<Tree>, classes: Option<Vec<String>>) -> Result<RandomForest> {
        let first = trees.first().ok_or_else(|| {
            Error::InvalidInput("from_trees: forest must have at least one tree".into())
        })?;
        let task = first.task();
        let n_features = first.n_features();
        for (i, t) in trees.iter().enumerate() {
            if t.task() != task {
                return Err(Error::InvalidInput(format!(
                    "from_trees: tree {i} task {:?} != tree 0 task {task:?}",
                    t.task()
                )));
            }
            if t.n_features() != n_features {
                return Err(Error::InvalidInput(format!(
                    "from_trees: tree {i} has {} features, tree 0 has {n_features}",
                    t.n_features()
                )));
            }
        }
        match (task, &classes) {
            (Task::Classification { n_classes }, Some(labels)) => {
                if labels.len() != n_classes {
                    return Err(Error::InvalidInput(format!(
                        "from_trees: {} class labels for an {n_classes}-class forest",
                        labels.len()
                    )));
                }
            }
            (Task::Classification { .. }, None) => {
                return Err(Error::InvalidInput(
                    "from_trees: classification forest requires class labels".into(),
                ));
            }
            (Task::Regression, Some(_)) => {
                return Err(Error::InvalidInput(
                    "from_trees: regression forest must not carry class labels".into(),
                ));
            }
            (Task::Regression, None) => {}
        }
        Ok(RandomForest {
            trees,
            task,
            n_features,
            classes,
        })
    }

    /// Assemble a forest from parts the fit path produced. Internal: the builder
    /// maintains the [`from_trees`](Self::from_trees) invariants, so this skips
    /// re-checking them (debug-asserted only).
    pub(crate) fn from_parts(
        trees: Vec<Tree>,
        task: Task,
        n_features: usize,
        classes: Option<Vec<String>>,
    ) -> RandomForest {
        debug_assert!(!trees.is_empty(), "from_parts: empty forest");
        debug_assert!(
            trees
                .iter()
                .all(|t| t.task() == task && t.n_features() == n_features),
            "from_parts: inconsistent trees"
        );
        RandomForest {
            trees,
            task,
            n_features,
            classes,
        }
    }

    /// The learning task.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of input features each tree expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Class labels in probability-column order (classification only; `None` for
    /// regression).
    pub fn classes(&self) -> Option<&[String]> {
        self.classes.as_deref()
    }

    /// Borrow tree `i` (for M5 persistence / M10 TreeSHAP). Panics if
    /// `i >= n_trees`.
    pub fn tree(&self, i: usize) -> &Tree {
        &self.trees[i]
    }

    /// Soft-vote class probabilities: the elementwise mean over trees of each
    /// tree's `predict_proba`, one `n_classes`-wide row per sample of `x`.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// forest is a regressor (API misuse, mirrors [`Tree::predict_proba`]).
    pub fn predict_proba(&self, x: &FeatureMatrix) -> Result<Vec<Vec<f64>>> {
        let n_classes = match self.task {
            Task::Classification { n_classes } => n_classes,
            Task::Regression => panic!("predict_proba called on a regression forest"),
        };
        self.check_features(x)?;
        let n_samples = x.n_samples();
        let mut acc = vec![vec![0.0f64; n_classes]; n_samples];
        for tree in &self.trees {
            // Each tree re-checks features and returns the same-shape rows.
            for (row, contrib) in acc.iter_mut().zip(tree.predict_proba(x)?) {
                for (a, c) in row.iter_mut().zip(contrib) {
                    *a += c;
                }
            }
        }
        let inv = 1.0 / self.trees.len() as f64;
        for row in acc.iter_mut() {
            for a in row.iter_mut() {
                *a *= inv;
            }
        }
        Ok(acc)
    }

    /// Predicted class **index** per sample: `argmax` of the soft-vote mean
    /// probability, taking the first (lowest-index) max on ties.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// forest is a regressor.
    pub fn predict_class_indices(&self, x: &FeatureMatrix) -> Result<Vec<usize>> {
        Ok(self
            .predict_proba(x)?
            .iter()
            .map(|row| argmax_first(row))
            .collect())
    }

    /// Regression prediction per sample: the mean over trees of each tree's leaf
    /// mean.
    ///
    /// Errors (`InvalidInput`) on a feature-count mismatch. **Panics** if the
    /// forest is a classifier.
    pub fn predict_values(&self, x: &FeatureMatrix) -> Result<Vec<f64>> {
        assert!(
            matches!(self.task, Task::Regression),
            "predict_values called on a classification forest"
        );
        self.check_features(x)?;
        let mut acc = vec![0.0f64; x.n_samples()];
        for tree in &self.trees {
            for (a, p) in acc.iter_mut().zip(tree.predict_values(x)?) {
                *a += p;
            }
        }
        let inv = 1.0 / self.trees.len() as f64;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        Ok(acc)
    }

    /// Mean-decrease-in-impurity feature importances: sum every tree's
    /// normalized MDI ([`Tree::feature_importances`]), then normalize the total
    /// to sum 1 (unless it is 0, e.g. an all-stump ensemble). Equals sklearn's
    /// forest `feature_importances_` to ~1e-9 (see the module note).
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.n_features];
        for tree in &self.trees {
            for (a, c) in acc.iter_mut().zip(tree.feature_importances()) {
                *a += c;
            }
        }
        let total: f64 = acc.iter().sum();
        if total > 0.0 {
            for a in acc.iter_mut() {
                *a /= total;
            }
        }
        acc
    }

    fn check_features(&self, x: &FeatureMatrix) -> Result<()> {
        if x.n_features() != self.n_features {
            return Err(Error::InvalidInput(format!(
                "forest expects {} features, got {}",
                self.n_features,
                x.n_features()
            )));
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // Two hand-built binary classifier trees, both splitting feature 0 at 0.5 but
    // with different leaf proportions, so the soft vote is a genuine average.
    //   tree A: left [1, 0], right [0, 1]
    //   tree B: left [0.5, 0.5], right [0.25, 0.75]
    fn cls_forest() -> RandomForest {
        let a = Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[1, -1, -1],
            &[2, -1, -1],
            &[0, -2, -2],
            &[0.5, -2.0, -2.0],
            &[0.5, 0.5, 1.0, 0.0, 0.0, 1.0],
            &[0.5, 0.0, 0.0],
            &[4, 2, 2],
            &[4.0, 2.0, 2.0],
            1,
        )
        .unwrap();
        let b = Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[1, -1, -1],
            &[2, -1, -1],
            &[0, -2, -2],
            &[0.5, -2.0, -2.0],
            &[0.375, 0.625, 0.5, 0.5, 0.25, 0.75],
            &[0.46875, 0.5, 0.375],
            &[8, 4, 4],
            &[8.0, 4.0, 4.0],
            1,
        )
        .unwrap();
        RandomForest::from_trees(vec![a, b], Some(vec!["x".into(), "y".into()])).unwrap()
    }

    #[test]
    fn soft_vote_and_argmax_tie() {
        let f = cls_forest();
        // Row 0.0 → both trees' left leaf: mean([1,0],[.5,.5]) = [.75,.25] → class 0.
        // Row 1.0 → both trees' right leaf: mean([0,1],[.25,.75]) = [.125,.875] → class 1.
        let x = FeatureMatrix::from_coo(2, 1, &[1], &[0], &[1.0]).unwrap();
        let proba = f.predict_proba(&x).unwrap();
        assert_eq!(proba, vec![vec![0.75, 0.25], vec![0.125, 0.875]]);
        assert_eq!(f.predict_class_indices(&x).unwrap(), vec![0, 1]);
    }

    #[test]
    fn argmax_takes_first_on_ties() {
        // A single-tree forest whose leaf ties both classes → class 0 wins.
        let t = Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[0.5, 0.5],
            &[0.5],
            &[2],
            &[2.0],
            1,
        )
        .unwrap();
        let f = RandomForest::from_trees(vec![t], Some(vec!["x".into(), "y".into()])).unwrap();
        let x = FeatureMatrix::from_coo(1, 1, &[0], &[0], &[1.0]).unwrap();
        assert_eq!(f.predict_class_indices(&x).unwrap(), vec![0]);
    }

    #[test]
    fn regression_forest_means_tree_values() {
        // Two stump regressors predicting 2.0 and 5.0 → forest predicts 3.5.
        let stump = |mean: f64| {
            Tree::from_sklearn(
                Task::Regression,
                &[-1],
                &[-1],
                &[-2],
                &[-2.0],
                &[mean],
                &[0.0],
                &[3],
                &[3.0],
                1,
            )
            .unwrap()
        };
        let f = RandomForest::from_trees(vec![stump(2.0), stump(5.0)], None).unwrap();
        let x = FeatureMatrix::from_coo(1, 1, &[0], &[0], &[1.0]).unwrap();
        assert_eq!(f.predict_values(&x).unwrap(), vec![3.5]);
    }

    #[test]
    fn feature_importances_sum_normalized() {
        // Two single-feature classifier trees each put all importance on feat 0.
        let f = cls_forest();
        let fi = f.feature_importances();
        assert_eq!(fi, vec![1.0]);
    }

    #[test]
    fn from_trees_rejects_inconsistent_forests() {
        let t2f = |n_features| {
            Tree::from_sklearn(
                Task::Classification { n_classes: 2 },
                &[-1],
                &[-1],
                &[-2],
                &[-2.0],
                &[0.5, 0.5],
                &[0.5],
                &[2],
                &[2.0],
                n_features,
            )
            .unwrap()
        };
        // Empty ensemble.
        assert!(RandomForest::from_trees(vec![], Some(vec!["a".into(), "b".into()])).is_err());
        // Mismatched feature counts.
        assert!(
            RandomForest::from_trees(vec![t2f(1), t2f(2)], Some(vec!["a".into(), "b".into()]))
                .is_err()
        );
        // Wrong class-label count.
        assert!(RandomForest::from_trees(vec![t2f(1)], Some(vec!["a".into()])).is_err());
        // Classifier without labels / regression with labels.
        assert!(RandomForest::from_trees(vec![t2f(1)], None).is_err());
        let reg = Tree::from_sklearn(
            Task::Regression,
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[1.0],
            &[0.0],
            &[1],
            &[1.0],
            1,
        )
        .unwrap();
        assert!(RandomForest::from_trees(vec![reg], Some(vec!["a".into()])).is_err());
    }

    #[test]
    fn mixed_task_trees_rejected() {
        let cls = Tree::from_sklearn(
            Task::Classification { n_classes: 2 },
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[0.5, 0.5],
            &[0.5],
            &[2],
            &[2.0],
            1,
        )
        .unwrap();
        let reg = Tree::from_sklearn(
            Task::Regression,
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[1.0],
            &[0.0],
            &[1],
            &[1.0],
            1,
        )
        .unwrap();
        assert!(RandomForest::from_trees(vec![cls, reg], None).is_err());
    }

    #[test]
    #[should_panic(expected = "regression forest")]
    fn predict_proba_panics_on_regressor() {
        let reg = Tree::from_sklearn(
            Task::Regression,
            &[-1],
            &[-1],
            &[-2],
            &[-2.0],
            &[1.0],
            &[0.0],
            &[1],
            &[1.0],
            1,
        )
        .unwrap();
        let f = RandomForest::from_trees(vec![reg], None).unwrap();
        let x = FeatureMatrix::from_coo(1, 1, &[0], &[0], &[1.0]).unwrap();
        let _ = f.predict_proba(&x);
    }

    #[test]
    fn feature_count_mismatch_errors() {
        let f = cls_forest(); // expects 1 feature
        let x = FeatureMatrix::from_coo(1, 2, &[0], &[0], &[1.0]).unwrap();
        assert!(f.predict_proba(&x).is_err());
    }
}
