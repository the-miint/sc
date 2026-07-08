#!/usr/bin/env python
"""Generate ground-truth decision-tree fixtures for sc-core's M3 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` conda env and uses
scikit-learn 1.4.2 as the oracle. For each synthetic, seeded case it fits a real
``DecisionTreeClassifier``/``DecisionTreeRegressor``, dumps the tree's exported
``tree_`` internals plus a query table, and records sklearn's ``predict`` /
``predict_proba`` / ``feature_importances_`` on that table. One JSON file per case
is written to ``oracle/fixtures/tree/`` and consumed by the hermetic Rust tests
in ``crates/sc-core/tests/oracle_tree.rs``.

These are the TIGHT-tier fixtures (~1e-9 vs sklearn): the Rust ``Tree::from_sklearn``
import path is validated to reproduce sklearn's evaluation exactly. The grow path
is validated separately by pure-Rust property tests (no fixtures).

Determinism: every case draws from its own ``numpy.random.default_rng`` seeded
from ``(SEED, nonce)``; trees are fit with ``random_state=0``; JSON is written
with ``sort_keys`` + fixed float repr, so re-running against the same env
reproduces byte-identical fixtures.

Verified sklearn 1.4.2 facts these fixtures encode (see the M3 plan):
- ``tree_.value`` is normalized class **proportions** (predict_proba(leaf) == value);
- entropy uses **log base 2**;
- thresholds are midpoints of the **float32**-cast feature values;
- MDI = per-node ``wn·imp − wn_L·imp_L − wn_R·imp_R`` / wn[root], normalized.
"""

import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tree"

GEN = {
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "qiime2_amplicon": "2024.10",
    "seed": SEED,
}

# The import path is exact arithmetic reproduction of sklearn's evaluation.
TOL_TIGHT = {"abs": 1e-9, "rel": 1e-9}


def rng(nonce):
    """A fresh, independent generator per case, stable under case reordering."""
    return np.random.default_rng([SEED, nonce])


def write_case(case):
    path = FIX_DIR / f"{case['id']}.json"
    with open(path, "w") as f:
        json.dump(case, f, sort_keys=True, indent=2)
        f.write("\n")


def dense_to_coo(xq, explicit_zero_cells=()):
    """Convert a dense query matrix to COO triplets. Nonzero cells are stored;
    zero cells are implicit unless listed in ``explicit_zero_cells`` (which are
    emitted as stored 0.0 to exercise the explicit-zero path)."""
    n, m = xq.shape
    ez = set(explicit_zero_cells)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(m):
            v = float(xq[i, j])
            if v != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(v)
            elif (i, j) in ez:
                rows.append(i)
                cols.append(j)
                vals.append(0.0)
    return {
        "rows": rows,
        "cols": cols,
        "vals": vals,
        "n_samples": int(n),
        "n_features": int(m),
    }


def classifier_case(cid, x, y, xq, criterion, coo, **fit_kwargs):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=0, **fit_kwargs)
    clf.fit(x, y)
    t = clf.tree_
    # classes_ are the integer labels 0..k-1, so predict already returns indices.
    assert list(clf.classes_) == list(range(len(clf.classes_))), "use 0..k labels"
    value = [t.value[i][0].tolist() for i in range(t.node_count)]
    return {
        "id": cid,
        "kind": "tree_classifier",
        "generator": GEN,
        "tolerance": TOL_TIGHT,
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "feature": t.feature.astype(int).tolist(),
        "threshold": t.threshold.tolist(),
        "value": value,
        "impurity": t.impurity.tolist(),
        "n_node_samples": t.n_node_samples.astype(int).tolist(),
        "weighted_n_node_samples": t.weighted_n_node_samples.tolist(),
        "n_classes": int(len(clf.classes_)),
        "criterion": criterion,
        "test_coo": coo,
        "expected": {
            "predict": [int(v) for v in clf.predict(xq).tolist()],
            "proba": clf.predict_proba(xq).tolist(),
            "mdi": clf.feature_importances_.tolist(),
        },
    }


def regressor_case(cid, x, y, xq, coo, **fit_kwargs):
    reg = DecisionTreeRegressor(criterion="squared_error", random_state=0, **fit_kwargs)
    reg.fit(x, y)
    t = reg.tree_
    value = [float(t.value[i][0][0]) for i in range(t.node_count)]
    return {
        "id": cid,
        "kind": "tree_regressor",
        "generator": GEN,
        "tolerance": TOL_TIGHT,
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "feature": t.feature.astype(int).tolist(),
        "threshold": t.threshold.tolist(),
        "value": value,
        "impurity": t.impurity.tolist(),
        "n_node_samples": t.n_node_samples.astype(int).tolist(),
        "weighted_n_node_samples": t.weighted_n_node_samples.tolist(),
        "criterion": "squared_error",
        "test_coo": coo,
        "expected": {
            "predict": reg.predict(xq).tolist(),
            "mdi": reg.feature_importances_.tolist(),
        },
    }


def gen_multiclass():
    # 3 features, 3 classes from a linear rule → a tree that uses several features.
    r = rng(1)
    n = 48
    x = r.uniform(-2.5, 2.5, (n, 3))
    lin = 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2]
    y = np.digitize(lin, bins=[-0.8, 0.8]).astype(int)  # → {0, 1, 2}
    xq = r.uniform(-2.5, 2.5, (12, 3))
    coo = dense_to_coo(xq)
    # Same data, both criteria — the entropy case is the arbiter for log base 2.
    write_case(classifier_case("tree_gini_multiclass", x, y, xq, "gini", coo, max_depth=3))
    write_case(
        classifier_case("tree_entropy_multiclass", x, y, xq, "entropy", coo, max_depth=3)
    )


def gen_binary():
    r = rng(4)
    n = 36
    x = r.uniform(-2.0, 2.0, (n, 2))
    y = (0.8 * x[:, 0] - 0.6 * x[:, 1] > 0).astype(int)
    xq = r.uniform(-2.0, 2.0, (10, 2))
    write_case(
        classifier_case("tree_binary", x, y, xq, "gini", dense_to_coo(xq), max_depth=3)
    )


def gen_regression():
    r = rng(3)
    n = 50
    x = r.uniform(-3.0, 3.0, (n, 3))
    y = 1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2] + r.normal(0, 0.3, n)
    xq = r.uniform(-3.0, 3.0, (12, 3))
    write_case(regressor_case("tree_regression", x, y, xq, dense_to_coo(xq), max_depth=3))


def gen_argmax_ties():
    # A leaf with tied class counts → predict must take the lowest class index.
    # Rows 0,1 share feature 0.0 with labels {0,1}; they land in one leaf whose
    # value is [0.5, 0.5, 0.0] → argmax is class 0.
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 2, 2])
    xq = np.array([[0.0], [0.4], [1.0], [3.0]])
    write_case(classifier_case("tree_argmax_ties", x, y, xq, "gini", dense_to_coo(xq)))


def gen_sparse_signed_zeros():
    # Fit on signed data; the QUERY table mixes negatives, implicit zeros, and
    # explicit stored zeros so the Rust RowView gather is exercised end to end.
    r = rng(6)
    n = 40
    x = r.uniform(-3.0, 3.0, (n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    clf_query = np.array(
        [
            [-2.0, 0.0],  # (0,1) explicit zero
            [0.0, -1.5],  # (1,0) implicit zero
            [0.0, 0.0],  # all-zero row: (2,0) explicit, (2,1) implicit
            [2.5, -2.5],
            [-0.3, 1.2],
            [1.7, 0.0],  # (5,1) implicit zero
        ]
    )
    coo = dense_to_coo(clf_query, explicit_zero_cells={(0, 1), (2, 0)})
    write_case(
        classifier_case(
            "tree_sparse_signed_zeros", x, y, clf_query, "gini", coo, max_depth=4
        )
    )


def gen_f32_boundary():
    # Threshold 0.5 is the midpoint of float32 0.0 and 1.0. A query value just
    # above 0.5 in f64 but rounding to 0.5 in f32 routes LEFT under sklearn's
    # float32 splitter — proving the f32 cast in traversal is load-bearing.
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(criterion="gini", random_state=0).fit(x, y)
    thr = float(clf.tree_.threshold[0])
    assert thr == 0.5, f"expected threshold 0.5, got {thr}"

    boundaries = []
    for k in range(25, 45):
        cand = thr + 2.0 ** (-k)
        # Diverges exactly when the raw-f64 and float32-cast comparisons disagree.
        if (cand <= thr) != (float(np.float32(cand)) <= thr):
            boundaries.append(cand)
        if len(boundaries) == 2:
            break
    assert len(boundaries) == 2, "could not construct f32-boundary query points"

    xq = np.array([[0.2], [0.8], [boundaries[0]], [boundaries[1]]])
    # Sanity: sklearn (float32 splitter) routes the boundary points LEFT (class 0).
    assert clf.predict(xq).tolist() == [0, 1, 0, 0]
    write_case(classifier_case("tree_f32_boundary", x, y, xq, "gini", dense_to_coo(xq)))


def main():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_multiclass()
    gen_binary()
    gen_regression()
    gen_argmax_ties()
    gen_sparse_signed_zeros()
    gen_f32_boundary()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} tree fixtures to {FIX_DIR}")
    print(
        f"  sklearn={GEN['sklearn']} scipy={GEN['scipy']} numpy={GEN['numpy']} "
        f"q2-amplicon={GEN['qiime2_amplicon']}"
    )


if __name__ == "__main__":
    main()
