#!/usr/bin/env python
"""Generate ground-truth TreeSHAP fixtures for sc-core's M10 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` conda env (scikit-learn 1.4.2
+ the ``shap`` library as the oracle) and writes one JSON file per case to
``oracle/fixtures/shap/``, consumed by the hermetic Rust tests in
``crates/sc-core/tests/oracle_shap.rs``.

TreeSHAP is a TIGHT-tier computation (design §8/§10): given a fixed forest,
path-dependent Shapley values are a deterministic function of the tree structure
+ per-node cover — no RNG, no background dataset. The oracle is
``shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")``
(Lundberg et al. 2018, Algorithm 2). q2 itself does not use SHAP, so the arbiter
is the ``shap`` library, not a q2 function.

For each case we fit a small seeded ``RandomForest*`` (``random_state=0``) on
seeded synthetic data — the same shape as the M4 forest TIGHT cases so the trees
are shallow and shap is fast — then dump every ``estimators_[t].tree_`` bundle +
a query table + shap's golden ``base_values``/``shap_values``.

**Version-robustness.** shap's multiclass ``shap_values`` output has drifted:
older shap returns a list of ``[n_samples, n_features]`` arrays (one per class);
shap >= ~0.45 returns a single ``[n_samples, n_features, n_classes]`` ndarray.
``normalize_cls_shap`` canonicalizes either to ``[sample][class][feature]``.
The resolved ``shap.__version__`` is stamped into every fixture's ``generator``
block.

Determinism: seeded ``numpy.random.default_rng``; ``random_state=0`` for each
fit; JSON written with ``sort_keys`` + a delete-then-write pass, so re-running
against the same env reproduces byte-identical fixtures.
"""

import json
from pathlib import Path

import numpy as np
import scipy
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shap"

GEN = {
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "qiime2_amplicon": "2024.10",
    "seed": SEED,
    "shap": shap.__version__,
}

TOL_TIGHT = {"abs": 1e-9, "rel": 1e-9}


def rng(nonce):
    return np.random.default_rng([SEED, nonce])


def write_case(case):
    path = FIX_DIR / f"{case['id']}.json"
    with open(path, "w") as f:
        json.dump(case, f, sort_keys=True, indent=2)
        f.write("\n")


def dense_to_coo(x):
    """Dense matrix → COO triplets, omitting implicit zeros."""
    n, m = x.shape
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(m):
            v = float(x[i, j])
            if v != 0.0:
                rows.append(int(i))
                cols.append(int(j))
                vals.append(v)
    return {
        "rows": rows,
        "cols": cols,
        "vals": vals,
        "n_samples": int(n),
        "n_features": int(m),
    }


def dump_cls_tree(t):
    return {
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "feature": t.feature.astype(int).tolist(),
        "threshold": t.threshold.tolist(),
        "value": [t.value[i][0].tolist() for i in range(t.node_count)],
        "impurity": t.impurity.tolist(),
        "n_node_samples": t.n_node_samples.astype(int).tolist(),
        "weighted_n_node_samples": t.weighted_n_node_samples.tolist(),
    }


def dump_reg_tree(t):
    return {
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "feature": t.feature.astype(int).tolist(),
        "threshold": t.threshold.tolist(),
        "value": [float(t.value[i][0][0]) for i in range(t.node_count)],
        "impurity": t.impurity.tolist(),
        "n_node_samples": t.n_node_samples.astype(int).tolist(),
        "weighted_n_node_samples": t.weighted_n_node_samples.tolist(),
    }


def normalize_cls_shap(sv, n_samples, n_features, n_classes):
    """Canonicalize shap's multiclass output to ``[sample][class][feature]``,
    robust to the list-of-arrays (old) vs 3D-ndarray (new) layouts."""
    if isinstance(sv, list):
        # list of [n_samples, n_features], one per class.
        arr = np.stack([np.asarray(a) for a in sv], axis=0)  # [class, sample, feat]
        arr = np.transpose(arr, (1, 0, 2))  # [sample, class, feat]
    else:
        arr = np.asarray(sv)
        if arr.shape == (n_samples, n_features, n_classes):
            arr = np.transpose(arr, (0, 2, 1))  # [sample, class, feat]
        elif arr.shape == (n_classes, n_samples, n_features):
            arr = np.transpose(arr, (1, 0, 2))
        elif arr.shape == (n_samples, n_classes, n_features):
            pass  # already canonical
        else:
            raise ValueError(f"unexpected shap_values shape {arr.shape}")
    assert arr.shape == (n_samples, n_classes, n_features), arr.shape
    return arr


def shap_classifier_case(cid, x, y, xq, criterion, n_estimators=6, tol=None, **fit_kwargs):
    clf = RandomForestClassifier(
        criterion=criterion, n_estimators=n_estimators, random_state=0, **fit_kwargs
    )
    clf.fit(x, y)
    assert list(clf.classes_) == list(range(len(clf.classes_))), "use 0..k labels"
    n_classes = len(clf.classes_)
    n_features = x.shape[1]
    ex = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    sv = normalize_cls_shap(ex.shap_values(xq), len(xq), n_features, n_classes)
    base = np.atleast_1d(np.asarray(ex.expected_value, dtype=float))
    assert base.shape == (n_classes,), base.shape
    # Sanity: additivity in probability space (shap RF output == predict_proba).
    recon = base[None, :] + sv.sum(axis=2)  # [sample, class]
    assert np.abs(recon - clf.predict_proba(xq)).max() < 1e-9, "shap not additive to proba"
    return {
        "id": cid,
        "kind": "forest_shap_classifier",
        "generator": GEN,
        "tolerance": tol or TOL_TIGHT,
        "trees": [dump_cls_tree(e.tree_) for e in clf.estimators_],
        "n_classes": int(n_classes),
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(n_features),
        "test_coo": dense_to_coo(xq),
        "expected": {
            "base_values": base.tolist(),
            "shap_values": sv.tolist(),
        },
    }


def shap_regressor_case(cid, x, y, xq, n_estimators=6, tol=None, **fit_kwargs):
    reg = RandomForestRegressor(
        criterion="squared_error", n_estimators=n_estimators, random_state=0, **fit_kwargs
    )
    reg.fit(x, y)
    n_features = x.shape[1]
    ex = shap.TreeExplainer(reg, feature_perturbation="tree_path_dependent")
    sv = np.asarray(ex.shap_values(xq), dtype=float)
    assert sv.shape == (len(xq), n_features), sv.shape
    base = float(np.atleast_1d(np.asarray(ex.expected_value, dtype=float))[0])
    recon = base + sv.sum(axis=1)
    assert np.abs(recon - reg.predict(xq)).max() < 1e-9, "shap not additive to prediction"
    return {
        "id": cid,
        "kind": "forest_shap_regressor",
        "generator": GEN,
        "tolerance": tol or TOL_TIGHT,
        "trees": [dump_reg_tree(e.tree_) for e in reg.estimators_],
        "n_features": int(n_features),
        "test_coo": dense_to_coo(xq),
        "expected": {
            "base_value": base,
            "shap_values": sv.tolist(),
        },
    }


def gen_shap_fixtures():
    # Small multiclass problem (mirrors the M4 forest TIGHT data): shallow trees,
    # simple values — TreeSHAP is exact and fast here.
    r = rng(1)
    n = 44
    x = r.uniform(-2.5, 2.5, (n, 3))
    lin = 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2]
    y = np.digitize(lin, bins=[-0.8, 0.8]).astype(int)  # {0, 1, 2}
    xq = r.uniform(-2.5, 2.5, (12, 3))
    write_case(shap_classifier_case("shap_gini_multiclass", x, y, xq, "gini", max_depth=3))
    # A deeper, wider ensemble stresses the EXTEND/UNWIND recursion (longer unique
    # paths, more repeated-feature splits) and the shap↔sc float agreement.
    write_case(
        shap_classifier_case(
            "shap_deep_multiclass", x, y, xq, "entropy", n_estimators=12, max_depth=6
        )
    )

    # Small regression problem (mirrors the M4 forest TIGHT regression data).
    r = rng(2)
    n = 50
    xr = r.uniform(-3.0, 3.0, (n, 3))
    yr = 1.5 * xr[:, 0] - 2.0 * xr[:, 1] + 0.5 * xr[:, 2] + r.normal(0, 0.3, n)
    xrq = r.uniform(-3.0, 3.0, (12, 3))
    write_case(shap_regressor_case("shap_regression", xr, yr, xrq, max_depth=3))


def main():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_shap_fixtures()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} shap fixtures to {FIX_DIR}")
    print(
        f"  sklearn={GEN['sklearn']} scipy={GEN['scipy']} numpy={GEN['numpy']} "
        f"shap={GEN['shap']} q2-amplicon={GEN['qiime2_amplicon']}"
    )


if __name__ == "__main__":
    main()
