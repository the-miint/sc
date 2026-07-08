#!/usr/bin/env python
"""Generate ground-truth RandomForest fixtures for sc-core's M4 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` conda env (scikit-learn 1.4.2
as the oracle) and writes one JSON file per case to ``oracle/fixtures/forest/``,
consumed by the hermetic Rust tests in ``crates/sc-core/tests/oracle_forest.rs``.

Two tiers (see the M4 plan):

- **TIGHT import** (``forest_classifier`` / ``forest_regressor``): fit a small
  real ``RandomForest*`` (``random_state=0``) on seeded synthetic data, dump every
  ``estimators_[t].tree_`` array bundle + ``classes_`` + a query table + the
  **forest-level** ``predict`` / ``predict_proba`` / ``feature_importances_``.
  Tol abs=rel=1e-9. These are the empirical arbiters for the aggregation math:
  soft-vote proba (mean over trees), mean-of-tree regression, and
  summed-then-normalized MDI (== sklearn's mean-of-normalized within 1e-9).

- **AGGREGATE** (``forest_aggregate``): for a whole dataset with a fixed
  **injected** train/test split, fit sklearn RF over K seeds on the train split
  and score on the test split (accuracy | r²+mse computed exactly as sc-core's
  ``metrics`` do), recording ``min/max/mean/std`` as a band plus ``ref_importances``
  (seed mean) and ``rank_corr_floor`` (min pairwise Spearman among the seeds).
  Our forest is deterministic under a fixed ``master_seed``; the band's margins
  are widened so our result sits inside. Robustness datasets (all-zero column,
  constant feature, tiny-n) are emitted with ``metric="invariant"`` — the Rust
  test then checks shape invariants only, no score band.

Determinism: seeded ``numpy.random.default_rng``; ``random_state`` fixed for both
the injected split and each fit; JSON written with ``sort_keys`` + a delete-then-
write pass, so re-running against the same env reproduces byte-identical fixtures.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import biom
import qiime2

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "forest"
DATA_DIR = Path(
    "/home/dtmcdonald/dev/q2-sample-classifier/q2_sample_classifier/tests/data"
)

GEN = {
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "qiime2_amplicon": "2024.10",
    "seed": SEED,
}

TOL_TIGHT = {"abs": 1e-9, "rel": 1e-9}
# Loose tolerance carried on aggregate fixtures (they assert bands, not values).
TOL_LOOSE = {"abs": 1e-6, "rel": 1e-6}
K_SEEDS = 25


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


# --------------------------------------------------------------------------- #
# TIGHT import fixtures
# --------------------------------------------------------------------------- #
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


def forest_classifier_case(cid, x, y, xq, criterion, n_estimators=6, **fit_kwargs):
    clf = RandomForestClassifier(
        criterion=criterion, n_estimators=n_estimators, random_state=0, **fit_kwargs
    )
    clf.fit(x, y)
    assert list(clf.classes_) == list(range(len(clf.classes_))), "use 0..k labels"
    return {
        "id": cid,
        "kind": "forest_classifier",
        "generator": GEN,
        "tolerance": TOL_TIGHT,
        "trees": [dump_cls_tree(e.tree_) for e in clf.estimators_],
        "n_classes": int(len(clf.classes_)),
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x.shape[1]),
        "criterion": criterion,
        "test_coo": dense_to_coo(xq),
        "expected": {
            "predict": [int(v) for v in clf.predict(xq).tolist()],
            "proba": clf.predict_proba(xq).tolist(),
            "mdi": clf.feature_importances_.tolist(),
        },
    }


def forest_regressor_case(cid, x, y, xq, n_estimators=6, **fit_kwargs):
    reg = RandomForestRegressor(
        criterion="squared_error", n_estimators=n_estimators, random_state=0, **fit_kwargs
    )
    reg.fit(x, y)
    return {
        "id": cid,
        "kind": "forest_regressor",
        "generator": GEN,
        "tolerance": TOL_TIGHT,
        "trees": [dump_reg_tree(e.tree_) for e in reg.estimators_],
        "n_features": int(x.shape[1]),
        "test_coo": dense_to_coo(xq),
        "expected": {
            "predict": reg.predict(xq).tolist(),
            "mdi": reg.feature_importances_.tolist(),
        },
    }


def gen_tight_import():
    # Small multiclass problem; both criteria arbitrate the soft-vote + MDI math.
    r = rng(1)
    n = 44
    x = r.uniform(-2.5, 2.5, (n, 3))
    lin = 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2]
    y = np.digitize(lin, bins=[-0.8, 0.8]).astype(int)  # {0, 1, 2}
    xq = r.uniform(-2.5, 2.5, (12, 3))
    write_case(forest_classifier_case("forest_gini_multiclass", x, y, xq, "gini", max_depth=3))
    write_case(
        forest_classifier_case("forest_entropy_multiclass", x, y, xq, "entropy", max_depth=3)
    )

    # Small regression problem for the mean-of-trees + MDI arbiter.
    r = rng(2)
    n = 50
    xr = r.uniform(-3.0, 3.0, (n, 3))
    yr = 1.5 * xr[:, 0] - 2.0 * xr[:, 1] + 0.5 * xr[:, 2] + r.normal(0, 0.3, n)
    xrq = r.uniform(-3.0, 3.0, (12, 3))
    write_case(forest_regressor_case("forest_regression", xr, yr, xrq, max_depth=3))


# --------------------------------------------------------------------------- #
# AGGREGATE fixtures
# --------------------------------------------------------------------------- #
def spearman(a, b):
    """Spearman rank correlation as Pearson over average-tie ranks — identical
    to the ranking sc-core's Rust aggregate test uses."""
    ra = rankdata(a).astype(float)
    rb = rankdata(b).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def spearman_floor(importances):
    vals = []
    K = len(importances)
    for i in range(K):
        for j in range(i + 1, K):
            vals.append(spearman(importances[i], importances[j]))
    return float(min(vals)) if vals else 1.0


def r2_and_mse(actual, pred):
    """r² (squared Pearson correlation) and MSE, exactly as sc-core's
    ``metrics::linear_regress`` computes them (x=actual, y=pred)."""
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    am, pm = actual.mean(), pred.mean()
    da, dp = actual - am, pred - pm
    ssaa = (da * da).sum()
    sspp = (dp * dp).sum()
    if ssaa == 0.0 or sspp == 0.0:
        r = 0.0
    else:
        r = np.clip((da * dp).sum() / np.sqrt(ssaa * sspp), -1.0, 1.0)
    mse = float(((actual - pred) ** 2).mean())
    return float(r * r), mse


def band(metric, arr, margin_abs, margin_rel):
    arr = np.asarray(arr, float)
    return {
        "metric": metric,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "margin_abs": float(margin_abs),
        "margin_rel": float(margin_rel),
    }


def injected_split(n, stratify, test_frac):
    idx = np.arange(n)
    train, test = train_test_split(
        idx,
        test_size=test_frac,
        random_state=SEED,
        shuffle=True,
        stratify=stratify,
    )
    return sorted(int(i) for i in train), sorted(int(i) for i in test)


def aggregate_classification_case(
    cid, X, labels, n_estimators, test_frac=0.3, acc_margin_abs=0.2, rank_margin=0.2
):
    labels = np.asarray(labels)
    train, test = injected_split(len(labels), stratify=labels, test_frac=test_frac)
    Xtr, Xte = X[train], X[test]
    ytr, yte = labels[train], labels[test]
    accs, imps = [], []
    for s in range(K_SEEDS):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=s)
        clf.fit(Xtr, ytr)
        accs.append(accuracy_score(yte, clf.predict(Xte)))
        imps.append(clf.feature_importances_)
    ref = np.mean(np.stack(imps), axis=0)
    return {
        "id": cid,
        "kind": "forest_aggregate",
        "generator": GEN,
        "tolerance": TOL_LOOSE,
        "task": "classification",
        "data_coo": dense_to_coo(X),
        "labels": [str(v) for v in labels.tolist()],
        "train_idx": train,
        "test_idx": test,
        "n_estimators": int(n_estimators),
        "band": band("accuracy", accs, acc_margin_abs, 0.0),
        "ref_importances": ref.tolist(),
        "rank_corr_floor": spearman_floor(imps),
        "rank_corr_margin": float(rank_margin),
    }


def aggregate_regression_case(
    cid, X, values, n_estimators, test_frac=0.3, rank_margin=0.2
):
    values = np.asarray(values, float)
    train, test = injected_split(len(values), stratify=None, test_frac=test_frac)
    Xtr, Xte = X[train], X[test]
    ytr, yte = values[train], values[test]
    r2s, mses, imps = [], [], []
    for s in range(K_SEEDS):
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=s)
        reg.fit(Xtr, ytr)
        r2, mse = r2_and_mse(yte, reg.predict(Xte))
        r2s.append(r2)
        mses.append(mse)
        imps.append(reg.feature_importances_)
    ref = np.mean(np.stack(imps), axis=0)
    return {
        "id": cid,
        "kind": "forest_aggregate",
        "generator": GEN,
        "tolerance": TOL_LOOSE,
        "task": "regression",
        "data_coo": dense_to_coo(X),
        "values": [float(v) for v in values.tolist()],
        "train_idx": train,
        "test_idx": test,
        "n_estimators": int(n_estimators),
        "band": band("r2", r2s, 0.05, 0.25),
        "mse_band": band("mse", mses, 0.0, 0.4),
        "ref_importances": ref.tolist(),
        "rank_corr_floor": spearman_floor(imps),
        "rank_corr_margin": float(rank_margin),
    }


def invariant_case(cid, task, X, target, n_estimators, test_frac=0.3):
    """Robustness case: emit the split + a metric='invariant' band. The Rust test
    checks shape invariants only (valid proba / in-range predictions / finite
    importances), no score band."""
    n = X.shape[0]
    if task == "classification":
        target = np.asarray(target)
        train, test = injected_split(n, stratify=target, test_frac=test_frac)
        payload_target = {"labels": [str(v) for v in target.tolist()]}
    else:
        target = np.asarray(target, float)
        train, test = injected_split(n, stratify=None, test_frac=test_frac)
        payload_target = {"values": [float(v) for v in target.tolist()]}
    case = {
        "id": cid,
        "kind": "forest_aggregate",
        "generator": GEN,
        "tolerance": TOL_LOOSE,
        "task": task,
        "data_coo": dense_to_coo(X),
        "train_idx": train,
        "test_idx": test,
        "n_estimators": int(n_estimators),
        "band": band("invariant", [0.0], 0.0, 0.0),
        "ref_importances": [0.0] * int(X.shape[1]),
        "rank_corr_floor": 0.0,
        "rank_corr_margin": 0.0,
    }
    case.update(payload_target)
    return case


# ---- real q2 datasets ----------------------------------------------------- #
def load_aligned(table_fp, map_fp, col, numeric, prevalence_top=None):
    t = qiime2.Artifact.load(str(table_fp)).view(biom.Table)
    md = pd.read_csv(map_fp, sep="\t", index_col=0)
    md.index = md.index.astype(str)
    target = md[col]
    if numeric:
        target = pd.to_numeric(target, errors="coerce")
    target = target.dropna()
    keep = sorted(s for s in (str(x) for x in t.ids(axis="sample")) if s in set(target.index))
    feat_ids = sorted(str(f) for f in t.ids(axis="observation"))
    dfmat = t.to_dataframe(dense=True)  # features x samples
    dfmat.index = dfmat.index.astype(str)
    dfmat.columns = dfmat.columns.astype(str)
    X = dfmat.loc[feat_ids, keep].T.values.astype(float)  # samples x features
    y = target.loc[keep].values
    if prevalence_top is not None and X.shape[1] > prevalence_top:
        prev = (X > 0).sum(axis=0)
        sel = np.sort(np.argsort(-prev, kind="stable")[:prevalence_top])
        X = X[:, sel]
    return X, y


def gen_aggregate_synthetic():
    # Separable multiclass: three well-separated Gaussian blobs.
    r = rng(10)
    centers = np.array([[0, 0], [6, 6], [-6, 6]], float)
    per = 20
    parts_x, parts_y = [], []
    for c, ctr in enumerate(centers):
        parts_x.append(ctr + r.normal(0, 0.8, (per, 2)))
        parts_y.append(np.full(per, c))
    # Pad with noise features so max_features='sqrt' genuinely subsamples.
    X = np.hstack([np.vstack(parts_x), r.normal(0, 1.0, (per * 3, 6))])
    y = np.concatenate(parts_y)
    write_case(
        aggregate_classification_case("agg_separable_multiclass", X, y, n_estimators=40)
    )

    # Imbalanced multiclass: classes of size 24 / 24 / 10.
    r = rng(11)
    sizes = [24, 24, 10]
    seps = [0.0, 5.0, 10.0]
    parts_x, parts_y = [], []
    for c, (sz, sep) in enumerate(zip(sizes, seps)):
        base = np.zeros(4)
        base[0] = sep
        parts_x.append(base + r.normal(0, 1.2, (sz, 4)))
        parts_y.append(np.full(sz, c))
    X = np.vstack(parts_x)
    y = np.concatenate(parts_y)
    write_case(
        aggregate_classification_case("agg_imbalanced_multiclass", X, y, n_estimators=40)
    )

    # Linear + noise regression (regressor uses max_features=all).
    r = rng(12)
    n = 90
    X = r.uniform(-3, 3, (n, 6))
    beta = np.array([2.0, -1.5, 0.0, 1.0, 0.0, -0.5])
    yv = X @ beta + r.normal(0, 0.8, n)
    write_case(aggregate_regression_case("agg_linear_regression", X, yv, n_estimators=40))


def gen_aggregate_real():
    Xc, yc = load_aligned(
        DATA_DIR / "chardonnay.table.qza", DATA_DIR / "chardonnay.map.txt", "Region", False
    )
    assert Xc.shape == (21, 281), Xc.shape
    write_case(
        aggregate_classification_case(
            "agg_chardonnay", Xc, yc, n_estimators=50, test_frac=0.3, acc_margin_abs=0.25
        )
    )

    Xe, ye = load_aligned(
        DATA_DIR / "ecam-table-maturity.qza",
        DATA_DIR / "ecam_map_maturity.txt",
        "month",
        True,
        prevalence_top=150,
    )
    assert Xe.shape == (126, 150), Xe.shape
    write_case(
        aggregate_regression_case("agg_ecam", Xe, ye, n_estimators=25, test_frac=0.25)
    )


def gen_robustness():
    # All-zero feature column: feature 3 is never observed.
    r = rng(20)
    n = 32
    X = np.abs(r.normal(0, 1, (n, 4)))
    X[:, 3] = 0.0
    y = (X[:, 0] > X[:, 1]).astype(int)
    write_case(invariant_case("robust_all_zero_column", "classification", X, y, 10))

    # Constant nonzero feature: feature 2 is 5.0 for every sample.
    r = rng(21)
    X = np.abs(r.normal(0, 1, (n, 4)))
    X[:, 2] = 5.0
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    write_case(invariant_case("robust_constant_feature", "classification", X, y, 10))

    # Tiny-n: only 8 samples, 2 classes of 4.
    r = rng(22)
    X = np.vstack([r.normal(0, 0.5, (4, 3)), 4.0 + r.normal(0, 0.5, (4, 3))])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    write_case(invariant_case("robust_tiny_n", "classification", X, y, 10, test_frac=0.25))


def main():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_tight_import()
    gen_aggregate_synthetic()
    gen_aggregate_real()
    gen_robustness()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} forest fixtures to {FIX_DIR}")
    print(
        f"  sklearn={GEN['sklearn']} scipy={GEN['scipy']} numpy={GEN['numpy']} "
        f"q2-amplicon={GEN['qiime2_amplicon']}"
    )


if __name__ == "__main__":
    main()
