#!/usr/bin/env python
"""Generate ground-truth nested/OOF-CV fixtures for sc-core's M8 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` conda env (scikit-learn 1.4.2
as the oracle) and writes one JSON file per case to ``oracle/fixtures/ncv/``,
consumed by the hermetic Rust tests in ``crates/sc-core/tests/oracle_ncv.rs``.

Ground truth is q2's ``classify_samples_ncv`` / ``regress_samples_ncv``
(``utilities._fit_and_predict_cv``): partition into ``cv`` folds
(``StratifiedKFold`` for classification, ``KFold`` for regression, both
``shuffle=True, random_state``); per fold fit a ``RandomForest`` on the training
complement and predict the held-out rows; concatenate the out-of-fold (OOF)
predictions over all samples.

Tier is **AGGREGATE** (see the M8 plan, mirroring M4's ``forest_aggregate``): the
outer folds are **injected** — we dump a per-sample ``fold_ids`` array, which the
Rust side rebuilds into the identical partition via ``CvFolds::from_fold_ids`` —
so q2 and our forest see the same folds. Only the forest RNG differs, so we band
the OOF score. Over ``K_SEEDS`` forest seeds (folds held fixed) we run the OOF
loop and compute the **overall** concatenated score — ``accuracy_score(y, oof)``
(classification) or squared-Pearson r² + MSE via the ``linear_regress`` formula
(regression, exactly as M4's ``forest_aggregate`` and NOT sklearn's ``r2_score``)
— recording ``min/max/mean/std`` as a band, plus ``ref_importances`` (seed-mean
of the fold-averaged importances) and ``rank_corr_floor`` (min pairwise Spearman
among the seeds' fold-averaged importances).

The per-fold ``mean_score``/``std_score`` q2 prints are pinned by the Rust
determinism golden instead — the primary parity target here is the low-variance
overall-OOF score.

Determinism: seeded ``numpy.random.default_rng``; ``random_state`` fixed for the
fold construction; JSON via ``sort_keys`` + delete-then-write, so re-running
against the same env reproduces byte-identical fixtures.
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
from sklearn.model_selection import KFold, StratifiedKFold

import biom
import qiime2

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ncv"
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

# Loose tolerance carried on aggregate fixtures (they assert bands, not values).
TOL_LOOSE = {"abs": 1e-6, "rel": 1e-6}
K_SEEDS = 25


def rng(nonce):
    return np.random.default_rng([SEED, nonce])


def write_case(case):
    with open(FIX_DIR / f"{case['id']}.json", "w") as f:
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


def make_folds(n, target, n_splits, stratify):
    """Return (fold_ids, folds) for a fixed shuffled (stratified) K-fold. ``folds``
    is the list of (train_idx, test_idx) arrays; ``fold_ids[i]`` is the test fold
    of sample ``i`` (what the Rust side rebuilds via ``CvFolds::from_fold_ids``)."""
    if stratify:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(n), target))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(n)))
    fold_ids = np.empty(n, dtype=int)
    for f, (_, test_idx) in enumerate(splits):
        fold_ids[test_idx] = f
    return fold_ids.tolist(), splits


def oof_run(X, target, folds, n_estimators, is_cls, random_state):
    """Run one OOF pass (folds fixed) with a forest seeded ``random_state``.
    Returns the overall OOF prediction array (original order) and the
    fold-averaged feature importances."""
    n = X.shape[0]
    oof = np.empty(n, dtype=object) if is_cls else np.zeros(n, dtype=float)
    imps = []
    for train_idx, test_idx in folds:
        est = (RandomForestClassifier if is_cls else RandomForestRegressor)(
            n_estimators=n_estimators, random_state=random_state
        )
        est.fit(X[train_idx], target[train_idx])
        oof[test_idx] = est.predict(X[test_idx])
        imps.append(est.feature_importances_)
    return oof, np.mean(np.stack(imps), axis=0)


def aggregate_classification_case(
    cid, X, labels, n_splits, n_estimators, acc_margin_abs=0.2, rank_margin=0.2
):
    labels = np.asarray([str(v) for v in labels])
    fold_ids, folds = make_folds(len(labels), labels, n_splits, stratify=True)
    accs, imps = [], []
    for s in range(K_SEEDS):
        oof, imp = oof_run(X, labels, folds, n_estimators, True, s)
        accs.append(accuracy_score(labels, oof))
        imps.append(imp)
    ref = np.mean(np.stack(imps), axis=0)
    return {
        "id": cid,
        "kind": "ncv_aggregate",
        "generator": GEN,
        "tolerance": TOL_LOOSE,
        "task": "classification",
        "data_coo": dense_to_coo(X),
        "labels": labels.tolist(),
        "fold_ids": [int(v) for v in fold_ids],
        "n_splits": int(n_splits),
        "n_estimators": int(n_estimators),
        "band": band("accuracy", accs, acc_margin_abs, 0.0),
        "ref_importances": ref.tolist(),
        "rank_corr_floor": spearman_floor(imps),
        "rank_corr_margin": float(rank_margin),
    }


def aggregate_regression_case(
    cid, X, values, n_splits, n_estimators, rank_margin=0.2
):
    values = np.asarray(values, float)
    fold_ids, folds = make_folds(len(values), values, n_splits, stratify=False)
    r2s, mses, imps = [], [], []
    for s in range(K_SEEDS):
        oof, imp = oof_run(X, values, folds, n_estimators, False, s)
        r2, mse = r2_and_mse(values, oof)
        r2s.append(r2)
        mses.append(mse)
        imps.append(imp)
    ref = np.mean(np.stack(imps), axis=0)
    return {
        "id": cid,
        "kind": "ncv_aggregate",
        "generator": GEN,
        "tolerance": TOL_LOOSE,
        "task": "regression",
        "data_coo": dense_to_coo(X),
        "values": [float(v) for v in values.tolist()],
        "fold_ids": [int(v) for v in fold_ids],
        "n_splits": int(n_splits),
        "n_estimators": int(n_estimators),
        "band": band("r2", r2s, 0.05, 0.25),
        "mse_band": band("mse", mses, 0.0, 0.4),
        "ref_importances": ref.tolist(),
        "rank_corr_floor": spearman_floor(imps),
        "rank_corr_margin": float(rank_margin),
    }


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


def gen_synthetic():
    # Separable multiclass: three well-separated Gaussian blobs + noise features
    # (so max_features='sqrt' genuinely subsamples). 3 classes x 20 = 60 samples.
    r = rng(10)
    centers = np.array([[0, 0], [6, 6], [-6, 6]], float)
    per = 20
    parts_x, parts_y = [], []
    for c, ctr in enumerate(centers):
        parts_x.append(ctr + r.normal(0, 0.8, (per, 2)))
        parts_y.append(np.full(per, c))
    X = np.hstack([np.vstack(parts_x), r.normal(0, 1.0, (per * 3, 6))])
    y = np.concatenate(parts_y)
    write_case(
        aggregate_classification_case(
            "agg_separable_multiclass", X, y, n_splits=5, n_estimators=40
        )
    )

    # Linear + noise regression (regressor uses max_features=all). n=90, 6 feats.
    r = rng(12)
    n = 90
    X = r.uniform(-3, 3, (n, 6))
    beta = np.array([2.0, -1.5, 0.0, 1.0, 0.0, -0.5])
    yv = X @ beta + r.normal(0, 0.8, n)
    write_case(
        aggregate_regression_case(
            "agg_linear_regression", X, yv, n_splits=5, n_estimators=40
        )
    )


def gen_real():
    # chardonnay (classification): min class = 3 -> n_splits=3; full feature set.
    Xc, yc = load_aligned(
        DATA_DIR / "chardonnay.table.qza", DATA_DIR / "chardonnay.map.txt", "Region", False
    )
    assert Xc.shape == (21, 281), Xc.shape
    write_case(
        aggregate_classification_case(
            "agg_chardonnay", Xc, yc, n_splits=3, n_estimators=50, acc_margin_abs=0.25
        )
    )

    # ecam (regression): top-150 prevalent features; n_splits=5.
    Xe, ye = load_aligned(
        DATA_DIR / "ecam-table-maturity.qza",
        DATA_DIR / "ecam_map_maturity.txt",
        "month",
        True,
        prevalence_top=150,
    )
    assert Xe.shape == (126, 150), Xe.shape
    write_case(
        aggregate_regression_case("agg_ecam", Xe, ye, n_splits=5, n_estimators=25)
    )


def main():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_synthetic()
    gen_real()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} ncv fixtures to {FIX_DIR}")
    print(
        f"  sklearn={GEN['sklearn']} scipy={GEN['scipy']} numpy={GEN['numpy']} "
        f"q2-amplicon={GEN['qiime2_amplicon']}"
    )


if __name__ == "__main__":
    main()
