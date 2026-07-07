#!/usr/bin/env python
"""Generate ground-truth metric fixtures for sc-core's M1 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` conda env and uses
q2-sample-classifier / scikit-learn / scipy as the oracle. For each synthetic,
seeded case it computes the exact metric values q2 would report and writes one
JSON file per case to ``oracle/fixtures/metrics/``. The committed fixtures are
then consumed by the hermetic Rust tests in ``crates/sc-core/tests``.

Determinism: every case draws from its own ``numpy.random.default_rng`` seeded
from ``(SEED, nonce)``, and JSON is written with ``sort_keys`` + fixed float
repr, so re-running against the same env reproduces byte-identical fixtures.

Oracle provenance for M1: the metric functions reproduced here
(``_linear_regress``, confusion/accuracy, ``_calculate_baseline_accuracy``, the
ROC helpers) are version-stable across the sklearn 1.4.2 line; see the M1 plan.
"""

import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

import q2_sample_classifier
from q2_sample_classifier.visuals import (
    _binarize_labels,
    _calculate_baseline_accuracy,
    _linear_regress,
    _roc_macro_average,
    _roc_micro_average,
    _roc_per_class,
)
import pandas as pd

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "metrics"

GEN = {
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "qiime2_amplicon": "2024.10",
    "seed": SEED,
}

# Tight tolerances: these functions are simple arithmetic reproductions, except
# the regression p-value, whose tail is computed by a different algorithm in
# statrs than in scipy — hence a looser relative tolerance there.
TOL_TIGHT = {"abs": 1e-9, "rel": 1e-9}
TOL_REGRESSION = {"abs": 1e-9, "rel": 1e-7}


def rng(nonce):
    """A fresh, independent generator per case, stable under case reordering."""
    return np.random.default_rng([SEED, nonce])


def write_case(case):
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    path = FIX_DIR / f"{case['id']}.json"
    with open(path, "w") as f:
        json.dump(case, f, sort_keys=True, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------
# Regression: scipy.stats.linregress(x=actual, y=pred) + sklearn MSE, via q2's
# _linear_regress (which takes pandas and returns the reported columns).
# --------------------------------------------------------------------------
def regression_case(cid, actual, pred, tol=TOL_REGRESSION):
    s_actual = pd.Series(np.asarray(actual, dtype=float), name="target")
    s_pred = pd.Series(np.asarray(pred, dtype=float))
    row = _linear_regress(s_actual, s_pred).iloc[0]
    return {
        "id": cid,
        "kind": "regression",
        "generator": GEN,
        "tolerance": tol,
        "actual": s_actual.tolist(),
        "pred": s_pred.tolist(),
        "expected": {
            "mse": float(row["Mean squared error"]),
            "r_value": float(row["r-value"]),
            "r_squared": float(row["r-squared"]),
            "p_value": float(row["P-value"]),
            "std_err": float(row["Std Error"]),
            "slope": float(row["Slope"]),
            "intercept": float(row["Intercept"]),
        },
    }


# --------------------------------------------------------------------------
# Classification: accuracy / confusion / baseline (+ ROC AUCs when proba given).
# Confusion + normalization reproduce visuals.py:137-145 exactly; baseline uses
# the pure q2 helper; ROC AUCs use q2's _binarize_labels / _roc_* helpers.
# --------------------------------------------------------------------------
def classification_case(cid, classes, y_true, y_pred, proba=None, tol=TOL_TIGHT):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    accuracy = float(accuracy_score(yt, yp))
    cm = confusion_matrix(yt, yp)  # labels = sorted unique union (no labels= arg)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    with np.errstate(all="ignore"):
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cmn = np.nan_to_num(cmn)

    _, _, baseline, ratio = _calculate_baseline_accuracy(pd.Series(yt), accuracy)

    expected = {
        "accuracy": accuracy,
        "baseline_accuracy": float(baseline),
        "accuracy_ratio": float(ratio),
        "labels": [int(x) for x in labels],
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cmn.tolist(),
    }

    case = {
        "id": cid,
        "kind": "classification",
        "generator": GEN,
        "tolerance": tol,
        "classes": list(classes),
        "y_true": [int(x) for x in yt.tolist()],
        "y_pred": [int(x) for x in yp.tolist()],
        "expected": expected,
    }

    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        k = proba.shape[1]
        col_classes = list(range(k))  # proba column order == class-index order
        binarized = _binarize_labels(pd.Series(yt), col_classes)
        fpr, tpr, roc_auc = _roc_per_class(binarized, proba, col_classes)
        fpr, tpr, roc_auc = _roc_micro_average(binarized, proba, fpr, tpr, roc_auc)
        fpr, tpr, roc_auc = _roc_macro_average(fpr, tpr, roc_auc, col_classes)
        expected["auc_per_class"] = [float(roc_auc[i]) for i in range(k)]
        expected["auc_micro"] = float(roc_auc["micro"])
        expected["auc_macro"] = float(roc_auc["macro"])
        case["proba"] = proba.tolist()

    return case


# --------------------------------------------------------------------------
# Direct sklearn.metrics.roc_curve dump — validates drop_intermediate exactly.
# The first threshold is inf in sklearn 1.4.2; stored as null (informational).
# --------------------------------------------------------------------------
def roc_curve_case(cid, y_true_binary, y_score, tol=TOL_TIGHT):
    yt = np.asarray(y_true_binary, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    fpr, tpr, thr = roc_curve(yt, ys)
    return {
        "id": cid,
        "kind": "roc_curve",
        "generator": GEN,
        "tolerance": tol,
        "y_true_binary": [int(x) for x in yt.tolist()],
        "y_score": ys.tolist(),
        "expected": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(t) if np.isfinite(t) else None for t in thr],
            "auc": float(auc(fpr, tpr)),
        },
    }


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gen_regression():
    # Clean strong-linear, a few sizes.
    for n, nonce in [(10, 1), (30, 2), (200, 3)]:
        r = rng(nonce)
        x = r.uniform(-5, 5, n)
        y = 2.3 * x - 0.7 + r.normal(0, 0.4, n)
        write_case(regression_case(f"reg_clean_n{n}", x, y))
    # Noisy / weak correlation.
    r = rng(4)
    x = r.uniform(0, 10, 50)
    y = 0.5 * x + r.normal(0, 5.0, 50)
    write_case(regression_case("reg_noisy_n50", x, y))
    # Negative slope.
    r = rng(5)
    x = r.normal(0, 3, 40)
    y = -1.8 * x + 4.0 + r.normal(0, 1.0, 40)
    write_case(regression_case("reg_negcorr_n40", x, y))


def gen_classification():
    # Balanced 3-class.
    r = rng(11)
    yt = np.repeat([0, 1, 2], 10)
    yp = yt.copy()
    flip = r.choice(len(yt), 8, replace=False)
    yp[flip] = (yp[flip] + r.integers(1, 3, 8)) % 3
    write_case(classification_case("cls_multiclass_balanced_k3", ["a", "b", "c"], yt, yp))

    # Imbalanced 4-class.
    r = rng(12)
    yt = np.concatenate([np.zeros(30), np.ones(12), np.full(6, 2), np.full(2, 3)]).astype(int)
    yp = yt.copy()
    flip = r.choice(len(yt), 15, replace=False)
    yp[flip] = r.integers(0, 4, 15)
    write_case(classification_case("cls_imbalanced_k4", ["w", "x", "y", "z"], yt, yp))

    # Binary.
    r = rng(13)
    yt = r.integers(0, 2, 40)
    yp = yt.copy()
    flip = r.choice(len(yt), 9, replace=False)
    yp[flip] = 1 - yp[flip]
    write_case(classification_case("cls_binary", ["neg", "pos"], yt, yp))

    # Perfect classifier.
    r = rng(14)
    yt = r.integers(0, 3, 24)
    write_case(classification_case("cls_perfect_k3", ["a", "b", "c"], yt, yt.copy()))

    # All wrong (every prediction differs from truth).
    r = rng(15)
    yt = r.integers(0, 3, 24)
    yp = (yt + 1 + r.integers(0, 2, 24)) % 3  # never equals yt
    assert np.all(yp != yt)
    write_case(classification_case("cls_all_wrong_k3", ["a", "b", "c"], yt, yp))

    # Zero-sum confusion row: class 2 is predicted but never true, so its true
    # row sums to zero and its normalized row is all zeros after nan_to_num.
    yt = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    yp = np.array([0, 2, 1, 2, 0, 1, 2, 1])
    assert 2 not in set(yt.tolist()) and 2 in set(yp.tolist())
    write_case(classification_case("cls_zero_row_k3", ["a", "b", "c"], yt, yp))


def gen_classification_proba():
    # 3-class continuous proba; labels sampled from the proba so AUC is informative.
    r = rng(21)
    n, k = 60, 3
    proba = softmax(r.normal(0, 1.5, (n, k)))
    yt = np.array([r.choice(k, p=proba[i]) for i in range(n)])
    yp = proba.argmax(axis=1)
    write_case(classification_case("cls_proba_k3", ["a", "b", "c"], yt, yp, proba=proba))

    # Binary proba.
    r = rng(22)
    n = 50
    p1 = r.beta(2, 2, n)
    proba = np.column_stack([1 - p1, p1])
    yt = (r.uniform(0, 1, n) < p1).astype(int)
    yp = proba.argmax(axis=1)
    write_case(classification_case("cls_proba_binary", ["neg", "pos"], yt, yp, proba=proba))

    # 3-class proba with tied scores (repeated proba rows) to exercise ties in
    # the ROC assembly (roc_curve threshold grouping + macro interpolation).
    r = rng(23)
    base = softmax(r.normal(0, 1.0, (10, 3)))
    proba = np.vstack([base, base, base])  # many exact score ties
    yt = np.array([r.integers(0, 3) for _ in range(proba.shape[0])])
    yp = proba.argmax(axis=1)
    write_case(classification_case("cls_proba_ties_k3", ["a", "b", "c"], yt, yp, proba=proba))


def gen_roc_curve():
    # Perfectly separable: all positives outscore all negatives.
    yt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ys = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    write_case(roc_curve_case("roc_separable", yt, ys))

    # Many tied scores (drop_intermediate + tie grouping stress).
    yt = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    ys = np.array([0.5, 0.5, 0.5, 0.7, 0.7, 0.3, 0.7, 0.3, 0.5, 0.5])
    write_case(roc_curve_case("roc_ties", yt, ys))

    # Typical random, two sizes.
    for n, nonce in [(50, 31), (200, 32)]:
        r = rng(nonce)
        yt = r.integers(0, 2, n)
        # Give positives a higher score on average, with overlap.
        ys = r.normal(yt * 0.8, 1.0)
        write_case(roc_curve_case(f"roc_random_n{n}", yt, ys))


def main():
    # Fresh output dir contents so removed cases don't linger.
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_regression()
    gen_classification()
    gen_classification_proba()
    gen_roc_curve()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} fixtures to {FIX_DIR}")
    print(f"  sklearn={GEN['sklearn']} scipy={GEN['scipy']} numpy={GEN['numpy']} "
          f"q2-amplicon={GEN['qiime2_amplicon']}")


if __name__ == "__main__":
    main()
