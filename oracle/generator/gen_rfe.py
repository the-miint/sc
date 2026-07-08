#!/usr/bin/env python
"""Generate ground-truth RFECV fixtures for sc-core's M6 oracle tests.

Runs inside the pinned ``qiime2-amplicon-2024.10`` env (scikit-learn 1.4.2).
Two tiers (see the M6 plan):

- **TIGHT bookkeeping** (``rfe_tight``): a mock estimator whose
  ``feature_importances_`` are the per-column means of X (so we can script
  distinct, position-independent importances by making column j constant == the
  importance) drives sklearn ``RFE``/``RFECV``. We dump the exact ``ranking_`` /
  ``support_`` (elimination down to 1 feature), the ``cv_results_`` score curve,
  the q2 sklearn-<1.5 x-axis reconstruction, and ``n_features_``. The Rust side's
  deterministic bookkeeping must reproduce all of these exactly.

- **AGGREGATE overlap** (``rfecv_aggregate``): q2's real RFECV config
  (``step``, ``cv``, ``min_features_to_select=1``, default scoring) run over K
  seeds on each dataset; we record the K selected feature-index sets, the min
  pairwise Jaccard among them (a self-consistency floor), and the chosen-count
  range. The Rust RFECV must overlap q2's within band. Datasets are
  prevalence-filtered so ``step=1`` stays tractable (documented adaptation).

Determinism: seeded ``numpy.random.default_rng``; JSON via ``sort_keys`` +
delete-then-write, so re-running reproduces byte-identical fixtures.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV

import biom
import qiime2

SEED = 20260707
FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "rfe"
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
TOL = {"abs": 1e-9, "rel": 1e-9}
K_SEEDS = 10


def rng(nonce):
    return np.random.default_rng([SEED, nonce])


def write_case(case):
    with open(FIX_DIR / f"{case['id']}.json", "w") as f:
        json.dump(case, f, sort_keys=True, indent=2)
        f.write("\n")


def dense_to_coo(x):
    n, m = x.shape
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(m):
            v = float(x[i, j])
            if v != 0.0:
                rows.append(int(i))
                cols.append(int(j))
                vals.append(v)
    return {"rows": rows, "cols": cols, "vals": vals,
            "n_samples": int(n), "n_features": int(m)}


# --------------------------------------------------------------------------- #
# TIGHT bookkeeping
# --------------------------------------------------------------------------- #
SCORE_BY_COUNT = {}


class Mock(ClassifierMixin, BaseEstimator):
    """feature_importances_ == per-column means; score scripted by feature count."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = X.mean(axis=0)
        return self

    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0])

    def score(self, X, y):
        return SCORE_BY_COUNT.get(self.n_features_in_, 0.0)


def resolved_step(step, n_features):
    if step < 1:
        return max(1, int(step * n_features))
    return int(step)


def q2_x_axis(n_features, step, n_grid):
    s = resolved_step(step, n_features)
    x = [n_features - (k * s) for k in range(n_grid - 1, -1, -1)]
    if x[0] < 1:
        x[0] = 1
    return x


def tight_case(cid, importances, step, scores_by_count):
    global SCORE_BY_COUNT
    importances = np.asarray(importances, float)
    n = len(importances)
    assert len(set(importances.tolist())) == n, "importances must be distinct"
    X = np.tile(importances, (12, 1))  # column j constant == importances[j]
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    rfe = RFE(Mock(), n_features_to_select=1, step=step).fit(X, y)

    SCORE_BY_COUNT = scores_by_count
    rfecv = RFECV(Mock(), step=step, cv=3, min_features_to_select=1).fit(X, y)
    mts = rfecv.cv_results_["mean_test_score"]
    x = q2_x_axis(n, step, len(mts))

    write_case({
        "id": cid,
        "kind": "rfe_tight",
        "generator": GEN,
        "tolerance": TOL,
        "importances": importances.tolist(),
        "step": float(step),
        "min_features_to_select": 1,
        "n_features": int(n),
        "expected": {
            "resolved_step": int(resolved_step(step, n)),
            "rfe_ranking": rfe.ranking_.astype(int).tolist(),
            "rfe_support": rfe.support_.astype(bool).tolist(),
            "grid_n_features": [int(v) for v in x],
            "grid_scores": [float(v) for v in mts],
            "n_optimal": int(rfecv.n_features_),
        },
    })


def gen_tight():
    # 6 distinct importances; scores peak at an interior count.
    imp6 = [0.50, 0.10, 0.90, 0.30, 0.70, 0.20]
    # step=1 -> grid counts ascending [1..6]; peak at 4.
    tight_case("rfe_step1_peak4", imp6, 1,
               {1: .55, 2: .60, 3: .70, 4: .92, 5: .80, 6: .74})
    # step=2 -> grid [1,2,4,6]; peak at 4.
    tight_case("rfe_step2_peak4", imp6, 2,
               {1: .40, 2: .60, 4: .90, 6: .55})
    # fractional step 0.34 -> resolved 2 -> grid [1,2,4,6]; peak at 2 (fewer wins).
    tight_case("rfe_frac034_peak2", imp6, 0.34,
               {1: .50, 2: .91, 4: .70, 6: .60})
    # larger: 10 distinct importances, step=1, peak at 3.
    imp10 = [0.55, 0.11, 0.99, 0.33, 0.77, 0.22, 0.44, 0.88, 0.66, 0.05]
    tight_case("rfe_step1_n10_peak3", imp10, 1,
               {c: (0.9 if c == 3 else 0.5 + 0.01 * c) for c in range(1, 11)})


# --------------------------------------------------------------------------- #
# AGGREGATE overlap vs q2
# --------------------------------------------------------------------------- #
def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def min_pairwise_jaccard(sets):
    vals = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return float(min(vals)) if vals else 1.0


def rfecv_selected(X, y, step, cv, n_estimators, is_classification, random_state):
    est = (RandomForestClassifier if is_classification else RandomForestRegressor)(
        n_estimators=n_estimators, random_state=random_state)
    sel = RFECV(est, step=step, cv=cv, min_features_to_select=1).fit(X, y)
    selected = sorted(int(i) for i in np.flatnonzero(sel.support_))
    return selected, int(sel.n_features_)


def aggregate_case(cid, task, X, target, step, cv, n_estimators,
                   jaccard_margin=0.1, n_optimal_margin=2):
    is_cls = task == "classification"
    sets, counts = [], []
    for s in range(K_SEEDS):
        sel, n_opt = rfecv_selected(X, target, step, cv, n_estimators, is_cls, s)
        sets.append(sel)
        counts.append(n_opt)
    case = {
        "id": cid,
        "kind": "rfecv_aggregate",
        "generator": GEN,
        "tolerance": TOL,
        "task": task,
        "data_coo": dense_to_coo(X),
        "step": float(step),
        "n_splits": int(cv),
        "n_estimators": int(n_estimators),
        "min_features_to_select": 1,
        "reference_sets": sets,
        "jaccard_floor": min_pairwise_jaccard(sets),
        "jaccard_margin": float(jaccard_margin),
        "n_optimal_min": int(min(counts)),
        "n_optimal_max": int(max(counts)),
        "n_optimal_margin": int(n_optimal_margin),
    }
    if is_cls:
        case["labels"] = [str(v) for v in np.asarray(target).tolist()]
    else:
        case["values"] = [float(v) for v in np.asarray(target, float).tolist()]
    write_case(case)


def load_aligned(table_fp, map_fp, col, numeric, prevalence_top):
    t = qiime2.Artifact.load(str(table_fp)).view(biom.Table)
    md = pd.read_csv(map_fp, sep="\t", index_col=0)
    md.index = md.index.astype(str)
    target = md[col]
    if numeric:
        target = pd.to_numeric(target, errors="coerce")
    target = target.dropna()
    keep = sorted(s for s in (str(x) for x in t.ids(axis="sample")) if s in set(target.index))
    feat_ids = sorted(str(f) for f in t.ids(axis="observation"))
    df = t.to_dataframe(dense=True)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    X = df.loc[feat_ids, keep].T.values.astype(float)
    y = target.loc[keep].values
    prev = (X > 0).sum(axis=0)
    sel = np.sort(np.argsort(-prev, kind="stable")[:prevalence_top])
    return X[:, sel], y


def gen_aggregate():
    # Synthetic classification: informative features 0,1 + noise; step=1, cv=3.
    r = rng(30)
    n = 30
    Xc = r.normal(0, 1, (n, 12))
    y = (Xc[:, 0] + Xc[:, 1] > 0).astype(int)
    Xc[:, 0] += 4 * y  # make 0,1 clearly informative
    Xc[:, 1] += 4 * y
    Xc = np.abs(Xc)  # counts-like, nonneg
    aggregate_case("agg_synth_classification", "classification", Xc,
                   np.array([str(v) for v in y]), step=1, cv=3, n_estimators=25)

    # Synthetic regression: linear in a few features + noise; step=1, cv=5.
    r = rng(31)
    n = 60
    Xr = np.abs(r.normal(0, 1, (n, 12)))
    beta = np.array([3.0, -2.0, 0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0])
    yr = Xr @ beta + r.normal(0, 0.5, n)
    aggregate_case("agg_synth_regression", "regression", Xr, yr,
                   step=1, cv=5, n_estimators=25)

    # chardonnay (classification): min class = 3 -> cv=3; top-40 prevalent features.
    Xch, ych = load_aligned(
        DATA_DIR / "chardonnay.table.qza", DATA_DIR / "chardonnay.map.txt",
        "Region", False, prevalence_top=40)
    aggregate_case("agg_chardonnay", "classification", Xch, ych,
                   step=1, cv=3, n_estimators=25)

    # ecam (regression): top-40 prevalent features; cv=5.
    Xe, ye = load_aligned(
        DATA_DIR / "ecam-table-maturity.qza", DATA_DIR / "ecam_map_maturity.txt",
        "month", True, prevalence_top=40)
    aggregate_case("agg_ecam", "regression", Xe, ye,
                   step=1, cv=5, n_estimators=25)


def main():
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIX_DIR.glob("*.json"):
        old.unlink()
    gen_tight()
    gen_aggregate()
    n = len(list(FIX_DIR.glob("*.json")))
    print(f"wrote {n} rfe fixtures to {FIX_DIR}")
    print(f"  sklearn={GEN['sklearn']} numpy={GEN['numpy']} q2-amplicon={GEN['qiime2_amplicon']}")


if __name__ == "__main__":
    main()
