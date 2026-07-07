//! Oracle tests for the M1 `metrics` module.
//!
//! Loads the committed JSON fixtures (generated against `qiime2-amplicon-2024.10`
//! by `oracle/generator/gen_metrics.py`) and asserts each pure-Rust metric
//! matches q2/scikit-learn/scipy to the per-case tolerance. Hermetic: no Python,
//! no network — just the checked-in fixtures.

use sc_core::metrics::{
    accuracy, accuracy_ratio, auc, baseline_accuracy, confusion_matrix,
    confusion_matrix_normalized, linear_regress, roc_auc_macro, roc_auc_micro, roc_auc_per_class,
    roc_curve,
};
use sc_oracle::{assert_close, assert_close_slice, load_dir, OracleCase, Payload};

fn fixtures() -> Vec<OracleCase> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../oracle/fixtures/metrics");
    let cases = load_dir(dir);
    assert!(!cases.is_empty(), "no fixtures loaded from {dir}");
    cases
}

#[test]
fn regression_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::Regression(c) = &case.payload else {
            continue;
        };
        n += 1;
        let (id, t, e) = (&case.id, &case.tolerance, &c.expected);
        let got = linear_regress(&c.actual, &c.pred);
        assert_close(got.mse, e.mse, t, &format!("{id}: mse"));
        assert_close(got.r_value, e.r_value, t, &format!("{id}: r_value"));
        assert_close(got.r_squared, e.r_squared, t, &format!("{id}: r_squared"));
        assert_close(got.p_value, e.p_value, t, &format!("{id}: p_value"));
        assert_close(got.std_err, e.std_err, t, &format!("{id}: std_err"));
        assert_close(got.slope, e.slope, t, &format!("{id}: slope"));
        assert_close(got.intercept, e.intercept, t, &format!("{id}: intercept"));
    }
    assert!(n >= 5, "expected >=5 regression fixtures, saw {n}");
}

#[test]
fn classification_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::Classification(c) = &case.payload else {
            continue;
        };
        n += 1;
        let (id, t, e) = (&case.id, &case.tolerance, &c.expected);

        let acc = accuracy(&c.y_true, &c.y_pred);
        assert_close(acc, e.accuracy, t, &format!("{id}: accuracy"));

        let base = baseline_accuracy(&c.y_true);
        assert_close(
            base,
            e.baseline_accuracy,
            t,
            &format!("{id}: baseline_accuracy"),
        );
        assert_close(
            accuracy_ratio(acc, base),
            e.accuracy_ratio,
            t,
            &format!("{id}: accuracy_ratio"),
        );

        let (labels, counts) = confusion_matrix(&c.y_true, &c.y_pred);
        let got_labels: Vec<i64> = labels.iter().map(|&x| x as i64).collect();
        assert_eq!(got_labels, e.labels, "{id}: confusion labels");
        let got_counts: Vec<Vec<i64>> = counts
            .iter()
            .map(|r| r.iter().map(|&x| x as i64).collect())
            .collect();
        assert_eq!(got_counts, e.confusion_matrix, "{id}: confusion counts");

        let norm = confusion_matrix_normalized(&counts);
        assert_eq!(
            norm.len(),
            e.confusion_matrix_normalized.len(),
            "{id}: normalized row count"
        );
        for (i, (gr, er)) in norm.iter().zip(&e.confusion_matrix_normalized).enumerate() {
            assert_close_slice(gr, er, t, &format!("{id}: normalized row {i}"));
        }
    }
    assert!(n >= 6, "expected >=6 classification fixtures, saw {n}");
}

#[test]
fn classification_auc_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::Classification(c) = &case.payload else {
            continue;
        };
        let Some(proba) = &c.proba else {
            continue;
        };
        n += 1;
        let (id, t, e) = (&case.id, &case.tolerance, &c.expected);

        let per = roc_auc_per_class(&c.y_true, proba);
        let exp_per = e.auc_per_class.as_ref().expect("auc_per_class present");
        assert_close_slice(&per, exp_per, t, &format!("{id}: auc_per_class"));

        let micro = roc_auc_micro(&c.y_true, proba);
        assert_close(
            micro,
            e.auc_micro.expect("auc_micro"),
            t,
            &format!("{id}: auc_micro"),
        );

        let macro_avg = roc_auc_macro(&c.y_true, proba);
        assert_close(
            macro_avg,
            e.auc_macro.expect("auc_macro"),
            t,
            &format!("{id}: auc_macro"),
        );
    }
    assert!(n >= 3, "expected >=3 proba fixtures, saw {n}");
}

#[test]
fn roc_curve_matches_oracle() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::RocCurve(c) = &case.payload else {
            continue;
        };
        n += 1;
        let (id, t, e) = (&case.id, &case.tolerance, &c.expected);

        let curve = roc_curve(&c.y_true_binary, &c.y_score);
        assert_close_slice(&curve.fpr, &e.fpr, t, &format!("{id}: fpr"));
        assert_close_slice(&curve.tpr, &e.tpr, t, &format!("{id}: tpr"));
        assert_close(auc(&curve.fpr, &curve.tpr), e.auc, t, &format!("{id}: auc"));

        // Thresholds are informational, but matching the point count and the
        // finite values is a strong `drop_intermediate` cross-check.
        assert_eq!(
            curve.thresholds.len(),
            e.thresholds.len(),
            "{id}: threshold count"
        );
        assert!(
            curve.thresholds[0].is_infinite(),
            "{id}: first threshold should be +inf"
        );
        for (i, exp) in e.thresholds.iter().enumerate() {
            if let Some(v) = exp {
                assert_close(curve.thresholds[i], *v, t, &format!("{id}: threshold[{i}]"));
            }
        }
    }
    assert!(n >= 4, "expected >=4 roc_curve fixtures, saw {n}");
}
