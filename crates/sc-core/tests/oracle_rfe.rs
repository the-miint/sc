//! AGGREGATE-tier oracle test for M6 RFECV: end-to-end selected-feature overlap
//! vs q2. Our RFECV (real forests, our seeded CV folds) is not sklearn-identical,
//! so selection is compared by Jaccard overlap within a band calibrated from q2's
//! own seed-to-seed spread, and the chosen count within a band. Deterministic
//! (fixed master seed). The TIGHT bookkeeping tier lives inline in
//! `rfe::bookkeeping` (it hits the pure fns directly).

use sc_core::{rfecv, FeatureMatrix, RfeStep, RfecvParams, Targets};
use sc_oracle::{load_dir, OracleCase, Payload, RfecvAggregateCase};

fn fixtures() -> Vec<OracleCase> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../oracle/fixtures/rfe");
    let cases = load_dir(dir);
    assert!(!cases.is_empty(), "no fixtures loaded from {dir}");
    cases
}

fn step_of(step: f64) -> RfeStep {
    if step >= 1.0 {
        RfeStep::Count(step as usize)
    } else {
        RfeStep::Fraction(step)
    }
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    use std::collections::BTreeSet;
    let sa: BTreeSet<usize> = a.iter().copied().collect();
    let sb: BTreeSet<usize> = b.iter().copied().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    sa.intersection(&sb).count() as f64 / sa.union(&sb).count() as f64
}

fn build_targets(c: &RfecvAggregateCase) -> Targets {
    match c.task.as_str() {
        "classification" => Targets::categorical(c.labels.as_ref().expect("labels")).unwrap(),
        "regression" => Targets::numeric(c.values.as_ref().expect("values")).unwrap(),
        other => panic!("unexpected task {other}"),
    }
}

#[test]
fn rfecv_selection_within_band_vs_q2() {
    let mut n = 0;
    for case in fixtures() {
        let Payload::RfecvAggregate(c) = &case.payload else {
            continue;
        };
        n += 1;
        let id = &case.id;
        let x = FeatureMatrix::from_coo(
            c.data_coo.n_samples,
            c.data_coo.n_features,
            &c.data_coo.rows,
            &c.data_coo.cols,
            &c.data_coo.vals,
        )
        .unwrap();
        let y = build_targets(c);

        let mut params = match c.task.as_str() {
            "classification" => RfecvParams::classifier_defaults(),
            "regression" => RfecvParams::regressor_defaults(),
            other => panic!("unexpected task {other}"),
        };
        params.forest.n_estimators = c.n_estimators;
        params.step = step_of(c.step);
        params.min_features_to_select = c.min_features_to_select;
        params.n_splits = c.n_splits;

        let result =
            rfecv(&x, &y, &params, 0).unwrap_or_else(|e| panic!("{id}: rfecv failed: {e}"));

        // Mean Jaccard of our selection vs each of q2's K seed selections.
        let mean_jaccard: f64 = c
            .reference_sets
            .iter()
            .map(|r| jaccard(&result.selected, r))
            .sum::<f64>()
            / c.reference_sets.len() as f64;
        let floor = c.jaccard_floor - c.jaccard_margin;
        assert!(
            mean_jaccard >= floor,
            "{id}: mean Jaccard {mean_jaccard:.3} < floor {floor:.3} \
             (q2 self-consistency {:.3} - margin {})",
            c.jaccard_floor,
            c.jaccard_margin
        );
        let lo = c.n_optimal_min.saturating_sub(c.n_optimal_margin);
        let hi = c.n_optimal_max + c.n_optimal_margin;
        assert!(
            (lo..=hi).contains(&result.n_optimal),
            "{id}: n_optimal {} outside [{lo}, {hi}]",
            result.n_optimal
        );
    }
    assert!(n >= 4, "expected >= 4 rfecv_aggregate fixtures, saw {n}");
}
