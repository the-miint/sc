//! Rust-side FFI tests: call the `extern "C"` entry points directly, building
//! Arrow inputs with arrow-rs and reading outputs back via `from_ffi`. The
//! headline check is that the marshaled fit+predict reproduces the direct
//! `sc-core` path byte-for-byte under the same seed.

use super::*;

use arrow_array::ffi::{from_ffi, to_ffi};
use arrow_array::{Array, Float64Array, Int64Array, StringArray};
use sc_core::{FeatureMatrix, MaxFeatures, RandomForest, RandomForestParams, Targets};
use std::mem::MaybeUninit;

fn ffi_i64(v: Vec<i64>) -> (FFI_ArrowArray, FFI_ArrowSchema) {
    to_ffi(&Int64Array::from(v).into_data()).unwrap()
}
fn ffi_f64(v: Vec<f64>) -> (FFI_ArrowArray, FFI_ArrowSchema) {
    to_ffi(&Float64Array::from(v).into_data()).unwrap()
}
fn ffi_utf8(v: &[&str]) -> (FFI_ArrowArray, FFI_ArrowSchema) {
    to_ffi(&StringArray::from(v.to_vec()).into_data()).unwrap()
}

fn make_table(ns: i64, nf: i64, rows: Vec<i64>, cols: Vec<i64>, vals: Vec<f64>) -> sc_coo_table_t {
    let (rows, rows_schema) = ffi_i64(rows);
    let (cols, cols_schema) = ffi_i64(cols);
    let (vals, vals_schema) = ffi_f64(vals);
    sc_coo_table_t {
        rows,
        rows_schema,
        cols,
        cols_schema,
        vals,
        vals_schema,
        n_samples: ns,
        n_features: nf,
        sample_ids: FFI_ArrowArray::empty(),
        sample_ids_schema: FFI_ArrowSchema::empty(),
        feature_ids: FFI_ArrowArray::empty(),
        feature_ids_schema: FFI_ArrowSchema::empty(),
    }
}

/// Dense rows → COO triplets (skipping zeros), matching sc-core's own test helper.
fn dense_coo(rows: &[&[f64]]) -> (i64, i64, Vec<i64>, Vec<i64>, Vec<f64>) {
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
    (n as i64, m as i64, r, c, v)
}

// A cleanly separable 2-feature, 2-class problem (mirrors forest::fit tests).
// (n_samples, n_features, rows, cols, vals, labels) — a fixture tuple.
#[allow(clippy::type_complexity)]
fn separable() -> (i64, i64, Vec<i64>, Vec<i64>, Vec<f64>, [&'static str; 6]) {
    let (ns, nf, r, c, v) = dense_coo(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[3.0, 3.0],
        &[3.0, 4.0],
        &[4.0, 3.0],
    ]);
    (ns, nf, r, c, v, ["a", "a", "a", "b", "b", "b"])
}

// Balanced 8-sample, 4-feature, 2-class problem (mirrors ncv/tuning tests).
#[allow(clippy::type_complexity)]
fn balanced8() -> (i64, i64, Vec<i64>, Vec<i64>, Vec<f64>, [&'static str; 8]) {
    let (ns, nf, r, c, v) = dense_coo(&[
        &[0.0, 0.0, 1.0, 2.0],
        &[0.0, 1.0, 3.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[0.0, 1.0, 2.0, 2.0],
        &[5.0, 5.0, 2.0, 0.0],
        &[5.0, 6.0, 0.0, 3.0],
        &[6.0, 5.0, 1.0, 1.0],
        &[6.0, 6.0, 0.0, 0.0],
    ]);
    (ns, nf, r, c, v, ["a", "a", "a", "a", "b", "b", "b", "b"])
}

/// A `sc_rf_params_t` equivalent to `classifier_defaults` with `max_features=Sqrt`
/// and `n_estimators=8` — the config the equality test also builds directly.
fn base_params(seed: u64) -> sc_rf_params_t {
    sc_rf_params_t {
        criterion: sc_criterion_t::SC_CRITERION_GINI,
        max_depth: 0,
        max_features: sc_max_features_t {
            kind: sc_max_features_kind_t::SC_MAX_FEATURES_SQRT,
            fraction: 0.0,
            count: 0,
        },
        min_samples_split: sc_min_samples_t {
            kind: sc_min_samples_kind_t::SC_MIN_SAMPLES_COUNT,
            count: 2,
            fraction: 0.0,
        },
        min_samples_leaf: sc_min_samples_t {
            kind: sc_min_samples_kind_t::SC_MIN_SAMPLES_COUNT,
            count: 1,
            fraction: 0.0,
        },
        min_weight_fraction_leaf: 0.0,
        min_impurity_decrease: 0.0,
        n_estimators: 8,
        bootstrap: true,
        max_samples: sc_max_samples_t {
            kind: sc_max_samples_kind_t::SC_MAX_SAMPLES_ALL,
            count: 0,
            fraction: 0.0,
        },
        random_state: seed,
        optimize_feature_selection: false,
        rfe_step: 1.0,
        parameter_tuning: false,
        cv: 5,
        n_threads: 0,
    }
}

fn new_ctx() -> *mut sc_context_t {
    let mut ctx: *mut sc_context_t = std::ptr::null_mut();
    assert_eq!(unsafe { sc_context_new(std::ptr::null(), &mut ctx) }, SC_OK);
    assert!(!ctx.is_null());
    ctx
}

/// Read an exported Utf8 array (predictions / class labels) back to owned strings.
unsafe fn read_utf8(a: FFI_ArrowArray, s: FFI_ArrowSchema) -> Vec<String> {
    let arr = StringArray::from(from_ffi(a, &s).unwrap());
    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
}

#[test]
fn marshaled_fit_predict_matches_direct() {
    let (ns, nf, r, c, v, labels) = separable();
    let seed = 42u64;

    // Direct sc-core path.
    let x = FeatureMatrix::from_coo(ns as usize, nf as usize, &r, &c, &v).unwrap();
    let y = Targets::categorical(&labels).unwrap();
    let mut p = RandomForestParams::classifier_defaults();
    p.n_estimators = 8;
    p.tree.max_features = MaxFeatures::Sqrt;
    let direct = RandomForest::fit(&x, &y, &p, seed).unwrap();
    let direct_labels: Vec<String> = direct
        .predict_class_indices(&x)
        .unwrap()
        .iter()
        .map(|&i| direct.classes().unwrap()[i].clone())
        .collect();

    // Marshaled C-API path with the identical config + seed.
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(seed);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );
        assert!(!model.is_null());

        let ptable = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict(ctx, model, &ptable, pa.as_mut_ptr(), ps.as_mut_ptr()),
            SC_OK
        );
        let got = read_utf8(pa.assume_init(), ps.assume_init());
        assert_eq!(got, direct_labels, "marshaled predict != direct predict");

        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn predict_proba_width_and_sum() {
    let (ns, nf, r, c, v, labels) = separable();
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(7);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let ptable = make_table(ns, nf, r, c, v);
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let mut ca = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut cs = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict_proba(
                ctx,
                model,
                &ptable,
                pa.as_mut_ptr(),
                ps.as_mut_ptr(),
                ca.as_mut_ptr(),
                cs.as_mut_ptr(),
            ),
            SC_OK
        );
        use arrow_array::FixedSizeListArray;
        let list = FixedSizeListArray::from(from_ffi(pa.assume_init(), &ps.assume_init()).unwrap());
        assert_eq!(list.len(), ns as usize);
        assert_eq!(list.value_length(), 2);
        let flat = list
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for row in 0..ns as usize {
            let s = flat.value(row * 2) + flat.value(row * 2 + 1);
            assert!((s - 1.0).abs() < 1e-9, "proba row {row} sums to {s}");
        }
        assert_eq!(
            read_utf8(ca.assume_init(), cs.assume_init()),
            vec!["a".to_string(), "b".into()]
        );
        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // index-driven cross-checks over parallel arrays
fn marshaled_shap_matches_direct() {
    use arrow_array::FixedSizeListArray;
    let (ns, nf, r, c, v, labels) = separable();
    let seed = 42u64;

    // Direct sc-core path: fit + tree_shap.
    let x = FeatureMatrix::from_coo(ns as usize, nf as usize, &r, &c, &v).unwrap();
    let y = Targets::categorical(&labels).unwrap();
    let mut p = RandomForestParams::classifier_defaults();
    p.n_estimators = 8;
    p.tree.max_features = MaxFeatures::Sqrt;
    let direct = RandomForest::fit(&x, &y, &p, seed).unwrap();
    let ds = direct.tree_shap(&x).unwrap();
    let proba = direct.predict_proba(&x).unwrap();
    let (no, nfeat) = (ds.n_outputs, ds.n_features);
    let width = no * nfeat;

    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(seed);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let stable = make_table(ns, nf, r, c, v);
        let mut result = MaybeUninit::<sc_shap_result_t>::uninit();
        assert_eq!(sc_shap(ctx, model, &stable, result.as_mut_ptr()), SC_OK);
        let result = result.assume_init();
        assert_eq!(result.n_outputs, no as i64);
        assert_eq!(result.n_features, nfeat as i64);

        // shap_values: FixedSizeList<f64>[width], one row per sample.
        let list = FixedSizeListArray::from(
            from_ffi(result.shap_values, &result.shap_values_schema).unwrap(),
        );
        assert_eq!(list.len(), ns as usize);
        assert_eq!(list.value_length(), width as i32);
        let flat = list
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // Each row equals the direct tree_shap, flattened output-major.
        for s in 0..ns as usize {
            for o in 0..no {
                for f in 0..nfeat {
                    let got = flat.value(s * width + o * nfeat + f);
                    assert!(
                        (got - ds.values[s][o][f]).abs() < 1e-12,
                        "shap[{s}][{o}][{f}]: marshaled {got} vs direct {}",
                        ds.values[s][o][f]
                    );
                }
            }
        }

        // base_values Float64[n_outputs] equal to the direct base.
        let base =
            Float64Array::from(from_ffi(result.base_values, &result.base_values_schema).unwrap());
        assert_eq!(base.len(), no);
        for o in 0..no {
            assert!((base.value(o) - ds.base_values[o]).abs() < 1e-12);
            // Additivity: base + Σ_f shap == predict_proba, per sample/class.
            for s in 0..ns as usize {
                let row_sum: f64 = (0..nfeat)
                    .map(|f| flat.value(s * width + o * nfeat + f))
                    .sum();
                assert!(
                    (base.value(o) + row_sum - proba[s][o]).abs() < 1e-9,
                    "additivity[{s}][{o}]"
                );
            }
        }

        // Classifier carries class labels aligned to the output axis.
        assert_eq!(
            read_utf8(result.classes, result.classes_schema),
            vec!["a".to_string(), "b".into()]
        );

        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn shap_regressor_leaves_classes_empty() {
    let (ns, nf, r, c, v) = dense_coo(&[
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 1.0],
        &[5.0, 0.0],
    ]);
    let targets = vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0];
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_f64(targets);
        let mut params = base_params(0);
        params.criterion = sc_criterion_t::SC_CRITERION_SQUARED_ERROR;
        params.max_features.kind = sc_max_features_kind_t::SC_MAX_FEATURES_ALL;
        params.n_estimators = 10;
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_regressor(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let stable = make_table(ns, nf, r, c, v);
        let mut result = MaybeUninit::<sc_shap_result_t>::uninit();
        assert_eq!(sc_shap(ctx, model, &stable, result.as_mut_ptr()), SC_OK);
        let result = result.assume_init();
        assert_eq!(result.n_outputs, 1);
        assert_eq!(result.n_features, nf);
        assert!(
            result.classes.is_released(),
            "regressor shap must leave classes empty"
        );
        let base =
            Float64Array::from(from_ffi(result.base_values, &result.base_values_schema).unwrap());
        assert_eq!(base.len(), 1);
        assert!(base.value(0).is_finite());
        // `base` (from_ffi) releases base_values on drop; every other field of
        // `result` is released by drop glue at scope end (partial move, no Drop
        // impl on the struct).

        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn serialize_deserialize_predict_round_trip() {
    let (ns, nf, r, c, v, labels) = separable();
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(123);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        // Predict before serialize.
        let ptable = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict(ctx, model, &ptable, pa.as_mut_ptr(), ps.as_mut_ptr()),
            SC_OK
        );
        let before = read_utf8(pa.assume_init(), ps.assume_init());

        // Serialize → deserialize.
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        assert_eq!(sc_model_serialize(model, &mut buf, &mut len), SC_OK);
        assert!(!buf.is_null() && len > 0);
        let mut reloaded: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(sc_model_deserialize(ctx, buf, len, &mut reloaded), SC_OK);
        sc_buffer_free(buf, len);

        // Predict with the reloaded model.
        let ptable2 = make_table(ns, nf, r, c, v);
        let mut pa2 = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps2 = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict(ctx, reloaded, &ptable2, pa2.as_mut_ptr(), ps2.as_mut_ptr()),
            SC_OK
        );
        let after = read_utf8(pa2.assume_init(), ps2.assume_init());
        assert_eq!(before, after, "reloaded predictions differ");

        sc_model_free(model);
        sc_model_free(reloaded);
        sc_context_free(ctx);
    }
}

#[test]
fn cross_validate_shape() {
    let (ns, nf, r, c, v, labels) = balanced8();
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r, c, v);
        let (t, ts) = ffi_utf8(&labels);
        let mut params = base_params(20260709);
        params.cv = 4;
        let mut result = MaybeUninit::<sc_cv_result_t>::uninit();
        assert_eq!(
            sc_cross_validate(ctx, &table, &t, &ts, &params, result.as_mut_ptr()),
            SC_OK
        );
        let result = result.assume_init();
        assert_eq!(result.predictions.len(), 8);
        assert!(!result.proba.is_released());
        assert_eq!(result.proba.len(), 8);
        assert_eq!(result.classes.len(), 2);
        assert_eq!(result.fold_scores.len(), 4);
        assert!(result.mean_score.is_finite() && result.std_score.is_finite());
        // Dropping `result` releases every output array (its FFI fields' Drop
        // calls arrow-rs's release callback).
        drop(result);
        sc_context_free(ctx);
    }
}

#[test]
fn cross_validate_regression_leaves_proba_classes_empty() {
    // The regression OOF path: predictions/fold_scores are Float64, and
    // proba/classes must be the released/empty sentinels (not live arrays).
    let (ns, nf, r, c, v) = dense_coo(&[
        &[1.0, 2.0, 0.0, 1.0],
        &[2.0, 0.0, 1.0, 0.0],
        &[3.0, 1.0, 0.0, 2.0],
        &[4.0, 0.0, 2.0, 1.0],
        &[5.0, 3.0, 0.0, 0.0],
        &[6.0, 0.0, 1.0, 3.0],
        &[7.0, 2.0, 0.0, 1.0],
        &[8.0, 1.0, 2.0, 0.0],
    ]);
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r, c, v);
        let (t, ts) = ffi_f64(targets);
        let mut params = base_params(20260709);
        params.criterion = sc_criterion_t::SC_CRITERION_SQUARED_ERROR;
        params.max_features.kind = sc_max_features_kind_t::SC_MAX_FEATURES_ALL;
        params.n_estimators = 8;
        params.cv = 4;
        let mut result = MaybeUninit::<sc_cv_result_t>::uninit();
        assert_eq!(
            sc_cross_validate(ctx, &table, &t, &ts, &params, result.as_mut_ptr()),
            SC_OK
        );
        let result = result.assume_init();
        assert_eq!(result.predictions.len(), 8);
        assert!(result.proba.is_released(), "regression proba must be empty");
        assert!(
            result.classes.is_released(),
            "regression classes must be empty"
        );
        assert_eq!(result.fold_scores.len(), 4);
        assert!(result.mean_score.is_finite());
        drop(result);
        sc_context_free(ctx);
    }
}

#[test]
fn regression_fit_predict_and_importances() {
    let (ns, nf, r, c, v) = dense_coo(&[
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 1.0],
        &[5.0, 0.0],
    ]);
    let targets = vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0];
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_f64(targets);
        let mut params = base_params(0);
        params.criterion = sc_criterion_t::SC_CRITERION_SQUARED_ERROR;
        params.max_features.kind = sc_max_features_kind_t::SC_MAX_FEATURES_ALL;
        params.n_estimators = 10;
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_regressor(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let ptable = make_table(ns, nf, r, c, v);
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict(ctx, model, &ptable, pa.as_mut_ptr(), ps.as_mut_ptr()),
            SC_OK
        );
        let arr = Float64Array::from(from_ffi(pa.assume_init(), &ps.assume_init()).unwrap());
        assert_eq!(arr.len(), 6);
        for i in 0..arr.len() {
            assert!((1.0..=9.0).contains(&arr.value(i)));
        }

        // Feature importances: length n_features, sums to ~1.
        let mut fa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut fs = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_feature_importances(model, fa.as_mut_ptr(), fs.as_mut_ptr()),
            SC_OK
        );
        let fi = Float64Array::from(from_ffi(fa.assume_init(), &fs.assume_init()).unwrap());
        assert_eq!(fi.len(), nf as usize);
        let sum: f64 = (0..fi.len()).map(|i| fi.value(i)).sum();
        assert!((sum - 1.0).abs() < 1e-9);

        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn model_feature_ids_round_trip_through_serialization() {
    // The training vocabulary is what lets a caller encode a *different*
    // dataset's features into this model's columns. sc only checks the feature
    // count, so without it a caller re-deriving its own encoding gets silently
    // wrong answers whenever two datasets differ by one feature each way.
    let (ns, nf, r, c, v, labels) = separable();
    unsafe {
        let ctx = new_ctx();
        let mut table = make_table(ns, nf, r, c, v);
        let (fids, fids_s) = ffi_utf8(&["feat_b", "feat_a"]);
        table.feature_ids = fids;
        table.feature_ids_schema = fids_s;
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(7);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let read_ids = |m: *const sc_model_t| {
            let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
            let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
            assert_eq!(
                sc_model_feature_ids(m, a.as_mut_ptr(), s.as_mut_ptr()),
                SC_OK
            );
            let arr = StringArray::from(from_ffi(a.assume_init(), &s.assume_init()).unwrap());
            (0..arr.len())
                .map(|i| arr.value(i).to_string())
                .collect::<Vec<_>>()
        };

        // Returned in the caller's column order, NOT sorted by sc: index i is
        // whatever the caller put in column i.
        let ids = read_ids(model);
        assert_eq!(ids, vec!["feat_b".to_string(), "feat_a".to_string()]);
        assert_eq!(ids.len(), nf as usize);

        // The vocabulary must survive the BLOB round trip, since that is how a
        // model reaches a later session.
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        assert_eq!(sc_model_serialize(model, &mut ptr, &mut len), SC_OK);
        let mut restored: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(sc_model_deserialize(ctx, ptr, len, &mut restored), SC_OK);
        assert_eq!(read_ids(restored), ids);

        sc_buffer_free(ptr, len);
        sc_model_free(restored);
        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn model_feature_ids_errors_when_absent() {
    // make_table leaves feature_ids as the released/empty sentinel, so this
    // model carries none. An explicit error beats an empty array a caller could
    // mistake for "this model has no features".
    let (ns, nf, r, c, v, labels) = separable();
    unsafe {
        let ctx = new_ctx();
        let table = make_table(ns, nf, r, c, v);
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(7);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_model_feature_ids(model, a.as_mut_ptr(), s.as_mut_ptr()),
            SC_ERR_INVALID_INPUT
        );

        sc_model_free(model);
        sc_context_free(ctx);
    }
}

#[test]
fn tuning_and_rfe_flag_paths_produce_valid_models() {
    let (ns, nf, r, c, v, labels) = balanced8();
    for (tuning, rfe) in [(true, false), (false, true)] {
        unsafe {
            let ctx = new_ctx();
            let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
            let (t, ts) = ffi_utf8(&labels);
            let mut params = base_params(9);
            params.n_estimators = 4;
            params.cv = 2;
            params.parameter_tuning = tuning;
            params.optimize_feature_selection = rfe;
            let mut model: *mut sc_model_t = std::ptr::null_mut();
            assert_eq!(
                sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
                SC_OK,
                "tuning={tuning} rfe={rfe}"
            );
            assert!(!model.is_null());

            // The model must predict every sample.
            let ptable = make_table(ns, nf, r.clone(), c.clone(), v.clone());
            let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
            let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
            assert_eq!(
                sc_predict(ctx, model, &ptable, pa.as_mut_ptr(), ps.as_mut_ptr()),
                SC_OK
            );
            let got = read_utf8(pa.assume_init(), ps.assume_init());
            assert_eq!(got.len(), 8);
            assert!(got.iter().all(|l| l == "a" || l == "b"));

            sc_model_free(model);
            sc_context_free(ctx);
        }
    }
}

#[test]
fn forced_error_battery() {
    let (ns, nf, r, c, v, _labels) = separable();
    unsafe {
        let ctx = new_ctx();

        // 1. Null pointer → SC_ERR_NULL_POINTER.
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        let params = base_params(0);
        let (t, ts) = ffi_utf8(&["a", "a", "a", "b", "b", "b"]);
        let code = sc_fit_classifier(ctx, std::ptr::null(), &t, &ts, &params, &mut model);
        assert_eq!(code, SC_ERR_NULL_POINTER);
        assert!(!sc_context_last_error(ctx).is_null());

        // 2. x/y sample mismatch → SC_ERR_INVALID_INPUT with a message.
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t2, ts2) = ffi_utf8(&["a", "b"]); // 2 labels, 6 samples
        let code = sc_fit_classifier(ctx, &table, &t2, &ts2, &params, &mut model);
        assert_eq!(code, SC_ERR_INVALID_INPUT);
        let msg = std::ffi::CStr::from_ptr(sc_context_last_error(ctx))
            .to_string_lossy()
            .into_owned();
        assert!(msg.contains("samples"), "unexpected message: {msg}");

        // 3. Wrong target dtype (numeric array to fit_classifier) → SC_ERR_ARROW.
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t3, ts3) = ffi_f64(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let code = sc_fit_classifier(ctx, &table, &t3, &ts3, &params, &mut model);
        assert_eq!(code, SC_ERR_ARROW);

        // 4. predict_proba on a regressor → SC_ERR_INVALID_INPUT (not a panic).
        let (rns, rnf, rr, rc, rv) = dense_coo(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let rtable = make_table(rns, rnf, rr.clone(), rc.clone(), rv.clone());
        let (rt, rts) = ffi_f64(vec![1.0, 2.0, 3.0, 4.0]);
        let mut reg_params = base_params(0);
        reg_params.criterion = sc_criterion_t::SC_CRITERION_SQUARED_ERROR;
        reg_params.max_features.kind = sc_max_features_kind_t::SC_MAX_FEATURES_ALL;
        reg_params.n_estimators = 4;
        let mut reg_model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_regressor(ctx, &rtable, &rt, &rts, &reg_params, &mut reg_model),
            SC_OK
        );
        let ptable = make_table(rns, rnf, rr, rc, rv);
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let mut ca = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut cs = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let code = sc_predict_proba(
            ctx,
            reg_model,
            &ptable,
            pa.as_mut_ptr(),
            ps.as_mut_ptr(),
            ca.as_mut_ptr(),
            cs.as_mut_ptr(),
        );
        assert_eq!(code, SC_ERR_INVALID_INPUT);
        sc_model_free(reg_model);

        // 5. Deserialize garbage → SC_ERR_DESERIALIZE.
        let garbage = [0u8; 16];
        let mut loaded: *mut sc_model_t = std::ptr::null_mut();
        let code = sc_model_deserialize(ctx, garbage.as_ptr(), garbage.len(), &mut loaded);
        assert_eq!(code, SC_ERR_DESERIALIZE);

        sc_context_free(ctx);
    }
}

#[test]
fn panic_barrier_maps_to_sc_err_panic() {
    // The public API guards every known sc-core panic site, so a panic can't be
    // reached through it by design. Test the barrier itself directly (this
    // module is inside the crate): a panicking closure yields SC_ERR_PANIC and
    // records a "panic:" message rather than unwinding across FFI.
    unsafe {
        let ctx = new_ctx();
        let code = guard(ctx, || panic!("boom"));
        assert_eq!(code, SC_ERR_PANIC);
        let msg = std::ffi::CStr::from_ptr(sc_context_last_error(ctx))
            .to_string_lossy()
            .into_owned();
        assert!(msg.contains("panic"), "unexpected message: {msg}");
        assert!(msg.contains("boom"));
        sc_context_free(ctx);
    }
}

#[test]
fn version_is_nonnull_and_matches_const() {
    // sc_version() hands back a 'static, NUL-terminated string equal to the
    // SC_VERSION const (and to the crate version). The pointer must never be
    // freed — it aliases a compile-time byte string.
    let p = sc_version();
    assert!(!p.is_null(), "sc_version returned NULL");
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_str()
        .expect("sc_version is valid UTF-8");
    assert_eq!(s, SC_VERSION);
    assert_eq!(s, env!("CARGO_PKG_VERSION"));
}

#[test]
fn context_with_dedicated_pool() {
    // n_threads > 0 builds a dedicated pool; fit must still succeed and match the
    // global-pool result (thread-count invariance from sc-core).
    let (ns, nf, r, c, v, labels) = separable();
    unsafe {
        let cfg = sc_config_t { n_threads: 2 };
        let mut ctx: *mut sc_context_t = std::ptr::null_mut();
        assert_eq!(sc_context_new(&cfg, &mut ctx), SC_OK);
        let table = make_table(ns, nf, r.clone(), c.clone(), v.clone());
        let (t, ts) = ffi_utf8(&labels);
        let params = base_params(42);
        let mut model: *mut sc_model_t = std::ptr::null_mut();
        assert_eq!(
            sc_fit_classifier(ctx, &table, &t, &ts, &params, &mut model),
            SC_OK
        );

        let ptable = make_table(ns, nf, r, c, v);
        let mut pa = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut ps = MaybeUninit::<FFI_ArrowSchema>::uninit();
        assert_eq!(
            sc_predict(ctx, model, &ptable, pa.as_mut_ptr(), ps.as_mut_ptr()),
            SC_OK
        );
        let pooled = read_utf8(pa.assume_init(), ps.assume_init());

        // Same seed on the global pool → identical labels.
        let x = FeatureMatrix::from_coo(
            ns as usize,
            nf as usize,
            &separable().2,
            &separable().3,
            &separable().4,
        )
        .unwrap();
        let y = Targets::categorical(&labels).unwrap();
        let mut p = RandomForestParams::classifier_defaults();
        p.n_estimators = 8;
        p.tree.max_features = MaxFeatures::Sqrt;
        let direct = RandomForest::fit(&x, &y, &p, 42).unwrap();
        let direct_labels: Vec<String> = direct
            .predict_class_indices(&x)
            .unwrap()
            .iter()
            .map(|&i| direct.classes().unwrap()[i].clone())
            .collect();
        assert_eq!(pooled, direct_labels);

        sc_model_free(model);
        sc_context_free(ctx);
    }
}
