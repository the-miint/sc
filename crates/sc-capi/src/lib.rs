//! `sc-capi`: the reentrant `sc_*` C-API — the **policy** layer of the M9 FFI
//! boundary (design §4/§5). It owns the opaque handles, the per-context rayon
//! pool, the `catch_unwind` panic barrier, `sc_status_t` error mapping +
//! last-error storage, the `#[repr(C)]` parameter/result structs, and the
//! workflow dispatch onto `sc-core`. All Arrow array conversion is delegated to
//! `sc-arrow`; all model math to `sc-core`.
//!
//! # Safety (shared C-ABI contract)
//!
//! Every `sc_*` entry point takes raw pointers supplied by a C caller. The
//! universal contract, assumed by all of them:
//!
//! * Non-null pointer arguments must point to valid, correctly-typed, aligned
//!   objects that live for the duration of the call. Null is checked and turned
//!   into `SC_ERR_NULL_POINTER` (never a crash), *except* the `free` functions,
//!   which treat null as a no-op.
//! * Input Arrow arrays are **borrowed**, never released (the caller keeps
//!   ownership). Output Arrow arrays are written into caller out-slots with an
//!   arrow-rs `release` callback the **caller must invoke**.
//! * `**out` handle slots receive a `Box::into_raw` pointer that must later be
//!   passed to the matching `sc_*_free`.
//! * A panic in any entry point is caught and converted to `SC_ERR_PANIC`; a
//!   panic never crosses the FFI boundary.
//!
//! Because these all share one contract documented here, per-function
//! `# Safety` sections would be pure repetition — hence the crate-level allow.
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

mod convert;

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

use sc_core::{ncv, FeatureMatrix, ModelBundle, NcvResult, OofPredictions, RandomForest, Task};

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Status returned by every fallible `sc_*` function. `0` is success; every
/// error is a distinct negative class.
pub type sc_status_t = i32;

pub const SC_OK: sc_status_t = 0;
pub const SC_ERR_INVALID_INPUT: sc_status_t = -1;
pub const SC_ERR_DESERIALIZE: sc_status_t = -2;
pub const SC_ERR_NULL_POINTER: sc_status_t = -3;
pub const SC_ERR_ARROW: sc_status_t = -4;
pub const SC_ERR_PANIC: sc_status_t = -99;
pub const SC_ERR_UNKNOWN: sc_status_t = -100;

// ---------------------------------------------------------------------------
// Parameter / config / result POD types (cbindgen emits these)
// ---------------------------------------------------------------------------

/// Context configuration. `n_threads <= 0` uses the global rayon pool; a
/// positive value builds a dedicated pool of that size.
#[repr(C)]
pub struct sc_config_t {
    pub n_threads: i32,
}

/// Split criterion (must match the task).
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum sc_criterion_t {
    SC_CRITERION_GINI = 0,
    SC_CRITERION_ENTROPY = 1,
    SC_CRITERION_SQUARED_ERROR = 2,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum sc_max_features_kind_t {
    SC_MAX_FEATURES_ALL = 0,
    SC_MAX_FEATURES_SQRT = 1,
    SC_MAX_FEATURES_LOG2 = 2,
    SC_MAX_FEATURES_FRACTION = 3,
    SC_MAX_FEATURES_COUNT = 4,
}

/// Tagged int-or-fraction `max_features` (sklearn semantics). `fraction` is read
/// only for `FRACTION`, `count` only for `COUNT`.
#[repr(C)]
pub struct sc_max_features_t {
    pub kind: sc_max_features_kind_t,
    pub fraction: f64,
    pub count: u64,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum sc_min_samples_kind_t {
    SC_MIN_SAMPLES_COUNT = 0,
    SC_MIN_SAMPLES_FRACTION = 1,
}

/// Tagged `min_samples_split` / `min_samples_leaf` (count or fraction of n).
#[repr(C)]
pub struct sc_min_samples_t {
    pub kind: sc_min_samples_kind_t,
    pub count: u64,
    pub fraction: f64,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum sc_max_samples_kind_t {
    SC_MAX_SAMPLES_ALL = 0,
    SC_MAX_SAMPLES_COUNT = 1,
    SC_MAX_SAMPLES_FRACTION = 2,
}

/// Tagged bootstrap sample size (`max_samples`).
#[repr(C)]
pub struct sc_max_samples_t {
    pub kind: sc_max_samples_kind_t,
    pub count: u64,
    pub fraction: f64,
}

/// The consolidated RandomForest workflow parameters (design §5). Carries the
/// per-tree + ensemble hyperparameters, the mandatory `random_state`, and the
/// workflow flags that select plain-fit / RFECV / tuning.
#[repr(C)]
pub struct sc_rf_params_t {
    pub criterion: sc_criterion_t,
    /// `<= 0` means unbounded depth (sklearn `None`).
    pub max_depth: i64,
    pub max_features: sc_max_features_t,
    pub min_samples_split: sc_min_samples_t,
    pub min_samples_leaf: sc_min_samples_t,
    pub min_weight_fraction_leaf: f64,
    pub min_impurity_decrease: f64,
    pub n_estimators: i64,
    pub bootstrap: bool,
    pub max_samples: sc_max_samples_t,
    pub random_state: u64,
    /// RFECV: eliminate features via cross-validated recursive elimination.
    pub optimize_feature_selection: bool,
    /// RFE step: `>= 1` = count, `(0,1)` = fraction (sklearn `step`).
    pub rfe_step: f64,
    /// RandomizedSearchCV over the built-in grid before fitting.
    pub parameter_tuning: bool,
    /// CV fold count for tuning / cross-validate (`<= 1` defaults to 5).
    pub cv: i64,
    /// Per-call thread hint (currently advisory; the context owns the pool).
    pub n_threads: i32,
}

/// A COO feature table (rows = samples, cols = features). `rows`/`cols` are
/// `Int64`, `vals` `Float64`, all length-nnz and index-aligned. `sample_ids` /
/// `feature_ids` are OPTIONAL `Utf8` labels (release == NULL means absent) used
/// only to label outputs / persisted models — there is no ID join (design §6).
#[repr(C)]
pub struct sc_coo_table_t {
    pub rows: FFI_ArrowArray,
    pub rows_schema: FFI_ArrowSchema,
    pub cols: FFI_ArrowArray,
    pub cols_schema: FFI_ArrowSchema,
    pub vals: FFI_ArrowArray,
    pub vals_schema: FFI_ArrowSchema,
    pub n_samples: i64,
    pub n_features: i64,
    pub sample_ids: FFI_ArrowArray,
    pub sample_ids_schema: FFI_ArrowSchema,
    pub feature_ids: FFI_ArrowArray,
    pub feature_ids_schema: FFI_ArrowSchema,
}

/// Output of `sc_cross_validate`. All Arrow arrays carry release callbacks the
/// caller must invoke. For regression, `proba`/`classes` are released/empty
/// (their `release` is NULL).
#[repr(C)]
pub struct sc_cv_result_t {
    /// OOF predictions per sample: `Utf8` labels (classification) / `Float64`
    /// values (regression).
    pub predictions: FFI_ArrowArray,
    pub predictions_schema: FFI_ArrowSchema,
    /// `FixedSizeList<Float64>[n_classes]` OOF probabilities (classification).
    pub proba: FFI_ArrowArray,
    pub proba_schema: FFI_ArrowSchema,
    /// `Utf8` class labels aligned to `proba` columns (classification).
    pub classes: FFI_ArrowArray,
    pub classes_schema: FFI_ArrowSchema,
    /// Per-fold `Float64` scores (accuracy / MSE).
    pub fold_scores: FFI_ArrowArray,
    pub fold_scores_schema: FFI_ArrowSchema,
    pub mean_score: f64,
    pub std_score: f64,
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Runtime resources: an optional dedicated rayon pool + last-error storage.
/// `last_error` uses interior mutability so `*const` entry points can still
/// record errors. One context per thread (no shared mutable state → reentrant).
pub struct sc_context_t {
    pool: Option<rayon::ThreadPool>,
    last_error: RefCell<Option<CString>>,
}

/// A trained model: an sc-core [`ModelBundle`] (forest + params + feature ids +
/// RFE selection). Pure serializable data — no runtime resources.
pub struct sc_model_t {
    bundle: ModelBundle,
}

// ---------------------------------------------------------------------------
// Error model + panic barrier
// ---------------------------------------------------------------------------

/// Internal error carrying a status code + message; mapped to `sc_status_t` and
/// the context's last-error string by [`guard`].
struct ApiError {
    code: sc_status_t,
    message: String,
}

impl ApiError {
    fn new(code: sc_status_t, message: impl Into<String>) -> Self {
        ApiError {
            code,
            message: message.into(),
        }
    }
    fn null(what: &str) -> Self {
        ApiError::new(SC_ERR_NULL_POINTER, format!("null pointer: {what}"))
    }
}

impl From<sc_core::Error> for ApiError {
    fn from(e: sc_core::Error) -> Self {
        let code = match &e {
            sc_core::Error::InvalidInput(_) => SC_ERR_INVALID_INPUT,
            sc_core::Error::Deserialize(_) => SC_ERR_DESERIALIZE,
            // `Error` is #[non_exhaustive]; a future variant maps to the catch-all.
            _ => SC_ERR_UNKNOWN,
        };
        ApiError::new(code, e.to_string())
    }
}

impl From<sc_arrow::MarshalError> for ApiError {
    fn from(e: sc_arrow::MarshalError) -> Self {
        match e {
            sc_arrow::MarshalError::Arrow(m) => ApiError::new(SC_ERR_ARROW, format!("arrow: {m}")),
            sc_arrow::MarshalError::Core(c) => ApiError::from(c),
        }
    }
}

/// Record `message` as the context's last error (if `ctx` is non-null). NUL
/// bytes are stripped so `CString::new` cannot fail.
fn record_error(ctx: *const sc_context_t, message: &str) {
    if ctx.is_null() {
        return;
    }
    let ctx = unsafe { &*ctx };
    let sanitized = message.replace('\0', " ");
    let cstr = CString::new(sanitized).unwrap_or_else(|_| CString::new("error").unwrap());
    *ctx.last_error.borrow_mut() = Some(cstr);
}

fn panic_message(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Run `f` behind the panic barrier, mapping its outcome to a status code and
/// recording any error message on `ctx`. Panics become `SC_ERR_PANIC`; they
/// never unwind across the FFI boundary.
fn guard<F>(ctx: *const sc_context_t, f: F) -> sc_status_t
where
    F: FnOnce() -> Result<(), ApiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SC_OK,
        Ok(Err(e)) => {
            record_error(ctx, &e.message);
            e.code
        }
        Err(p) => {
            record_error(ctx, &format!("panic: {}", panic_message(p)));
            SC_ERR_PANIC
        }
    }
}

/// Panic barrier for the `void`-returning `free` functions: a panic in a
/// destructor is swallowed rather than allowed to unwind across FFI (which would
/// abort the host). Nothing to report — these have no status/context.
fn guard_void<F: FnOnce()>(f: F) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

/// Panic barrier for `sc_context_last_error`: on a panic, return NULL rather
/// than unwind across FFI.
fn guard_cstr<F: FnOnce() -> *const c_char>(f: F) -> *const c_char {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(std::ptr::null())
}

/// Null-check a raw pointer and borrow it, or error with `what`.
unsafe fn req_ref<'a, T>(p: *const T, what: &str) -> Result<&'a T, ApiError> {
    if p.is_null() {
        Err(ApiError::null(what))
    } else {
        Ok(&*p)
    }
}

/// Run a (possibly-parallel) sc-core computation on the context's pool, or the
/// global pool if the context has none.
fn run_in_pool<R, G>(ctx: &sc_context_t, g: G) -> R
where
    G: FnOnce() -> R + Send,
    R: Send,
{
    match &ctx.pool {
        Some(pool) => pool.install(g),
        None => g(),
    }
}

// ---------------------------------------------------------------------------
// Import helpers (borrow the caller's table in place)
// ---------------------------------------------------------------------------

unsafe fn import_matrix(table: &sc_coo_table_t) -> Result<FeatureMatrix, ApiError> {
    Ok(sc_arrow::import_coo(
        table.n_samples,
        table.n_features,
        &table.rows,
        &table.rows_schema,
        &table.cols,
        &table.cols_schema,
        &table.vals,
        &table.vals_schema,
    )?)
}

/// If the model was fit with RFE, its forest was trained on a feature *subset*;
/// project `x` onto those original-feature columns before predicting. Otherwise
/// borrow `x` unchanged.
fn features_for<'a>(
    bundle: &ModelBundle,
    x: &'a FeatureMatrix,
) -> Result<Cow<'a, FeatureMatrix>, ApiError> {
    match &bundle.rfe_selection {
        None => Ok(Cow::Borrowed(x)),
        Some(sel) => {
            for &c in sel {
                if c >= x.n_features() {
                    return Err(ApiError::new(
                        SC_ERR_INVALID_INPUT,
                        format!(
                            "predict: rfe-selected column {c} >= input n_features {}",
                            x.n_features()
                        ),
                    ));
                }
            }
            Ok(Cow::Owned(x.select_columns(sel)))
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_context_new(
    config: *const sc_config_t,
    out: *mut *mut sc_context_t,
) -> sc_status_t {
    guard(std::ptr::null(), || {
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        let n_threads = if config.is_null() {
            0
        } else {
            (*config).n_threads
        };
        let pool = if n_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads as usize)
                    .build()
                    .map_err(|e| {
                        ApiError::new(SC_ERR_INVALID_INPUT, format!("thread pool: {e}"))
                    })?,
            )
        } else {
            None
        };
        let ctx = Box::new(sc_context_t {
            pool,
            last_error: RefCell::new(None),
        });
        *out = Box::into_raw(ctx);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn sc_context_free(ctx: *mut sc_context_t) {
    guard_void(|| {
        if !ctx.is_null() {
            drop(Box::from_raw(ctx));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn sc_context_last_error(ctx: *const sc_context_t) -> *const c_char {
    guard_cstr(|| {
        if ctx.is_null() {
            return std::ptr::null();
        }
        // The returned pointer aliases the stored CString; it is valid until the
        // next call that records an error on this context, or until the context
        // is freed (standard C last-error lifetime).
        match &*(*ctx).last_error.borrow() {
            Some(s) => s.as_ptr(),
            None => std::ptr::null(),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn sc_model_free(model: *mut sc_model_t) {
    guard_void(|| {
        if !model.is_null() {
            drop(Box::from_raw(model));
        }
    });
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_fit_classifier(
    ctx: *mut sc_context_t,
    table: *const sc_coo_table_t,
    targets: *const FFI_ArrowArray,
    targets_schema: *const FFI_ArrowSchema,
    params: *const sc_rf_params_t,
    out_model: *mut *mut sc_model_t,
) -> sc_status_t {
    guard(ctx, || {
        fit_impl(ctx, table, targets, targets_schema, params, out_model, true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn sc_fit_regressor(
    ctx: *mut sc_context_t,
    table: *const sc_coo_table_t,
    targets: *const FFI_ArrowArray,
    targets_schema: *const FFI_ArrowSchema,
    params: *const sc_rf_params_t,
    out_model: *mut *mut sc_model_t,
) -> sc_status_t {
    guard(ctx, || {
        fit_impl(
            ctx,
            table,
            targets,
            targets_schema,
            params,
            out_model,
            false,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn fit_impl(
    ctx: *mut sc_context_t,
    table: *const sc_coo_table_t,
    targets: *const FFI_ArrowArray,
    targets_schema: *const FFI_ArrowSchema,
    params: *const sc_rf_params_t,
    out_model: *mut *mut sc_model_t,
    is_classification: bool,
) -> Result<(), ApiError> {
    let ctx = req_ref(ctx, "ctx")?;
    let table = req_ref(table, "table")?;
    let targets = req_ref(targets, "targets")?;
    let targets_schema = req_ref(targets_schema, "targets_schema")?;
    let params = req_ref(params, "params")?;
    if out_model.is_null() {
        return Err(ApiError::null("out_model"));
    }

    let x = import_matrix(table)?;
    let y = if is_classification {
        sc_arrow::import_targets_categorical(targets, targets_schema)?
    } else {
        sc_arrow::import_targets_numeric(targets, targets_schema)?
    };
    let feature_ids = sc_arrow::import_optional_ids(&table.feature_ids, &table.feature_ids_schema)?;
    let seed = params.random_state;

    let mut bundle = if params.parameter_tuning {
        let tp = convert::tuning_params(params, is_classification);
        let (bundle, _) = run_in_pool(ctx, || sc_core::fit_tuned(&x, &y, &tp, seed))?;
        bundle
    } else if params.optimize_feature_selection {
        let rp = convert::rfecv_params(params);
        let (bundle, _) = run_in_pool(ctx, || sc_core::rfe::fit_selected(&x, &y, &rp, seed))?;
        bundle
    } else {
        let rf = convert::rf_params(params);
        let forest = run_in_pool(ctx, || RandomForest::fit(&x, &y, &rf, seed))?;
        ModelBundle {
            forest,
            params: Some(rf),
            feature_ids: None,
            rfe_selection: None,
        }
    };
    // The tuning / RFE helpers don't carry feature ids; attach the imported set.
    if bundle.feature_ids.is_none() {
        bundle.feature_ids = feature_ids;
    }

    *out_model = Box::into_raw(Box::new(sc_model_t { bundle }));
    Ok(())
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_predict(
    ctx: *mut sc_context_t,
    model: *const sc_model_t,
    table: *const sc_coo_table_t,
    out_pred: *mut FFI_ArrowArray,
    out_pred_schema: *mut FFI_ArrowSchema,
) -> sc_status_t {
    guard(ctx, || {
        let _ctx = req_ref(ctx, "ctx")?;
        let model = req_ref(model, "model")?;
        let table = req_ref(table, "table")?;
        if out_pred.is_null() || out_pred_schema.is_null() {
            return Err(ApiError::null("out_pred"));
        }
        let x = import_matrix(table)?;
        let xf = features_for(&model.bundle, &x)?;
        let forest = &model.bundle.forest;
        match forest.task() {
            Task::Classification { .. } => {
                let idx = forest.predict_class_indices(&xf)?;
                let classes = forest.classes().expect("classifier has classes");
                let labels: Vec<String> = idx.iter().map(|&i| classes[i].clone()).collect();
                sc_arrow::export_utf8(&labels, out_pred, out_pred_schema)?;
            }
            Task::Regression => {
                let vals = forest.predict_values(&xf)?;
                sc_arrow::export_f64(&vals, out_pred, out_pred_schema)?;
            }
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn sc_predict_proba(
    ctx: *mut sc_context_t,
    model: *const sc_model_t,
    table: *const sc_coo_table_t,
    out_proba: *mut FFI_ArrowArray,
    out_proba_schema: *mut FFI_ArrowSchema,
    out_classes: *mut FFI_ArrowArray,
    out_classes_schema: *mut FFI_ArrowSchema,
) -> sc_status_t {
    guard(ctx, || {
        let _ctx = req_ref(ctx, "ctx")?;
        let model = req_ref(model, "model")?;
        let table = req_ref(table, "table")?;
        if out_proba.is_null()
            || out_proba_schema.is_null()
            || out_classes.is_null()
            || out_classes_schema.is_null()
        {
            return Err(ApiError::null("out_proba/out_classes"));
        }
        // Put both out-slots in a fully releasable (empty) state before any real
        // export, so an error partway through never leaves a live array beside
        // uninitialized memory (a caller's "release every field" cleanup stays
        // safe regardless of the status code).
        sc_arrow::write_empty(out_proba, out_proba_schema);
        sc_arrow::write_empty(out_classes, out_classes_schema);
        let forest = &model.bundle.forest;
        // Guard the regressor case *before* calling predict_proba (which panics
        // on a regressor) — return a clean error instead.
        let n_classes = match forest.task() {
            Task::Classification { n_classes } => n_classes,
            Task::Regression => {
                return Err(ApiError::new(
                    SC_ERR_INVALID_INPUT,
                    "predict_proba: model is a regressor",
                ))
            }
        };
        let x = import_matrix(table)?;
        let xf = features_for(&model.bundle, &x)?;
        let proba = forest.predict_proba(&xf)?;
        sc_arrow::export_proba(&proba, n_classes, out_proba, out_proba_schema)?;
        let classes: Vec<String> = forest.classes().expect("classifier has classes").to_vec();
        sc_arrow::export_utf8(&classes, out_classes, out_classes_schema)?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Cross-validate (OOF)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_cross_validate(
    ctx: *mut sc_context_t,
    table: *const sc_coo_table_t,
    targets: *const FFI_ArrowArray,
    targets_schema: *const FFI_ArrowSchema,
    params: *const sc_rf_params_t,
    out_result: *mut sc_cv_result_t,
) -> sc_status_t {
    guard(ctx, || {
        let ctx = req_ref(ctx, "ctx")?;
        let table = req_ref(table, "table")?;
        let targets = req_ref(targets, "targets")?;
        let targets_schema = req_ref(targets_schema, "targets_schema")?;
        let params = req_ref(params, "params")?;
        if out_result.is_null() {
            return Err(ApiError::null("out_result"));
        }
        // Put the whole result struct in a fully releasable (empty) state before
        // any import / fit / export, so any early return leaves every field
        // well-defined and safe to release — never a live array beside
        // uninitialized memory.
        init_empty_cv_result(out_result);
        // Dispatch classification vs regression on the target dtype.
        let (y, is_classification) = match targets_schema.format() {
            "u" => (
                sc_arrow::import_targets_categorical(targets, targets_schema)?,
                true,
            ),
            "g" => (
                sc_arrow::import_targets_numeric(targets, targets_schema)?,
                false,
            ),
            other => {
                return Err(ApiError::new(
                    SC_ERR_ARROW,
                    format!("cross_validate: unsupported target Arrow format {other:?}"),
                ))
            }
        };
        let x = import_matrix(table)?;
        let np = convert::ncv_params(params, is_classification);
        let seed = params.random_state;
        let result = run_in_pool(ctx, || ncv(&x, &y, &np, seed))?;
        write_cv_result(&result, out_result)?;
        Ok(())
    })
}

/// Fill every field of an (uninitialized) `sc_cv_result_t` with empty/released
/// sentinels + zero scores, so a partial failure leaves the whole struct
/// well-defined and safely releasable.
unsafe fn init_empty_cv_result(out: *mut sc_cv_result_t) {
    use std::ptr::addr_of_mut;
    sc_arrow::write_empty(
        addr_of_mut!((*out).predictions),
        addr_of_mut!((*out).predictions_schema),
    );
    sc_arrow::write_empty(
        addr_of_mut!((*out).proba),
        addr_of_mut!((*out).proba_schema),
    );
    sc_arrow::write_empty(
        addr_of_mut!((*out).classes),
        addr_of_mut!((*out).classes_schema),
    );
    sc_arrow::write_empty(
        addr_of_mut!((*out).fold_scores),
        addr_of_mut!((*out).fold_scores_schema),
    );
    addr_of_mut!((*out).mean_score).write(0.0);
    addr_of_mut!((*out).std_score).write(0.0);
}

unsafe fn write_cv_result(
    result: &NcvResult,
    out: *mut sc_cv_result_t,
) -> Result<(), sc_arrow::MarshalError> {
    use std::ptr::addr_of_mut;
    match &result.predictions {
        OofPredictions::Classification {
            classes,
            codes,
            proba,
        } => {
            let labels: Vec<String> = codes.iter().map(|&c| classes[c].clone()).collect();
            sc_arrow::export_utf8(
                &labels,
                addr_of_mut!((*out).predictions),
                addr_of_mut!((*out).predictions_schema),
            )?;
            sc_arrow::export_proba(
                proba,
                classes.len(),
                addr_of_mut!((*out).proba),
                addr_of_mut!((*out).proba_schema),
            )?;
            sc_arrow::export_utf8(
                classes,
                addr_of_mut!((*out).classes),
                addr_of_mut!((*out).classes_schema),
            )?;
        }
        OofPredictions::Regression { values } => {
            sc_arrow::export_f64(
                values,
                addr_of_mut!((*out).predictions),
                addr_of_mut!((*out).predictions_schema),
            )?;
            sc_arrow::write_empty(
                addr_of_mut!((*out).proba),
                addr_of_mut!((*out).proba_schema),
            );
            sc_arrow::write_empty(
                addr_of_mut!((*out).classes),
                addr_of_mut!((*out).classes_schema),
            );
        }
    }
    sc_arrow::export_f64(
        &result.fold_scores,
        addr_of_mut!((*out).fold_scores),
        addr_of_mut!((*out).fold_scores_schema),
    )?;
    addr_of_mut!((*out).mean_score).write(result.mean_score);
    addr_of_mut!((*out).std_score).write(result.std_score);
    Ok(())
}

// ---------------------------------------------------------------------------
// Feature importances
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_feature_importances(
    model: *const sc_model_t,
    out: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> sc_status_t {
    guard(std::ptr::null(), || {
        let model = req_ref(model, "model")?;
        if out.is_null() || out_schema.is_null() {
            return Err(ApiError::null("out"));
        }
        let fi = model.bundle.forest.feature_importances();
        sc_arrow::export_f64(&fi, out, out_schema)?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Persistence (byte buffers only)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sc_model_serialize(
    model: *const sc_model_t,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> sc_status_t {
    guard(std::ptr::null(), || {
        let model = req_ref(model, "model")?;
        if out_ptr.is_null() || out_len.is_null() {
            return Err(ApiError::null("out_ptr/out_len"));
        }
        let bytes = model.bundle.to_bytes().into_boxed_slice();
        let len = bytes.len();
        *out_len = len;
        // Hand the buffer's raw data pointer to the caller; reclaimed via
        // sc_buffer_free (which rebuilds the same Box<[u8]>).
        *out_ptr = Box::into_raw(bytes) as *mut u8;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn sc_buffer_free(ptr: *mut u8, len: usize) {
    guard_void(|| {
        if ptr.is_null() {
            return;
        }
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice as *mut [u8]));
    });
}

#[no_mangle]
pub unsafe extern "C" fn sc_model_deserialize(
    ctx: *mut sc_context_t,
    ptr: *const u8,
    len: usize,
    out_model: *mut *mut sc_model_t,
) -> sc_status_t {
    guard(ctx, || {
        let _ctx = req_ref(ctx, "ctx")?;
        if ptr.is_null() {
            return Err(ApiError::null("ptr"));
        }
        if out_model.is_null() {
            return Err(ApiError::null("out_model"));
        }
        let bytes = std::slice::from_raw_parts(ptr, len);
        let bundle = ModelBundle::from_bytes(bytes)?;
        *out_model = Box::into_raw(Box::new(sc_model_t { bundle }));
        Ok(())
    })
}

#[cfg(test)]
mod tests;
