//! Arrow C Data Interface marshaling between the pure [`sc_core`] engine and
//! native (C/C++) callers — the *mechanical* layer of the M9 FFI boundary.
//!
//! This crate is deliberately thin and policy-free: it converts Arrow arrays to
//! and from sc-core's [`FeatureMatrix`] / [`Targets`] / result vectors and
//! nothing more. Opaque handles, the panic barrier, error-code mapping, and
//! workflow orchestration live one layer up in `sc-capi`.
//!
//! All `unsafe` in the `sc` project is confined to this crate and `sc-capi`;
//! `sc-core` stays `#![forbid(unsafe_code)]`. Two ownership contracts, matching
//! the Arrow C Data Interface producer/consumer split (design §5/§6):
//!
//! * **Import borrows, never moves.** [`import_coo`] / the `import_targets_*`
//!   functions read the caller's `FFI_ArrowArray` buffers *in place* and never
//!   invoke the array's `release` callback — the caller keeps ownership of its
//!   inputs. Imports require `null_count == 0` and `offset == 0` (microbiome COO
//!   tables and target vectors are dense and unsliced); anything else is a
//!   [`MarshalError::Arrow`].
//! * **Export owns, then hands off.** The `export_*` functions build a fresh
//!   arrow-rs array, `to_ffi` it (installing arrow-rs's `release` callback), and
//!   write it into the caller's out-slot. The **consumer** releases it.

use std::sync::Arc;

use arrow_array::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Array, FixedSizeListArray, Float64Array, StringArray};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field};

use sc_core::{FeatureMatrix, Targets};

// Arrow C Data Interface format codes (spec §"Data type description — format
// strings"). We accept exactly these three physical layouts at the boundary.
const FMT_INT64: &str = "l";
const FMT_FLOAT64: &str = "g";
const FMT_UTF8: &str = "u";

/// A failure marshaling data across the Arrow boundary.
///
/// `Arrow` = the C Data Interface structure was malformed or an unsupported
/// layout (wrong dtype, nulls, a non-zero offset, a bad buffer). `Core` = the
/// data was well-formed Arrow but `sc-core` rejected its *content*. `sc-capi`
/// maps the two to `SC_ERR_ARROW` and `SC_ERR_INVALID_INPUT` respectively.
#[derive(Debug)]
pub enum MarshalError {
    /// Malformed / unsupported Arrow structure.
    Arrow(String),
    /// Well-formed Arrow, but `sc-core` rejected the content.
    Core(sc_core::Error),
}

impl MarshalError {
    fn arrow(msg: impl Into<String>) -> Self {
        MarshalError::Arrow(msg.into())
    }
}

impl From<sc_core::Error> for MarshalError {
    fn from(e: sc_core::Error) -> Self {
        MarshalError::Core(e)
    }
}

impl std::fmt::Display for MarshalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarshalError::Arrow(m) => write!(f, "arrow: {m}"),
            MarshalError::Core(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MarshalError {}

// ---------------------------------------------------------------------------
// Import (borrow in place)
// ---------------------------------------------------------------------------

/// Read the pointer stored at `buffers[index]` of an Arrow array.
///
/// # Safety
/// `a` must be a well-formed `FFI_ArrowArray` (its `buffers` array holds at
/// least `n_buffers` pointers). The returned pointer is only read here — the
/// caller must keep `a` alive for as long as it is used.
unsafe fn buffer_ptr(a: &FFI_ArrowArray, index: usize) -> Result<*const u8, MarshalError> {
    if a.buffers.is_null() || (a.n_buffers as usize) <= index {
        return Err(MarshalError::arrow(format!(
            "array has {} buffers; need at least {}",
            a.n_buffers,
            index + 1
        )));
    }
    // The buffers array is a C array of `*const void`; read the pointer at
    // `index` (unaligned to match arrow-rs's own consumer, though a conformant
    // producer aligns it).
    Ok(std::ptr::read_unaligned(
        (a.buffers as *const *const u8).add(index),
    ))
}

/// Reject a primitive array that is not the dense, unsliced, non-null layout we
/// require, and check its dtype against `fmt`.
fn check_primitive(
    a: &FFI_ArrowArray,
    s: &FFI_ArrowSchema,
    fmt: &str,
    name: &str,
) -> Result<(), MarshalError> {
    if s.format() != fmt {
        return Err(MarshalError::arrow(format!(
            "{name}: expected Arrow format {fmt:?}, got {:?}",
            s.format()
        )));
    }
    if a.null_count != 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: null_count {} must be 0 (nulls unsupported at the boundary)",
            a.null_count
        )));
    }
    if a.offset != 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: offset {} must be 0 (sliced arrays unsupported)",
            a.offset
        )));
    }
    if a.length < 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: negative length {}",
            a.length
        )));
    }
    Ok(())
}

/// Borrow a dense, non-null `Int64`/`Float64` data buffer as a `&[T]`.
///
/// # Safety
/// `a`/`s` must be a matching, well-formed array/schema pair; the returned
/// slice borrows the caller's buffer and must not outlive it.
unsafe fn borrow_primitive<'a, T: Copy>(
    a: &'a FFI_ArrowArray,
    s: &FFI_ArrowSchema,
    fmt: &str,
    name: &str,
) -> Result<&'a [T], MarshalError> {
    check_primitive(a, s, fmt, name)?;
    let n = a.length as usize;
    if n == 0 {
        return Ok(&[]);
    }
    // buffers = [validity, data]; the data buffer is index 1.
    let p = buffer_ptr(a, 1)? as *const T;
    if p.is_null() {
        return Err(MarshalError::arrow(format!("{name}: null data buffer")));
    }
    if !(p as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(MarshalError::arrow(format!(
            "{name}: data buffer is not {}-byte aligned",
            std::mem::align_of::<T>()
        )));
    }
    Ok(std::slice::from_raw_parts(p, n))
}

/// Borrow a dense, non-null `Utf8` array as owned `String`s.
///
/// Returns owned strings (not borrowed `&str`) so the result can outlive the
/// borrowed buffers — used for `feature_ids`/`sample_ids` that flow into a
/// persisted model.
///
/// # Safety
/// `a`/`s` must be a matching, well-formed array/schema pair kept alive for the
/// duration of the call.
unsafe fn borrow_utf8(
    a: &FFI_ArrowArray,
    s: &FFI_ArrowSchema,
    name: &str,
) -> Result<Vec<String>, MarshalError> {
    if s.format() != FMT_UTF8 {
        return Err(MarshalError::arrow(format!(
            "{name}: expected Arrow format {FMT_UTF8:?} (utf8), got {:?}",
            s.format()
        )));
    }
    if a.null_count != 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: null_count {} must be 0",
            a.null_count
        )));
    }
    if a.offset != 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: offset {} must be 0",
            a.offset
        )));
    }
    if a.length < 0 {
        return Err(MarshalError::arrow(format!(
            "{name}: negative length {}",
            a.length
        )));
    }
    let n = a.length as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    // buffers = [validity, offsets(i32), data(u8)].
    let off_p = buffer_ptr(a, 1)? as *const i32;
    if off_p.is_null() {
        return Err(MarshalError::arrow(format!("{name}: null offsets buffer")));
    }
    if !(off_p as usize).is_multiple_of(std::mem::align_of::<i32>()) {
        return Err(MarshalError::arrow(format!(
            "{name}: offsets buffer is not 4-byte aligned"
        )));
    }
    let offsets = std::slice::from_raw_parts(off_p, n + 1);
    let total = offsets[n];
    if total < 0 || offsets[0] != 0 {
        return Err(MarshalError::arrow(format!("{name}: malformed offsets")));
    }
    let data_p = buffer_ptr(a, 2)?;
    let data: &[u8] = if total == 0 || data_p.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(data_p, total as usize)
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (lo, hi) = (offsets[i], offsets[i + 1]);
        if lo < 0 || hi < lo || hi as usize > data.len() {
            return Err(MarshalError::arrow(format!(
                "{name}: offsets out of range at index {i}"
            )));
        }
        let s = std::str::from_utf8(&data[lo as usize..hi as usize])
            .map_err(|e| MarshalError::arrow(format!("{name}: invalid utf8 at index {i}: {e}")))?;
        out.push(s.to_owned());
    }
    Ok(out)
}

/// Import a COO feature matrix by borrowing the three component buffers in place
/// (`rows`/`cols` `Int64`, `vals` `Float64`). Rows are samples, cols are
/// features. Delegates content validation (dimension/range/length) to
/// [`FeatureMatrix::from_coo`].
///
/// # Safety
/// Every array/schema pair must be a well-formed, matched `FFI_ArrowArray` /
/// `FFI_ArrowSchema` that outlives the call. The caller retains ownership; this
/// never calls any `release` callback.
#[allow(clippy::too_many_arguments)]
pub unsafe fn import_coo(
    n_samples: i64,
    n_features: i64,
    rows: &FFI_ArrowArray,
    rows_s: &FFI_ArrowSchema,
    cols: &FFI_ArrowArray,
    cols_s: &FFI_ArrowSchema,
    vals: &FFI_ArrowArray,
    vals_s: &FFI_ArrowSchema,
) -> Result<FeatureMatrix, MarshalError> {
    let ns = usize::try_from(n_samples)
        .map_err(|_| MarshalError::arrow(format!("negative n_samples {n_samples}")))?;
    let nf = usize::try_from(n_features)
        .map_err(|_| MarshalError::arrow(format!("negative n_features {n_features}")))?;
    let r: &[i64] = borrow_primitive(rows, rows_s, FMT_INT64, "rows")?;
    let c: &[i64] = borrow_primitive(cols, cols_s, FMT_INT64, "cols")?;
    let v: &[f64] = borrow_primitive(vals, vals_s, FMT_FLOAT64, "vals")?;
    Ok(FeatureMatrix::from_coo(ns, nf, r, c, v)?)
}

/// Import a `Utf8` target array into categorical [`Targets`] (sorted-unique
/// encoding, aligned to the COO sample order).
///
/// # Safety
/// See [`import_coo`]: `a`/`s` must be a well-formed, matched pair.
pub unsafe fn import_targets_categorical(
    a: &FFI_ArrowArray,
    s: &FFI_ArrowSchema,
) -> Result<Targets, MarshalError> {
    let labels = borrow_utf8(a, s, "targets")?;
    Ok(Targets::categorical(&labels)?)
}

/// Import a `Float64` target array into numeric [`Targets`].
///
/// # Safety
/// See [`import_coo`]: `a`/`s` must be a well-formed, matched pair.
pub unsafe fn import_targets_numeric(
    a: &FFI_ArrowArray,
    s: &FFI_ArrowSchema,
) -> Result<Targets, MarshalError> {
    let v: &[f64] = borrow_primitive(a, s, FMT_FLOAT64, "targets")?;
    Ok(Targets::numeric(v)?)
}

/// Import an *optional* `Utf8` id array (e.g. `feature_ids`). A released array
/// (`release == NULL`, the Arrow convention for "absent") returns `None`.
///
/// # Safety
/// See [`import_coo`]: `a`/`s` must be a well-formed, matched pair (or an
/// absent/released array).
pub unsafe fn import_optional_ids(
    a: &FFI_ArrowArray,
    s: &FFI_ArrowSchema,
) -> Result<Option<Vec<String>>, MarshalError> {
    if a.release.is_none() {
        return Ok(None);
    }
    Ok(Some(borrow_utf8(a, s, "ids")?))
}

// ---------------------------------------------------------------------------
// Export (own, then hand off with a release callback)
// ---------------------------------------------------------------------------

/// `to_ffi` `data` and write the owning `FFI_ArrowArray`/`FFI_ArrowSchema` into
/// the caller's out-slots. The arrow-rs `release` callback travels with them;
/// the consumer releases.
///
/// # Safety
/// `out_a`/`out_s` must be valid, writable, uninitialized (or don't-care)
/// out-slots. Their previous contents are overwritten without being dropped.
unsafe fn write_ffi(
    data: ArrayData,
    out_a: *mut FFI_ArrowArray,
    out_s: *mut FFI_ArrowSchema,
) -> Result<(), MarshalError> {
    let (ffi_a, ffi_s) = to_ffi(&data).map_err(|e| MarshalError::arrow(e.to_string()))?;
    std::ptr::write(out_a, ffi_a);
    std::ptr::write(out_s, ffi_s);
    Ok(())
}

/// Export a `Float64` array (regression predictions, feature importances, fold
/// scores).
///
/// # Safety
/// See [`write_ffi`].
pub unsafe fn export_f64(
    values: &[f64],
    out_a: *mut FFI_ArrowArray,
    out_s: *mut FFI_ArrowSchema,
) -> Result<(), MarshalError> {
    write_ffi(
        Float64Array::from(values.to_vec()).into_data(),
        out_a,
        out_s,
    )
}

/// Export a `Utf8` array (classification prediction labels, class labels).
///
/// # Safety
/// See [`write_ffi`].
pub unsafe fn export_utf8(
    values: &[String],
    out_a: *mut FFI_ArrowArray,
    out_s: *mut FFI_ArrowSchema,
) -> Result<(), MarshalError> {
    let arr = StringArray::from_iter_values(values.iter());
    write_ffi(arr.into_data(), out_a, out_s)
}

/// Export equal-width `Float64` rows as `FixedSizeList<Float64>[width]`, one list
/// per row (row-major flatten). Used for soft-vote probabilities
/// (`width == n_classes`) and per-sample TreeSHAP rows
/// (`width == n_outputs * n_features`). `width` must be `>= 1` and every row must
/// have exactly that width.
///
/// # Safety
/// See [`write_ffi`].
pub unsafe fn export_fixed_size_list(
    rows: &[Vec<f64>],
    width: usize,
    out_a: *mut FFI_ArrowArray,
    out_s: *mut FFI_ArrowSchema,
) -> Result<(), MarshalError> {
    if width == 0 {
        return Err(MarshalError::arrow("fixed_size_list: width must be >= 1"));
    }
    let mut flat = Vec::with_capacity(rows.len() * width);
    for (i, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(MarshalError::arrow(format!(
                "fixed_size_list: row {i} width {} != {width}",
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    export_fixed_size_list_flat(flat, width, out_a, out_s)
}

/// Export an already row-major buffer of `n_rows * width` doubles as
/// `FixedSizeList<Float64>[width]`, taking ownership: the `Vec` becomes the
/// child array's buffer without a copy. Use it when the caller already holds the
/// flat layout — for TreeSHAP's samples × outputs × features matrix, the
/// row-slice form above would hold the data twice while exporting.
///
/// # Safety
/// See [`write_ffi`].
pub unsafe fn export_fixed_size_list_flat(
    values: Vec<f64>,
    width: usize,
    out_a: *mut FFI_ArrowArray,
    out_s: *mut FFI_ArrowSchema,
) -> Result<(), MarshalError> {
    if width == 0 {
        return Err(MarshalError::arrow("fixed_size_list: width must be >= 1"));
    }
    if !values.len().is_multiple_of(width) {
        return Err(MarshalError::arrow(format!(
            "fixed_size_list: {} values is not a whole number of rows of width {width}",
            values.len()
        )));
    }
    let values = Arc::new(Float64Array::from(values));
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    let list = FixedSizeListArray::try_new(field, width as i32, values, None)
        .map_err(|e| MarshalError::arrow(e.to_string()))?;
    write_ffi(list.into_data(), out_a, out_s)
}

/// Write an *empty/released* array+schema into the out-slots — the sentinel for
/// "no data here" (e.g. the probability + class-label slots of a regression
/// cross-validation result). The consumer sees `release == NULL`.
///
/// # Safety
/// `out_a`/`out_s` must be valid, writable out-slots (contents overwritten
/// without being dropped).
pub unsafe fn write_empty(out_a: *mut FFI_ArrowArray, out_s: *mut FFI_ArrowSchema) {
    std::ptr::write(out_a, FFI_ArrowArray::empty());
    std::ptr::write(out_s, FFI_ArrowSchema::empty());
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ffi::from_ffi;
    use arrow_array::{Int64Array, Int8Array};
    use std::mem::MaybeUninit;

    // Build FFI array+schema from an arrow-rs array (a well-formed producer).
    fn to_ffi_pair(arr: impl Array) -> (FFI_ArrowArray, FFI_ArrowSchema) {
        to_ffi(&arr.into_data()).unwrap()
    }

    #[test]
    fn imports_coo_equal_to_direct_from_coo() {
        let (rows, rs) = to_ffi_pair(Int64Array::from(vec![0i64, 2, 1]));
        let (cols, cs) = to_ffi_pair(Int64Array::from(vec![0i64, 1, 0]));
        let (vals, vs) = to_ffi_pair(Float64Array::from(vec![5.0, 7.0, 9.0]));
        let got = unsafe { import_coo(3, 2, &rows, &rs, &cols, &cs, &vals, &vs).unwrap() };
        let direct =
            FeatureMatrix::from_coo(3, 2, &[0, 2, 1], &[0, 1, 0], &[5.0, 7.0, 9.0]).unwrap();
        assert_eq!(got, direct);
    }

    #[test]
    fn rejects_wrong_dtype() {
        // rows given a Float64 array where Int64 is required.
        let (rows, rs) = to_ffi_pair(Float64Array::from(vec![0.0, 1.0]));
        let (cols, cs) = to_ffi_pair(Int64Array::from(vec![0i64, 1]));
        let (vals, vs) = to_ffi_pair(Float64Array::from(vec![1.0, 2.0]));
        let err = unsafe { import_coo(2, 2, &rows, &rs, &cols, &cs, &vals, &vs).unwrap_err() };
        assert!(matches!(err, MarshalError::Arrow(_)), "got {err:?}");
    }

    #[test]
    fn rejects_nulls_and_offsets() {
        // An array with a null → null_count != 0 → rejected.
        let (with_null, ws) = to_ffi_pair(Int64Array::from(vec![Some(1i64), None, Some(3)]));
        let (cols, cs) = to_ffi_pair(Int64Array::from(vec![0i64, 0, 0]));
        let (vals, vs) = to_ffi_pair(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let err = unsafe { import_coo(3, 1, &with_null, &ws, &cols, &cs, &vals, &vs).unwrap_err() };
        assert!(matches!(err, MarshalError::Arrow(_)), "got {err:?}");

        // offset != 0 → rejected. arrow-rs's `to_ffi` bakes any slice offset into
        // the exported buffers (normalizing offset to 0), so set the field
        // directly to exercise the guard — a raw C producer can present one. The
        // offset check fires before any buffer is read, so this is safe.
        let (mut rows, rs) = to_ffi_pair(Int64Array::from(vec![0i64, 1]));
        rows.offset = 1;
        let (cols2, cs2) = to_ffi_pair(Int64Array::from(vec![0i64, 0]));
        let (vals2, vs2) = to_ffi_pair(Float64Array::from(vec![1.0, 2.0]));
        let err = unsafe { import_coo(4, 1, &rows, &rs, &cols2, &cs2, &vals2, &vs2).unwrap_err() };
        assert!(matches!(err, MarshalError::Arrow(_)), "got {err:?}");
    }

    #[test]
    fn imports_categorical_and_numeric_targets() {
        let (arr, s) = to_ffi_pair(StringArray::from(vec!["b", "a", "b", "c"]));
        let t = unsafe { import_targets_categorical(&arr, &s).unwrap() };
        assert_eq!(t, Targets::categorical(&["b", "a", "b", "c"]).unwrap());

        let (arr, s) = to_ffi_pair(Float64Array::from(vec![1.5, -2.0, 3.25]));
        let t = unsafe { import_targets_numeric(&arr, &s).unwrap() };
        assert_eq!(t, Targets::numeric(&[1.5, -2.0, 3.25]).unwrap());
    }

    #[test]
    fn optional_ids_absent_is_none() {
        let a = FFI_ArrowArray::empty();
        let s = FFI_ArrowSchema::empty();
        let got = unsafe { import_optional_ids(&a, &s).unwrap() };
        assert_eq!(got, None);

        let (arr, sch) = to_ffi_pair(StringArray::from(vec!["f0", "f1"]));
        let got = unsafe { import_optional_ids(&arr, &sch).unwrap() };
        assert_eq!(got, Some(vec!["f0".to_string(), "f1".into()]));
    }

    #[test]
    fn rejects_unsupported_dtype_for_targets() {
        // Int8 targets: neither Float64 (numeric) nor Utf8 (categorical).
        let (arr, s) = to_ffi_pair(Int8Array::from(vec![1i8, 2, 3]));
        assert!(matches!(
            unsafe { import_targets_numeric(&arr, &s) },
            Err(MarshalError::Arrow(_))
        ));
        assert!(matches!(
            unsafe { import_targets_categorical(&arr, &s) },
            Err(MarshalError::Arrow(_))
        ));
    }

    // Round-trip an exported array back through from_ffi (which installs a
    // consumer that calls the release callback on drop — proves the callback is
    // valid and the values survive).
    #[test]
    fn exports_f64_round_trip() {
        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let values = [1.0, 2.5, -3.0];
        unsafe {
            export_f64(&values, a.as_mut_ptr(), s.as_mut_ptr()).unwrap();
            let a = a.assume_init();
            let s = s.assume_init();
            let data = from_ffi(a, &s).unwrap();
            let arr = Float64Array::from(data);
            assert_eq!(arr.values(), &values);
        }
    }

    #[test]
    fn exports_utf8_round_trip() {
        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let labels = vec!["x".to_string(), "yy".into(), "".into()];
        unsafe {
            export_utf8(&labels, a.as_mut_ptr(), s.as_mut_ptr()).unwrap();
            let data = from_ffi(a.assume_init(), &s.assume_init()).unwrap();
            let arr = StringArray::from(data);
            let got: Vec<&str> = (0..arr.len()).map(|i| arr.value(i)).collect();
            assert_eq!(got, vec!["x", "yy", ""]);
        }
    }

    #[test]
    fn exports_fixed_size_list() {
        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let rows = vec![vec![0.75, 0.25], vec![0.1, 0.9]];
        unsafe {
            export_fixed_size_list(&rows, 2, a.as_mut_ptr(), s.as_mut_ptr()).unwrap();
            let data = from_ffi(a.assume_init(), &s.assume_init()).unwrap();
            let list = FixedSizeListArray::from(data);
            assert_eq!(list.len(), 2);
            assert_eq!(list.value_length(), 2);
            let flat = list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(flat.values(), &[0.75, 0.25, 0.1, 0.9]);
        }
        // Wrong-width row is rejected.
        let bad = vec![vec![0.5, 0.5], vec![1.0]];
        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        let err =
            unsafe { export_fixed_size_list(&bad, 2, a.as_mut_ptr(), s.as_mut_ptr()).unwrap_err() };
        assert!(matches!(err, MarshalError::Arrow(_)));
    }

    #[test]
    fn write_empty_is_released() {
        let mut a = MaybeUninit::<FFI_ArrowArray>::uninit();
        let mut s = MaybeUninit::<FFI_ArrowSchema>::uninit();
        unsafe {
            write_empty(a.as_mut_ptr(), s.as_mut_ptr());
            assert!(a.assume_init().is_released());
            // schema drops harmlessly (its release is a no-op when empty).
            let _ = s.assume_init();
        }
    }
}
