//! ROC curves and AUC, ported from scikit-learn (`roc_curve`, `auc`) and q2's
//! per-class / micro / macro assembly (`visuals.py:218-348`).
//!
//! The delicate part is `roc_curve`'s `drop_intermediate=True`: collinear
//! interior points are dropped. AUC is unchanged by that drop, but the retained
//! FPR grid feeds macro-average interpolation, so it must be reproduced exactly.

/// A single ROC curve. `thresholds[0]` is `+inf` (sklearn 1.4.2's prepended
/// point); `fpr`/`tpr` begin at `(0, 0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RocCurve {
    pub fpr: Vec<f64>,
    pub tpr: Vec<f64>,
    pub thresholds: Vec<f64>,
}

/// Port of `sklearn.metrics.roc_curve(y_true_bin, y_score, drop_intermediate=True)`,
/// with `pos_label = 1`.
///
/// `y_true_bin` holds 0/1 labels. Panics if the slices differ in length or are
/// empty. FPR/TPR are `NaN` if there are no negatives/positives respectively
/// (mirroring sklearn's divide-by-zero), so callers should ensure both classes
/// are present.
pub fn roc_curve(y_true_bin: &[u8], y_score: &[f64]) -> RocCurve {
    assert_eq!(
        y_true_bin.len(),
        y_score.len(),
        "roc_curve: y_true/y_score length mismatch"
    );
    assert!(!y_true_bin.is_empty(), "roc_curve: empty input");
    let n = y_score.len();

    // Descending order by score: numpy's `argsort(kind="mergesort")[::-1]` is a
    // stable ascending sort, reversed. Rust's `sort_by` is stable. `total_cmp`
    // gives a total order (equal to `partial_cmp` for finite scores) so a stray
    // non-finite score degrades deterministically instead of panicking.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| y_score[a].total_cmp(&y_score[b]));
    order.reverse();

    // `_binary_clf_curve`: a (fp, tp, threshold) triple at each score boundary
    // (the last index of every run of equal scores). Ties inside a run don't
    // affect the cumulative counts sampled at the boundary.
    let mut fps: Vec<u64> = Vec::new();
    let mut tps: Vec<u64> = Vec::new();
    let mut thr: Vec<f64> = Vec::new();
    let mut cum_tp: u64 = 0;
    for pos in 0..n {
        let idx = order[pos];
        cum_tp += u64::from(y_true_bin[idx]);
        let boundary = pos + 1 == n || y_score[order[pos + 1]] != y_score[idx];
        if boundary {
            tps.push(cum_tp);
            fps.push((pos as u64 + 1) - cum_tp);
            thr.push(y_score[idx]);
        }
    }

    // drop_intermediate: keep endpoints, and any interior point whose second
    // difference in fps or tps is nonzero (i.e. not collinear with neighbours).
    let m = fps.len();
    if m > 2 {
        let mut keep = vec![false; m];
        keep[0] = true;
        keep[m - 1] = true;
        for i in 1..m - 1 {
            let d2f = fps[i + 1] as i128 - 2 * fps[i] as i128 + fps[i - 1] as i128;
            let d2t = tps[i + 1] as i128 - 2 * tps[i] as i128 + tps[i - 1] as i128;
            keep[i] = d2f != 0 || d2t != 0;
        }
        let mut w = 0;
        for r in 0..m {
            if keep[r] {
                fps[w] = fps[r];
                tps[w] = tps[r];
                thr[w] = thr[r];
                w += 1;
            }
        }
        fps.truncate(w);
        tps.truncate(w);
        thr.truncate(w);
    }

    // Prepend the (0, 0) point (threshold +inf), then normalize by the totals
    // (fps[-1] = #negatives, tps[-1] = #positives). sklearn replaces the *entire*
    // fpr (resp. tpr) array with NaN when there are no negatives (resp.
    // positives) rather than only the interior points — reproduce that so a
    // single-class fold degrades to a NaN curve exactly as sklearn does.
    let total_neg = *fps.last().expect("at least one boundary");
    let total_pos = *tps.last().expect("at least one boundary");
    let mut thresholds = Vec::with_capacity(thr.len() + 1);
    thresholds.push(f64::INFINITY);
    thresholds.extend_from_slice(&thr);
    RocCurve {
        fpr: normalize_with_prepended_zero(&fps, total_neg),
        tpr: normalize_with_prepended_zero(&tps, total_pos),
        thresholds,
    }
}

/// Prepend a leading 0 and divide by `total`, or return an all-`NaN` array of
/// the same length when `total == 0` — matching sklearn's `fps / fps[-1]` with
/// its `fps[-1] <= 0` degenerate guard.
fn normalize_with_prepended_zero(counts: &[u64], total: u64) -> Vec<f64> {
    let len = counts.len() + 1;
    if total == 0 {
        return vec![f64::NAN; len];
    }
    let total = total as f64;
    let mut out = Vec::with_capacity(len);
    out.push(0.0);
    out.extend(counts.iter().map(|&c| c as f64 / total));
    out
}

/// Trapezoidal area under `(x, y)` with sklearn's monotonic-direction handling
/// (`sklearn.metrics.auc`). Panics if `x` is not monotonic or has < 2 points.
pub fn auc(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "auc: x/y length mismatch");
    assert!(x.len() >= 2, "auc: need >= 2 points");
    // sklearn: only if some step is negative do we check for fully-decreasing.
    let direction = if x.windows(2).any(|w| w[1] - w[0] < 0.0) {
        assert!(
            x.windows(2).all(|w| w[1] - w[0] <= 0.0),
            "auc: x is neither non-decreasing nor non-increasing"
        );
        -1.0
    } else {
        1.0
    };
    let mut area = 0.0;
    for i in 0..x.len() - 1 {
        area += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0;
    }
    direction * area
}

/// Per-class AUC (one-vs-rest), aligned to proba column order. `y_true` holds
/// class indices; `proba[i]` is the length-`k` probability row for sample `i`.
pub fn roc_auc_per_class(y_true: &[usize], proba: &[Vec<f64>]) -> Vec<f64> {
    let k = proba_width(y_true, proba);
    (0..k)
        .map(|c| {
            let (bin, scores) = binary_problem(y_true, proba, c);
            let curve = roc_curve(&bin, &scores);
            auc(&curve.fpr, &curve.tpr)
        })
        .collect()
}

/// Micro-average AUC: ROC over the row-major-raveled one-hot labels vs. probas.
pub fn roc_auc_micro(y_true: &[usize], proba: &[Vec<f64>]) -> f64 {
    let k = proba_width(y_true, proba);
    let mut bin = Vec::with_capacity(y_true.len() * k);
    let mut scores = Vec::with_capacity(y_true.len() * k);
    for (i, row) in proba.iter().enumerate() {
        for (c, &score) in row.iter().enumerate() {
            bin.push(u8::from(y_true[i] == c));
            scores.push(score);
        }
    }
    let curve = roc_curve(&bin, &scores);
    auc(&curve.fpr, &curve.tpr)
}

/// Macro-average AUC: mean per-class TPR interpolated onto the union of all
/// per-class FPR grids, then integrated (`_roc_macro_average`).
pub fn roc_auc_macro(y_true: &[usize], proba: &[Vec<f64>]) -> f64 {
    let k = proba_width(y_true, proba);
    let curves: Vec<RocCurve> = (0..k)
        .map(|c| {
            let (bin, scores) = binary_problem(y_true, proba, c);
            roc_curve(&bin, &scores)
        })
        .collect();

    // all_fpr = np.unique(concat of per-class fprs): sorted, exact-dedup.
    // `total_cmp` matches `partial_cmp` for finite fpr but avoids a panic if a
    // single-class column produced a NaN fpr (the macro AUC then degrades to
    // NaN, as it does in numpy, rather than crashing).
    let mut all_fpr: Vec<f64> = curves.iter().flat_map(|c| c.fpr.iter().copied()).collect();
    all_fpr.sort_by(f64::total_cmp);
    all_fpr.dedup();

    let mut mean_tpr = vec![0.0; all_fpr.len()];
    for c in &curves {
        for (slot, &x) in mean_tpr.iter_mut().zip(&all_fpr) {
            *slot += interp(x, &c.fpr, &c.tpr);
        }
    }
    for v in &mut mean_tpr {
        *v /= k as f64;
    }
    auc(&all_fpr, &mean_tpr)
}

/// The one-vs-rest binary problem for class `c`: 0/1 labels (`1` where the true
/// class is `c`) paired with proba column `c`. Shared by the per-class and
/// macro paths so the one-hot encoding stays defined in one place.
fn binary_problem(y_true: &[usize], proba: &[Vec<f64>], c: usize) -> (Vec<u8>, Vec<f64>) {
    let bin = y_true.iter().map(|&y| u8::from(y == c)).collect();
    let scores = proba.iter().map(|row| row[c]).collect();
    (bin, scores)
}

/// Number of proba columns (= class count), with shared shape assertions.
fn proba_width(y_true: &[usize], proba: &[Vec<f64>]) -> usize {
    assert_eq!(
        y_true.len(),
        proba.len(),
        "roc auc: y_true/proba length mismatch"
    );
    assert!(!proba.is_empty(), "roc auc: empty input");
    proba[0].len()
}

/// Faithful port of `numpy.interp` for the values this module needs: `xp` is
/// non-decreasing; out-of-range queries clamp to the endpoints; a query landing
/// exactly on a knot (including a repeated one) returns that knot's `fp`, which
/// matches numpy picking the *last* index of a duplicate run.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // Largest index j with xp[j] <= x (exists since x >= xp[0]).
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_of_unit_triangle() {
        // Perfect classifier: TPR jumps to 1 at FPR 0 → AUC 1.
        assert_eq!(auc(&[0.0, 0.0, 1.0], &[0.0, 1.0, 1.0]), 1.0);
    }

    #[test]
    fn no_negatives_makes_whole_fpr_nan() {
        // All-positive labels → no negatives. sklearn returns an all-NaN fpr
        // (including the prepended point), a finite tpr rising to 1, and does
        // not crash.
        let c = roc_curve(&[1, 1, 1], &[0.1, 0.2, 0.3]);
        assert!(c.fpr.iter().all(|v| v.is_nan()), "fpr should be all NaN");
        assert_eq!(c.tpr[0], 0.0);
        assert_eq!(*c.tpr.last().unwrap(), 1.0);
    }

    #[test]
    fn interp_returns_top_of_duplicate_knot() {
        // Duplicated x=0 knot: numpy returns the last (top) fp, here 0.5.
        let xp = [0.0, 0.0, 1.0];
        let fp = [0.0, 0.5, 1.0];
        assert_eq!(interp(0.0, &xp, &fp), 0.5);
        assert_eq!(interp(0.5, &xp, &fp), 0.75);
    }
}
