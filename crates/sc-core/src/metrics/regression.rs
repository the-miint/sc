//! Regression report: a faithful port of `scipy.stats.linregress` (with
//! `x = actual`, `y = pred`) plus `sklearn.mean_squared_error`, matching
//! q2's `_linear_regress` (`visuals.py:86-99`).

use statrs::distribution::{ContinuousCDF, StudentsT};

/// The columns q2 reports for a regression, in `_linear_regress` order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionReport {
    pub mse: f64,
    pub r_value: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub std_err: f64,
    pub slope: f64,
    pub intercept: f64,
}

/// Ordinary least-squares regression of `pred` on `actual` (note the
/// orientation: `x = actual`, `y = pred`, matching q2), returning the same
/// statistics scipy's `linregress` does plus the mean squared error.
///
/// Panics if the slices differ in length or have fewer than three points
/// (scipy special-cases `n < 3`; q2 never calls it there and neither do we).
pub fn linear_regress(actual: &[f64], pred: &[f64]) -> RegressionReport {
    assert_eq!(
        actual.len(),
        pred.len(),
        "linear_regress: actual/pred length mismatch"
    );
    let n = actual.len();
    assert!(n >= 3, "linear_regress: need >=3 points, got {n}");
    let nf = n as f64;
    let (x, y) = (actual, pred);

    let xmean = x.iter().sum::<f64>() / nf;
    let ymean = y.iter().sum::<f64>() / nf;

    // Divide-by-N (bias=1) moments, matching scipy's `np.cov(x, y, bias=1)`.
    // The 1/N cancels in every ratio below, but keeping it mirrors scipy.
    let (mut ssxm, mut ssym, mut ssxym) = (0.0, 0.0, 0.0);
    let mut mse = 0.0;
    for i in 0..n {
        let dx = x[i] - xmean;
        let dy = y[i] - ymean;
        ssxm += dx * dx;
        ssym += dy * dy;
        ssxym += dx * dy;
        let e = x[i] - y[i];
        mse += e * e;
    }
    ssxm /= nf;
    ssym /= nf;
    ssxym /= nf;
    mse /= nf;

    // Pearson r, clamped to [-1, 1] exactly as scipy does.
    let r = if ssxm == 0.0 || ssym == 0.0 {
        0.0
    } else {
        (ssxym / (ssxm * ssym).sqrt()).clamp(-1.0, 1.0)
    };

    let slope = ssxym / ssxm;
    let intercept = ymean - slope * xmean;

    let df = nf - 2.0;
    const TINY: f64 = 1e-20;
    let t = r * (df / ((1.0 - r + TINY) * (1.0 + r + TINY))).sqrt();
    // scipy: p = 2 * t.sf(|t|, df). statrs's `sf` is the survival function, more
    // accurate in the tail than `1 - cdf`.
    let dist = StudentsT::new(0.0, 1.0, df).expect("df > 0 for n >= 3");
    let p_value = 2.0 * dist.sf(t.abs());

    let std_err = ((1.0 - r * r) * ssym / ssxm / df).sqrt();

    RegressionReport {
        mse,
        r_value: r,
        r_squared: r * r,
        p_value,
        std_err,
        slope,
        intercept,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line_is_perfectly_correlated() {
        // pred = 2*actual + 1 exactly → r = 1, slope 2, intercept 1, mse 0.
        let actual = [0.0, 1.0, 2.0, 3.0, 4.0];
        let pred = [1.0, 3.0, 5.0, 7.0, 9.0];
        let rep = linear_regress(&actual, &pred);
        assert!((rep.slope - 2.0).abs() < 1e-12);
        assert!((rep.intercept - 1.0).abs() < 1e-12);
        assert!((rep.r_value - 1.0).abs() < 1e-12);
        // MSE measures actual-vs-pred distance, unrelated to r: errors are
        // -1..-5, so mean(e²) = (1+4+9+16+25)/5 = 11.
        assert!((rep.mse - 11.0).abs() < 1e-12);
    }
}
