//! Split criteria: impurity measures for classification and regression.
//!
//! These mirror scikit-learn's `_criterion.pyx` for sklearn 1.4.2 exactly (the
//! pinned oracle target):
//! - **gini** `1 − Σ p_c²`
//! - **entropy** `−Σ p_c·log2(p_c)` over `p_c > 0` — **log base 2**, not the
//!   natural log (verified empirically against the oracle; the design doc's
//!   "natural log" is wrong for 1.4.2 and the entropy fixture is the arbiter).
//! - **squared_error** (regressor) = node variance
//!   `mean(y²) − mean(y)² = sumsq/n − (sum/n)²`.
//!
//! `p_c` are the `value` proportions sklearn stores per node, so recomputing an
//! impurity from a node's stored `value` must reproduce its stored `impurity`
//! (the import oracle test asserts exactly this, which nails the log base).

use super::params::Criterion;

/// Gini impurity of a class-proportion vector `p` (entries sum to 1):
/// `1 − Σ p_c²`.
pub fn gini(p: &[f64]) -> f64 {
    // Proportions already have implicit denominator 1, so pass n = 1.0.
    impurity_from_counts(p, 1.0, Criterion::Gini)
}

/// Shannon entropy of a class-proportion vector `p` in **bits**:
/// `−Σ p_c·log2(p_c)` over the strictly-positive entries.
pub fn entropy(p: &[f64]) -> f64 {
    impurity_from_counts(p, 1.0, Criterion::Entropy)
}

/// Population variance from running moments — the regressor `squared_error`
/// impurity: `sumsq/n − (sum/n)²`. Uses sklearn's exact algebraic form (so its
/// cancellation behavior matches too). `n` must be `> 0`.
pub fn variance(sum: f64, sumsq: f64, n: f64) -> f64 {
    let mean = sum / n;
    sumsq / n - mean * mean
}

/// Classification impurity from **weighted class counts** and their total `n`
/// (`n > 0`), selected by `crit`. The single source of truth for gini/entropy:
/// [`gini`]/[`entropy`] delegate here with `n = 1.0`, so proportions-in and
/// counts-in can never drift apart.
///
/// Panics (internal misuse) if called with [`Criterion::SquaredError`], which is
/// a regression criterion with no class-count form.
pub(crate) fn impurity_from_counts(counts: &[f64], n: f64, crit: Criterion) -> f64 {
    match crit {
        Criterion::Gini => {
            let sq: f64 = counts
                .iter()
                .map(|&c| {
                    let p = c / n;
                    p * p
                })
                .sum();
            1.0 - sq
        }
        Criterion::Entropy => {
            let mut e = 0.0;
            for &c in counts {
                if c > 0.0 {
                    let p = c / n;
                    e -= p * p.log2();
                }
            }
            e
        }
        Criterion::SquaredError => {
            unreachable!("impurity_from_counts called with the regression criterion")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_of_pure_and_uniform() {
        assert_eq!(gini(&[1.0, 0.0, 0.0]), 0.0);
        // Two equal classes: 1 - (0.25 + 0.25) = 0.5.
        assert_eq!(gini(&[0.5, 0.5]), 0.5);
        // Three equal classes: 1 - 3*(1/9) = 2/3.
        assert!((gini(&[1.0 / 3.0; 3]) - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn entropy_uses_log2_not_ln() {
        // Three equal classes → log2(3) ≈ 1.585 (ln would give ≈ 1.099).
        let e = entropy(&[1.0 / 3.0; 3]);
        assert!((e - 3.0_f64.log2()).abs() < 1e-12, "entropy {e} != log2(3)");
        // A pure node has zero entropy.
        assert_eq!(entropy(&[1.0, 0.0]), 0.0);
        // Binary uniform = exactly 1 bit.
        assert!((entropy(&[0.5, 0.5]) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn variance_matches_population_form() {
        // y = [1, 2, 10, 12]: mean 6.25, var 23.1875.
        let (sum, sumsq, n) = (25.0, 249.0, 4.0);
        assert!((variance(sum, sumsq, n) - 23.1875).abs() < 1e-12);
    }

    #[test]
    fn counts_and_proportions_agree() {
        // impurity_from_counts on raw counts equals the proportion-based fns.
        let counts = [2.0, 3.0, 5.0];
        let n = 10.0;
        let p: Vec<f64> = counts.iter().map(|&c| c / n).collect();
        assert_eq!(impurity_from_counts(&counts, n, Criterion::Gini), gini(&p));
        assert_eq!(
            impurity_from_counts(&counts, n, Criterion::Entropy),
            entropy(&p)
        );
    }
}
