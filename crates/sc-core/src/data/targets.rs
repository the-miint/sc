//! Prediction targets: categorical (classification) or numeric (regression),
//! caller-aligned to the feature matrix's sample order.
//!
//! Categorical labels are encoded by **sorted-unique** into `classes` + per-
//! sample `codes`. q2 uses no `LabelEncoder`; it passes raw labels to
//! `RandomForestClassifier`, which sorts unique labels into `classes_`. Encoding
//! here the same way keeps class order (and therefore probability-column order)
//! consistent with sklearn and with the M1 metrics conventions.

use crate::{Error, Result};

/// Targets for one column of samples.
#[derive(Debug, Clone, PartialEq)]
pub enum Targets {
    /// Classification: `classes` are the sorted-unique labels (sklearn
    /// `classes_` order); `codes[i]` indexes sample `i`'s label into `classes`.
    Categorical {
        classes: Vec<String>,
        codes: Vec<usize>,
    },
    /// Regression: raw numeric targets.
    Numeric(Vec<f64>),
}

impl Targets {
    /// Encode categorical labels by sorted-unique. Errors on empty input.
    pub fn categorical<S: AsRef<str>>(labels: &[S]) -> Result<Self> {
        if labels.is_empty() {
            return Err(Error::InvalidInput(
                "categorical targets: empty input".into(),
            ));
        }
        let mut classes: Vec<String> = labels.iter().map(|s| s.as_ref().to_owned()).collect();
        classes.sort();
        classes.dedup();
        let codes = labels
            .iter()
            .map(|s| {
                classes
                    .binary_search_by(|c| c.as_str().cmp(s.as_ref()))
                    .expect("label is in the sorted-unique class set")
            })
            .collect();
        Ok(Targets::Categorical { classes, codes })
    }

    /// Store numeric targets. Errors on empty input or any non-finite value
    /// (regression cannot consume NaN/inf; callers clean/drop upstream).
    pub fn numeric(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::InvalidInput("numeric targets: empty input".into()));
        }
        if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "numeric targets: non-finite value {bad}"
            )));
        }
        Ok(Targets::Numeric(values.to_vec()))
    }

    /// Number of samples (named to match `FeatureMatrix`/`Split`/`CvFolds`).
    pub fn n_samples(&self) -> usize {
        match self {
            Targets::Categorical { codes, .. } => codes.len(),
            Targets::Numeric(v) => v.len(),
        }
    }

    /// Number of classes (categorical only).
    pub fn n_classes(&self) -> Option<usize> {
        match self {
            Targets::Categorical { classes, .. } => Some(classes.len()),
            Targets::Numeric(_) => None,
        }
    }

    /// The sorted-unique class labels (categorical only).
    pub fn classes(&self) -> Option<&[String]> {
        match self {
            Targets::Categorical { classes, .. } => Some(classes),
            Targets::Numeric(_) => None,
        }
    }

    /// Per-sample class codes (categorical only).
    pub fn codes(&self) -> Option<&[usize]> {
        match self {
            Targets::Categorical { codes, .. } => Some(codes),
            Targets::Numeric(_) => None,
        }
    }

    /// The stratification view the partitioner consumes: `Some((codes, k))` for
    /// categorical (classification → stratify on), `None` for numeric
    /// (regression → stratify off). Encodes q2's stratify-by-target policy.
    pub fn strata(&self) -> Option<(&[usize], usize)> {
        match self {
            Targets::Categorical { classes, codes } => Some((codes, classes.len())),
            Targets::Numeric(_) => None,
        }
    }

    /// Reindex the targets by the sample subset `idx`, returning targets whose
    /// sample `k` is the original sample `idx[k]`. The bootstrap bridge that
    /// pairs with [`FeatureMatrix::select_rows`](crate::FeatureMatrix::select_rows):
    /// `idx` may repeat a row (with replacement) and need not be sorted.
    ///
    /// The categorical **`classes` set is preserved unchanged** — even if a
    /// bootstrap sample omits a class entirely, `n_classes` (and therefore the
    /// per-tree probability-column width) stays stable across the forest so
    /// per-tree `value` vectors align for averaging. Panics if any index is
    /// `>= n_samples` (indices come from a validated split/bootstrap draw).
    pub fn select(&self, idx: &[usize]) -> Targets {
        let n = self.n_samples();
        match self {
            Targets::Categorical { classes, codes } => {
                let sub = idx
                    .iter()
                    .map(|&i| {
                        assert!(i < n, "Targets::select: index {i} out of range [0, {n})");
                        codes[i]
                    })
                    .collect();
                Targets::Categorical {
                    classes: classes.clone(),
                    codes: sub,
                }
            }
            Targets::Numeric(v) => Targets::Numeric(
                idx.iter()
                    .map(|&i| {
                        assert!(i < n, "Targets::select: index {i} out of range [0, {n})");
                        v[i]
                    })
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_is_sorted_unique() {
        let t = Targets::categorical(&["b", "a", "b", "c", "a"]).unwrap();
        assert_eq!(
            t.classes(),
            Some(["a".to_string(), "b".into(), "c".into()].as_slice())
        );
        // codes invert to the original labels via classes[code].
        assert_eq!(t.codes(), Some([1, 0, 1, 2, 0].as_slice()));
        assert_eq!(t.n_classes(), Some(3));
        assert_eq!(t.n_samples(), 5);
        assert!(t.strata().is_some());
    }

    #[test]
    fn numeric_has_no_strata() {
        let t = Targets::numeric(&[1.5, -2.0, 3.25]).unwrap();
        assert_eq!(t.n_samples(), 3);
        assert_eq!(t.n_classes(), None);
        assert!(t.strata().is_none());
    }

    #[test]
    fn rejects_empty_and_nonfinite() {
        assert!(Targets::categorical::<&str>(&[]).is_err());
        assert!(Targets::numeric(&[]).is_err());
        assert!(Targets::numeric(&[1.0, f64::NAN]).is_err());
        assert!(Targets::numeric(&[1.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn select_reindexes_and_allows_repeats() {
        let cat = Targets::categorical(&["a", "b", "c", "a"]).unwrap();
        // Bootstrap-style draw: out of order, with a repeat.
        let sub = cat.select(&[3, 3, 1]);
        assert_eq!(sub.codes(), Some([0, 0, 1].as_slice())); // a, a, b
        assert_eq!(sub.n_samples(), 3);

        let num = Targets::numeric(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            num.select(&[2, 0, 2]),
            Targets::numeric(&[3.0, 1.0, 3.0]).unwrap()
        );
    }

    #[test]
    fn select_preserves_full_class_set_when_a_class_is_absent() {
        // A subset that omits class "c" must still carry all three classes so
        // per-tree probability columns stay full-width across a bootstrap forest.
        let cat = Targets::categorical(&["a", "b", "c"]).unwrap();
        let sub = cat.select(&[0, 1]); // only a, b sampled
        assert_eq!(sub.n_classes(), Some(3), "class set must survive selection");
        assert_eq!(
            sub.classes(),
            Some(["a".to_string(), "b".into(), "c".into()].as_slice())
        );
        assert_eq!(sub.codes(), Some([0, 1].as_slice()));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn select_panics_on_oob_index() {
        Targets::numeric(&[1.0, 2.0]).unwrap().select(&[0, 2]);
    }
}
