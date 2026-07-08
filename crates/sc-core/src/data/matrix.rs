//! Internal feature matrix: COO ingestion into compressed-sparse-column (CSC)
//! storage. Rows are samples, columns are features — mirroring the sparse matrix
//! q2 fits (biom → `DictVectorizer`) and the wide, sparse shape of microbiome
//! tables. CSC gives later milestones contiguous per-feature columns for split
//! search. This crate is Arrow-free; COO triplets arrive as plain slices (the
//! Arrow C Data Interface marshaling is a separate outer crate).

use crate::{Error, Result};

/// A sample-by-feature matrix in compressed-sparse-column form.
///
/// `col_ptr` has length `n_features + 1`; column `c` owns the entries
/// `col_ptr[c]..col_ptr[c+1]` of `row_idx`/`values`, with `row_idx` ascending
/// within each column. Zeros are implicit.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    n_samples: usize,
    n_features: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

impl FeatureMatrix {
    /// Build from COO triplets (`rows`/`cols` are sample/feature indices,
    /// `vals` the values), rows = samples, cols = features.
    ///
    /// Duplicate `(row, col)` pairs are **summed** (scipy COO→CSR semantics; a
    /// summed-to-zero entry is retained as an explicit zero, matching scipy's
    /// default of not pruning). Errors (`InvalidInput`) on: `rows/cols/vals`
    /// length mismatch; a negative or out-of-range row/col index;
    /// `n_samples == 0` or `n_features == 0`.
    pub fn from_coo(
        n_samples: usize,
        n_features: usize,
        rows: &[i64],
        cols: &[i64],
        vals: &[f64],
    ) -> Result<Self> {
        if n_samples == 0 || n_features == 0 {
            return Err(Error::InvalidInput(format!(
                "from_coo: dimensions must be > 0 (got {n_samples}x{n_features})"
            )));
        }
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err(Error::InvalidInput(format!(
                "from_coo: rows/cols/vals length mismatch ({}, {}, {})",
                rows.len(),
                cols.len(),
                vals.len()
            )));
        }
        let mut triples = Vec::with_capacity(rows.len());
        for ((&r, &c), &v) in rows.iter().zip(cols).zip(vals) {
            if r < 0 || r as u64 >= n_samples as u64 {
                return Err(Error::InvalidInput(format!(
                    "from_coo: row index {r} out of range [0, {n_samples})"
                )));
            }
            if c < 0 || c as u64 >= n_features as u64 {
                return Err(Error::InvalidInput(format!(
                    "from_coo: col index {c} out of range [0, {n_features})"
                )));
            }
            triples.push((c as usize, r as usize, v));
        }
        let (col_ptr, row_idx, values) = build_csc(n_features, triples);
        Ok(FeatureMatrix {
            n_samples,
            n_features,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Number of samples (rows).
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of features (columns).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of stored (nonzero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate the stored `(sample_row, value)` pairs of one feature column,
    /// ascending by row. The per-feature access split search will use in M3.
    ///
    /// Panics if `feature >= n_features`.
    pub fn column(&self, feature: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (s, e) = (self.col_ptr[feature], self.col_ptr[feature + 1]);
        self.row_idx[s..e]
            .iter()
            .copied()
            .zip(self.values[s..e].iter().copied())
    }

    /// Materialize the sample-row subset `idx` as a new matrix whose rows are
    /// remapped to `0..idx.len()` in the order given (feature columns
    /// unchanged). This is the bridge that turns a [`crate::Split`] /
    /// [`crate::CvFolds`] index list into trainable data.
    ///
    /// `idx` may repeat a row (that sample's entries are duplicated into each
    /// target position). Panics if any index is `>= n_samples` — indices are
    /// expected to come from a validated `Split`/`CvFolds`.
    pub fn select_rows(&self, idx: &[usize]) -> FeatureMatrix {
        // For each old row, the new positions it maps to.
        let mut positions: Vec<Vec<usize>> = vec![Vec::new(); self.n_samples];
        for (new_pos, &old) in idx.iter().enumerate() {
            assert!(
                old < self.n_samples,
                "select_rows: index {old} out of range [0, {})",
                self.n_samples
            );
            positions[old].push(new_pos);
        }
        let mut triples = Vec::new();
        for c in 0..self.n_features {
            for k in self.col_ptr[c]..self.col_ptr[c + 1] {
                let v = self.values[k];
                for &new_pos in &positions[self.row_idx[k]] {
                    triples.push((c, new_pos, v));
                }
            }
        }
        let (col_ptr, row_idx, values) = build_csc(self.n_features, triples);
        FeatureMatrix {
            n_samples: idx.len(),
            n_features: self.n_features,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Materialize the feature-column subset `idx` as a new matrix whose columns
    /// are remapped to `0..idx.len()` in the order given (sample rows unchanged) —
    /// the bridge RFECV uses to fit on a surviving feature subset.
    ///
    /// Cheap in CSC: each chosen column's stored entries are already contiguous
    /// and row-sorted, so this copies slices and renumbers columns (no resort /
    /// no dup-summing). `idx` may repeat a column (its entries are copied into
    /// each target column). Panics if any index is `>= n_features`.
    pub fn select_columns(&self, idx: &[usize]) -> FeatureMatrix {
        let mut col_ptr = vec![0usize; idx.len() + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for (new_c, &old) in idx.iter().enumerate() {
            assert!(
                old < self.n_features,
                "select_columns: index {old} out of range [0, {})",
                self.n_features
            );
            let (s, e) = (self.col_ptr[old], self.col_ptr[old + 1]);
            row_idx.extend_from_slice(&self.row_idx[s..e]);
            values.extend_from_slice(&self.values[s..e]);
            col_ptr[new_c + 1] = col_ptr[new_c] + (e - s);
        }
        FeatureMatrix {
            n_samples: self.n_samples,
            n_features: idx.len(),
            col_ptr,
            row_idx,
            values,
        }
    }
}

/// Assemble CSC arrays from `(col, row, val)` triples: sort by `(col, row)`, sum
/// duplicate `(col, row)` pairs, and build the column pointers. Shared by
/// `from_coo` and `select_rows`.
fn build_csc(
    n_features: usize,
    mut triples: Vec<(usize, usize, f64)>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut col_ptr = vec![0usize; n_features + 1];
    let mut row_idx = Vec::with_capacity(triples.len());
    let mut values = Vec::with_capacity(triples.len());
    let mut i = 0;
    while i < triples.len() {
        let (c, r, mut v) = triples[i];
        let mut j = i + 1;
        while j < triples.len() && triples[j].0 == c && triples[j].1 == r {
            v += triples[j].2;
            j += 1;
        }
        row_idx.push(r);
        values.push(v);
        col_ptr[c + 1] += 1;
        i = j;
    }
    for c in 0..n_features {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, row_idx, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reconstruct a dense [n_samples][n_features] view for assertions. `c`
    // indexes both the column accessor and the output cell, so a range loop is
    // the natural form here.
    #[allow(clippy::needless_range_loop)]
    fn dense(m: &FeatureMatrix) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; m.n_features()]; m.n_samples()];
        for c in 0..m.n_features() {
            for (r, v) in m.column(c) {
                d[r][c] = v;
            }
        }
        d
    }

    #[test]
    fn ingests_coo_to_csc() {
        // 3 samples x 2 features.
        let m = FeatureMatrix::from_coo(3, 2, &[0, 2, 1], &[0, 1, 0], &[5.0, 7.0, 9.0]).unwrap();
        assert_eq!(m.n_samples(), 3);
        assert_eq!(m.n_features(), 2);
        assert_eq!(m.nnz(), 3);
        assert_eq!(
            dense(&m),
            vec![vec![5.0, 0.0], vec![9.0, 0.0], vec![0.0, 7.0]]
        );
    }

    #[test]
    fn duplicate_entries_are_summed() {
        // Two entries at (row 0, col 0) → summed to 3.0.
        let m = FeatureMatrix::from_coo(2, 1, &[0, 0, 1], &[0, 0, 0], &[1.0, 2.0, 4.0]).unwrap();
        assert_eq!(m.nnz(), 2);
        assert_eq!(dense(&m), vec![vec![3.0], vec![4.0]]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(FeatureMatrix::from_coo(0, 1, &[], &[], &[]).is_err());
        assert!(FeatureMatrix::from_coo(2, 2, &[0, 1], &[0], &[1.0, 2.0]).is_err()); // len mismatch
        assert!(FeatureMatrix::from_coo(2, 2, &[2], &[0], &[1.0]).is_err()); // row OOB
        assert!(FeatureMatrix::from_coo(2, 2, &[0], &[-1], &[1.0]).is_err()); // negative col
    }

    #[test]
    fn select_rows_materializes_subset() {
        let m = FeatureMatrix::from_coo(4, 2, &[0, 1, 2, 3], &[0, 1, 0, 1], &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        // Take rows [3, 1] → new row 0 = old 3, new row 1 = old 1.
        let sub = m.select_rows(&[3, 1]);
        assert_eq!(sub.n_samples(), 2);
        assert_eq!(sub.n_features(), 2);
        assert_eq!(dense(&sub), vec![vec![0.0, 4.0], vec![0.0, 2.0]]);
    }

    #[test]
    fn select_columns_materializes_subset() {
        // 3 samples x 3 features, one nonzero per (sample, feature) diagonal-ish.
        let m = FeatureMatrix::from_coo(
            3,
            3,
            &[0, 1, 2, 0, 2],
            &[0, 1, 2, 2, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        // dense = [[1,0,4],[0,2,0],[5,0,3]]. Take features [2, 0] → new col 0 =
        // old 2, new col 1 = old 0 (rows unchanged).
        let sub = m.select_columns(&[2, 0]);
        assert_eq!(sub.n_samples(), 3);
        assert_eq!(sub.n_features(), 2);
        assert_eq!(
            dense(&sub),
            vec![vec![4.0, 1.0], vec![0.0, 0.0], vec![3.0, 5.0]]
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn select_columns_panics_on_oob() {
        let m = FeatureMatrix::from_coo(2, 2, &[0, 1], &[0, 1], &[1.0, 2.0]).unwrap();
        m.select_columns(&[0, 2]);
    }
}
