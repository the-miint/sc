//! Classification metrics: accuracy, confusion matrix (raw + row-normalized),
//! and the majority-class baseline. Ports `visuals.py:137-145` and
//! `_calculate_baseline_accuracy` (`visuals.py:202-207`).

/// Fraction of predictions that match the truth (`sklearn.accuracy_score`).
///
/// Panics if the slices differ in length or are empty.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "accuracy: y_true/y_pred length mismatch"
    );
    assert!(!y_true.is_empty(), "accuracy: empty input");
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
    correct as f64 / y_true.len() as f64
}

/// Confusion matrix with labels = the sorted unique union of `y_true ∪ y_pred`
/// (sklearn's default when no `labels=` argument is given).
///
/// Returns `(labels, counts)` where `counts[i][j]` is the number of samples with
/// true label `labels[i]` predicted as `labels[j]`. Rows are true labels,
/// columns predicted labels.
///
/// Panics if the slices differ in length or are empty.
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize]) -> (Vec<usize>, Vec<Vec<u64>>) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "confusion_matrix: y_true/y_pred length mismatch"
    );
    assert!(!y_true.is_empty(), "confusion_matrix: empty input");

    let mut labels: Vec<usize> = y_true.iter().chain(y_pred).copied().collect();
    labels.sort_unstable();
    labels.dedup();

    let k = labels.len();
    let mut counts = vec![vec![0u64; k]; k];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        // `labels` is sorted, so binary_search gives each value's row/column.
        let i = labels.binary_search(&t).expect("true label in union");
        let j = labels.binary_search(&p).expect("pred label in union");
        counts[i][j] += 1;
    }
    (labels, counts)
}

/// Row-normalize a confusion matrix: each cell divided by its row sum, with
/// all-zero rows (a label that is never a true value) mapped to zeros —
/// reproducing `cm/cm.sum(axis=1)` followed by `np.nan_to_num`.
pub fn confusion_matrix_normalized(counts: &[Vec<u64>]) -> Vec<Vec<f64>> {
    counts
        .iter()
        .map(|row| {
            let sum: u64 = row.iter().sum();
            if sum == 0 {
                // 0/0 → NaN → nan_to_num → 0 for every cell in the row.
                vec![0.0; row.len()]
            } else {
                let s = sum as f64;
                row.iter().map(|&c| c as f64 / s).collect()
            }
        })
        .collect()
}

/// Majority-class baseline: the largest true-class count divided by `n`
/// (`_calculate_baseline_accuracy`). Panics if `y_true` is empty.
pub fn baseline_accuracy(y_true: &[usize]) -> f64 {
    assert!(!y_true.is_empty(), "baseline_accuracy: empty input");
    let max_label = *y_true.iter().max().expect("non-empty");
    let mut counts = vec![0u64; max_label + 1];
    for &c in y_true {
        counts[c] += 1;
    }
    let largest = counts.into_iter().max().expect("non-empty");
    largest as f64 / y_true.len() as f64
}

/// Ratio of achieved accuracy to the majority-class baseline.
pub fn accuracy_ratio(accuracy: f64, baseline: f64) -> f64 {
    accuracy / baseline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_baseline() {
        let y = [0, 0, 1, 2];
        assert_eq!(accuracy(&y, &y), 1.0);
        // largest true-class count is 2 (label 0) out of 4.
        assert_eq!(baseline_accuracy(&y), 0.5);
        assert_eq!(accuracy_ratio(1.0, 0.5), 2.0);
    }

    #[test]
    fn zero_sum_row_normalizes_to_zero() {
        // Label 2 is predicted but never true → its row sums to zero.
        let (labels, counts) = confusion_matrix(&[0, 1], &[0, 2]);
        assert_eq!(labels, vec![0, 1, 2]);
        let norm = confusion_matrix_normalized(&counts);
        assert_eq!(norm[2], vec![0.0, 0.0, 0.0]);
    }
}
