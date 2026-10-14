//! Parallel/sequential execution shim.
//!
//! `indexed_map` returns an identically-ordered `Vec` regardless of the
//! `parallel` feature, so results are invariant to thread count. This is the
//! seam the forest's per-tree build uses in M4 (through its fallible form,
//! `try_indexed_map`).

use std::collections::TryReserveError;

/// Map `f` over `0..n`, collecting into an ordered `Vec`.
///
/// With the `parallel` feature this fans out over rayon; without it, it runs
/// sequentially. Either way the output order matches `0..n`.
///
/// The trait bounds are identical in both feature configurations — only the
/// body changes — so a closure that compiles here compiles regardless of
/// whether `parallel` is enabled. Cargo features must stay additive: gating the
/// bounds on the feature would let a `--no-default-features` build accept a
/// non-`Send`/`Sync` closure that then fails once feature unification turns
/// `parallel` back on elsewhere in the build graph.
///
/// Panics if the `n` outputs cannot be allocated; when `n` is caller input, use
/// [`try_indexed_map`].
pub fn indexed_map<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Send + Sync,
{
    try_indexed_map(n, f).expect("indexed_map: cannot allocate the output")
}

/// [`indexed_map`] with the output reserved fallibly: if `n` outputs cannot be
/// allocated, this returns `Err` before `f` ever runs.
///
/// The output is sized by `n` up front, and an infallible reservation hands an
/// unallocatable `n` to the global alloc-error handler, which aborts the
/// process — no `Result`, and no `catch_unwind` at an FFI boundary, can contain
/// that. Use this whenever `n` is caller input (the forest's `n_estimators`).
/// The output's order and contents are exactly `indexed_map`'s.
pub fn try_indexed_map<R, F>(n: usize, f: F) -> Result<Vec<R>, TryReserveError>
where
    R: Send,
    F: Fn(usize) -> R + Send + Sync,
{
    let mut out = Vec::new();
    out.try_reserve_exact(n)?;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Fills `out` in place: rayon reserves only capacity it lacks, here none.
        (0..n).into_par_iter().map(f).collect_into_vec(&mut out);
    }
    #[cfg(not(feature = "parallel"))]
    {
        out.extend((0..n).map(f));
    }
    Ok(out)
}

/// Fill `buf` in consecutive `width`-sized chunks, calling `f(i, chunk)` for
/// chunk `i`.
///
/// The in-place sibling of [`indexed_map`]: each chunk is written by exactly one
/// call that depends only on its index, so the result is identical with or
/// without the `parallel` feature and at any thread count. Writing straight into
/// one caller-owned buffer avoids a per-index allocation and the copy that would
/// join them. A trailing partial chunk is passed as-is; `width == 0` is a no-op.
pub fn fill_chunks<T, F>(buf: &mut [T], width: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if width == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
    #[cfg(not(feature = "parallel"))]
    {
        buf.chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::{fill_chunks, indexed_map};
    use crate::tree_seed;

    // Deterministic, pure per-index value; no shared state.
    fn work(i: usize) -> u64 {
        tree_seed(0xABCD, i as u64)
    }

    /// Directly backs the "same result regardless of thread count" invariant by
    /// running the same map under pools of different sizes.
    #[test]
    fn output_is_invariant_to_thread_count() {
        let n = 1000;
        let reference: Vec<u64> = (0..n).map(work).collect();
        for k in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .unwrap();
            let got = pool.install(|| indexed_map(n, work));
            assert_eq!(got, reference, "output changed with {k} threads");
        }
    }

    /// Same invariant for the in-place form, with a width that leaves no chunk
    /// boundary aligned to a thread split by accident.
    #[test]
    fn fill_chunks_is_invariant_to_thread_count() {
        let (n, width) = (257usize, 3usize);
        let fill = |i: usize, chunk: &mut [u64]| {
            for (j, v) in chunk.iter_mut().enumerate() {
                *v = work(i) ^ j as u64;
            }
        };
        let mut reference = vec![0u64; n * width];
        for (i, chunk) in reference.chunks_mut(width).enumerate() {
            fill(i, chunk);
        }
        for k in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .unwrap();
            let mut got = vec![0u64; n * width];
            pool.install(|| fill_chunks(&mut got, width, fill));
            assert_eq!(got, reference, "output changed with {k} threads");
        }
    }
}
