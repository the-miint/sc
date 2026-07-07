//! Parallel/sequential execution shim.
//!
//! `indexed_map` returns an identically-ordered `Vec` regardless of the
//! `parallel` feature, so results are invariant to thread count. This is the
//! seam the forest's per-tree build uses in M4.

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
pub fn indexed_map<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::indexed_map;
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
}
