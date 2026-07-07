//! Deterministic RNG boundary.
//!
//! No code outside this module names a concrete PRNG. Reproducibility
//! (invariant across platforms and library versions) and thread-count
//! invariance both depend on the primitives here. See `sc-design.html` §7.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Derive an independent, deterministic per-tree seed from a master seed and a
/// tree index.
///
/// Pure function so every tree's seed can be computed independently (no shared
/// generator to serialize across threads), which is what makes forest output
/// invariant to thread count.
pub fn tree_seed(master_seed: u64, index: u64) -> u64 {
    // SplitMix64 finalizer over `master_seed + index * GOLDEN`. Stateless and
    // well-mixed, so distinct indices yield distinct, decorrelated seeds.
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut z = master_seed.wrapping_add(index.wrapping_mul(GOLDEN));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic, value-stable, portable RNG.
///
/// Wraps `ChaCha8Rng` and exposes only the sampling operations callers need,
/// so the concrete generator never leaks into the rest of the crate. Never
/// `StdRng`/`SmallRng` (whose streams are not stable across versions/platforms).
pub struct DetRng(ChaCha8Rng);

impl DetRng {
    /// Construct from an explicit `u64` seed.
    pub fn from_seed(seed: u64) -> Self {
        DetRng(ChaCha8Rng::seed_from_u64(seed))
    }

    /// Construct the RNG for tree `index` under `master_seed`.
    pub fn for_tree(master_seed: u64, index: u64) -> Self {
        Self::from_seed(tree_seed(master_seed, index))
    }

    /// Draw the next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Draw an unbiased uniform integer in `[0, bound)`.
    ///
    /// Portable across x86/ARM/wasm32 (pure `u64`, no `u128`, no unsafe). Uses
    /// rejection on the incomplete low block: with `2^64 = q·bound + r`, the
    /// range `[r, 2^64)` has length `q·bound`, a whole multiple of `bound`, so
    /// every residue occurs exactly `q` times there — rejecting `x < r` (where
    /// `r = 2^64 mod bound`, computed as `(-bound) mod bound`) removes the bias
    /// of a plain `x % bound`.
    ///
    /// This algorithm is **frozen**: its output is a cross-platform/version
    /// reproducibility contract (see the golden vectors in
    /// `tests/determinism.rs`). Panics if `bound == 0`.
    pub fn uniform_u64(&mut self, bound: u64) -> u64 {
        assert!(bound != 0, "uniform_u64: bound must be nonzero");
        let thresh = bound.wrapping_neg() % bound; // 2^64 mod bound
        loop {
            let x = self.next_u64();
            if x >= thresh {
                return x % bound;
            }
        }
    }

    /// Shuffle `slice` in place with the Fisher–Yates (Durstenfeld) algorithm.
    ///
    /// **Frozen direction**: `i` runs from `len-1` down to `1`, swapping index
    /// `i` with `uniform_u64(i + 1)`. No-op for `len <= 1`. Deterministic and
    /// portable because every index draw goes through [`Self::uniform_u64`]. We
    /// deliberately do not use `rand`'s `SliceRandom::shuffle`, whose algorithm
    /// could drift across versions. See the golden vectors in
    /// `tests/determinism.rs`.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.uniform_u64(i as u64 + 1) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DetRng;

    #[test]
    fn uniform_bound_one_is_always_zero() {
        let mut r = DetRng::from_seed(1);
        for _ in 0..100 {
            assert_eq!(r.uniform_u64(1), 0);
        }
    }

    #[test]
    fn uniform_draws_stay_below_bound() {
        let mut r = DetRng::from_seed(7);
        for bound in [2u64, 3, 6, 10, 97, 1000] {
            for _ in 0..10_000 {
                assert!(r.uniform_u64(bound) < bound);
            }
        }
    }

    #[test]
    #[should_panic(expected = "bound must be nonzero")]
    fn uniform_zero_bound_panics() {
        DetRng::from_seed(0).uniform_u64(0);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<usize> = (0..500).collect();
        DetRng::from_seed(123).shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>(), "not a permutation");
    }

    #[test]
    fn shuffle_is_noop_for_trivial_slices() {
        let mut empty: [u8; 0] = [];
        DetRng::from_seed(0).shuffle(&mut empty);
        let mut one = [42u8];
        DetRng::from_seed(0).shuffle(&mut one);
        assert_eq!(one, [42]);
    }

    #[test]
    fn shuffle_is_seed_deterministic() {
        let base: Vec<u32> = (0..64).collect();
        let mut a = base.clone();
        let mut b = base.clone();
        DetRng::from_seed(9).shuffle(&mut a);
        DetRng::from_seed(9).shuffle(&mut b);
        assert_eq!(a, b, "same seed must give the same permutation");
        let mut c = base.clone();
        DetRng::from_seed(10).shuffle(&mut c);
        assert_ne!(a, c, "different seeds should (almost surely) differ");
    }
}
