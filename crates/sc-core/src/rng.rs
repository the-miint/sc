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
}
