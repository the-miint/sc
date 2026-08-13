# Third-party licenses

`sc` is BSD-3-Clause (see [LICENSE](LICENSE)). Sections (a)–(c) below catalog the
third-party *crates*, grouped by *where the code ends up*. Every license there is
on the `deny.toml` allow-list, and `cargo deny check` enforces that at each
release.

Section (d) covers what `cargo deny` cannot see: the upstream project `sc`
reimplements ([q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier))
and the Python oracle environment used to generate the committed test fixtures.
Six of those fixtures embed data derived from the upstream test suite, so (d)
reproduces the applicable BSD-3-Clause notice in full. For the papers to cite,
see [CITATION.cff](CITATION.cff).

This list is compiled from `cargo tree` / `cargo metadata` (not from memory)
against the committed `Cargo.lock`. Regenerate it if the lockfile changes — note
each tier needs a different edge filter, and the build-time tier (c) in
particular requires `-e build` (or `-e normal,build`), since `-e normal` alone
never surfaces build-dependencies such as cbindgen and its subtree:

```sh
# (a) shipped runtime — the wasm surface (persist superset covers the bare set):
cargo tree -p sc-core --target wasm32-unknown-unknown --no-default-features --features persist -e normal
# (b) native FFI runtime — subtract (a) from this:
cargo tree -p sc-capi -e normal
# (c) build-time-only — cbindgen and its full transitive closure:
cargo tree -p cbindgen                 # cbindgen's own tree
cargo metadata --format-version 1 --all-features   # for the license fields
```

Versions are the major/minor lines pinned in `Cargo.lock`; a crate offering a
license choice (`A OR B`) is used under the terms compatible with BSD-3-Clause
redistribution.

## (a) Shipped in the `sc-core` library — wasm **and** native

These are the runtime crates linked into every `sc-core` build, including the
`wasm32-unknown-unknown` artifact. All are permissive (no copyleft); this is the
surface `ci/wasm-audit.sh` guards.

| Crate | Version | License |
| --- | --- | --- |
| rand | 0.10 | MIT OR Apache-2.0 |
| rand_core | 0.10 | MIT OR Apache-2.0 |
| rand_chacha | 0.10 | MIT OR Apache-2.0 |
| ppv-lite86 | 0.2 | MIT OR Apache-2.0 |
| zerocopy | 0.8 | BSD-2-Clause OR Apache-2.0 OR MIT |
| statrs | 0.18 | MIT |
| approx | 0.5 | Apache-2.0 |
| num-traits | 0.2 | MIT OR Apache-2.0 |

With the default-on `persist` feature (serde + postcard; still wasm-clean):

| Crate | Version | License |
| --- | --- | --- |
| serde | 1.0 | MIT OR Apache-2.0 |
| serde_core | 1.0 | MIT OR Apache-2.0 |
| postcard | 1.1 | MIT OR Apache-2.0 |
| cobs | 0.3 | MIT OR Apache-2.0 |
| thiserror | 2.0 | MIT OR Apache-2.0 |

## (b) Native FFI stack only — `sc-arrow` / `sc-capi` (never wasm)

These are linked only into the native FFI libraries (the `sc-capi` staticlib /
cdylib), never into the shipped wasm artifact. The Arrow C-Data-Interface stack
and the rayon thread pool dominate; the rest are their transitive dependencies.

| Crate | Version | License |
| --- | --- | --- |
| arrow-array | 59 | Apache-2.0 AND MIT |
| arrow-buffer | 59 | Apache-2.0 |
| arrow-data | 59 | Apache-2.0 |
| arrow-schema | 59 | Apache-2.0 |
| ahash | 0.8 | MIT OR Apache-2.0 |
| hashbrown | 0.17 | MIT OR Apache-2.0 |
| half | 2.7 | MIT OR Apache-2.0 |
| bytes | 1.12 | MIT |
| chrono | 0.4 | MIT OR Apache-2.0 |
| iana-time-zone | 0.1 | MIT OR Apache-2.0 |
| num-bigint | 0.4 | MIT OR Apache-2.0 |
| num-complex | 0.4 | MIT OR Apache-2.0 |
| num-integer | 0.1 | MIT OR Apache-2.0 |
| bitflags | 2 | MIT OR Apache-2.0 |
| cfg-if | 1 | MIT OR Apache-2.0 |
| once_cell | 1 | MIT OR Apache-2.0 |
| libc | 0.2 | MIT OR Apache-2.0 |
| libm | 0.2 | MIT |
| either | 1 | MIT OR Apache-2.0 |
| rayon | 1 | MIT OR Apache-2.0 |
| rayon-core | 1 | MIT OR Apache-2.0 |
| crossbeam-deque | 0.8 | MIT OR Apache-2.0 |
| crossbeam-epoch | 0.9 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8 | MIT OR Apache-2.0 |
| getrandom | 0.3 | MIT OR Apache-2.0 |

`getrandom` is pulled by `ahash`'s runtime RNG and is **native-only** — `sc-core`
disables OS entropy precisely so `getrandom` never reaches the wasm surface
(seeds are always explicit; see design §11). Its per-target backends
(`r-efi`, `wasi`, `wasip2`; multi-licensed, all containing an Apache-2.0 or MIT
option) are compiled only for their respective targets, not the native Linux
artifact.

## (c) Build-time only — not linked into or shipped with any artifact

These run at compile time (code generators / proc-macros / build scripts) and
leave no code in the produced libraries.

| Crate | Version | License | Role |
| --- | --- | --- | --- |
| cbindgen | 0.29 | MPL-2.0 | Regenerates `include/sc.h` for `sc-capi` |
| serde_derive | 1.0 | MIT OR Apache-2.0 | `#[derive(Serialize/Deserialize)]` |
| thiserror-impl | 2.0 | MIT OR Apache-2.0 | `#[derive(Error)]` |
| zerocopy-derive | 0.8 | BSD-2-Clause OR Apache-2.0 OR MIT | `zerocopy` derives |
| syn | 2.0 | MIT OR Apache-2.0 | proc-macro parsing |
| quote | 1.0 | MIT OR Apache-2.0 | proc-macro codegen |
| proc-macro2 | 1.0 | MIT OR Apache-2.0 | proc-macro tokens |
| unicode-ident | 1.0 | (MIT OR Apache-2.0) AND Unicode-3.0 | ident validation |
| tiny-keccak | 2.0 | CC0-1.0 | `ahash` compile-time-rng (native build) |
| const-random | 0.1 | MIT OR Apache-2.0 | `ahash` compile-time-rng |
| const-random-macro | 0.1 | MIT OR Apache-2.0 | `ahash` compile-time-rng |

cbindgen (the only MPL-2.0 crate, and a build tool — see below) additionally
pulls its full parsing + scratch-file toolchain at build time, none of which
lands in any shipped artifact. All are MIT and/or Apache-2.0 except where noted:

- TOML/config parsing: `toml`, `toml_datetime`, `toml_parser`, `serde_json`,
  `serde_spanned`, `winnow` (two versions, 0.7 + 1.0), `indexmap`, `equivalent`,
  `hashbrown`, `heck`, `log`, `itoa`, `memchr`, `zmij`.
- Temp-file scratch space (`tempfile`) and its OS/entropy backends: `fastrand`,
  `rustix`, `linux-raw-sys`, `windows-sys` (Windows only), plus a build-only
  `getrandom` 0.4 and its per-target `r-efi` backend (multi-licensed with an
  Apache-2.0/MIT option) — distinct from the native-runtime getrandom in (b).
- Shared build helpers already listed above (`syn`/`quote`/`proc-macro2`/
  `unicode-ident`, `libc`, `cfg-if`, `once_cell`, `bitflags`).

Two license terms here exist **only** because of build/native trees and never
touch the wasm artifact: `MPL-2.0` (cbindgen — a build tool; its file-level
copyleft covers cbindgen's own sources, not the generated header nor our
BSD-3-Clause code) and `CC0-1.0` (tiny-keccak — a public-domain dedication under
`ahash`). Both are documented in `deny.toml`. `Unicode-3.0` enters via
`unicode-ident`, a compile-time proc-macro helper.

Not listed: the dev-only `sc-oracle` crate and its `serde` / `serde_json`
dependency (both MIT OR Apache-2.0) are used solely by the test suite — they are
never compiled into `sc-core`, the native FFI libraries, or the wasm artifact.

## (d) Upstream reference implementation and oracle environment

`cargo deny` sees only the Cargo graph, so it cannot cover this tier: the project
`sc` reimplements, and the Python environment whose outputs are committed as the
oracle fixtures under [`oracle/fixtures/`](oracle/fixtures). None of it is a
build or runtime dependency of any `sc` artifact — reproducing the fixtures
(`make regen-fixtures`) is the only operation that needs it, and the fixtures
themselves are consumed only by the test suite. Versions below are the resolved
`qiime2-amplicon-2024.10` environment recorded in every fixture's `generator`
block; licenses are the package metadata from that environment.

| Project | Version | License | Role in `sc` |
| --- | --- | --- | --- |
| [q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier) | 2024.10.0 | BSD-3-Clause | **The reimplemented upstream.** Defines the workflows and reported metrics; `visuals.py` is the arbiter for `sc-core::metrics`, and its test suite supplies two datasets (below) |
| qiime2 | 2024.10.1 | BSD-3-Clause | Loads the `.qza` tables in the aggregate generators |
| scikit-learn | 1.4.2 | BSD-3-Clause | Numeric oracle for trees, forests, RFECV, CV splits, metrics |
| scipy | 1.13.0 | BSD-3-Clause | Oracle for `linregress` / rank statistics |
| numpy | 1.26.4 | BSD-3-Clause | Seeded generation of the synthetic cases |
| pandas | 2.2.2 | BSD-3-Clause | Sample-metadata handling in the generators |
| biom-format | 2.1.15 | BSD-3-Clause | Feature-table view behind the `.qza` loads |
| [shap](https://github.com/shap/shap) | 0.49.1 | MIT | Oracle for path-dependent TreeSHAP (`gen_shap.py`); not used by q2 itself |

`sc-core`'s implementation is written from the documented behavior of these
libraries and validated numerically against them; no upstream source is vendored
into any `sc` crate. `crates/sc-core/src/metrics/` is the closest coupling — a
deliberate port of the values `q2-sample-classifier`'s `visuals.py` reports — and
is attributed as such in the module documentation.

### Redistributed fixture data — BSD-3-Clause notice retained

Most fixtures are built from seeded synthetic data and carry no third-party
copyright. **Six do not.** These embed feature-table values and sample metadata
derived from real datasets distributed in the `q2-sample-classifier` test suite
(upstream paths relative to `q2_sample_classifier/tests/data/`), and are
therefore redistributions of third-party material:

| Fixture | Upstream source files | Samples × features |
| --- | --- | --- |
| `oracle/fixtures/forest/agg_chardonnay.json` | `chardonnay.table.qza` + `chardonnay.map.txt` (`Region`) | 21 × 281 |
| `oracle/fixtures/ncv/agg_chardonnay.json` | same | 21 × 281 |
| `oracle/fixtures/rfe/agg_chardonnay.json` | same | 21 × 40 |
| `oracle/fixtures/forest/agg_ecam.json` | `ecam-table-maturity.qza` + `ecam_map_maturity.txt` (`month`) | 126 × 150 |
| `oracle/fixtures/ncv/agg_ecam.json` | same | 126 × 150 |
| `oracle/fixtures/rfe/agg_ecam.json` | same | 126 × 40 |

The values are not verbatim copies — the generators load each table through
`qiime2`/`biom`, apply the filtering and feature subsetting described in each
generator, and emit COO triplets alongside the reference score bands — but they
are *derived from* the upstream files, so BSD-3-Clause condition 1 applies and
the notice is retained here in full:

```
BSD 3-Clause License

Copyright (c) 2017-2026, QIIME 2 development team.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Since the fixtures are read only by the test suite, this material never reaches
`sc-core`, the native FFI libraries, or the shipped wasm artifact — no binary
distribution of `sc` contains it.

The two datasets originate from published studies; cite those, not the test
suite, if you reuse the data. They are listed with the rest of the references in
[CITATION.cff](CITATION.cff).
