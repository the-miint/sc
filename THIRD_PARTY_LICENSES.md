# Third-party licenses

`sc` is BSD-3-Clause (see [LICENSE](LICENSE)). It depends on the third-party
crates below, grouped by *where the code ends up*. Every license here is on the
`deny.toml` allow-list, and `cargo deny check` enforces that at each release.

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
