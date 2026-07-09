//! Regenerate the C header (`include/sc.h` at the repo root) on every build, so
//! it can never drift from the `#[no_mangle]` surface. The header is a build
//! product (gitignored); cbindgen is a build-time-only dependency.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Repo root is two levels up from crates/sc-capi.
    let header = crate_dir.join("../../include/sc.h");

    let mut config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("read crates/sc-capi/cbindgen.toml");

    // Emit `#define SC_VERSION "x.y.z"` into the header, keyed off the crate
    // version so it can't drift. cbindgen won't render the `SC_VERSION` Rust
    // const (it can't evaluate `env!`), so we inject the define ourselves via
    // `after_includes` (appended to, not replacing, the Arrow struct block).
    // `sc_version()` returns the same string at runtime; a Rust FFI test and the
    // C harness both assert the two agree.
    let version = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");
    let define = format!("\n#define SC_VERSION \"{version}\"\n");
    config.after_includes = Some(match config.after_includes.take() {
        Some(existing) => format!("{existing}{define}"),
        None => define,
    });

    let bindings = cbindgen::generate_with_config(&crate_dir, config)
        .expect("cbindgen failed to generate the C header");

    std::fs::create_dir_all(header.parent().unwrap()).expect("create include/");
    bindings.write_to_file(&header);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/convert.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
    // Emitting any `rerun-if-*` disables Cargo's "watch the whole package"
    // default, so we must explicitly track the injected version: a bare version
    // bump (no source change) would otherwise skip this script and leave a stale
    // `#define SC_VERSION` in include/sc.h, drifting from `sc_version()`.
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");
}
