//! Regenerate the C header (`include/sc.h` at the repo root) on every build, so
//! it can never drift from the `#[no_mangle]` surface. The header is a build
//! product (gitignored); cbindgen is a build-time-only dependency.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Repo root is two levels up from crates/sc-capi.
    let header = crate_dir.join("../../include/sc.h");

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("read crates/sc-capi/cbindgen.toml");

    let bindings = cbindgen::generate_with_config(&crate_dir, config)
        .expect("cbindgen failed to generate the C header");

    std::fs::create_dir_all(header.parent().unwrap()).expect("create include/");
    bindings.write_to_file(&header);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/convert.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
}
