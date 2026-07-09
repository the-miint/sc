#!/usr/bin/env bash
# M11 wasm dependency-surface gate.
#
# The shipped crate — sc-core built for wasm32-unknown-unknown — must link none
# of the native-only heavy dependencies. This codifies design §11's "no
# rayon / getrandom / filesystem / arrow-ipc in the wasm build" so a future
# dependency bump can't silently regress the WASM story. CI runs it; `make
# wasm-audit` runs it locally.
#
# It inspects both shipped feature combos (bare and +persist). `-e normal` keeps
# the scan to runtime edges — crucially, it drops the *dev*-dependency `rayon`
# (declared for the native thread-count-invariance test), which would otherwise
# false-positive here.
set -euo pipefail

TARGET="wasm32-unknown-unknown"

# Crate-NAME prefixes that must never appear in the wasm runtime tree. `arrow`
# covers arrow-array/-ipc/-schema/…; `crossbeam` covers the crossbeam-* crates
# rayon pulls; `getrandom` is the OS-entropy backend we deliberately never link.
FORBIDDEN=(rayon getrandom crossbeam arrow)

FEATURE_SETS=(
  "--no-default-features"
  "--no-default-features --features persist"
)

fail=0
for features in "${FEATURE_SETS[@]}"; do
  echo "== cargo tree -p sc-core --target ${TARGET} ${features} -e normal =="
  # --prefix none => one "name vX.Y.Z" line per unique crate (no tree glyphs),
  # so the crate name is the first token and easy to anchor on.
  # shellcheck disable=SC2086  # ${features} must word-split into separate flags
  #   (e.g. "--no-default-features --features persist"); the values are trusted
  #   literals from FEATURE_SETS, so this is intentional, not an injection risk.
  tree="$(cargo tree -p sc-core --target "${TARGET}" ${features} -e normal --prefix none)"
  for crate in "${FORBIDDEN[@]}"; do
    # Anchor on the name field so `arrow` catches `arrow-array` but never
    # false-hits a substring buried inside some other crate name.
    if hits="$(printf '%s\n' "${tree}" | grep -E "^${crate}([ -]|\$)")"; then
      echo "  FORBIDDEN wasm dependency matching '${crate}':"
      printf '%s\n' "${hits}" | sed 's/^/      /'
      fail=1
    fi
  done
done

if [ "${fail}" -ne 0 ]; then
  echo "wasm dep surface DIRTY — a forbidden crate reached the shipped tree (see above)" >&2
  exit 1
fi
echo "wasm dep surface clean"
