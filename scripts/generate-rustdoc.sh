#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-target/rustdoc}"

if [[ ! -f "Cargo.toml" ]]; then
  echo "Please run this script in the root directory of the project that contains Cargo.toml." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export RUSTDOCFLAGS="--cfg docsrs"

cargo doc --no-deps --document-private-items --target-dir "$OUTPUT_DIR"

echo "rustdoc has been generated to $OUTPUT_DIR/doc"
