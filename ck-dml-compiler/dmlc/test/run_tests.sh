#!/usr/bin/env bash
# Minimal stand-in for LLVM's lit: runs the "// RUN:" lines of each .mlir file.
#   %s is replaced by the path of the test file.
# Usage: run_tests.sh <path-to-dml-opt> [test.mlir ...]
set -u
if [ $# -lt 1 ]; then echo "usage: $0 <path-to-dml-opt> [tests...]" >&2; exit 2; fi
DML_OPT_BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
HERE=$(cd "$(dirname "$0")" && pwd)
LLVM_BIN=${LLVM_BIN:-/usr/lib/llvm-18/bin}
# Make `dml-opt`, `FileCheck` and `not` resolvable inside RUN lines.
TOOLS=$(mktemp -d); trap 'rm -rf "$TOOLS"' EXIT
ln -s "$DML_OPT_BIN" "$TOOLS/dml-opt"
export PATH="$TOOLS:$LLVM_BIN:$PATH"

[ $# -eq 0 ] && set -- "$HERE"/*.mlir
pass=0; fail=0
for test in "$@"; do
  ok=1; ran=0
  while IFS= read -r line; do
    cmd=${line#*RUN:}; cmd=${cmd//%s/$test}; ran=1
    if ! out=$(bash -o pipefail -c "$cmd" 2>&1); then
      ok=0; echo "FAIL $(basename "$test")"; echo "  RUN:$cmd"; echo "$out" | sed 's/^/  | /' | head -20
    fi
  done < <(grep -E '^// RUN:' "$test")
  if [ $ran -eq 0 ]; then ok=0; echo "FAIL $(basename "$test") (no RUN line)"; fi
  if [ $ok -eq 1 ]; then echo "PASS $(basename "$test")"; pass=$((pass+1)); else fail=$((fail+1)); fi
done
echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
