#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_dir="$repo_root/third_party/sprintz/cpp/Compress"
output="$repo_root/evidence/sprintz_cli"
system=$(uname -s)
machine=$(uname -m)

case "$system:$machine" in
  Darwin:arm64)
    compiler=${CXX:-/usr/bin/clang++}
    arch_flags="-arch x86_64 -mavx2 -mbmi2"
    ;;
  Darwin:x86_64|Linux:x86_64)
    compiler=${CXX:-c++}
    arch_flags="-mavx2 -mbmi2"
    ;;
  *)
    echo "unsupported Sprintz build target: $system $machine" >&2
    echo "the verified helper requires x86-64 with AVX2 and BMI2" >&2
    exit 1
    ;;
esac

# shellcheck disable=SC2086
"$compiler" $arch_flags -std=c++17 -O3 -DNDEBUG \
  -I"$source_dir" \
  "$repo_root/evidence/sprintz_cli.cpp" \
  "$source_dir/sprintz_delta_rle.cpp" \
  "$source_dir/format.cpp" \
  -o "$output"

echo "$output"
