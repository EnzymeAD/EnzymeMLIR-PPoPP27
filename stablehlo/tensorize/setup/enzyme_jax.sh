#!/bin/bash
# Build enzymexlamlir-opt at the Enzyme-JAX revision used for the results (ENZYME_JAX_REV in
# scripts/env.sh; 5d830a0, which includes the fix for EnzymeAD/Enzyme-JAX#3590). Needs bazel
# (bazelisk), clang and git, as in the top-level Dockerfile. With the public remote cache the build
# takes ~10 min on a large machine.
# Usage: setup/enzyme_jax.sh -> build/Enzyme-JAX/bazel-bin/enzymexlamlir-opt
set -e
source "$(dirname "$0")/../scripts/env.sh"
src=$BUILD/Enzyme-JAX
mkdir -p "$BUILD"
[ -d "$src/.git" ] || git clone https://github.com/EnzymeAD/Enzyme-JAX "$src"
git -C "$src" fetch -q origin
git -C "$src" checkout -q "$ENZYME_JAX_REV"
(cd "$src" && CC=${CC:-clang} bazel build --config=public_cache -c opt //:enzymexlamlir-opt)
ls -l "$src/bazel-bin/enzymexlamlir-opt"
