#!/bin/bash
# (Only needed to re-run synthesis, scripts/synth.sh.) Apply our fixes to the Tensorize submodule and
# build it with its own scripts (builds LLVM/MLIR and MLIR-HLO first: several hours), plus its
# Python 3.10 environment. Dependencies: see tensorize/Dockerfile (ninja, lld, clang, clang-format,
# jq, bison, flex, libboost, ...).
# Usage: setup/tensorize.sh -> tensorize/build, build/venv-tensorize
set -e
source "$(dirname "$0")/../scripts/env.sh"
cd "$TENSORIZE"
if git apply --check "$HERE/patches/tensorize-fixes.patch" 2> /dev/null; then
  git apply "$HERE/patches/tensorize-fixes.patch"
  echo "applied patches/tensorize-fixes.patch"
else
  git apply --reverse --check "$HERE/patches/tensorize-fixes.patch" && echo "patch already applied"
fi
./build_tools/build_dependencies.sh
./build_tools/build.sh
venv=$BUILD/venv-tensorize
python3.10 -m venv "$venv"
"$venv/bin/pip" install -r requirements.txt pymlir==0.5 pyparsing==3.3.3 toml==0.10.2
