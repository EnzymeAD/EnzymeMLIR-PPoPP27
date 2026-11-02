#!/bin/bash
# Python environment for the timing runs: jax 0.11.2 and enzyme-ad 0.0.15 (for enzyme_ad.jax.hlo_call,
# which runs Enzyme-JAX's StableHLO inside a jitted JAX function).
# Usage: setup/jax.sh [cuda12|cpu]   (default cuda12) -> build/venv-jax
set -e
source "$(dirname "$0")/../scripts/env.sh"
variant=${1:-cuda12}
venv=$BUILD/venv-jax
PYTHON=${PYTHON:-python3.12}
mkdir -p "$BUILD"
if command -v uv > /dev/null; then
  uv venv --python "$PYTHON" "$venv"
  uv pip install --python "$venv/bin/python" -r "$HERE/requirements/jax-$variant.txt"
else
  "$PYTHON" -m venv "$venv"
  "$venv/bin/pip" install -r "$HERE/requirements/jax-$variant.txt"
fi
"$venv/bin/python" -c 'import jax, enzyme_ad.jax; print("jax", jax.__version__, jax.devices())'
