#!/bin/bash
# Creates .venv/ for the benchmark: legoESM (submodule, from its uv.lock), CUDA JAX, and Enzyme-JAX.
# usage: ./setup_env.sh                   released enzyme-ad from PyPI (version below)
#        ./setup_env.sh path/to/enzyme_ad-*.whl   a locally built Enzyme-JAX wheel instead
# requires: uv (https://docs.astral.sh/uv/), CUDA 12 driver
set -euo pipefail
cd "$(dirname "$0")"
JAX_VERSION=0.10.0
ENZYME_AD=${1:-enzyme-ad==0.0.16}
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

git submodule update --init legoESM
(cd legoESM && uv sync --frozen --python 3.13)
uv pip install --python .venv/bin/python "jax[cuda12]==$JAX_VERSION"
uv pip install --python .venv/bin/python --no-deps --reinstall "$ENZYME_AD"
.venv/bin/python -c "
import importlib.metadata as m, jax, enzyme_ad.jax
print('jax', jax.__version__, 'enzyme-ad', m.version('enzyme-ad'), jax.devices())"
