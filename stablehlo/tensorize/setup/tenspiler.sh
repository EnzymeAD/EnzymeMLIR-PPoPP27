#!/bin/bash
# (Only needed to re-run Tenspiler synthesis, scripts/tenspiler_synth.sh; the timings use the shipped
# programs/tenspiler code.) Clone Tenspiler at the revision used, apply our one fix (Rosette's
# Bitwuzla path was hard-coded to the authors' machine), and install its toolchain locally (no root):
# Racket 8.7 + Rosette 4.0, Bitwuzla 0.7.0 (built from source), the repo's cvc5 1.1.2, LLVM/Clang
# 11.1.0 (release tarball; also provides the libc++ headers the frontend expects), the AddEmptyBlocks
# LLVM pass, and a Python 3.10 venv. Needs uv, git, curl, cmake, a C/C++ compiler and GMP.
# Usage: setup/tenspiler.sh -> build/tenspiler (+ build/tenspiler/env.sh)
set -e
source "$(dirname "$0")/../scripts/env.sh"
TENSPILER_REV=${TENSPILER_REV:-9705480954ec7b67bb7f1d29bc5566515635df30}
T=$BUILD/tenspiler
D=$T/.deps
mkdir -p "$BUILD"
[ -d "$T/.git" ] || git clone https://github.com/tenspiler/tenspiler "$T"
git -C "$T" checkout -q "$TENSPILER_REV"
git -C "$T" apply --check "$HERE/patches/tenspiler-bitwuzla-path.patch" 2> /dev/null \
  && git -C "$T" apply "$HERE/patches/tenspiler-bitwuzla-path.patch"
mkdir -p "$D"

# Python (versions from Tenspiler's poetry.lock; the TensorFlow/PyTorch/MLX packages are not needed:
# synthesis and code generation never import them)
uv venv --python 3.10 "$T/.venv"
VIRTUAL_ENV=$T/.venv uv pip install llvmlite==0.39.1 mypy==1.13.0 "networkx>=2.6.3,<3" numpy==1.26.2 \
  pydash "pyparsing>=3.0.8,<4" regex==2023.10.3 meson ninja

# Racket 8.7 + Rosette
if [ ! -x "$D/racket/bin/racket" ]; then
  curl -sSL -o "$D/racket.sh" https://mirror.racket-lang.org/installers/8.7/racket-8.7-x86_64-linux-cs.sh
  sh "$D/racket.sh" --in-place --dest "$D/racket"
fi
"$D/racket/bin/raco" pkg install --auto -i --batch --skip-installed rosette

# Bitwuzla 0.7.0 (Rosette's solver)
if [ ! -x "$D/bitwuzla/bin/bitwuzla" ]; then
  rm -rf "$D/bitwuzla-src"
  git clone --depth 1 --branch 0.7.0 https://github.com/bitwuzla/bitwuzla "$D/bitwuzla-src"
  (cd "$D/bitwuzla-src" && PATH=$T/.venv/bin:$PATH ./configure.py release --prefix "$D/bitwuzla" \
    && cd build && PATH=$T/.venv/bin:$PATH meson compile && PATH=$T/.venv/bin:$PATH meson install)
fi

# cvc5: the static binary shipped in the repo, behind a wrapper (Tenspiler passes --tlimit=1, i.e.
# 1 ms, which can never prove anything; the wrapper substitutes $CVC5_TLIMIT_MS, default 300 s)
mkdir -p "$D/cvc5-bin"
cat > "$D/cvc5-bin/cvc5" << EOF
#!/usr/bin/env bash
args=()
for a in "\$@"; do
  case "\$a" in --tlimit=*) args+=("--tlimit=\${CVC5_TLIMIT_MS:-300000}") ;; *) args+=("\$a") ;; esac
done
exec "$T/cvc5" "\${args[@]}"
EOF
chmod +x "$D/cvc5-bin/cvc5" "$T/cvc5"

# LLVM/Clang 11.1.0 and the AddEmptyBlocks pass
L=$D/clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04
if [ ! -x "$L/bin/clang++" ]; then
  curl -sSL https://github.com/llvm/llvm-project/releases/download/llvmorg-11.1.0/clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz \
    | tar -xJ -C "$D"
fi
mkdir -p "$T/llvm-pass/build"
(cd "$T/llvm-pass/build" && cmake .. -DLLVM_DIR="$L/lib/cmake/llvm" -DCMAKE_CXX_COMPILER="$L/bin/clang++" \
  -DCMAKE_C_COMPILER="$L/bin/clang" > /dev/null && make -s)

cat > "$T/env.sh" << EOF
# Source this to use the local Tenspiler toolchain.
export PATH="$T/.venv/bin:$D/racket/bin:$D/bitwuzla/bin:$D/cvc5-bin:$L/bin:\$PATH"
export BITWUZLA_PATH="$D/bitwuzla/bin/bitwuzla"
export PYTHONPATH="$T\${PYTHONPATH:+:\$PYTHONPATH}"
export TENSPILER="$T"
EOF
echo "Tenspiler ready: source $T/env.sh"
