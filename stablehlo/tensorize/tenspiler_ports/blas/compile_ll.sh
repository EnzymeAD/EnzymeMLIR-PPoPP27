#!/usr/bin/env bash
# Linux equivalent of tenspiler/llama/cpp/compile-add-blocks.sh:
# clang++ 11 with LLVM-11's bundled libc++ headers (-stdlib=libc++, same std::__1 names as the
# shipped macOS .ll), then the AddEmptyBlocks pass + lowerinvoke/unreachableblockelim/instnamer,
# then the loop analysis.  Usage: compile_ll.sh file.cc [outdir]
set -e
T=${TENSPILER:?set TENSPILER to the Tenspiler checkout (with its .deps from setup)}
L=$T/.deps/clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04/bin
f=$1; out=${2:-$(dirname $f)}; base=$(basename ${f%.*})
mkdir -p $out
$L/clang++ -stdlib=libc++ -O0 -I$T/headers -c -emit-llvm -fno-discard-value-names -S "$f" -o "$out/$base.ll"
$L/opt -load $T/llvm-pass/build/addEmptyBlocks/libAddEmptyBlocksPass.so -addEmptyBlock -lowerinvoke --unreachableblockelim -instnamer -S "$out/$base.ll" > "$out/$base.tmp.ll"
mv "$out/$base.tmp.ll" "$out/$base.ll"
$L/opt -analyze -loops "$out/$base.ll" > "$out/$base.loops"
echo "wrote $out/$base.ll $out/$base.loops"
