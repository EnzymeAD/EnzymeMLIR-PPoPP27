"""Specialize the Kiriko PolyBench/C 3.2 MLIR kernels to their benchmark sizes.

Kiriko's kernels take their loop bounds as leading i32 arguments. This replaces every
use of those arguments in the body with the constant that the kernel's main() (in
<k>_prep_mlir.c) passes, keeping the signature unchanged, then canonicalizes so the
affine bounds become constants. Writes benchmarks/kiriko/<k>.mlir.

Usage: python scripts/kiriko_specialize.py KIRIKO/polybench-c-mlir-3.2 [--opt enzymexlamlir-opt] [--out benchmarks/kiriko]
"""
import argparse
import glob
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def call_args(prep, name):
    src = open(prep).read()
    main = src[src.index("int main("):]
    ints = dict(re.findall(r"\bint\s+(\w+)\s*=\s*(-?\d+)\s*;", main))
    m = re.search(rf"\b{re.escape(name)}\s*\((.*?)\);", main, re.S)
    args = [a.strip() for a in m.group(1).split(",")]
    return args, ints


def specialize(kernel, prep, opt):
    src = open(kernel).read()
    m = re.search(r"func\.func @(\w+)\((.*?)\)\s*(attributes\s*\{[^}]*\})?\s*\{", src, re.S)
    name, params = m.group(1), m.group(2)
    args, ints = call_args(prep, name)
    consts = []
    for pos, (arg, ty) in enumerate(re.findall(r"(%arg\d+)\s*:\s*([^,{]+?)(?:\s*\{[^}]*\})?\s*(?:,|$)", params)):
        if ty.strip() != "i32":
            continue
        val = ints.get(args[pos])
        if val is None and re.fullmatch(r"-?\d+", args[pos]):
            val = args[pos]
        if val is None:
            raise SystemExit(f"{kernel}: no constant for {arg} (call arg {args[pos]!r})")
        consts.append((arg, val))
    head, body = src[:m.end()], src[m.end():]
    for arg, val in consts:
        body = re.sub(rf"{arg}\b", f"%size_{arg[1:]}", body)
    defs = "".join(f"\n    %size_{arg[1:]} = arith.constant {val} : i32" for arg, val in consts)
    text = head + defs + body
    out = subprocess.run([opt, "--canonicalize", "-"], input=text, capture_output=True, text=True, check=True).stdout
    return out, consts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="Kiriko checkout's polybench-c-mlir-3.2 directory")
    p.add_argument("--opt", default=os.environ.get("OPT", "enzymexlamlir-opt"))
    p.add_argument("--out", default=os.path.join(ROOT, "benchmarks/kiriko"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for kernel in sorted(glob.glob(os.path.join(a.dir, "**/MLIR/*_kernel.mlir"), recursive=True)):
        k = os.path.basename(kernel)[: -len("_kernel.mlir")]
        prep = os.path.join(os.path.dirname(kernel), f"{k}_prep_mlir.c")
        out, consts = specialize(kernel, prep, a.opt)
        open(os.path.join(a.out, f"{k}.mlir"), "w").write(out)
        print(k, " ".join(f"{x}={v}" for x, v in consts))


if __name__ == "__main__":
    main()
