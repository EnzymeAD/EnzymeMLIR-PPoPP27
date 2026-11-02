"""Prepare a Tensorize benchmark kernel for Enzyme-JAX's raise-affine-to-stablehlo.

The original kernel is kept unchanged (only made private). For each kernel we add:
  - @<name>_wrap: takes every arg as a memref (scalars as 0-d memrefs), loads the
    0-d memrefs and calls the original kernel. raise-affine-to-stablehlo only
    raises funcs whose args are all memrefs.
  - @main: public entry that calls @<name>_wrap via enzymexla.jit_call on tensors,
    mirroring what Reactant emits. It returns the kernel's output buffers (each
    aliased to its operand): those named with --outputs, or by default every buffer
    the kernel stores to. Naming them leaves scratch buffers (tmp, sum, ...) out, so
    the raised code doesn't compute their final values.

Kernels that return values (scalars or 0-d memrefs, e.g. a reduction returning its
alloca) get one extra 0-d memref parameter per result in the wrapper, which the result
is copied into. @main feeds those from zero constants and returns them after the
output buffers, so its parameters stay exactly the kernel's arguments.

Usage: python benchmark/memrefify.py kernel.mlir [--outputs arg4,arg5] > kernel.in.mlir
"""
import argparse
import re
import sys
from dataclasses import dataclass

SCALAR = r"(?:i\d+|f16|bf16|f32|f64|index)"
ARG = re.compile(r"%(\w+):\s*(memref<[^>]*>|" + SCALAR + r"\b)")
FUNC = re.compile(r"func\.func\s+@([\w.$-]+)\((.*?)\)([^\n]*)\{[ \t]*$", re.S | re.M)
RESULTS = re.compile(r"->\s*\(?\s*([^(){]*?)\s*\)?\s*(?:attributes|$)")
STORE = re.compile(r"(?:affine|memref)\.store\s+[^,]+,\s*%(\w+)\[")


@dataclass
class Kernel:
    name: str
    args: list  # [(name, type)], scalars as plain element types
    outputs: list  # indices of the args returned from @main
    results: list  # [(type, element type)] of the kernel's results (scalars or 0-d memrefs)

    @property
    def memref_types(self):
        return [ty if ty.startswith("memref<") else f"memref<{ty}>" for _, ty in self.args]


def func_body(src, start):
    """Return the text of the region whose '{' ends right before `start`."""
    depth = 1
    for i in range(start, len(src)):
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        if depth == 0:
            return src[start:i]
    raise ValueError("unbalanced braces")


def parse_kernels(src, output_names=None):
    kernels = []
    for m in FUNC.finditer(src):
        args = ARG.findall(m.group(2))
        names = [name for name, _ in args]
        stored = set(STORE.findall(func_body(src, m.end())))
        results = parse_results(m.group(1), m.group(3))
        if output_names:
            for name in output_names:
                if name not in names:
                    sys.exit(f"error: @{m.group(1)} has no argument %{name}")
                if name not in stored:
                    print(f"warning: @{m.group(1)} never stores to output %{name}", file=sys.stderr)
            outputs = [names.index(name) for name in output_names]
        else:
            outputs = [i for i, name in enumerate(names) if name in stored]
        if not outputs and not results:
            print(f"warning: no stores found in @{m.group(1)}, returning all buffers", file=sys.stderr)
            outputs = [i for i, (_, ty) in enumerate(args) if ty.startswith("memref<")]
        kernels.append(Kernel(m.group(1), args, outputs, results))
    return kernels


def parse_results(name, signature_tail):
    """[(type, element type)] of a kernel's results; only scalars and 0-d memrefs are supported."""
    m = RESULTS.search(signature_tail.strip())
    if not m:
        return []
    results = []
    for ty in (t.strip() for t in m.group(1).split(",") if t.strip()):
        elem = re.fullmatch(r"memref<(" + SCALAR + r")>|(" + SCALAR + r")", ty)
        if not elem:
            sys.exit(f"error: @{name} returns {ty}; only scalars and 0-d memrefs are supported")
        results.append((ty, elem.group(1) or elem.group(2)))
    return results


def emit_wrapper(k):
    params, loads, operands = [], [], []
    for i, ((_, ty), mty) in enumerate(zip(k.args, k.memref_types)):
        params.append(f"%m{i}: {mty}")
        if ty.startswith("memref<"):
            operands.append(f"%m{i}")
        else:
            loads.append(f"    %s{i} = affine.load %m{i}[] : {mty}\n")
            operands.append(f"%s{i}")
    call_types = ", ".join(ty for _, ty in k.args)
    # One out-param per result; the returned value (or 0-d memref) is copied into it.
    copies = []
    for j, (ty, elem) in enumerate(k.results):
        params.append(f"%o{j}: memref<{elem}>")
        value = f"%res#{j}"
        if ty.startswith("memref<"):
            copies.append(f"    %v{j} = affine.load %res#{j}[] : {ty}\n")
            value = f"%v{j}"
        copies.append(f"    affine.store {value}, %o{j}[] : memref<{elem}>\n")
    result_types = ", ".join(ty for ty, _ in k.results)
    call = f"func.call @{k.name}({', '.join(operands)}) : ({call_types}) -> ({result_types})"
    if k.results:
        call = f"%res:{len(k.results)} = " + call
    return (
        f"  func.func private @{k.name}_wrap({', '.join(params)}) {{\n"
        + "".join(loads)
        + f"    {call}\n"
        + "".join(copies)
        + "    return\n"
        "  }\n"
    )


def emit_main(k, main_name):
    params = [t.replace("memref<", "tensor<", 1) for t in k.memref_types]
    # jit_call operands: the kernel's args, then one zero 0-d tensor per result slot.
    tensors = params + [f"tensor<{e}>" for _, e in k.results]
    zeros = "".join(
        f"    %z{j} = stablehlo.constant dense<{'0.000000e+00' if e.startswith(('f', 'bf')) else '0'}> : tensor<{e}>\n"
        for j, (_, e) in enumerate(k.results)
    )
    operands = [f"%a{i}" for i in range(len(params))] + [f"%z{j}" for j in range(len(k.results))]
    outputs = k.outputs + list(range(len(params), len(tensors)))
    in_sig = ", ".join(tensors)
    out_sig = ", ".join(tensors[i] for i in outputs)
    n = len(outputs)
    # A single result is not a tuple, so it's aliased with empty output_tuple_indices.
    aliases = ",\n      ".join(
        f"#stablehlo.output_operand_alias<output_tuple_indices = [{'' if n == 1 else out}], "
        f"operand_index = {arg}, operand_tuple_indices = []>"
        for out, arg in enumerate(outputs)
    )
    return (
        f"  func.func @{main_name}({', '.join(f'%a{i}: {t}' for i, t in enumerate(params))}) -> ({out_sig}) {{\n"
        + zeros
        + f"    %r:{n} = enzymexla.jit_call @{k.name}_wrap ({', '.join(operands)}) "
        f"<{{output_operand_aliases = [\n      {aliases}]}}> : ({in_sig}) -> ({out_sig})\n"
        f"    return {', '.join(f'%r#{i}' for i in range(n))} : {out_sig}\n"
        "  }\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel", help="MLIR file with the kernel func(s)")
    parser.add_argument("--outputs", help="comma-separated output args, e.g. arg4,arg5 (default: all stored-to)")
    args = parser.parse_args()
    src = open(args.kernel).read()
    # The benchmarks use pre-LLVM-17 op names; upstream renamed them (same NaN semantics).
    src = re.sub(r"\barith\.maxf\b", "arith.maximumf", src)
    src = re.sub(r"\barith\.minf\b", "arith.minimumf", src)
    kernels = parse_kernels(src, args.outputs.split(",") if args.outputs else None)
    text = FUNC.sub(lambda m: m.group(0).replace("func.func @", "func.func private @", 1), src)
    extra = []
    for i, k in enumerate(kernels):
        extra.append(emit_wrapper(k))
        extra.append(emit_main(k, "main" if len(kernels) == 1 else f"main{i}"))
    cut = text.rstrip().rfind("}")
    sys.stdout.write(text[:cut] + "".join(extra) + text[cut:])


if __name__ == "__main__":
    main()
