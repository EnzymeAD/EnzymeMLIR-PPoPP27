# Known issues and caveats

Things to keep in mind when reading the tables. All of them were found while running this comparison.

## Tensorize programs

- **Wrong programs (not timed).** These Tensorize programs compute something other than their kernel.
  They are listed in `meta/<suite>/tensorize_broken.txt`, and `time.sh` reports them as broken:

  | suite | kernel | problem |
  |---|---|---|
  | polybench | syrk, syr2k | `restore_sizes` misses `np.tri(5)`: the mask stays 5x5 instead of 1200x1200 |
  | blend | color_burn_8 | `restore_sizes` misses `np.eye(3)`: a 3x3 mask against 1000x1000 arrays |
  | blend | darken_blend_8, lighten_blend_8 | `where`'s condition and value are swapped |
  | blend | dissolve_blend_8 | the expression does not match the loop |
  | darknet | ol_l2_cpu1 | `2 ** (a - b)` instead of `(b - a) ** 2` |
  | dspstone | pin_down | `restore_sizes` misses `np.full([3], 1)` |
  | utdsp | histogram | `restore_sizes` misses `np.full([3], 0)` |

- **Partial programs.** For bicg and mvt, Tensorize computes only one of the kernel's two outputs (q
  and x2). `raise.sh` therefore also raises a version that returns only that output (`<suite>_1out`).
  The tables compare against that version, so both sides compute the same thing.
- **Synthesis timeout.** Each `sympy.solve` call has a wall-clock limit (`TENSORIZE_SOLVE_TIMEOUT`,
  1 s by default). On a loaded machine this rejects correct sketches. Kiriko gemm, 2mm and syrk
  failed with 1 s and succeeded with 20 s; the shipped programs for those three come from the 20 s
  run. Synthesis failures in `programs/tensorize/synth_times.csv` may be partly due to load.
- **Our fixes** (`patches/tensorize-fixes.patch`, applied by `setup/tensorize.sh`):
  - the solve timeout above, made configurable;
  - sympy treats integral floats as different from integers (`A > 0` vs `A > 0.0`), so they are now
    compared as equal;
  - an `AssertionError` raised inside sympy/mpmath no longer crashes synthesis;
  - `scf.if` is now emitted as Python, not C.

## Kernel definitions

These kernels in the suites don't compute what their name says. Both tools are compared on the
kernel as written.

- **blas/gemv** reads `x[i]` (the row index) instead of `x[j]`. It computes `x[i] * sum_j A[i][j]`,
  not `A·x`.
- **blend**:
  - **color_burn_8** computes `32 - (32-(32-b))/a`, i.e. `32 - b/a`, not `32 - (32-b)/a`.
  - **color_dodge_8** divides by zero (inf/NaN) where `a = 32`.
  - **dissolve_blend_8** ignores its second image: it stores the scalar
    `opacity - (fmod(rand, 100) + 1)/100` into every element.
- **llama** (relative to Tenspiler's llama kernels):
  - **rmsnorm_part2** divides by the constant 3.0, not by `size`.
  - **transformer_part3** computes `1/(b·(1+e^-a))`, not SiLU `x/(1+e^-x)`.
- **polybench/covariance** is shrunk to 5x3 in the Tensorize suite, so its timings are dominated by
  launch overhead.
- **dsp/matinit**: the function is named `@matadd` (it is an initialisation, `A[i][j] = c`).

## Enzyme-JAX (build `5d830a0`)

- **Very slow kernels.** In some kernels, Enzyme-JAX keeps nested sequential `while` loops.
  - polybench and kiriko symm: one call takes more than 30 min on an RTX 5090, and 8-17 min on CPU.
  - Kiriko cholesky, ludcmp and trmm: one call takes more than 30 min on an RTX 5090.
  - These kernels, plus kiriko dynprog and floyd-warshall (next item), are in `$SKIP`.
  - Others are timed but slow: polybench trmm (~5.8 s vs 1.7 ms for Tensorize), Kiriko trisolv
    (~78 s), and Kiriko adi (~1.3 s).
  - Enzyme-JAX `5ff3f33` raises the same IR as `5d830a0` for all PolyBench and Kiriko kernels
    except fdtd-apml, so these results also hold at that revision.
- **[EnzymeAD/Enzyme-JAX#3598](https://github.com/EnzymeAD/Enzyme-JAX/issues/3598)** (open). The
  pattern `remove_loop_carried_dependencies_from_while_load_operations` makes an in-place loop read
  its initial buffer, which forces XLA to copy the whole buffer every time the loop is entered.
  - Affects Kiriko durbin (~0.1 s before #3587, ~37 s at `5d830a0` on CPU), dynprog and
    floyd-warshall (2·N² full-matrix copies).
  - Output is unchanged. Removing the pattern from the pipeline restores the run time.
- **[#3590](https://github.com/EnzymeAD/Enzyme-JAX/issues/3590)** (fixed by #3592, included here).
  `iter_args` loops with a nonzero lower bound crashed the raising (Kiriko ludcmp, reg_detect).
- **Inherently sequential kernels are slower on GPU than on CPU.** Each loop iteration becomes small
  kernel launches. Examples: cholesky, durbin, ludcmp, Kiriko trmm, reg_detect, trisolv,
  jacobi-1d. Run with `JAX_PLATFORMS=cpu` to compare.

## Tenspiler

- **Integers only.** Tenspiler synthesizes and verifies integer programs: `vector<int>`, integer
  division, and `exp`/`sqrt` replaced by identity stubs that it treats as uninterpreted functions.
  - Float code appears only at code generation, which prints the stubs as `np.exp`/`np.sqrt`.
  - The generated code's float behaviour was checked only by testing against float64 references.
- **Vacuous verification.** Tenspiler's own cvc5 check reports "verified", but its background
  axioms are contradictory on their own:
  - `matrix_transpose`'s definition is unsatisfiable;
  - several "prepend" axioms are false at index 0.

  After fixing them, cvc5 proves only llama softmax_part1 and softmax_part3 within 300 s.
- **Ports.** blas and dsp are not Tenspiler benchmarks; we ported them (`tenspiler_ports/`).
  - Two needed a small added TensIR op: ger (`vec_outer`) and dsp matmul (`matrix_matrix_mul`).
  - The ports return new arrays where our kernels update in place.
  - matinit and vfill are written as `c + 0*A`, which gives NaN for non-finite inputs.
- **Wrong or mismatched generated code.**
  - llama transformer_part1 uses floor division `//` for a float divide.
  - blend color_burn_8 and color_dodge_8 match Tenspiler's C++ only for pixels in [0, 32].
  - blend dissolve_blend_8 does not match Tenspiler's C++.
- **Timed alone.** Tenspiler's code is timed on its own, while Tensorize and Enzyme-JAX are timed
  alternately call by call.

## Timing

- **Launch overhead.** Most software-suite kernels run in 0.04-0.1 ms on GPU, which is dominated by
  launch overhead. Differences below ~0.03 ms are noise.
- **First-call effect.** The first timed call after the warm-up is consistently slower: a median of
  1.35x the others in our runs. Tensorize runs first in each round, so without discarding it
  Tensorize looked ~10% slower on tiny kernels. `DISCARD=1` (the default) drops it.
- **Load sensitivity.** CPU timings in particular depend on machine load. `run_info.txt` records the
  load at the start of a run.
