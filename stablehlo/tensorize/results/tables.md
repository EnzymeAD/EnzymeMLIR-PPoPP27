# Tensorize vs Enzyme-JAX vs Tenspiler

Times in ms (harmonic mean of the kept timed calls); prep and XLA compile in s. T/E < 1: Tensorize faster. — : not available (see status).

## polybench

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| 2mm | 75.7 | 3.2 | 2.06 | 0.085 | 0.072 | 2.05 | 1.00 |  |
| 3mm | 51.4 | 4 | 3.35 | 0.09 | 0.093 | 3.36 | 1.00 |  |
| atax | 54 | 0.16 | 0.117 | 0.085 | 0.065 | 0.113 | 1.03 |  |
| bicg | 41.2 | 0.16 | 0.0855 | 0.082 | 0.11 | 0.0865 | 0.99 | both sides compute only Tensorize's output |
| correlation | — | — | — | 0.1 | 2.8 | 2.49 |  | Tensorize: no synthesized program |
| covariance | 111 | 0.21 | 0.08 | 0.094 | 0.16 | 0.0782 | 1.02 | suite runs it at 5x3 |
| doitgen | 61.8 | 2.2 | 0.678 | 0.088 | 0.058 | 0.67 | 1.01 |  |
| gemm | 45.5 | 2.4 | 1.58 | 0.09 | 0.071 | 1.56 | 1.02 |  |
| gemver | 246 | 0.2 | 0.122 | 0.087 | 0.13 | 0.116 | 1.05 |  |
| gesummv | 43 | 0.14 | 0.0908 | 0.088 | 0.047 | 0.0765 | 1.19 |  |
| mvt | 35 | 0.18 | 0.0945 | 0.083 | 0.13 | 0.098 | 0.96 | both sides compute only Tensorize's output |
| symm | 68.7 | — | — | 0.12 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call) |
| syr2k | 91.6 | — | — | 0.089 | 0.17 | 16.6 |  | Tensorize: broken: restore_sizes misses np.tri(5) -> mask stays 5x5 instead of 1200x1200 |
| syrk | 79.3 | — | — | 0.085 | 0.15 | 16.6 |  | Tensorize: broken: restore_sizes misses np.tri(5) -> mask stays 5x5 instead of 1200x1200 |
| trmm | 64.5 | 2.3 | 1.73 | 0.093 | 0.14 | 5,820 | 0.00 |  |

T/E over 11 kernels: geomean 0.49, median 1.01

## kiriko

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| 2mm | 177 | 2.2 | 2.65 | 0.088 | 0.071 | 2.63 | 1.01 |  |
| 3mm | 96.6 | 2.2 | 3.85 | 0.088 | 0.066 | 3.94 | 0.98 |  |
| adi | — | — | — | 0.11 | 0.2 | 1,340 |  | Tensorize: no synthesized program |
| atax | — | — | — | 0.084 | 0.17 | 0.342 |  | Tensorize: no synthesized program |
| bicg | 74.4 | 0.14 | 0.163 | 0.085 | 0.038 | 0.159 | 1.02 | both sides compute only Tensorize's output |
| cholesky | — | — | — | 0.096 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis fail |
| correlation | — | — | — | 0.1 | 2.7 | 1.32 |  | Tensorize: no synthesized program |
| covariance | 144 | 2.3 | 1.25 | 0.094 | 0.16 | 1.25 | 1.00 |  |
| doitgen | 233 | 2 | 0.377 | 0.09 | 0.066 | 0.376 | 1.00 |  |
| durbin | — | — | — | 0.097 | 0.23 | 157,878 |  | Tensorize: no synthesized program |
| dynprog | — | — | — | 0.11 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis fail |
| fdtd-2d | — | — | — | 0.091 | 0.16 | 2.8 |  | Tensorize: no synthesized program |
| fdtd-apml | — | — | — | 0.24 | 0.31 | 634 |  | Tensorize: no synthesized program |
| floyd-warshall | — | — | — | 0.086 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis timeout |
| gemm | 220 | 2.2 | 1.32 | 0.085 | 0.064 | 1.32 | 1.00 |  |
| gemver | 429 | 0.27 | 0.429 | 0.09 | 0.22 | 0.38 | 1.13 |  |
| gesummv | 279 | 0.18 | 0.251 | 0.085 | 0.047 | 0.243 | 1.03 |  |
| gramschmidt | — | — | — | 0.093 | 0.25 | 7.71 |  | Tensorize: no synthesized program |
| jacobi-1d-imper | — | — | — | 0.088 | 0.15 | 0.851 |  | Tensorize: no synthesized program |
| jacobi-2d-imper | — | — | — | 0.089 | 0.15 | 0.583 |  | Tensorize: no synthesized program |
| lu | — | — | — | 0.096 | 0.16 | 10.8 |  | Tensorize: no synthesized program |
| ludcmp | — | — | — | 0.12 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis fail |
| mvt | 131 | 0.17 | 0.198 | 0.086 | 0.12 | 0.157 | 1.26 | both sides compute only Tensorize's output |
| reg_detect | — | — | — | 0.099 | 0.19 | 433 |  | Tensorize: no synthesized program |
| seidel-2d | — | — | — | 0.098 | 0.18 | 170,136 |  | Tensorize: no synthesized program |
| symm | — | — | — | 0.12 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis fail |
| syr2k | — | — | — | 0.091 | 0.17 | 10.4 |  | Tensorize: no synthesized program |
| syrk | 78 | 2.3 | 1.32 | 0.083 | 0.068 | 1.33 | 0.99 |  |
| trisolv | — | — | — | 0.089 | 0.25 | 77,780 |  | Tensorize: no synthesized program |
| trmm | — | — | — | 0.086 | — | — |  | not timed ($SKIP: > 30 min per Enzyme-JAX call); Tensorize synthesis fail |

T/E over 10 kernels: geomean 1.04, median 1.01

## blas

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | Tenspiler synth | Tenspiler XLA | Tenspiler ms | Tenspiler/E | status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dot | 12.9 | 0.12 | 0.0792 | 0.083 | 0.033 | 0.0654 | 1.21 | 4.2 | 0.12 | 0.0636 | 0.97 |  |
| gemv | 24.7 | 0.12 | 0.0838 | 0.087 | 0.083 | 0.0698 | 1.20 | 4.2 | 0.091 | 0.152 | 2.18 | kernel reads x[i], not x[j]: computes x[i]*sum_j A[i][j], not A·x |
| ger | 19.2 | 0.063 | 0.0717 | 0.085 | 0.036 | 0.0642 | 1.12 | 6.2 | 0.028 | 0.0823 | 1.28 |  |
| gemv (Tenspiler only) |  |  |  |  |  |  |  | 4.2 | 0.09 | 0.0817 |  | no Tensorize/Enzyme-JAX counterpart |

T/E over 3 kernels: geomean 1.18, median 1.20

## blend

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | Tenspiler synth | Tenspiler XLA | Tenspiler ms | Tenspiler/E | status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| color_burn_8 | 70.7 | — | — | 0.087 | 0.067 | 0.086 |  | 24.4 | 0.11 | 0.133 | (1.54) | Tensorize: broken: restore_sizes misses np.eye(3) -> 3x3 mask against 1000x1000 arrays; Tenspiler: different computation (our MLIR computes 32 - b/a) |
| color_dodge_8 | 140 | 0.064 | 0.0897 | 0.085 | 0.035 | 0.0792 | 1.13 | 24.4 | 0.078 | 0.15 | (1.90) | Tenspiler: different computation (our MLIR divides by 0 at a = 32) |
| darken_blend_8 | 128 | — | — | 0.086 | 0.057 | 0.0792 |  | 18.3 | 0.026 | 0.087 | 1.10 | Tensorize: broken: synthesized where(a, b, a > b) instead of where(a > b, b, a) |
| dissolve_blend_8 | 207 | — | — | 0.085 | 0.063 | 0.0823 |  | 68.6 | 0.039 | 0.111 | (1.35) | Tensorize: broken: synthesized expression does not match the loop (rel err 3e2); Tenspiler: different computation (our MLIR ignores `active`) |
| lighten_blend_8 | 91.6 | — | — | 0.083 | 0.055 | 0.065 |  | 24.4 | 0.025 | 0.0677 | 1.04 | Tensorize: broken: synthesized where(a, b, a < b) instead of where(a < b, b, a) |
| linear_burn_8 | 40.9 | 0.056 | 0.055 | 0.084 | 0.027 | 0.0594 | 0.93 | 12.3 | 0.025 | 0.0801 | 1.35 |  |
| linear_dodge_8 | 14.1 | 0.052 | 0.0679 | 0.084 | 0.026 | 0.0666 | 1.02 | 10.3 | 0.023 | 0.0688 | 1.03 |  |
| multiply_blend_8 | 20.9 | 0.055 | 0.0584 | 0.084 | 0.028 | 0.0578 | 1.01 | 10.3 | 0.064 | 0.111 | (1.93) | Tenspiler: Tenspiler divides as integers (floor), ours in float |
| normal_blend_8 | 46.2 | 0.057 | 0.0613 | 0.083 | 0.029 | 0.0617 | 0.99 | 4.2 | 0.028 | 0.0936 | 1.52 |  |
| normal_blend_f | 32.5 | 0.059 | 0.0541 | 0.086 | 0.027 | 0.0506 | 1.07 | 4.2 | 0.026 | 0.169 | 3.33 |  |
| overlay_blend_8 | — | — | — | 0.084 | 0.06 | 0.099 |  | 54.5 | 0.067 | 0.103 | (1.04) | Tensorize: no synthesized program; Tenspiler: Tenspiler divides as integers (floor), ours in float |
| screen_blend_8 | 29.5 | 0.061 | 0.0579 | 0.084 | 0.03 | 0.0551 | 1.05 | 22.4 | 0.063 | 0.141 | (2.56) | Tenspiler: Tenspiler divides as integers (floor), ours in float |

T/E over 7 kernels: geomean 1.03, median 1.02

## darknet

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| gemm_nn | 38.8 | 2.5 | 1.2 | 0.085 | 0.12 | 1.2 | 1.00 |  |
| gemm_nt | 36.8 | 3.2 | 1.34 | 0.083 | 0.074 | 1.29 | 1.04 |  |
| gemm_tn | 38.9 | 2.4 | 1.19 | 0.084 | 0.075 | 1.19 | 1.00 |  |
| gemm_tt | 39 | 2.1 | 1.28 | 0.085 | 0.072 | 1.26 | 1.02 |  |
| mag_array | 8 | 0.11 | 0.0585 | 0.081 | 0.037 | 0.0543 | 1.08 |  |
| matrix_add_matrix | 7.3 | 0.054 | 0.0553 | 0.084 | 0.027 | 0.0541 | 1.02 |  |
| mse_array | 8.3 | 0.11 | 0.0635 | 0.081 | 0.035 | 0.07 | 0.91 |  |
| mult_add_into_cpu | 10.3 | 0.054 | 0.0551 | 0.084 | 0.026 | 0.0521 | 1.06 |  |
| ol_l2_cpu1 | 33.6 | — | — | 0.084 | 0.055 | 0.0781 |  | Tensorize: broken: synthesized 2 ** (a - b) instead of (b - a) ** 2 |
| ol_l2_cpu2 | 11 | 0.052 | 0.0579 | 0.082 | 0.026 | 0.0554 | 1.04 |  |
| scale_array | 6.7 | 0.053 | 0.0549 | 0.085 | 0.025 | 0.0549 | 1.00 |  |
| scale_matrix | 6.7 | 0.058 | 0.0603 | 0.081 | 0.028 | 0.0601 | 1.00 |  |
| sum_array | 7 | 0.11 | 0.0648 | 0.085 | 0.033 | 0.0676 | 0.96 |  |
| translate_array | 7.1 | 0.053 | 0.0549 | 0.082 | 0.024 | 0.0538 | 1.02 |  |

T/E over 13 kernels: geomean 1.01, median 1.02

## dsp

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | Tenspiler synth | Tenspiler XLA | Tenspiler ms | Tenspiler/E | status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matadd | 7.7 | 0.054 | 0.0628 | 0.08 | 0.027 | 0.0609 | 1.03 | 4.3 | 0.055 | 0.0685 | 1.13 |  |
| matinit | 42.6 | 0.058 | 0.0657 | 0.078 | 0.026 | 0.0601 | 1.09 | 4.2 | 0.029 | 0.149 | 2.48 |  |
| matmul | 16.1 | 2.6 | 1.22 | 0.083 | 0.11 | 1.21 | 1.01 | 10.3 | 2.2 | 1.2 | 0.99 |  |
| matscal | 7 | 0.058 | 0.0662 | 0.083 | 0.027 | 0.0637 | 1.04 | 4.3 | 0.03 | 0.128 | 2.01 |  |
| matsub | 7.8 | 0.056 | 0.0592 | 0.085 | 0.027 | 0.056 | 1.06 | 4.3 | 0.031 | 0.0719 | 1.29 |  |
| vadd | 7.2 | 0.052 | 0.0576 | 0.082 | 0.027 | 0.0552 | 1.04 | 4.2 | 0.024 | 0.0669 | 1.21 |  |
| vcopy | 33.9 | 0.028 | 0.0588 | 0.081 | 0.006 | 0.0608 | 0.97 | 2.2 | 0.006 | 0.129 | 2.13 |  |
| vfill | 42 | 0.054 | 0.0583 | 0.08 | 0.024 | 0.0672 | 0.87 | 2.2 | 0.027 | 0.0998 | 1.48 |  |
| vmul | 7.9 | 0.052 | 0.0574 | 0.082 | 0.025 | 0.0538 | 1.07 | 4.2 | 0.023 | 0.0673 | 1.25 |  |
| vneg | 6.4 | 0.056 | 0.0579 | 0.083 | 0.025 | 0.0605 | 0.96 | 2.2 | 0.023 | 0.0668 | 1.10 |  |
| voffset | 7.2 | 0.053 | 0.0675 | 0.082 | 0.026 | 0.0656 | 1.03 | 2.2 | 0.024 | 0.0893 | 1.36 |  |
| vrecip | 6.8 | 0.055 | 0.0532 | 0.085 | 0.027 | 0.0508 | 1.05 | 2.2 | 0.03 | 0.0617 | 1.21 |  |
| vscal | 6.9 | 0.055 | 0.0746 | 0.087 | 0.025 | 0.0659 | 1.13 | 4.2 | 0.021 | 0.149 | 2.26 |  |
| vsub | 10.5 | 0.055 | 0.0857 | 0.081 | 0.024 | 0.0718 | 1.19 | 4.2 | 0.02 | 0.107 | 1.49 |  |
| w_vec | 14.2 | 0.053 | 0.0651 | 0.083 | 0.024 | 0.0579 | 1.13 | 4.2 | 0.02 | 0.15 | 2.60 |  |

T/E over 15 kernels: geomean 1.04, median 1.04

## dspstone

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| mat1x3 | 15.5 | 0.13 | 0.0613 | 0.085 | 0.04 | 0.0582 | 1.05 |  |
| matrix1 | 24.9 | 3.1 | 1.21 | 0.085 | 0.072 | 1.21 | 1.00 |  |
| matrix2 | 25 | 3.1 | 1.25 | 0.087 | 1.8 | 1.26 | 0.99 |  |
| n_real_updates | 14.7 | 0.056 | 0.0589 | 0.081 | 0.027 | 0.0531 | 1.11 |  |
| pin_down | 15.5 | — | — | 0.083 | 0.052 | 0.0615 |  | Tensorize: broken: restore_sizes misses np.full([3], 1) in the return -> length 3 instead of 1000 |

T/E over 4 kernels: geomean 1.04, median 1.03

## llama

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | Tenspiler synth | Tenspiler XLA | Tenspiler ms | Tenspiler/E | status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 22 | 0.14 | 0.0587 | 0.087 | 0.042 | 0.0601 | 0.98 | 6.2 | 0.13 | 0.0595 | 0.99 |  |
| rmsnorm_part1 | 7.9 | 0.11 | 0.0646 | 0.083 | 0.034 | 0.0631 | 1.02 | 2.1 | 0.065 | 0.0741 | 1.18 |  |
| rmsnorm_part2 | 28 | 0.06 | 0.0745 | 0.084 | 0.029 | 0.0719 | 1.04 | 4.2 | 0.03 | 0.133 | (1.85) | Tenspiler: different computation: ours divides by 3.0, not size |
| softmax_part1 | 7 | 0.13 | 0.0575 | 0.082 | 0.043 | 0.0549 | 1.05 | 4.2 | 0.085 | 0.0655 | 1.19 |  |
| softmax_part2 | 15.7 | 0.059 | 0.0668 | 0.081 | 0.03 | 0.0627 | 1.06 | 4.2 | 0.029 | 0.155 | 2.47 |  |
| softmax_part3 | 7.4 | 0.11 | 0.0734 | 0.084 | 0.033 | 0.0644 | 1.14 | 2.1 | 0.06 | 0.06 | 0.93 |  |
| softmax_part4 | 8.1 | 0.06 | 0.0788 | 0.084 | 0.032 | 0.0787 | 1.00 | 2.2 | 0.028 | 0.0972 | 1.23 |  |
| transformer_part3 | 29.4 | 0.068 | 0.0805 | 0.085 | 0.037 | 0.0669 | 1.20 | 2.2 | 0.033 | 0.118 | (1.77) | Tenspiler: different computation: ours is 1/(b(1+e^-a)), not SiLU |
| transformer_part4 | 7.9 | 0.052 | 0.0644 | 0.082 | 0.024 | 0.0575 | 1.12 | 2.2 | 0.022 | 0.0803 | 1.40 |  |
| transformer_part1 (Tenspiler only) |  |  |  |  |  |  |  | 16.2 | 0.076 | 0.128 |  | no Tensorize/Enzyme-JAX counterpart |
| transformer_part2 (Tenspiler only) |  |  |  |  |  |  |  | 8.2 | 0.076 | 0.053 |  | no Tensorize/Enzyme-JAX counterpart |

T/E over 9 kernels: geomean 1.07, median 1.05

## makespeare

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| sum_of_squares | 7.7 | 0.11 | 0.0606 | 0.084 | 0.036 | 0.0588 | 1.03 |  |

T/E over 1 kernels: geomean 1.03, median 1.03

## mathfu

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| diveq | 8.4 | 0.057 | 0.0541 | 0.083 | 0.027 | 0.0522 | 1.04 |  |
| diveq_sca | 8.6 | 0.058 | 0.0657 | 0.084 | 0.026 | 0.0662 | 0.99 |  |
| len | 8.8 | 0.11 | 0.0625 | 0.084 | 0.035 | 0.0619 | 1.01 |  |
| len_sq | 8.3 | 0.11 | 0.0708 | 0.083 | 0.034 | 0.066 | 1.07 |  |
| lerp | 37.3 | 0.056 | 0.0622 | 0.083 | 0.027 | 0.0581 | 1.07 |  |
| matmul_sca | 8.1 | 0.053 | 0.0666 | 0.083 | 0.026 | 0.0628 | 1.06 |  |
| muleq | 6.5 | 0.053 | 0.0615 | 0.082 | 0.026 | 0.0613 | 1.00 |  |
| muleq_sca | 7.2 | 0.053 | 0.0541 | 0.083 | 0.024 | 0.053 | 1.02 |  |
| negate | 6.2 | 0.058 | 0.0741 | 0.08 | 0.025 | 0.0701 | 1.06 |  |
| pluseq | 6.9 | 0.051 | 0.0622 | 0.08 | 0.024 | 0.0589 | 1.06 |  |
| subeq | 7.5 | 0.051 | 0.0544 | 0.082 | 0.023 | 0.054 | 1.01 |  |
| subeq_sca | 7.9 | 0.054 | 0.0604 | 0.083 | 0.025 | 0.0602 | 1.00 |  |

T/E over 12 kernels: geomean 1.03, median 1.03

## simpl_array

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| array_inc | 6 | 0.058 | 0.0518 | 0.083 | 0.023 | 0.0506 | 1.02 |  |
| array_sum | 7.7 | 0.11 | 0.0593 | 0.084 | 0.033 | 0.0536 | 1.11 |  |
| cube_in_place | — | — | — | 0.083 | 0.052 | 0.0769 |  | Tensorize: no synthesized program |
| fourth_in_place | 17.8 | 0.066 | 0.0632 | 0.084 | 0.024 | 0.0594 | 1.06 |  |
| sum_elts | 8 | 0.11 | 0.0673 | 0.084 | 0.033 | 0.065 | 1.03 |  |

T/E over 4 kernels: geomean 1.06, median 1.05

## utdsp

| kernel | Tensorize synth | Tensorize XLA | Tensorize ms | Enzyme raise | Enzyme XLA | Enzyme ms | T/E | status / notes |
|---|---|---|---|---|---|---|---|---|
| dct | 34.7 | 2.6 | 2.36 | 0.085 | 0.13 | 2.37 | 1.00 |  |
| fir_small | 10.1 | 0.12 | 0.0577 | 0.083 | 0.036 | 0.0594 | 0.97 |  |
| histogram | 15.6 | — | — | 0.082 | 0.055 | 0.0585 |  | Tensorize: broken: restore_sizes misses np.full([3], 0) in the return -> length 3 instead of 1000 |
| lmsfir1 | 9.9 | 0.12 | 0.0603 | 0.084 | 0.035 | 0.0574 | 1.05 |  |
| lmsfir2 | 10.4 | 0.054 | 0.0727 | 0.084 | 0.026 | 0.0647 | 1.12 |  |
| mult_big | 17.1 | 2.5 | 1.18 | 0.086 | 0.12 | 1.18 | 1.00 |  |

T/E over 5 kernels: geomean 1.03, median 1.00

**All suites:** T/E over 94 kernels: geomean 0.95, median 1.03
