import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

# colorblind-friendly color map taken from https://personal.sron.nl/~pault/
rgb_colors = [[68,119,170], [238,102,119], [102,204,238], [204,187,68], [34,136,51], [238,102,119], [170,51,119], [187,187,187]]
rgba_colors = [x + [256] for x in rgb_colors]
rgba_colors_norm = np.vstack(np.array([np.array(x) / 256.0 for x in rgba_colors]))
colormap = ListedColormap(rgba_colors_norm)

# Hydra:
# - AMD Ryzen Threadripper PRO 7995WX 96-Cores
# - NVIDIA GeForce RTX 5090 32 GB VRAM
raw_runtimes = {
  "LBM": { # 100 Iterations, same as Futhark's default
    "EnzymeLLVM": [0.200386, 0.201904, 0.203658, 0.201001, 0.203199, 0.200504, 0.200924, 0.201260, 0.200509, 0.206771, 0.203641, 0.203939],
    "EnzymeMLIR": [0.09356, 0.09414, 0.09445, 0.09368, 0.09795, 0.09434, 0.09350, 0.09435, 0.09369, 0.09365, 0.092651, 0.092592],
    "EnzymeMLIR (atomics)": [0.231313, 0.231457, 0.231497, 0.231324, 0.231200, 0.231647, 0.231522, 0.231302, 0.231461, 0.231534, 0.231311, 0.231614],
    "Futhark": [0.106473, 0.106657, 0.106408, 0.106458, 0.106414, 0.106593, 0.106556, 0.106639, 0.106585, 0.106513, 0.106408, 0.106593]
  },
  "RSBench": { # 10200 lookups
    "EnzymeLLVM": [3.042, 2.993, 2.927, 2.890, 2.896, 2.864, 2.867, 2.984, 2.908, 3.046, 3.094, 2.823],
    "EnzymeMLIR": [0.116, 0.116, 0.117, 0.117, 0.115, 0.116, 0.116, 0.115, 0.117, 0.115, 0.117, 0.115],
    "Futhark": [0.032211, 0.032203, 0.032201, 0.032214, 0.032203, 0.032198, 0.032264, 0.032212, 0.032209, 0.032219, 0.032205, 0.032208]
  },
  "Oceananigans.jl": { # 128 timesteps
    "EnzymeLLVM": [349.546833574, 342.069749719, 344.715206062, 346.54683372, 343.944704052, 345.79549403, 345.649731097, 348.770533088, 347.740546824, 346.74585043, 345.628197242, 349.288836495],
    "EnzymeMLIR": [429.331878947, 429.376601515, 429.351666216, 429.685426457, 429.573605157, 429.041916229, 428.66163381, 429.762206607, 430.22280294, 430.418672853, 429.830867455, 430.306217239],
  },
  # Using "BothRev", "HLOOpt" for the EnzymeMLIR variant
  "JaxMD": {
    "JAX": [
      0.004247298,
      0.004232009,
      0.004273964999999999,
      0.00483082,
      0.004721407,
      0.003978538,
      0.003955935,
      0.004169931,
      0.004244073,
      0.004265748000000001,
      0.004249628,
      0.004181559
    ],
    "EnzymeMLIR": [
      0.0014570639999999999,
      0.0014565790000000002,
      0.001474706,
      0.001458184,
      0.001456396,
      0.001505784,
      0.001473603,
      0.001451721,
      0.001464392,
      0.001459246,
      0.001461742,
      0.001476168
    ],
  },
  "LegoESM": { # 240 timesteps, binomial b=n to emulate JAX
    # TODO: need to sample properly
    "JAX": [27.48] * 12,
    "EnzymeMLIR": [24.35] * 12,
  }
}

peak_memory_bytes = {
    "LBM": {
        "EnzymeMLIR": 4.448601600,
        "EnzymeLLVM": 5.096601600,
        "Futhark": 19.241677052,
    },
    "RSBench": {
        "EnzymeMLIR": 0.057819523,
        "EnzymeLLVM": 0.057826179,
        "Futhark": 0.0111029444,
    },
    "Oceananigans.jl": { # "enzyme_naive" and "reactant_full" at 120 timesteps
        "EnzymeMLIR": 22.326607872,
        "EnzymeLLVM": 237.87667456,
    },
    "JaxMD": {
        "EnzymeMLIR": 0.172748544,
        "JAX": 0.440552192,
    },
    "LegoESM": { # 240 tsteps, binomial b=n
        "EnzymeMLIR": 17.68,
        "JAX": 27.63,
    },
}

#
# Format the data
#
def prepare_speedup_df(data_dict):
  columns = [f"run{i+1}" for i in range(12)]
  benchmarks = data_dict.keys()
  methods = list(set(list(data_dict["LBM"].keys()) + list(data_dict["JaxMD"].keys())))
  idx = pd.MultiIndex.from_product((benchmarks, methods))

  data = np.zeros((len(idx), len(columns)))
  data.fill(np.nan)
  results = pd.DataFrame(data=data, index=idx, columns=columns)
  for benchmark, variants in data_dict.items():
    for variant, data in variants.items():
      results.loc[benchmark, variant][:len(data)] = data

  return results

def prepare_memory_df(data_dict):
    df = pd.DataFrame(data_dict)
    return df

results = prepare_speedup_df(raw_runtimes)

# Bootstrap confidence intervals
data = np.zeros((len(results.index), 2))
data.fill(np.nan)
intervals = pd.DataFrame(data=data, index=results.index, columns=["ci_lo", "ci_hi"])
n_resamples = 10_000
n_cols = results.shape[1]
stat_fn = np.median
ci = 0.99
alpha = (1 - ci) / 2
for benchmark in results.index.levels[0]:
    baseline = "JAX" if benchmark in ["JaxMD", "LegoESM"] else "EnzymeLLVM"
    benchresults = results.loc[benchmark]
    var_samples = benchresults.values[:, np.random.randint(0, n_cols, size=(n_resamples, n_cols))]
    base_samples = benchresults.loc[baseline].values[np.random.randint(0, n_cols, size=(n_resamples, n_cols))]
    ratios = stat_fn(base_samples, axis=1) / stat_fn(var_samples, axis=2)
    ci_lo = np.percentile(ratios, 100 * alpha, axis=1)
    ci_hi = np.percentile(ratios, 100 * (1 - alpha), axis=1)
    intervals.loc[benchmark, "ci_lo"] = ci_lo
    intervals.loc[benchmark, "ci_hi"] = ci_hi
medians = results.median(axis=1).unstack(level=1)

JAX_BENCHMARKS = ["JaxMD", "LegoESM"]
LLVM_BENCHMARKS = ["LBM", "RSBench", "Oceananigans.jl"] # Benchmarks where EnzymeLLVM is the baseline
memory_df = prepare_memory_df(peak_memory_bytes)

jax_mem = memory_df[JAX_BENCHMARKS]
jax_mem_reductions = (jax_mem.loc["JAX"] / jax_mem).T
jax_mem_reductions = jax_mem_reductions.drop(columns=["EnzymeLLVM", "JAX", "Futhark"])
llvm_mem = memory_df[LLVM_BENCHMARKS]
llvm_mem_reductions = (llvm_mem.loc["EnzymeLLVM"] / llvm_mem).T
llvm_mem_reductions = llvm_mem_reductions.drop(columns=["EnzymeLLVM", "JAX"])

jax_medians = medians.loc[JAX_BENCHMARKS]
jax_speedups = (jax_medians["JAX"] / jax_medians.T).T
jax_speedups = jax_speedups.drop(columns=[c for c in jax_speedups.columns if c not in ["EnzymeMLIR"]])
cpp_medians = medians.loc[LLVM_BENCHMARKS]
cpp_speedups = (cpp_medians["EnzymeLLVM"] / cpp_medians.T).T
cpp_speedups = cpp_speedups.drop(columns=["JAX", "EnzymeMLIR (atomics)", "EnzymeLLVM"])

#
# Plotting
#
def plot_speedups_on_axis(ax, df, yticks, baseline):
    xs = np.arange(df.shape[0]) * df.shape[1]
    width = 0.3
    offset = 0.03
    for i, c in enumerate(df.columns):
        ci_low = (df[c] - intervals.unstack(level=1)["ci_lo"][c].loc[df.index]).values
        ci_high = (intervals.unstack(level=1)["ci_hi"][c].loc[df.index] - df[c]).values
        ax.bar(
        xs + width * float(i) + (i - 1) * offset,
        df[c] - 1.0,
        bottom=1.0,
        yerr=[ci_low, ci_high],
        capsize=15,
        width=width,
        zorder=3,
        edgecolor="#444",
        label=c,
        color=colormap(i))
    plot_baseline_in_front = False
    baseline_style = "dotted" if baseline == "JAX" else "dashed"
    ax.plot([-1, xs[-1] + 3], [1.0] * 2, linestyle=baseline_style, color="black", linewidth=2, zorder=4 if plot_baseline_in_front else 2, label=baseline)

    ax.grid(axis='y', zorder=1)
    ax.set_xticks(xs + width * ((df.shape[1]-1)/2.0), df.index)
    ax.set_xlim(-0.5, df.shape[0] * df.shape[1] - 0.5)

    ax.set_yscale("log")
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(yticks)
    ax.legend()

def plot_mem_on_axis(ax, df, yticks, baseline):
    xs = np.arange(df.shape[0]) * df.shape[1]
    width = 0.3
    offset = 0.03
    for i, c in enumerate(df.columns):
        ax.bar(
            xs + width * float(i) + (i - 1) * offset,
            df[c] - 1.0,
            bottom=1.0,
            width=width,
            zorder=3,
            edgecolor="#444",
            label=c,
            color=colormap(i))
    plot_baseline_in_front = False
    baseline_style = "dotted" if baseline == "JAX" else "dashed"
    ax.plot([-1, xs[-1] + 3], [1.0] * 2, linestyle=baseline_style, color="black", linewidth=2, zorder=4 if plot_baseline_in_front else 2, label=baseline)

    ax.grid(axis='y', zorder=1)
    ax.set_xticks(xs + width * ((df.shape[1]-1)/2.0), df.index)
    ax.set_xlim(-0.5, df.shape[0] * df.shape[1] - 0.5)

    ax.set_yscale("log")
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(yticks)
    ax.legend()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, width_ratios=[0.5, 0.5], height_ratios=[0.55, 0.45])
ax1.set_ylabel("Speedup (log scale)")
ax3.set_ylabel("Memory Reduction")
fig.set_size_inches(9, 4)
plot_speedups_on_axis(ax1, cpp_speedups, [0.5, 1, 2, 4, 8, 16, 32, 64, 128], "EnzymeLLVM")
plot_speedups_on_axis(ax2, jax_speedups, [0.5, 1, 2, 4], "JAX")
plot_mem_on_axis(ax3, llvm_mem_reductions, [0.25, 0.5, 1, 2, 4, 8], "EnzymeLLVM")
plot_mem_on_axis(ax4, jax_mem_reductions, [0.5, 1, 2, 4], "JAX")

fig.tight_layout()
fig.savefig("enzymemlir_speedups.pdf")
