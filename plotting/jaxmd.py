"""Render JAX-MD reference or freshly collected measurements (also used by the notebook)."""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS = {"PreRev": "EnzymeMLIR\nPre AD", "PostRev": "EnzymeMLIR\nPost AD", "BothRev": "EnzymeMLIR\nPre + Post AD"}
PIPELINES = {"HLOOpt": "HLO Opts.", "DefOpt": "All (No Inline)", "IDefOpt": "All (Inline)"}
COLORS = {"HLOOpt": "#fc8d62", "DefOpt": "#66c2a5", "IDefOpt": "#8da0cb"}


def resolve_paths(source="reference", input_path=None, output_dir=None, *, root=REPO_ROOT):
    if source not in ("reference", "new"):
        raise ValueError(f"Invalid source {source!r}: expected reference or new")
    default = root / "results" / "results_JAXMD.json"
    if source == "new":
        default = root / "results" / "jaxmd" / "results_JAXMD.json"
    input_path = Path(input_path).resolve() if input_path else default.resolve()
    output_dir = Path(output_dir).resolve() if output_dir else root / "results" / "plots" / "jaxmd" / source
    if not input_path.is_file():
        raise FileNotFoundError(f"No JAX-MD {source} data at {input_path}. Run the benchmark first or pass --input JSON.")
    return input_path, output_dir.resolve()


def load_results(path):
    samples = pd.read_json(path)
    if samples.empty:
        raise ValueError(f"No measurements in {path}")
    if len(samples[["benchmark", "backend"]].drop_duplicates()) != 1:
        raise ValueError("Select a JSON file containing one benchmark/backend; combining workloads would change the speedup baseline.")
    samples = samples.explode(["time_s", "peak_memory_bytes"])
    samples = samples.astype({"time_s": float, "peak_memory_bytes": float})
    return samples.groupby(["benchmark", "pipeline", "backend", "key"], sort=False).agg(
        nsamples=("time_s", "size"),
        time_avg_s=("time_s", "mean"), time_std_s=("time_s", "std"),
        time_min_s=("time_s", "min"), time_max_s=("time_s", "max"),
        mem_avg_bytes=("peak_memory_bytes", "mean"), mem_std_bytes=("peak_memory_bytes", "std"),
        mem_min_bytes=("peak_memory_bytes", "min"), mem_max_bytes=("peak_memory_bytes", "max"),
    ).reset_index()


def make_figures(df):
    avg = df.pivot_table(index="key", columns="pipeline", values="time_avg_s")
    mem = df.pivot_table(index="key", columns="pipeline", values="mem_avg_bytes") / 2**20
    baseline = avg.loc["BothRev", "Jax"]
    speedup = baseline / avg.loc[list(KEYS), list(PIPELINES)]
    figures = {}
    width = 0.25
    offsets = (np.arange(len(PIPELINES)) - (len(PIPELINES) - 1) / 2) * width
    bar_kw = dict(edgecolor="black", linewidth=0.8)
    for name, values, base, ylabel, title in (
        ("runtime", avg * 1e3, baseline * 1e3, "Runtime (ms)", "Runtime"),
        ("memory", mem, mem.loc["BothRev", "Jax"], "Peak memory usage [MiB]", "Peak Memory"),
    ):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(0, base, width=len(PIPELINES) * width, color="lightgray", label="No opt", **bar_kw)
        x = np.arange(1, len(KEYS) + 1)
        for off, (pipe, label) in zip(offsets, PIPELINES.items()):
            ax.bar(x + off, values.loc[list(KEYS), pipe], width, color=COLORS[pipe], label=label, **bar_kw)
        ax.set_xticks(np.arange(len(KEYS) + 1), ["JAX", *KEYS.values()])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        figures[name] = fig
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(KEYS))
    for off, (pipe, label) in zip(offsets, PIPELINES.items()):
        ax.bar(x + off, speedup[pipe], width, color=COLORS[pipe], label=label, **bar_kw)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, list(KEYS.values()))
    ax.set_ylabel('Speedup over "No opt"')
    # Keep the reference scale, while allowing a new run's bars to remain visible.
    ax.set_ylim(0, max(3.6, float(speedup.max().max()) * 1.1))
    ax.legend(frameon=True, ncols=len(PIPELINES), loc="lower center", bbox_to_anchor=(0.5, 0.86))
    ax.set_title("Speedup")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    figures["speedup"] = fig
    return figures


def save_figures(figures, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"jaxmd20-{name}.{extension}", bbox_inches="tight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("reference", "new"), default=os.environ.get("PLOT_SOURCE", "reference"))
    parser.add_argument("--input", default=os.environ.get("PLOT_INPUT"), help="Override the selected JSON file")
    parser.add_argument("--output-dir", default=os.environ.get("PLOT_OUTPUT_DIR"))
    args = parser.parse_args(argv)
    input_path, output_dir = resolve_paths(args.source, args.input, args.output_dir)
    print(f"JAX-MD plot source: {args.source}\nInput: {input_path}\nOutput: {output_dir}")
    figures = make_figures(load_results(input_path))
    save_figures(figures, output_dir)
    for figure in figures.values():
        plt.close(figure)


if __name__ == "__main__":
    main()
