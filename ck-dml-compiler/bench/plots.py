"""Draw the presentation figures from the benchmark CSV files.

Usage (from the repo root):
    python bench/plots.py --results results

Writes results/figures/*.png at 200 dpi. Every number plotted is a median taken
from results/bench.csv, so the figures can be regenerated at any time.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

from common import SYSTEM_NAMES, Timings, read_csv  # noqa: E402

# One fixed colour per system, the same in every figure.
# The palette was checked for colour-blind separation; labels sit next to the
# marks as well, so no figure depends on colour alone.
COLORS = {
    "numpy_ref": "#2a78d6",
    "econml_loop": "#eb6834",
    "compiled": "#1baf7a",
    "compiled_nopush": "#eda100",
    "compiled_nocse": "#eda100",
    "numpy_pushdown": "#e87ba4",
    "econml_smart": "#e87ba4",
}
# Plain-language names for the profiler's function names (EconML 0.17.1).
PROFILE_NAMES = {
    "<built-in method _functools.reduce>": "cross product of features and treatment",
    "conc": "add the intercept column to X",
    "<method 'repeat' of 'numpy.ndarray' objects>": "repeat T0 and T1 into n-row arrays",
    "<method 'reduce' of 'numpy.ufunc' objects>": "NumPy ufunc reduce",
    "predict": "the actual prediction",
    "<built-in method numpy._core._multiarray_umath.c_einsum>": "einsum",
    "full": "allocate constant arrays",
    "effect": "effect() itself",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    """Quiet axes and grid, so the data is the loudest thing on the page."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def save(figure, folder: Path, name: str) -> None:
    """Write one figure and close it."""
    figure.savefig(folder / name, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {folder / name}")


def fig_profile(results: Path, folder: Path) -> None:
    """Where one EconML effect() call spends its time."""
    rows = read_csv(results / "econml_profile.csv")
    if not rows:
        print("  skipped fig_profile.png (run bench/profile_econml.py first)")
        return

    # Slowest function at the top
    rows = rows[::-1]
    labels = [PROFILE_NAMES.get(row["function"], f"{row['function']}  ({row['file']})") for row in rows]
    millis = [float(row["self_seconds"]) * 1e3 for row in rows]

    figure, axis = plt.subplots(figsize=(8, 3.6))
    bars = axis.barh(labels, millis, color=COLORS["econml_loop"], height=0.6)
    axis.bar_label(bars, labels=[f"{value:.1f} ms" for value in millis], padding=4, color=INK)
    axis.set_xlabel("self time, milliseconds")
    axis.set_title(f"One EconML effect() call on {int(rows[0]['rows']):,} rows: where the time goes")
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.15)
    for bar, row in zip(bars, rows):
        # The one bar that is real work gets the "compiled" colour
        if row["function"] == "predict":
            bar.set_color(COLORS["compiled"])
    save(figure, folder, "fig_profile.png")


def fig_workloads(timings: Timings, folder: Path) -> None:
    """Each system on each workload at the largest size, one panel per workload."""
    threads = timings.thread_counts()[-1]
    panels = (
        ("effects", "Per-row effects", ("econml_loop", "econml_smart", "numpy_ref", "compiled")),
        ("intervals", "Per-row 95% intervals", ("econml_loop", "numpy_ref", "compiled")),
    )

    figure, axes = plt.subplots(1, len(panels), figsize=(11, 3.4))
    for axis, (workload, title, systems) in zip(axes, panels):
        n = timings.values(4, workload)[-1]
        K = max(k for k in timings.values(5, workload)
                if timings.get(workload, "numpy_ref", "blas", "end_to_end", n, k))

        # (label, seconds, colour, filled marker?) for every row of the panel
        points = []
        for system in systems:
            if system.startswith("compiled"):
                end_to_end = timings.get(workload, system, threads, "end_to_end", n, K)
                preloaded = timings.get(workload, system, threads, "compute_only", n, K)
                points.append((f"dmlc compiled, {threads} threads", end_to_end, COLORS[system], True))
                points.append(("dmlc compiled, data already loaded", preloaded, COLORS[system], False))
            else:
                seconds = timings.get(workload, system, "blas", "end_to_end", n, K)
                points.append((SYSTEM_NAMES[system], seconds, COLORS[system], True))
        points = [point for point in points if point[1] is not None][::-1]

        # A dot per system on a log axis: the times span three orders of magnitude
        for row, (label, seconds, color, filled) in enumerate(points):
            axis.plot(seconds * 1e3, row, "o", markersize=9, color=color,
                      markerfacecolor=color if filled else SURFACE, markeredgewidth=2)
            axis.annotate(f"{seconds * 1e3:,.0f} ms", (seconds * 1e3, row),
                          textcoords="offset points", xytext=(10, -3), color=INK)
        axis.set_yticks(range(len(points)), [point[0] for point in points])
        axis.set_xscale("log")
        axis.set_ylim(-0.6, len(points) - 0.4)
        axis.margins(x=0.25)
        axis.set_xlabel("median time, milliseconds (log scale)")
        axis.set_title(f"{title}: {n:,} rows, {K} scenarios")
        axis.grid(axis="y", visible=False)
    figure.tight_layout()
    save(figure, folder, "fig_workloads.png")


def line_chart(axis, series, x_label, title) -> None:
    """Draw labelled lines. `series` is a list of (label, xs, seconds, colour, dashed)."""
    for label, xs, seconds, color, dashed in series:
        millis = [value * 1e3 for value in seconds]
        axis.plot(xs, millis, "o--" if dashed else "o-", color=color, linewidth=2, markersize=6)

    # Name each line at its right-hand end, in ink rather than the line colour.
    # Labels are placed bottom to top and pushed up when two would overlap.
    ends = sorted((seconds[-1] * 1e3, label, xs[-1]) for label, xs, seconds, _, _ in series)
    lowest_free = 0.0
    for value, label, x in ends:
        y = max(value, lowest_free)
        axis.annotate(f"{label}  {value:,.1f} ms", (x, y),
                      textcoords="offset points", xytext=(8, -3), color=INK, fontsize=9)
        lowest_free = y * 1.3   # 1.3x apart on a log axis is about one line of text

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel("median time, ms (log scale)")
    axis.set_title(title)


def fig_pushdown(timings: Timings, folder: Path) -> None:
    """Average effect for many models: the workload where mean pushdown pays off."""
    threads = timings.thread_counts()[-1]
    n, K = timings.values(4, "ate")[-1], timings.values(5, "ate")[-1]
    models = timings.values(6, "ate")

    # Left: NumPy arrays in and out. Right: X is already inside the runtime,
    # so only the computation is timed (NumPy has no loading step either way).
    panels = (("end_to_end", "NumPy arrays in, NumPy arrays out"),
              ("compute_only", "data already loaded"))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for axis, (timing, note) in zip(axes, panels):
        wanted = (
            ("numpy_ref", "blas", "end_to_end", "NumPy"),
            ("numpy_pushdown", "blas", "end_to_end", "NumPy, pushdown by hand"),
            ("compiled_nopush", threads, timing, "dmlc without pushdown"),
            ("compiled", threads, timing, "dmlc with pushdown"),
        )
        series = []
        for system, thread, system_timing, label in wanted:
            seconds = [timings.get("ate", system, thread, system_timing, n, K, M) for M in models]
            if all(value is not None for value in seconds):
                series.append((label, models, seconds, COLORS[system], False))

        line_chart(axis, series, "number of models (M)", f"Average effect of M models: {note}")
        axis.set_xticks(models, [str(M) for M in models])
        axis.margins(x=0.45)
    axes[1].set_ylabel("")
    figure.suptitle(f"{n:,} rows, {K} scenarios per model, {threads} threads", x=0.01, ha="left",
                    color=MUTED, fontsize=10)
    figure.tight_layout()
    save(figure, folder, "fig_pushdown.png")


def fig_scaling(timings: Timings, folder: Path) -> None:
    """Time against the number of rows, for effects and for intervals."""
    threads = timings.thread_counts()[-1]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    for axis, workload, title in zip(axes, ("effects", "intervals"), ("Per-row effects", "Per-row 95% intervals")):
        rows = timings.values(4, workload)

        # The scenario count that was measured at every row count
        K = next(k for k in timings.values(5, workload)
                 if all(timings.get(workload, "numpy_ref", "blas", "end_to_end", n, k) for n in rows))

        wanted = (
            ("econml_loop", "blas", "end_to_end", "EconML", False),
            ("numpy_ref", "blas", "end_to_end", "NumPy", False),
            ("compiled", threads, "end_to_end", "dmlc", False),
            ("compiled", threads, "compute_only", "dmlc, data loaded", True),
        )
        series = []
        for system, thread, timing, label, dashed in wanted:
            seconds = [timings.get(workload, system, thread, timing, n, K) for n in rows]
            if all(value is not None for value in seconds):
                series.append((label, rows, seconds, COLORS[system], dashed))
        line_chart(axis, series, "rows scored (n)", f"{title}: {K} scenarios")
        axis.margins(x=0.4)
    figure.tight_layout()
    save(figure, folder, "fig_scaling.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="folder holding the CSV files")
    args = parser.parse_args()

    results = Path(args.results)
    folder = results / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    style()
    timings = Timings(results)
    fig_profile(results, folder)
    fig_workloads(timings, folder)
    fig_pushdown(timings, folder)
    fig_scaling(timings, folder)


if __name__ == "__main__":
    main()
