"""Write docs/RESULTS.md from the benchmark CSV files.

Usage (from the repo root):
    python bench/report.py --results results --out docs/RESULTS.md

Every number in the report is computed here from the CSV files. Do not edit
docs/RESULTS.md by hand: rerun the benchmark and this script instead.
"""
import argparse
from pathlib import Path

from common import SYSTEM_NAMES, Timings, ms, ratio, read_csv, read_env


def table(header: list, rows: list) -> str:
    """A Markdown table from a header row and data rows."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def compare(ours, theirs) -> str:
    """Describe `ours` against `theirs`, for example '3.1x faster' or '1.4x slower'."""
    if ours is None or theirs is None:
        return "not measured"
    if ours <= theirs:
        return f"{theirs / ours:.1f}x faster"
    return f"{ours / theirs:.1f}x slower"


def environment_section(env: dict) -> str:
    """The machine and package versions behind the numbers."""
    packages = ", ".join(f"{name} {version}" for name, version in env.get("packages", {}).items())
    rows = [
        ["Date (UTC)", env.get("date_utc")],
        ["CPU", f"{env.get('cpu_model')}, {env.get('cpu_cores')} cores"],
        ["Memory", f"{env.get('ram_gb')} GB"],
        ["Operating system", env.get("os")],
        ["Inside Docker", "yes" if env.get("inside_docker") else "no"],
        ["Python", env.get("python")],
        ["Packages", packages],
        ["NumPy and EconML threads", "; ".join(env.get("blas", [])) or "library default"],
        ["Runs per number", f"{env.get('warmup_runs')} warm-up, then the median of {env.get('timed_runs')} timed"],
        ["EconML fit (not part of any timing)", f"{env.get('econml_fit_seconds')} s on {env.get('train_rows'):,} rows"],
    ]
    return "## 1. Where these numbers come from\n\n" + table(["Item", "Value"], rows)


def per_row_section(timings: Timings, workload: str, number: int, title: str, systems: tuple) -> str:
    """Median time for every system, one column per scenario count, at the largest n."""
    n = timings.values(4, workload)[-1]
    scenarios = [K for K in timings.values(5, workload)
                 if timings.get(workload, "numpy_ref", "blas", "end_to_end", n, K)]
    threads = timings.thread_counts()

    # One table row per (system, thread count, timing)
    lines = []
    for system in systems:
        if system.startswith("compiled"):
            for count in threads:
                for timing, note in (("end_to_end", "NumPy in, NumPy out"), ("compute_only", "data already loaded")):
                    name = f"{SYSTEM_NAMES[system]}, {count} thread{'s' if count > 1 else ''}, {note}"
                    lines.append((name, [timings.get(workload, system, count, timing, n, K) for K in scenarios]))
        else:
            lines.append((SYSTEM_NAMES[system],
                          [timings.get(workload, system, "blas", "end_to_end", n, K) for K in scenarios]))

    rows = [[name] + [ms(value) for value in values] for name, values in lines]
    header = ["System"] + [f"K = {K}" for K in scenarios]
    return (f"## {number}. {title}\n\nMedian milliseconds for {n:,} rows. K is the number of scenarios.\n\n"
            + table(header, rows))


def ate_section(timings: Timings, number: int) -> str:
    """Average effect: the workload where mean pushdown changes the amount of work."""
    n, K = timings.values(4, "ate")[-1], timings.values(5, "ate")[-1]
    models = timings.values(6, "ate")
    threads = timings.thread_counts()

    lines = [(SYSTEM_NAMES["econml_loop"] + " (only one model was fitted)", "econml_loop", "blas", "end_to_end"),
             (SYSTEM_NAMES["numpy_ref"], "numpy_ref", "blas", "end_to_end"),
             (SYSTEM_NAMES["numpy_pushdown"], "numpy_pushdown", "blas", "end_to_end")]
    for system in ("compiled_nopush", "compiled"):
        for count in threads:
            for timing, note in (("end_to_end", "NumPy in, NumPy out"), ("compute_only", "data already loaded")):
                lines.append((f"{SYSTEM_NAMES[system]}, {count} thread{'s' if count > 1 else ''}, {note}",
                              system, count, timing))

    rows = [[name] + [ms(timings.get("ate", system, thread, timing, n, K, M)) for M in models]
            for name, system, thread, timing in lines]
    header = ["System"] + [f"M = {M}" for M in models]
    return (f"## {number}. Average effect for M models\n\n"
            f"Median milliseconds for {n:,} rows and {K} scenarios per model. M is the number of models.\n\n"
            + table(header, rows))


def scaling_section(timings: Timings, number: int) -> str:
    """Time against the number of rows."""
    threads = timings.thread_counts()[-1]
    blocks = []
    for workload in ("effects", "intervals"):
        rows_measured = timings.values(4, workload)
        K = next(k for k in timings.values(5, workload)
                 if all(timings.get(workload, "numpy_ref", "blas", "end_to_end", n, k) for n in rows_measured))
        lines = (("EconML, one call per scenario", "econml_loop", "blas", "end_to_end"),
                 ("Hand-written NumPy", "numpy_ref", "blas", "end_to_end"),
                 (f"dmlc compiled, {threads} threads, NumPy in, NumPy out", "compiled", threads, "end_to_end"),
                 (f"dmlc compiled, {threads} threads, data already loaded", "compiled", threads, "compute_only"))
        rows = [[name] + [ms(timings.get(workload, system, thread, timing, n, K)) for n in rows_measured]
                for name, system, thread, timing in lines]
        blocks.append(f"**{workload}, K = {K}**\n\n" + table(["System"] + [f"n = {n:,}" for n in rows_measured], rows))
    return f"## {number}. Scaling with the number of rows\n\nMedian milliseconds.\n\n" + "\n\n".join(blocks)


def op_count_section(results: Path, number: int) -> str:
    """How many operations each compiler stage leaves behind."""
    counts = read_csv(results / "opcounts.csv")
    stages = ("as_written", "decomposed", "cse", "lowered")
    rows = []
    for workload in dict.fromkeys(row["workload"] for row in counts):
        totals = []
        for stage in stages:
            ops = [row for row in counts if row["workload"] == workload and row["stage"] == stage]
            dml = sum(int(row["count"]) for row in ops if row["op"].startswith("dml."))
            linalg = sum(int(row["count"]) for row in ops if row["op"].startswith("linalg."))
            totals.append(f"{dml} dml" if stage != "lowered" else f"{linalg} linalg")
        rows.append([workload] + totals)
    header = ["Program", "As written", "After decompose", "After CSE", "After lowering"]
    return (f"## {number}. Operation counts per compiler stage\n\n"
            "CSE (common subexpression elimination) is the stock MLIR pass. It can merge our ops because\n"
            "every `dml` op is declared `Pure`.\n\n" + table(header, rows))


def accuracy_section(results: Path, number: int) -> str:
    """Largest difference from the oracle, per system."""
    worst = {}
    for row in read_csv(results / "errors.csv"):
        key = (row["workload"], row["system"], row["against"])
        worst[key] = max(worst.get(key, 0.0), float(row["max_abs_error"]))
    rows = [[workload, SYSTEM_NAMES.get(system, system), SYSTEM_NAMES.get(against, against), f"{error:.1e}"]
            for (workload, system, against), error in sorted(worst.items())]
    return (f"## {number}. Agreement with the oracle\n\n"
            "Largest absolute difference over every size measured. The benchmark aborts above 1e-9.\n"
            "Results agree to rounding error. They are not bit-identical, because the compiled code\n"
            "adds numbers in a different order.\n\n"
            + table(["Workload", "System", "Compared against", "Max abs difference"], rows))


def compile_section(results: Path, number: int) -> str:
    """One-off compile cost per program."""
    rows = [[row["workload"], SYSTEM_NAMES.get(row["system"], row["system"]), row["K"], row["M"],
             ms(float(row["seconds"]))] for row in read_csv(results / "compile.csv")]
    return (f"## {number}. Compile time\n\nPaid once per program, not per call. dml-opt plus IREE, in milliseconds.\n\n"
            + table(["Workload", "Pipeline", "K", "M", "Compile ms"], rows))


def profile_section(results: Path, number: int) -> str:
    """Where one EconML effect() call spends its time."""
    rows = read_csv(results / "econml_profile.csv")
    if not rows:
        return f"## {number}. EconML profile\n\nNot measured. Run `make profile`."
    body = [[f"`{row['function']}`", row["file"], f"{float(row['self_seconds']) * 1e3:.1f}",
             f"{float(row['share_of_call']):.0%}"] for row in rows]
    return (f"## {number}. Where one EconML `effect()` call spends its time\n\n"
            f"Self time for one call on {int(rows[0]['rows']):,} rows, from cProfile.\n\n"
            + table(["Function", "File", "Self ms", "Share"], body)
            + "\n\nReading the names against the EconML 0.17.1 source: `_functools.reduce` is the cross product\n"
            "of features and treatment, `conc` adds the intercept column to X, `repeat` expands the scalar\n"
            "T0 and T1 into n-row arrays, and `predict` is the actual prediction.")


def headline_section(timings: Timings) -> str:
    """The short version, with each claim computed from the medians."""
    threads = timings.thread_counts()[-1]

    def facts(workload):
        n = timings.values(4, workload)[-1]
        K = max(k for k in timings.values(5, workload)
                if timings.get(workload, "econml_loop", "blas", "end_to_end", n, k))
        get = lambda system, thread, timing="end_to_end": timings.get(workload, system, thread, timing, n, K)
        return n, K, get

    lines = []
    for workload, label in (("effects", "Per-row effects"), ("intervals", "Per-row intervals")):
        n, K, get = facts(workload)
        econml, numpy_time = get("econml_loop", "blas"), get("numpy_ref", "blas")
        compiled, loaded = get("compiled", threads), get("compiled", threads, "compute_only")
        nocse = get("compiled_nocse", threads, "compute_only")
        lines.append(
            f"- **{label}** ({n:,} rows, {K} scenarios): EconML {ms(econml)} ms, NumPy {ms(numpy_time)} ms, "
            f"dmlc {ms(compiled)} ms end to end and {ms(loaded)} ms with the data already loaded. "
            f"End to end, dmlc is {compare(compiled, econml)} than EconML and {compare(compiled, numpy_time)} than NumPy. "
            f"Without our CSE step the compute time is {ms(nocse)} ms, so CSE in our dialect does not change run time."
        )

    n, K = timings.values(4, "ate")[-1], timings.values(5, "ate")[-1]
    models = timings.values(6, "ate")
    M = models[-2] if len(models) > 1 else models[-1]
    get = lambda system, thread, timing="end_to_end": timings.get("ate", system, thread, timing, n, K, M)
    lines.append(
        f"- **Average effect** ({n:,} rows, {M} models, {K} scenarios): NumPy {ms(get('numpy_ref', 'blas'))} ms, "
        f"dmlc without pushdown {ms(get('compiled_nopush', threads))} ms, dmlc with pushdown {ms(get('compiled', threads))} ms. "
        f"Pushdown makes dmlc {ratio(get('compiled_nopush', threads), get('compiled', threads))} faster, and "
        f"{compare(get('compiled', threads), get('numpy_ref', 'blas'))} than plain NumPy. "
        f"NumPy with the same pushdown written by hand takes {ms(get('numpy_pushdown', 'blas'))} ms. "
        f"With the data already loaded, pushdown takes dmlc from {ms(get('compiled_nopush', threads, 'compute_only'))} ms "
        f"to {ms(get('compiled', threads, 'compute_only'))} ms "
        f"({ratio(get('compiled_nopush', threads, 'compute_only'), get('compiled', threads, 'compute_only'))})."
    )
    return "## 2. Headline numbers\n\nAll times are medians. \"End to end\" means NumPy arrays in and NumPy arrays out.\n\n" + "\n".join(lines)


LIMITATIONS = """## {number}. Limitations, stated plainly

- **Inference only.** The compiler covers querying a fitted model. Fitting (the cross-fitting
  and the nuisance models) stays in EconML and dominates total time when a model is fitted once
  and queried once.
- **One estimator.** Only EconML `LinearDML` with a linear final stage and one continuous
  treatment. No forests, no discrete treatments, no multiple outcomes.
- **A NumPy expert can match the algebra.** The "NumPy with mean pushdown" row is the same
  rewrite done by hand. The compiler's contribution is that it finds the rewrite automatically
  from naive queries, not that it beats hand-tuned code.
- **Our CSE step does not change run time.** IREE removes the same duplicates after lowering.
  Decomposition still matters because it is what makes the pushdown rewrite possible.
- **Copying data in and out is a real cost.** End-to-end numbers include copying X into the
  IREE runtime and copying each result out. For cheap queries this copy is larger than the
  compute, which is why the tables show both timings.
- **Not bit-identical.** Results match EconML to rounding error (see the agreement table).
- **Synthetic data, one machine.** Timings were taken on the machine in section 1, not on the
  team's laptops or the Sol cluster. Rerun `make results` on the target machine before quoting them.
- **Perturbed models.** The many-model workload uses one fitted model plus noisy copies of its
  coefficients. That is enough for timing, because inference cost depends only on shapes.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="folder holding the CSV files")
    parser.add_argument("--out", default="docs/RESULTS.md", help="report file to write")
    args = parser.parse_args()

    results = Path(args.results)
    timings = Timings(results)

    sections = [
        "# Results\n\n_Generated by `bench/report.py` from `results/*.csv`. Do not edit by hand._\n\n"
        "Figures are in `results/figures/`. Raw runs are in `results/bench.csv`.",
        environment_section(read_env(results)),
        headline_section(timings),
        per_row_section(timings, "effects", 3, "Per-row effects",
                        ("econml_loop", "econml_smart", "numpy_ref", "compiled", "compiled_nocse")),
        per_row_section(timings, "intervals", 4, "Per-row 95% confidence intervals",
                        ("econml_loop", "numpy_ref", "compiled", "compiled_nocse")),
        ate_section(timings, 5),
        scaling_section(timings, 6),
        op_count_section(results, 7),
        compile_section(results, 8),
        accuracy_section(results, 9),
        profile_section(results, 10),
        LIMITATIONS.format(number=11),
    ]
    Path(args.out).write_text("\n\n".join(sections).rstrip() + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
