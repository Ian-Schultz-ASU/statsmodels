"""Benchmark: time every system on every workload and write the raw runs to CSV.

Usage (from the repo root):
    python bench/run.py --out results            # full run, about 10 minutes
    python bench/run.py --out /tmp/q --quick     # small sizes, about 1 minute

Nothing in this file computes a summary. It only records raw timings, so every
number in docs/RESULTS.md can be traced back to a row in results/bench.csv.
"""
import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dmlc import data, emit, pipeline, reference  # noqa: E402

# Any result further than this from its oracle aborts the benchmark (FR-14).
MAX_ALLOWED_ERROR = 1e-9

# Number of scenarios used for the average-effect workload.
ATE_SCENARIOS = 8


@dataclass(frozen=True)
class Plan:
    """Sizes for one benchmark run."""

    train_rows: int        # rows used to fit the EconML model
    warmup: int            # untimed runs before measuring
    repeats: int           # timed runs per measurement
    row_sweep: tuple       # n values, measured at `main_scenarios`
    main_scenarios: int    # K used for the row sweep
    effect_scenarios: tuple    # K values for per-row effects at the largest n
    interval_scenarios: tuple  # K values for per-row intervals at the largest n
    model_sweep: tuple     # M values for the average-effect workload
    features: int = 10     # d


FULL = Plan(
    train_rows=200_000,
    warmup=2,
    repeats=7,
    row_sweep=(10_000, 100_000, 1_000_000),
    main_scenarios=20,
    effect_scenarios=(1, 5, 20, 50),
    interval_scenarios=(1, 5, 20),
    model_sweep=(1, 5, 25, 50),
)

QUICK = Plan(
    train_rows=20_000,
    warmup=1,
    repeats=3,
    row_sweep=(2_000, 20_000),
    main_scenarios=5,
    effect_scenarios=(1, 5),
    interval_scenarios=(1, 5),
    model_sweep=(1, 3),
)


def scenario_deltas(count: int) -> np.ndarray:
    """`count` treatment changes spread between 0.1 and 2.0."""
    return np.linspace(0.1, 2.0, count)


@dataclass
class Recorder:
    """Collects every row that ends up in the CSV files."""

    timings: list = field(default_factory=list)
    compiles: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    op_counts: list = field(default_factory=list)


@dataclass
class Bench:
    """Everything the workload functions share."""

    plan: Plan
    estimator: object                    # fitted econml LinearDML
    model: reference.FittedLinearDML     # its beta and cov
    record: Recorder
    modules: dict = field(default_factory=dict)   # compiled-module cache

    def measure(self, workload, system, threads, timing, sizes, function):
        """Time `function` and record one row per timed run.

        Args:
            workload: "effects", "intervals" or "ate".
            system: name from the table in docs/PRD.md (FR-10).
            threads: 1, the core count, or "blas" for NumPy and EconML.
            timing: "end_to_end" or "compute_only".
            sizes: dict with n, K and M for this measurement.
            function: zero-argument callable to time.

        Returns:
            Whatever the last call to `function` returned.
        """
        # Untimed runs first: they fill caches and start thread pools
        for _ in range(self.plan.warmup):
            function()

        result = None
        for run_index in range(self.plan.repeats):
            start = time.perf_counter()
            result = function()
            seconds = time.perf_counter() - start
            self.record.timings.append(
                dict(workload=workload, system=system, threads=threads, timing=timing,
                     n=sizes["n"], d=self.plan.features, K=sizes["K"], M=sizes["M"],
                     run_idx=run_index, seconds=seconds)
            )

        median = float(np.median([r["seconds"] for r in self.record.timings[-self.plan.repeats:]]))
        print(f"  {workload:9s} {system:16s} threads={str(threads):5s} {timing:12s} "
              f"n={sizes['n']:<8d} K={sizes['K']:<3d} M={sizes['M']:<3d} median={median * 1e3:10.3f} ms", flush=True)
        return result

    def check(self, workload, system, threads, sizes, got, want, against):
        """Record the max abs error of `got` against an oracle, and abort if it is too large."""
        error = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
        self.record.errors.append(
            dict(workload=workload, system=system, threads=threads, n=sizes["n"], K=sizes["K"],
                 M=sizes["M"], against=against, max_abs_error=error)
        )
        if not error <= MAX_ALLOWED_ERROR:
            raise SystemExit(f"ABORT: {workload}/{system} differs from {against} by {error:.3e}")

    def compiled_module(self, workload, system, sizes, mlir_text, pipeline_name) -> bytes:
        """Compile once per (workload, pipeline, K, M) and record the compile time."""
        key = (workload, pipeline_name, sizes["K"], sizes["M"])
        if key not in self.modules:
            start = time.perf_counter()
            self.modules[key] = pipeline.compile_module(mlir_text, pipeline_name)
            seconds = time.perf_counter() - start
            self.record.compiles.append(
                dict(workload=workload, system=system, K=sizes["K"], M=sizes["M"], seconds=seconds)
            )
        return self.modules[key]


def thread_settings():
    """(label for the Runner, thread count written to the CSV) for both IREE drivers."""
    return (("single", 1), ("multi", os.cpu_count()))


def bench_effects(bench: Bench, n: int, K: int) -> None:
    """Per-row effects for K scenarios on one model."""
    sizes = dict(n=n, K=K, M=1)
    deltas = scenario_deltas(K)
    X = data.make_features(n, bench.plan.features)
    estimator, model = bench.estimator, bench.model

    # EconML as normally used: one call per scenario. This is the oracle.
    want = bench.measure(
        "effects", "econml_loop", "blas", "end_to_end", sizes,
        lambda: np.stack([np.ravel(estimator.effect(X, T0=0, T1=delta)) for delta in deltas], axis=1),
    )

    # A careful EconML user: effect sizes once, then one multiply per scenario
    got = bench.measure(
        "effects", "econml_smart", "blas", "end_to_end", sizes,
        lambda: np.multiply.outer(np.ravel(estimator.const_marginal_effect(X)), deltas),
    )
    bench.check("effects", "econml_smart", "blas", sizes, got, want, "econml_loop")

    # Hand-written NumPy
    got = bench.measure("effects", "numpy_ref", "blas", "end_to_end", sizes,
                        lambda: reference.effects(model, X, deltas))
    bench.check("effects", "numpy_ref", "blas", sizes, got, want, "econml_loop")

    # Compiled, with and without our CSE step
    text = emit.emit_effects(bench.plan.features, deltas)
    for system, pipeline_name in (("compiled", "full"), ("compiled_nocse", "nocse")):
        module = bench.compiled_module("effects", system, sizes, text, pipeline_name)
        for label, threads in thread_settings():
            runner = pipeline.Runner(module, label)

            # NumPy arrays in, K NumPy vectors out: includes copying X into the
            # runtime and copying each result out. Stacking the vectors into one
            # (n, K) array is only needed for the check, so it is not timed.
            got = bench.measure("effects", system, threads, "end_to_end", sizes,
                                lambda: runner(X, model.beta))
            bench.check("effects", system, threads, sizes, np.stack(got, axis=1), want, "econml_loop")

            # Inputs already inside the runtime: the data-is-loaded-once case
            X_device, beta_device = runner.to_device(X), runner.to_device(model.beta)
            bench.measure("effects", system, threads, "compute_only", sizes,
                          lambda: runner.call_raw(X_device, beta_device))
            del runner, X_device, beta_device
    gc.collect()


def bench_intervals(bench: Bench, n: int, K: int) -> None:
    """Per-row 95% confidence intervals for K scenarios on one model."""
    sizes = dict(n=n, K=K, M=1)
    deltas = scenario_deltas(K)
    X = data.make_features(n, bench.plan.features)
    estimator, model = bench.estimator, bench.model

    def econml_intervals():
        # One (lo, hi) pair per scenario, stacked as lo0, hi0, lo1, hi1, ...
        columns = []
        for delta in deltas:
            lo, hi = estimator.effect_interval(X, T0=0, T1=delta, alpha=0.05)
            columns += [np.ravel(lo), np.ravel(hi)]
        return np.stack(columns, axis=1)

    def numpy_intervals():
        lo, hi = reference.effect_intervals(model, X, deltas, alpha=0.05)
        return np.stack([lo, hi], axis=2).reshape(n, 2 * K)

    want = bench.measure("intervals", "econml_loop", "blas", "end_to_end", sizes, econml_intervals)

    got = bench.measure("intervals", "numpy_ref", "blas", "end_to_end", sizes, numpy_intervals)
    bench.check("intervals", "numpy_ref", "blas", sizes, got, want, "econml_loop")

    text = emit.emit_intervals(bench.plan.features, deltas, alpha=0.05)
    for system, pipeline_name in (("compiled", "full"), ("compiled_nocse", "nocse")):
        module = bench.compiled_module("intervals", system, sizes, text, pipeline_name)
        for label, threads in thread_settings():
            runner = pipeline.Runner(module, label)

            got = bench.measure("intervals", system, threads, "end_to_end", sizes,
                                lambda: runner(X, model.beta, model.cov))
            bench.check("intervals", system, threads, sizes, np.stack(got, axis=1), want, "econml_loop")

            X_device = runner.to_device(X)
            beta_device, cov_device = runner.to_device(model.beta), runner.to_device(model.cov)
            bench.measure("intervals", system, threads, "compute_only", sizes,
                          lambda: runner.call_raw(X_device, beta_device, cov_device))
            del runner, X_device, beta_device, cov_device
    gc.collect()


def bench_ate(bench: Bench, n: int, M: int) -> None:
    """Average effect for M models by ATE_SCENARIOS scenarios on one population."""
    K = ATE_SCENARIOS
    sizes = dict(n=n, K=K, M=M)
    deltas = scenario_deltas(K)
    X = data.make_features(n, bench.plan.features)
    models = data.perturbed_models(bench.model, M)
    betas = [m.beta for m in models]

    def numpy_plain():
        # One pass over X per model
        return np.stack([reference.ate(m, X, deltas) for m in models])

    def numpy_pushdown():
        # One pass over X in total: the same algebra the compiler applies
        xbar = X.mean(axis=0)
        return np.stack([reference.ate_pushdown(m, X, deltas, xbar) for m in models])

    want = bench.measure("ate", "numpy_ref", "blas", "end_to_end", sizes, numpy_plain)

    got = bench.measure("ate", "numpy_pushdown", "blas", "end_to_end", sizes, numpy_pushdown)
    bench.check("ate", "numpy_pushdown", "blas", sizes, got, want, "numpy_ref")

    # Only one model was really fitted, so EconML is measured at M = 1 only
    if M == 1:
        estimator = bench.estimator
        got = bench.measure(
            "ate", "econml_loop", "blas", "end_to_end", sizes,
            lambda: np.array([[estimator.ate(X, T0=0, T1=delta) for delta in deltas]]),
        )
        bench.check("ate", "numpy_ref", "blas", sizes, want, got, "econml_loop")

    text = emit.emit_ates(bench.plan.features, M, deltas)
    for system, pipeline_name in (("compiled", "full"), ("compiled_nopush", "nopush")):
        module = bench.compiled_module("ate", system, sizes, text, pipeline_name)
        for label, threads in thread_settings():
            runner = pipeline.Runner(module, label)

            got = bench.measure("ate", system, threads, "end_to_end", sizes,
                                lambda: np.array(runner(X, *betas)).reshape(M, K))
            bench.check("ate", system, threads, sizes, got, want, "numpy_ref")

            X_device = runner.to_device(X)
            betas_device = [runner.to_device(b) for b in betas]
            bench.measure("ate", system, threads, "compute_only", sizes,
                          lambda: runner.call_raw(X_device, *betas_device))
            del runner, X_device, betas_device
    gc.collect()


def record_op_counts(bench: Bench) -> None:
    """Count operations after each compiler stage, for the op-count table."""
    d, K = bench.plan.features, bench.plan.main_scenarios
    M = max(bench.plan.model_sweep[:-1]) if len(bench.plan.model_sweep) > 1 else bench.plan.model_sweep[0]
    programs = {
        f"effects K={K}": (emit.emit_effects(d, scenario_deltas(K)), "--dml-decompose"),
        f"intervals K={K}": (emit.emit_intervals(d, scenario_deltas(K)), "--dml-decompose"),
        f"ate M={M} K={ATE_SCENARIOS}": (emit.emit_ates(d, M, scenario_deltas(ATE_SCENARIOS)), "--dml-decompose"),
        f"ate M={M} K={ATE_SCENARIOS} (no pushdown)":
            (emit.emit_ates(d, M, scenario_deltas(ATE_SCENARIOS)), "--dml-decompose-no-pushdown"),
    }
    for name, (text, decompose_flag) in programs.items():
        decomposed = pipeline.opt(text, decompose_flag)
        shared = pipeline.opt(decomposed, "--cse")
        lowered = pipeline.opt(shared, "--dml-lower-to-linalg")
        stages = (("as_written", text), ("decomposed", decomposed), ("cse", shared), ("lowered", lowered))
        for stage, stage_text in stages:
            for op, count in sorted(pipeline.count_ops(stage_text).items()):
                bench.record.op_counts.append(dict(workload=name, stage=stage, op=op, count=count))


def describe_environment(plan: Plan, quick: bool, fit_seconds: float) -> dict:
    """Machine and package details, stored next to the numbers they produced."""
    from importlib.metadata import version

    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        ram_gb = None

    try:
        from threadpoolctl import threadpool_info
        blas = [f"{p.get('internal_api')} {p.get('num_threads')} threads" for p in threadpool_info()]
    except ImportError:
        blas = []

    packages = ("numpy", "scipy", "scikit-learn", "econml", "iree-base-compiler", "iree-base-runtime")
    return dict(
        date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        mode="quick" if quick else "full",
        cpu_model=cpu_model,
        cpu_cores=os.cpu_count(),
        ram_gb=ram_gb,
        os=platform.platform(),
        inside_docker=Path("/.dockerenv").exists(),
        python=platform.python_version(),
        packages={name: version(name) for name in packages},
        blas=blas,
        warmup_runs=plan.warmup,
        timed_runs=plan.repeats,
        train_rows=plan.train_rows,
        econml_fit_seconds=round(fit_seconds, 3),
    )


def write_csv(path: Path, rows: list) -> None:
    """Write a list of dicts as CSV, using the first row's keys as the header."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results", help="folder for the CSV files")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a fast sanity check")
    args = parser.parse_args()

    plan = QUICK if args.quick else FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Fit the one EconML model every workload queries
    print(f"fitting LinearDML on {plan.train_rows:,} rows ...", flush=True)
    start = time.perf_counter()
    estimator = data.fit_econml(plan.features, plan.train_rows)
    fit_seconds = time.perf_counter() - start
    bench = Bench(plan, estimator, reference.extract(estimator), Recorder())

    # The first IREE compile in a process is slow; do one that is never timed
    pipeline.compile_module(emit.emit_effects(plan.features, [1.0]))

    largest = plan.row_sweep[-1]

    # Rows sweep at the main scenario count
    for n in plan.row_sweep:
        bench_effects(bench, n, plan.main_scenarios)
        bench_intervals(bench, n, plan.main_scenarios)

    # Scenario sweep at the largest row count (the main point is already done)
    for K in plan.effect_scenarios:
        if K != plan.main_scenarios:
            bench_effects(bench, largest, K)
    for K in plan.interval_scenarios:
        if K != plan.main_scenarios:
            bench_intervals(bench, largest, K)

    # Model sweep for the average effect
    for M in plan.model_sweep:
        bench_ate(bench, largest, M)

    record_op_counts(bench)

    write_csv(out / "bench.csv", bench.record.timings)
    write_csv(out / "compile.csv", bench.record.compiles)
    write_csv(out / "errors.csv", bench.record.errors)
    write_csv(out / "opcounts.csv", bench.record.op_counts)
    (out / "env.json").write_text(json.dumps(describe_environment(plan, args.quick, fit_seconds), indent=2) + "\n")

    worst = max(row["max_abs_error"] for row in bench.record.errors)
    print(f"done: {len(bench.record.timings)} timed runs, worst error {worst:.2e}, files in {out}/")


if __name__ == "__main__":
    main()
