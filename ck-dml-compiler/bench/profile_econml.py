"""Profile one EconML effect() call, to show where its time goes.

Usage (from the repo root):
    python bench/profile_econml.py --out results

Writes results/econml_profile.csv with the functions that used the most time
themselves (self time, not counting the functions they call).
"""
import argparse
import cProfile
import csv
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dmlc import data  # noqa: E402

# How many functions to keep in the table.
TOP = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results", help="folder for econml_profile.csv")
    parser.add_argument("--rows", type=int, default=1_000_000, help="rows to score")
    parser.add_argument("--train-rows", type=int, default=200_000, help="rows used to fit")
    args = parser.parse_args()

    # Same model and features as the benchmark
    estimator = data.fit_econml(n_train=args.train_rows)
    X = data.make_features(args.rows)

    # One untimed call first, so one-off setup work is not counted
    estimator.effect(X, T0=0, T1=1)

    # Profile exactly one call
    profiler = cProfile.Profile()
    profiler.enable()
    estimator.effect(X, T0=0, T1=1)
    profiler.disable()

    # stats maps (file, line, function) -> (calls, primitive calls, self time, cumulative, callers)
    stats = pstats.Stats(profiler).stats
    total = sum(entry[2] for entry in stats.values())
    slowest = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)[:TOP]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "econml_profile.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["function", "file", "line", "calls", "self_seconds", "share_of_call", "rows"])
        for (file, line, function), (calls, _, self_seconds, _, _) in slowest:
            writer.writerow([function, Path(file).name, line, calls,
                             f"{self_seconds:.6f}", f"{self_seconds / total:.4f}", args.rows])
            print(f"  {self_seconds * 1e3:8.2f} ms  {self_seconds / total:6.1%}  {function}  ({Path(file).name}:{line})")
    print(f"total {total * 1e3:.1f} ms for one effect() call on {args.rows:,} rows")


if __name__ == "__main__":
    main()
