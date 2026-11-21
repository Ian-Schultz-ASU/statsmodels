"""Helpers shared by the figure and report scripts: read the CSVs, take medians."""
import csv
import json
import statistics
from pathlib import Path

# Display names, used in every figure and table so they always match.
SYSTEM_NAMES = {
    "econml_loop": "EconML, one call per scenario",
    "econml_smart": "EconML, careful user",
    "numpy_ref": "Hand-written NumPy",
    "numpy_pushdown": "NumPy with mean pushdown",
    "compiled": "dmlc compiled",
    "compiled_nocse": "dmlc without our CSE step",
    "compiled_nopush": "dmlc without mean pushdown",
}


def read_csv(path: Path) -> list:
    """Read a CSV file into a list of dicts. A missing file gives an empty list."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_env(results: Path) -> dict:
    """Machine and package details written by bench/run.py."""
    path = results / "env.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Timings:
    """Median seconds per measurement, looked up by what was measured."""

    def __init__(self, results: Path):
        # Group the raw runs by everything except the run index
        runs = {}
        for row in read_csv(results / "bench.csv"):
            key = (row["workload"], row["system"], row["threads"], row["timing"],
                   int(row["n"]), int(row["K"]), int(row["M"]))
            runs.setdefault(key, []).append(float(row["seconds"]))

        # One median per measurement
        self.medians = {key: statistics.median(values) for key, values in runs.items()}

    def get(self, workload, system, threads, timing, n, K, M=1):
        """Median seconds, or None when that combination was not measured."""
        return self.medians.get((workload, system, str(threads), timing, n, K, M))

    def values(self, position: int, workload: str) -> list:
        """Sorted distinct values of one key field (4 = n, 5 = K, 6 = M) for a workload."""
        return sorted({key[position] for key in self.medians if key[0] == workload})

    def thread_counts(self) -> list:
        """Sorted IREE thread counts that were measured, for example [1, 4]."""
        return sorted({int(key[2]) for key in self.medians if key[2] != "blas"})


def ms(seconds) -> str:
    """Format seconds as milliseconds with three significant digits."""
    if seconds is None:
        return "n/a"
    return f"{seconds * 1e3:.3g}" if seconds * 1e3 < 1000 else f"{seconds * 1e3:,.0f}"


def ratio(slow, fast) -> str:
    """How many times faster `fast` is than `slow`, for example '6.2x'."""
    if slow is None or fast is None:
        return "n/a"
    value = slow / fast
    return f"{value:.1f}x" if value < 100 else f"{value:,.0f}x"
