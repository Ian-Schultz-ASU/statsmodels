"""Run the compiler: dml-opt passes, IREE (Intermediate Representation Execution Environment) code generation, execution.

The full pipeline, replacing repeated EconML calls with one compiled program:

    1. emit: batch all queries into one MLIR module  (emit.py)
    2. dml-opt: decompose query ops, CSE (Common Subexpression Elimination)
       merges shared work, mean pushdown optimizes averages  (DMLPasses.cpp)
    3. IREE: compile optimized MLIR to native CPU code
    4. Runner: execute once, return all answers at once

Example:
    text = emit.emit_effects(d, deltas)       # step 1: naive MLIR, one op per query
    module = compile_module(text)              # steps 2-3: optimize + compile
    outputs = Runner(module)(X, beta)          # step 4: run, get list of NumPy arrays
"""
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

# Pass pipelines, in the order dml-opt applies them.
# "full" is the optimized path: decompose breaks query ops into smaller ops,
# --cse merges identical ops (e.g., two X @ beta[1:] become one), and
# --dml-lower-to-linalg converts to standard dialects that IREE can compile.
PIPELINES = {
    "full": ("--dml-decompose", "--cse", "--dml-lower-to-linalg"),
    "nocse": ("--dml-decompose", "--dml-lower-to-linalg"),           # skip CSE, for benchmarking
    "nopush": ("--dml-decompose-no-pushdown", "--cse", "--dml-lower-to-linalg"),  # skip mean pushdown
}

# IREE flags. The f64 flags are required: without them IREE silently computes
# in float32 and results no longer match EconML.
IREE_FLAGS = (
    "--iree-input-demote-f64-to-f32=false",
    "--iree-vm-target-extension-f64",
    "--iree-vm-target-truncate-unsupported-floats=false",
    "--iree-llvmcpu-target-cpu=host",
    "--iree-opt-level=O3",
)

# IREE runtime drivers: one thread, or a pool with one worker per core.
DRIVERS = {"single": "local-sync", "multi": "local-task"}


def find_dml_opt() -> str:
    """Locate the dml-opt binary: $DML_OPT, then PATH, then the usual build folders."""
    candidates = [os.environ.get("DML_OPT"), shutil.which("dml-opt")]
    repo = Path(__file__).resolve().parents[2]
    candidates += [Path("/build/tools/dml-opt/dml-opt"), repo / "dmlc/build/tools/dml-opt/dml-opt"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("dml-opt not found. Build it (make build) or set DML_OPT=/path/to/dml-opt.")


def opt(mlir_text: str, *passes: str) -> str:
    """Run dml-opt with the given pass flags and return the resulting IR text."""
    result = subprocess.run([find_dml_opt(), *passes], input=mlir_text, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"dml-opt {' '.join(passes)} failed:\n{result.stderr}")
    return result.stdout


def count_ops(mlir_text: str) -> dict:
    """Count dml.* and linalg.* operations in IR text (linalg.yield excluded)."""
    names = re.findall(r"(?<![\w.])((?:dml|linalg)\.[a-z_]+)", mlir_text)
    return dict(Counter(name for name in names if name != "linalg.yield"))


def lower(mlir_text: str, pipeline: str = "full") -> str:
    """Apply a named pass pipeline. The result contains only standard dialects."""
    return opt(mlir_text, *PIPELINES[pipeline])


def compile_iree(lowered_text: str) -> bytes:
    """Compile lowered IR to an IREE module for the host CPU."""
    from iree.compiler import compile_str

    return compile_str(lowered_text, target_backends=["llvm-cpu"], extra_args=list(IREE_FLAGS))


def compile_module(mlir_text: str, pipeline: str = "full") -> bytes:
    """Query-level MLIR text in, executable IREE module out.

    This replaces EconML entirely for inference: the returned module runs
    all batched queries in one call, with shared work eliminated by the passes.
    """
    return compile_iree(lower(mlir_text, pipeline))


class Runner:
    """Executes the `@query` function of a compiled module.

    threads="single" runs on the calling thread; "multi" uses one worker per core.
    """

    def __init__(self, module: bytes, threads: str = "multi"):
        import iree.runtime as rt

        self._rt = rt
        self.context = rt.SystemContext(config=rt.Config(DRIVERS[threads]))
        self.context.add_vm_module(rt.VmModule.from_flatbuffer(self.context.instance, module, warn_if_copy=False))
        self.function = self.context.modules.module.query

    def to_device(self, array: np.ndarray):
        """Copy a NumPy array into the runtime once, so later calls skip the copy."""
        return self._rt.asdevicearray(self.context.config.device, np.ascontiguousarray(array, dtype=np.float64))

    def call_raw(self, *arrays):
        """Run and return IREE's own buffers, without converting to NumPy."""
        return self.function(*arrays)

    def __call__(self, *arrays) -> list:
        """Run and return a list of NumPy arrays that own their memory."""
        raw = self.function(*arrays)
        raw = raw if isinstance(raw, (tuple, list)) else (raw,)
        # Copy while `raw` is alive: a view would point at an IREE buffer that
        # is freed as soon as `raw` goes out of scope.
        return [np.array(np.asarray(r), copy=True) for r in raw]
