"""Write a batch of analyst queries as MLIR text in the `dml` dialect.

This is the entry point to the compiler pipeline. Instead of calling EconML
once per query (which rebuilds X @ beta[1:] every time), we emit all queries
into one MLIR module. The compiler then finds shared subexpressions (like
X @ beta[1:] appearing in every query) and eliminates the repeated work.

The emitter is deliberately naive: one query op per question, with no sharing.
Finding the shared work is the compiler's job, not the emitter's.

The row count is dynamic (`?`) so one compiled module serves any dataset size.
The feature count `d` is static because the lowering slices beta and cov by it.
"""
from .reference import z_score


def _f64(value: float) -> str:
    """A float64 literal that round-trips exactly through MLIR text."""
    return f"{float(value):.17e}"


def _module(args, body, results, result_types) -> str:
    types = ", ".join(result_types)
    return (
        "module {\n"
        f"  func.func public @query({', '.join(args)}) -> ({types}) {{\n"
        + "\n".join(f"    {line}" for line in body)
        + f"\n    return {', '.join(results)} : {types}\n"
        "  }\n"
        "}\n"
    )


def emit_effects(d: int, deltas) -> str:
    """K per-row effect queries on one model.

    Each delta is a treatment change (T1 - T0). EconML would need one
    est.effect() call per delta, each recomputing X @ beta[1:]. Here we
    emit all K queries; CSE (Common Subexpression Elimination) merges the
    shared X @ beta[1:] into one computation.

    Signature: @query(X: (?, d), beta: (d+1,)) -> K tensors of shape (?,)
    """
    X, B, V = f"tensor<?x{d}xf64>", f"tensor<{d + 1}xf64>", "tensor<?xf64>"
    body, results = [], []
    for k, delta in enumerate(deltas):
        body.append(f'%e{k} = "dml.effect"(%X, %beta) {{delta = {_f64(delta)} : f64}} : ({X}, {B}) -> {V}')
        results.append(f"%e{k}")
    return _module([f"%X: {X}", f"%beta: {B}"], body, results, [V] * len(results))


def emit_intervals(d: int, deltas, alpha: float = 0.05) -> str:
    """K per-row confidence-interval queries on one model.

    Signature: @query(X, beta, cov: (d+1, d+1)) -> lo0, hi0, lo1, hi1, ...
    """
    p = d + 1
    X, B, C, V = f"tensor<?x{d}xf64>", f"tensor<{p}xf64>", f"tensor<{p}x{p}xf64>", "tensor<?xf64>"
    z = _f64(z_score(alpha))
    body, results = [], []
    for k, delta in enumerate(deltas):
        body.append(
            f'%lo{k}, %hi{k} = "dml.effect_interval"(%X, %beta, %cov) '
            f"{{delta = {_f64(delta)} : f64, z = {z} : f64}} : ({X}, {B}, {C}) -> ({V}, {V})"
        )
        results += [f"%lo{k}", f"%hi{k}"]
    return _module([f"%X: {X}", f"%beta: {B}", f"%cov: {C}"], body, results, [V] * len(results))


def emit_ates(d: int, n_models: int, deltas) -> str:
    """Average effect for every (model, scenario) pair on one population.

    EconML would loop: for each model, for each delta, call est.ate().
    Each call computes X @ beta[1:] then averages over n rows. Our
    compiler applies mean pushdown: mean(X @ b) == mean(X) @ b, reducing
    an O(n*d) matmul to an O(d) dot product per model.

    Signature: @query(X, b0, b1, ..., b{M-1}) -> M*K rank-0 tensors,
    ordered model by model, scenario by scenario within a model.
    """
    X, B, V = f"tensor<?x{d}xf64>", f"tensor<{d + 1}xf64>", "tensor<f64>"
    args = [f"%X: {X}"] + [f"%b{m}: {B}" for m in range(n_models)]
    body, results = [], []
    for m in range(n_models):
        for k, delta in enumerate(deltas):
            body.append(f'%a{m}_{k} = "dml.ate"(%X, %b{m}) {{delta = {_f64(delta)} : f64}} : ({X}, {B}) -> {V}')
            results.append(f"%a{m}_{k}")
    return _module(args, body, results, [V] * len(results))
