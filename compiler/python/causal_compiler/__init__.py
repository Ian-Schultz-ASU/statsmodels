"""Causal inference compiler components."""

from .adapter import ContrastInputs, evaluate_ols_contrast, extract_ols_contrast
from .compiler import compile_iree, lower_mlir, run_iree
from .contrast import treatment_effect
from .mlir import emit_treatment_effect_mlir

__all__ = [
    "ContrastInputs",
    "compile_iree",
    "emit_treatment_effect_mlir",
    "evaluate_ols_contrast",
    "extract_ols_contrast",
    "lower_mlir",
    "run_iree",
    "treatment_effect",
]
