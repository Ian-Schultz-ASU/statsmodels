"""MLIR IR emitter for causal inference operations."""

import json

from .adapter import ContrastInputs


def emit_treatment_effect_mlir(inputs: ContrastInputs) -> str:
    """Emit a generic MLIR module containing ``ci.treatment_effect``.

    The data matrix and coefficient vector are passed as function arguments so
    large datasets are never embedded as IR constants.

    Parameters
    ----------
    inputs : ContrastInputs
        Extracted model parameters, indices, and data shapes from the
        ``statsmodels`` adapter.

    Returns
    -------
    str
        MLIR module text defining ``@counterfactual``.
    """
    row_count, interaction_count = inputs.interaction_data.shape
    coefficient_count = inputs.coefficients.size
    interaction_columns = ", ".join(
        json.dumps(col) for col in inputs.interaction_columns
    )
    coefficient_indices = ", ".join(str(idx) for idx in inputs.interaction_indices)

    return f"""module {{
  func.func public @counterfactual(
      %data: tensor<{row_count}x{interaction_count}xf64>,
      %coefficients: tensor<{coefficient_count}xf64>
  ) -> tensor<{row_count}xf64> {{
    %effects = "ci.treatment_effect"(%data, %coefficients) {{
      control_value = {inputs.control_value:.17e} : f64,
      interaction_columns = [{interaction_columns}],
      interaction_coefficients = array<i64: {coefficient_indices}>,
      main_effect_index = {inputs.main_effect_index} : i64,
      treated_value = {inputs.treated_value:.17e} : f64,
      treatment_name = {json.dumps(inputs.treatment_var)}
    }} : (
      tensor<{row_count}x{interaction_count}xf64>,
      tensor<{coefficient_count}xf64>
    ) -> tensor<{row_count}xf64>
    return %effects : tensor<{row_count}xf64>
  }}
}}"""
