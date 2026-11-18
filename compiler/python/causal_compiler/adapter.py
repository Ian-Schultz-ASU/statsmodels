"""Statsmodels adapter for extracting causal contrast inputs."""

from dataclasses import dataclass
from numbers import Real
import re

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from .contrast import treatment_effect


@dataclass(frozen=True)
class ContrastInputs:
    """Extracted arrays and parameter indices for an OLS treatment contrast.

    Attributes
    ----------
    interaction_data : ndarray
        Two-dimensional array of covariates that interact with the treatment.
    coefficients : ndarray
        Full fitted parameter vector from the OLS model.
    main_effect_index : int
        Index of the treatment main effect in ``coefficients``.
    interaction_indices : tuple of int
        Indices of the treatment-interaction terms in ``coefficients``.
    interaction_columns : tuple of str
        DataFrame column names corresponding to ``interaction_data`` columns.
    treatment_var : str
        Name of the treatment column.
    control_value : float
        Baseline (control) treatment value.
    treated_value : float
        Counterfactual (treated) treatment value.
    """

    interaction_data: np.ndarray
    coefficients: np.ndarray
    main_effect_index: int
    interaction_indices: tuple[int, ...]
    interaction_columns: tuple[str, ...]
    treatment_var: str
    control_value: float
    treated_value: float

    @property
    def treatment_delta(self) -> float:
        """Treated treatment value minus control treatment value."""
        return float(self.treated_value - self.control_value)

    @property
    def main_effect(self) -> float:
        """Fitted coefficient of the treatment main effect."""
        return float(self.coefficients[self.main_effect_index])

    @property
    def interaction_coefficients(self) -> np.ndarray:
        """Fitted coefficients for the treatment-interaction terms."""
        return self.coefficients[list(self.interaction_indices)]


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)


def extract_ols_contrast(
    results,
    treatment_var: str,
    treatment_values: tuple[Real, Real] | list[Real],
    data: pd.DataFrame,
) -> ContrastInputs:
    """Extract contrast inputs from a fitted formula-based OLS result.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted ``statsmodels`` formula OLS results object.
    treatment_var : str
        Name of the numeric treatment column.
    treatment_values : tuple of float or list of float
        Pair ``(control_value, treated_value)`` of distinct finite numbers.
    data : pandas.DataFrame
        Scoring dataset containing the treatment and interaction columns.

    Returns
    -------
    ContrastInputs
        Extracted arrays, parameter indices, and treatment delta.

    Raises
    ------
    TypeError
        If inputs have unsupported types or non-numeric columns.
    ValueError
        If the model is not formula-based OLS, lacks required terms, or has
        unsupported transformed/higher-order treatment interactions.
    """
    model = getattr(results, "model", None)
    if not isinstance(model, OLS):
        raise TypeError("results must be fitted OLS results")
    if not hasattr(results, "params"):
        raise ValueError("results must contain fitted parameters")
    if not hasattr(model, "formula"):
        raise ValueError("results must come from a formula-based OLS model")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row")
    if not isinstance(treatment_var, str) or not treatment_var:
        raise TypeError("treatment_var must be a non-empty column name")
    if treatment_var not in data.columns:
        raise ValueError(f"treatment column {treatment_var!r} is missing from data")
    if not pd.api.types.is_numeric_dtype(data[treatment_var]):
        raise TypeError("treatment column must contain numeric values")
    if (
        not isinstance(treatment_values, (tuple, list))
        or len(treatment_values) != 2
        or not all(_is_finite_real(v) for v in treatment_values)
        or treatment_values[0] == treatment_values[1]
    ):
        raise ValueError(
            "treatment_values must be a pair of distinct finite numeric values"
        )

    names = list(model.exog_names)
    if treatment_var not in names:
        raise ValueError(f"model must include a main effect for {treatment_var!r}")

    interaction_columns: list[str] = []
    interaction_indices: list[int] = []
    pattern = re.compile(rf"\b{re.escape(treatment_var)}\b")

    for index, name in enumerate(names):
        if name == treatment_var:
            continue
        factors = name.split(":")
        if not any(pattern.search(factor) for factor in factors):
            continue
        if treatment_var not in factors:
            raise ValueError("transformed treatment terms are not supported")
        if len(factors) != 2 or factors.count(treatment_var) != 1:
            raise ValueError(
                "only two-factor interactions with one treatment factor are supported"
            )
        col = factors[1] if factors[0] == treatment_var else factors[0]
        if col not in data.columns:
            raise ValueError("interaction factors must be raw numeric data columns")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise TypeError("interaction columns must contain numeric values")
        interaction_columns.append(col)
        interaction_indices.append(index)

    if not interaction_columns:
        raise ValueError("the model must include at least one treatment interaction")

    coefficients = np.asarray(results.params, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size != len(names):
        raise ValueError("fitted parameters do not match model design columns")
    if not np.isfinite(coefficients).all():
        raise ValueError("fitted coefficients must all be finite")

    interaction_data = data[interaction_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(interaction_data).all():
        raise ValueError("interaction data must contain only finite values")

    control_value, treated_value = treatment_values
    return ContrastInputs(
        interaction_data=interaction_data,
        coefficients=coefficients,
        main_effect_index=names.index(treatment_var),
        interaction_indices=tuple(interaction_indices),
        interaction_columns=tuple(interaction_columns),
        treatment_var=treatment_var,
        control_value=float(control_value),
        treated_value=float(treated_value),
    )


def evaluate_ols_contrast(
    results,
    treatment_var: str,
    treatment_values: tuple[Real, Real] | list[Real],
    data: pd.DataFrame,
) -> pd.Series:
    """Compute row-level OLS treatment contrasts from a fitted model.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted ``statsmodels`` formula OLS results object.
    treatment_var : str
        Name of the numeric treatment column.
    treatment_values : tuple of float or list of float
        Pair ``(control_value, treated_value)`` of distinct finite numbers.
    data : pandas.DataFrame
        Scoring dataset containing the treatment and interaction columns.

    Returns
    -------
    pandas.Series
        Predicted treated-minus-control contrast indexed like ``data``.
    """
    inputs = extract_ols_contrast(results, treatment_var, treatment_values, data)
    effects = treatment_effect(
        inputs.interaction_data,
        main_effect=inputs.main_effect,
        interaction_coefficients=inputs.interaction_coefficients,
        treatment_delta=inputs.treatment_delta,
    )
    return pd.Series(effects, index=data.index, name="treatment_effect")
