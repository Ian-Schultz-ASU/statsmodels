"""Reference implementations for causal contrast calculations."""

import numpy as np


def treatment_effect(
    interaction_data: np.ndarray,
    main_effect: float,
    interaction_coefficients: np.ndarray,
    treatment_delta: float,
) -> np.ndarray:
    """Compute a row-level OLS treatment contrast from interaction terms.

    The operation is:

    ``treatment_delta * (main_effect + interaction_data @ interaction_coefficients)``

    Parameters
    ----------
    interaction_data : ndarray
        Matrix with one observation per row and one treatment-interaction
        feature per column.
    main_effect : float
        Fitted coefficient of the treatment main effect.
    interaction_coefficients : ndarray
        Fitted coefficients corresponding to the columns of
        ``interaction_data``.
    treatment_delta : float
        Treated treatment value minus control treatment value.

    Returns
    -------
    ndarray
        Treatment contrast for each observation.

    Raises
    ------
    ValueError
        If the input array dimensions or feature counts do not match.
    """
    interaction_data = np.asarray(interaction_data, dtype=np.float64)
    interaction_coefficients = np.asarray(
        interaction_coefficients, dtype=np.float64
    )
    if interaction_data.ndim != 2:
        raise ValueError("interaction_data must be a two-dimensional array")
    if interaction_coefficients.ndim != 1:
        raise ValueError(
            "interaction_coefficients must be a one-dimensional array"
        )
    if interaction_data.shape[1] != interaction_coefficients.size:
        raise ValueError(
            "interaction_data columns must match interaction_coefficients"
        )

    return treatment_delta * (
        main_effect + interaction_data @ interaction_coefficients
    )
