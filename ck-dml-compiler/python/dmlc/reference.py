"""NumPy reference for the inference path of EconML's LinearDML.

This file is the correctness oracle for the compiler. Every function is the
plain formula from docs/PRD.md section 3.2, written to be read, not to be fast.

Shapes: n rows, d features, p = d + 1.
    X      (n, d)   scoring features
    beta   (p,)     final-stage coefficients, intercept first
    cov    (p, p)   covariance matrix of beta
    deltas (K,)     one treatment change T1 - T0 per scenario
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class FittedLinearDML:
    """The two arrays a fitted LinearDML needs for inference.

    After fitting, the helper models and residuals are gone. All inference
    (effects, confidence intervals, averages) is arithmetic on these two
    arrays alone, which is why this is all the compiler needs as input.

    beta: (p,) OLS (Ordinary Least Squares) coefficients, intercept first.
          p = d + 1 because beta[0] is the intercept (baseline effect when
          all features are zero) and beta[1:] are the per-feature slopes.
    cov:  (p, p) covariance matrix of beta, measuring uncertainty in each
          coefficient. Used to compute standard errors and confidence intervals.
    """

    beta: np.ndarray
    cov: np.ndarray

    @property
    def d(self) -> int:
        return self.beta.size - 1


def extract(est) -> FittedLinearDML:
    """Pull beta and cov out of a fitted econml.dml.LinearDML.

    This is the bridge from EconML to our compiler. EconML stores the OLS
    (Ordinary Least Squares) result in private attributes of its final-stage
    model. We copy them into FittedLinearDML, and from here on the compiler
    never touches EconML again.

    `_param` and `_param_var` are private attributes of EconML's final model.
    They are stable for the EconML version pinned in requirements.txt.
    """
    final = est.model_final_

    # beta: the OLS coefficients from the final stage, intercept first.
    beta = np.asarray(final._param, dtype=np.float64).ravel()

    # cov: the covariance of beta, used for standard errors and intervals.
    cov = np.asarray(final._param_var, dtype=np.float64)
    return FittedLinearDML(beta, cov)


def z_score(alpha: float) -> float:
    """Normal quantile for a two-sided (1 - alpha) interval. 1.96 for alpha = 0.05."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def cate(m: FittedLinearDML, X: np.ndarray) -> np.ndarray:
    """Effect size per row: theta_i = beta[0] + x_i . beta[1:].  Shape (n,).

    CATE (Conditional Average Treatment Effect): the effect for one patient
    given their features. beta[0] is the intercept (baseline), beta[1:] tilt
    the effect up or down per feature.
    """
    return m.beta[0] + X @ m.beta[1:]


def cate_stderr(m: FittedLinearDML, X: np.ndarray) -> np.ndarray:
    """Standard error of theta_i: sqrt([1, x_i] cov [1, x_i]^T).  Shape (n,)."""
    phi = np.hstack([np.ones((X.shape[0], 1)), X])          # (n, p)
    variance = np.einsum("ij,ij->i", phi @ m.cov, phi)      # row-wise quadratic form
    return np.sqrt(variance)


def effects(m: FittedLinearDML, X: np.ndarray, deltas) -> np.ndarray:
    """effect[i, k] = deltas[k] * theta_i.  Shape (n, K)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    # calls cate() so the theta is recomputed every time this function is called, Compiler fixes this 
    # Also If delta were baked into theta instead of kept separate, you could not share one theta computation across 20 scenarios. 
    #   - This separation is what makes the compiler's shared-work optimization possible.
    return np.multiply.outer(cate(m, X), deltas)


def effect_intervals(m: FittedLinearDML, X: np.ndarray, deltas, alpha: float = 0.05):
    """Normal confidence interval of each effect. Returns (lo, hi), each (n, K)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    center = np.multiply.outer(cate(m, X), deltas)
    half_width = z_score(alpha) * np.multiply.outer(cate_stderr(m, X), np.abs(deltas))
    return center - half_width, center + half_width


def ate(m: FittedLinearDML, X: np.ndarray, deltas) -> np.ndarray:
    """Average effect per scenario: deltas[k] * mean_i(theta_i).  Shape (K,)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    return deltas * np.mean(cate(m, X))


def ate_pushdown(m: FittedLinearDML, X: np.ndarray, deltas, xbar=None) -> np.ndarray:
    """Same value as `ate`, using mean(X @ b) == mean(X) @ b.

    Pass a precomputed `xbar = X.mean(axis=0)` to share it across models.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if xbar is None:
        xbar = X.mean(axis=0)
    return deltas * (m.beta[0] + xbar @ m.beta[1:])
