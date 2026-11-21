"""Synthetic data and the EconML model that serves as the external "oracle"."""
import warnings

import numpy as np

from .reference import FittedLinearDML, extract


def make_data(n: int, d_x: int = 10, d_w: int = 10, seed: int = 0):
    """Build partially linear data with a known effect.

    The true effect is theta(x) = 1 + 0.5*x[0] - 0.3*x[1], so after fitting
    beta should be close to [1.0, 0.5, -0.3, 0.0, ...]. The intercept (beta[0])
    is the baseline effect when all features are zero.

    theta(x) = 1 + 0.5 * x[0] - 0.3 * x[1]   (column indices wrap when d_x < 3)
    T = 0.5 * x[0] + w[0] + noise
    Y = theta(x) * T + x[2] + w[1] + noise

    Returns (Y, T, X, W, theta) with X of shape (n, d_x) and W of shape (n, d_w).
    """
    rng = np.random.default_rng(seed)

    # X (features): columns the effect varies with. W (controls): remove confounding only.
    X = rng.normal(size=(n, d_x))
    W = rng.normal(size=(n, d_w))

    # Treatment depends on X and W, so W is a confounder that must be cleaned out.
    T = 0.5 * X[:, 0] + W[:, 0] + rng.normal(size=n)

    # True per-row effect: only X columns 0 and 1 matter.
    theta = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1 % d_x]

    # Outcome = effect * treatment + confounding from X and W + noise.
    Y = theta * T + X[:, 2 % d_x] + W[:, 1 % d_w] + rng.normal(size=n)
    return Y, T, np.ascontiguousarray(X), W, theta


def make_features(n: int, d_x: int = 10, seed: int = 1) -> np.ndarray:
    """Scoring features only, C-contiguous float64, shape (n, d_x)."""
    return make_data(n, d_x=d_x, seed=seed)[2]


def fit_econml(d_x: int = 10, n_train: int = 200_000, seed: int = 0):
    """Fit the LinearDML used everywhere in this project. Returns the estimator.

    DML (Double Machine Learning) final stage: OLS (Ordinary Least Squares) on
    T_resid * [1, X] against Y_resid. Produces beta (p,) and cov (p, p), which
    are all the compiler needs for inference.
    """
    from econml.dml import LinearDML
    from sklearn.linear_model import LinearRegression

    Y, T, X, W, _ = make_data(n_train, d_x=d_x, seed=seed)

    # Two helper models clean confounders; cv=5 fits them out-of-fold to avoid overfitting.
    est = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), cv=5, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(Y, T, X=X, W=W)
    return est


def perturbed_models(base: FittedLinearDML, n_models: int, seed: int = 3):
    """`n_models` models on the same features: the fitted one, then noisy copies.

    Stands in for many outcome metrics analysed on one population. Only the
    coefficients differ, which is all the inference path depends on.
    """
    rng = np.random.default_rng(seed)
    models = [base]
    for _ in range(n_models - 1):
        models.append(FittedLinearDML(base.beta + 0.1 * rng.normal(size=base.beta.size), base.cov))
    return models


__all__ = ["make_data", "make_features", "fit_econml", "perturbed_models", "extract"]
