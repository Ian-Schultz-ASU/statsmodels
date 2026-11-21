"""FR-1: the NumPy reference agrees with EconML (the external oracle)."""
import numpy as np
import pytest

from dmlc import data, reference

from conftest import DELTAS, TOLERANCE, fitted_estimator, fitted_model

ROW_COUNTS = (1, 257, 5000)
DATA_SEEDS = (1, 2)


@pytest.fixture(params=[(n, seed) for n in ROW_COUNTS for seed in DATA_SEEDS], ids=lambda p: f"n={p[0]}-seed={p[1]}")
def X(request, d):
    n, seed = request.param
    return data.make_features(n, d_x=d, seed=seed)


def test_extracted_parameters_have_expected_shapes(d):
    model = fitted_model(d)
    assert model.beta.shape == (d + 1,)
    assert model.cov.shape == (d + 1, d + 1)
    np.testing.assert_allclose(model.cov, model.cov.T, rtol=0, atol=1e-15)


def test_cate_matches_const_marginal_effect(d, X):
    want = fitted_estimator(d).const_marginal_effect(X).ravel()
    np.testing.assert_allclose(reference.cate(fitted_model(d), X), want, **TOLERANCE)


def test_effects_match_econml(d, X):
    est = fitted_estimator(d)
    got = reference.effects(fitted_model(d), X, DELTAS)
    assert got.shape == (X.shape[0], len(DELTAS))
    for k, delta in enumerate(DELTAS):
        want = np.ravel(est.effect(X, T0=0, T1=delta))
        np.testing.assert_allclose(got[:, k], want, **TOLERANCE)


def test_effect_depends_only_on_the_treatment_change(d, X):
    est = fitted_estimator(d)
    want = np.ravel(est.effect(X, T0=0.7, T1=1.7))        # same change as 0 -> 1
    got = reference.effects(fitted_model(d), X, [1.0])[:, 0]
    np.testing.assert_allclose(got, want, **TOLERANCE)


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_intervals_match_econml(d, X, alpha):
    est = fitted_estimator(d)
    lo, hi = reference.effect_intervals(fitted_model(d), X, DELTAS, alpha)
    for k, delta in enumerate(DELTAS):
        want_lo, want_hi = est.effect_interval(X, T0=0, T1=delta, alpha=alpha)
        np.testing.assert_allclose(lo[:, k], np.ravel(want_lo), **TOLERANCE)
        np.testing.assert_allclose(hi[:, k], np.ravel(want_hi), **TOLERANCE)
    assert np.all(lo <= hi)


def test_stderr_matches_econml_inference(d, X):
    want = np.ravel(fitted_estimator(d).const_marginal_effect_inference(X).stderr)
    np.testing.assert_allclose(reference.cate_stderr(fitted_model(d), X), want, **TOLERANCE)


def test_ate_matches_econml(d, X):
    est = fitted_estimator(d)
    got = reference.ate(fitted_model(d), X, DELTAS)
    for k, delta in enumerate(DELTAS):
        np.testing.assert_allclose(got[k], est.ate(X, T0=0, T1=delta), **TOLERANCE)


def test_pushdown_equals_plain_ate(d, X):
    model = fitted_model(d)
    np.testing.assert_allclose(
        reference.ate_pushdown(model, X, DELTAS), reference.ate(model, X, DELTAS), **TOLERANCE
    )
