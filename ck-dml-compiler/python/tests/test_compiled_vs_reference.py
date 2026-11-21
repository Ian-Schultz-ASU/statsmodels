"""FR-2, FR-3, FR-4: compiled output agrees with the NumPy reference."""
import numpy as np
import pytest

from dmlc import data, pipeline, reference

from conftest import DELTAS, TOLERANCE, compiled, fitted_model

THREADS = ("single", "multi")
ROW_COUNTS = (1, 257, 5000)


def columns(outputs):
    """Stack a list of (n,) results into an (n, K) array."""
    return np.stack(outputs, axis=1)


@pytest.mark.parametrize("threads", THREADS)
def test_effects(d, threads):
    model = fitted_model(d)
    run = pipeline.Runner(compiled("effects", d), threads)
    for n in ROW_COUNTS:                       # FR-3: one module, several row counts
        X = data.make_features(n, d_x=d)
        got = columns(run(X, model.beta))
        np.testing.assert_allclose(got, reference.effects(model, X, DELTAS), **TOLERANCE)


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_intervals(d, threads, alpha):
    model = fitted_model(d)
    run = pipeline.Runner(compiled("intervals", d, alpha=alpha), threads)
    for n in ROW_COUNTS:
        X = data.make_features(n, d_x=d)
        outputs = run(X, model.beta, model.cov)          # lo0, hi0, lo1, hi1, ...
        want_lo, want_hi = reference.effect_intervals(model, X, DELTAS, alpha)
        np.testing.assert_allclose(columns(outputs[0::2]), want_lo, **TOLERANCE)
        np.testing.assert_allclose(columns(outputs[1::2]), want_hi, **TOLERANCE)


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("pipeline_name", ["full", "nopush"])
@pytest.mark.parametrize("n_models", [1, 3])
def test_average_effects(d, threads, pipeline_name, n_models):
    models = data.perturbed_models(fitted_model(d), n_models)
    run = pipeline.Runner(compiled("ates", d, pipeline_name, n_models=n_models), threads)
    for n in ROW_COUNTS:
        X = data.make_features(n, d_x=d)
        outputs = run(X, *[m.beta for m in models])
        got = np.array([float(o) for o in outputs]).reshape(n_models, len(DELTAS))
        want = np.stack([reference.ate(m, X, DELTAS) for m in models])
        np.testing.assert_allclose(got, want, **TOLERANCE)


@pytest.mark.parametrize("kind", ["effects", "intervals"])
def test_cse_does_not_change_results(d, kind):
    """FR-4: the pipeline without --cse computes the same numbers."""
    model = fitted_model(d)
    args = (model.beta,) if kind == "effects" else (model.beta, model.cov)
    X = data.make_features(257, d_x=d)
    with_cse = pipeline.Runner(compiled(kind, d, "full"), "single")(X, *args)
    without = pipeline.Runner(compiled(kind, d, "nocse"), "single")(X, *args)
    for a, b in zip(with_cse, without):
        np.testing.assert_allclose(a, b, **TOLERANCE)


def test_results_own_their_memory(d):
    """Outputs must stay valid after the runner and IREE buffers are gone."""
    model = fitted_model(d)
    X = data.make_features(257, d_x=d)
    run = pipeline.Runner(compiled("effects", d), "single")
    outputs = run(X, model.beta)
    want = reference.effects(model, X, DELTAS)
    del run
    assert all(o.flags.owndata for o in outputs)
    np.testing.assert_allclose(columns(outputs), want, **TOLERANCE)
