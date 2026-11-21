"""Shared fixtures: fitted models and compiled modules are built once per session."""
import functools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmlc import data, emit, pipeline, reference  # noqa: E402

FEATURE_COUNTS = (1, 3, 10)          # d
DELTAS = (-1.5, 0.1, 1.0, 2.0)       # treatment changes, including a negative one
TOLERANCE = dict(rtol=1e-9, atol=1e-12)


@functools.lru_cache(maxsize=None)
def fitted_estimator(d: int):
    """EconML LinearDML fitted on 20,000 rows with d features."""
    return data.fit_econml(d_x=d, n_train=20_000)


@functools.lru_cache(maxsize=None)
def fitted_model(d: int) -> reference.FittedLinearDML:
    return reference.extract(fitted_estimator(d))


@functools.lru_cache(maxsize=None)
def compiled(kind: str, d: int, pipeline_name: str = "full", alpha: float = 0.05, n_models: int = 1) -> bytes:
    """Compile one query batch. Cached, because IREE compilation takes about a second."""
    if kind == "effects":
        text = emit.emit_effects(d, DELTAS)
    elif kind == "intervals":
        text = emit.emit_intervals(d, DELTAS, alpha)
    elif kind == "ates":
        text = emit.emit_ates(d, n_models, DELTAS)
    else:
        raise ValueError(kind)
    return pipeline.compile_module(text, pipeline_name)


@pytest.fixture(params=FEATURE_COUNTS, ids=lambda d: f"d={d}")
def d(request):
    return request.param
