"""The emitter and pass pipeline, checked from Python (op counts per stage)."""
import pytest

from dmlc import emit, pipeline


def test_emitted_effects_are_one_query_per_scenario():
    text = emit.emit_effects(3, [0.5, 1.0, 2.0])
    assert pipeline.count_ops(text) == {"dml.effect": 3}
    assert "tensor<?x3xf64>" in text and "tensor<4xf64>" in text


def test_decompose_then_cse_shares_the_effect_size():
    text = emit.emit_effects(3, [0.5, 1.0, 2.0])
    decomposed = pipeline.opt(text, "--dml-decompose")
    assert pipeline.count_ops(decomposed) == {"dml.cate": 3, "dml.scale": 3}
    shared = pipeline.opt(decomposed, "--cse")
    assert pipeline.count_ops(shared) == {"dml.cate": 1, "dml.scale": 3}


def test_intervals_share_effect_size_and_stderr():
    text = emit.emit_intervals(3, [0.5, 1.0, 2.0])
    shared = pipeline.opt(text, "--dml-decompose", "--cse")
    assert pipeline.count_ops(shared) == {"dml.cate": 1, "dml.cate_stderr": 1, "dml.axpby": 6}


def test_pushdown_leaves_one_pass_over_the_data():
    text = emit.emit_ates(3, n_models=4, deltas=[1.0, 2.0])
    pushed = pipeline.opt(text, "--dml-decompose", "--cse")
    assert pipeline.count_ops(pushed) == {"dml.col_mean": 1, "dml.ate_from_mean": 8}
    not_pushed = pipeline.opt(text, "--dml-decompose-no-pushdown", "--cse")
    assert pipeline.count_ops(not_pushed) == {"dml.cate": 4, "dml.mean_scale": 8}


@pytest.mark.parametrize("name", sorted(pipeline.PIPELINES))
def test_every_pipeline_removes_all_dml_ops(name):
    for text in (emit.emit_effects(3, [1.0]), emit.emit_intervals(3, [1.0]), emit.emit_ates(3, 2, [1.0])):
        lowered = pipeline.lower(text, name)
        assert not any(op.startswith("dml.") for op in pipeline.count_ops(lowered))


def test_lowering_a_query_op_is_reported():
    with pytest.raises(RuntimeError, match="run --dml-decompose before"):
        pipeline.opt(emit.emit_effects(3, [1.0]), "--dml-lower-to-linalg")


def test_float_literals_round_trip():
    delta = 0.1 + 0.2                                   # not exactly representable
    text = pipeline.opt(emit.emit_effects(1, [delta]), "--dml-decompose")
    printed = text.split("a = ")[1].split(" :")[0]
    assert float(printed) == delta
