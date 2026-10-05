from types import SimpleNamespace
import pytest
import torch
from test_meta_training import model, update
from worldttn.checkpoint import make_optimizer
from worldttn.parallel_checkpoint import save_training_checkpoint


@pytest.mark.parametrize("change", [None, "model", "rng", "data", "optimizer"])
def test_compare_complete_bundles_checks_tensor_contents_rng_cursor_and_adam(tmp_path, change):
    from tools.ttn_compare_resume import compare, diagnose
    m = model()
    optimizer = make_optimizer(m)
    update(m, optimizer)
    engine = SimpleNamespace(model=m, rank=0, world=1, mode="single", reshard=lambda: None)
    a, b = tmp_path / "a/last.pt", tmp_path / "b/last.pt"
    save_training_checkpoint(a, engine, optimizer, 1, {"cursor": 1}, {"tbptt": 2})
    if change == "model":
        with torch.no_grad(): m.ttn_system.generators.u.add_(.01)
    if change == "rng": torch.rand(1)
    if change == "optimizer": next(iter(optimizer.state.values()))["exp_avg"].add_(.001)
    save_training_checkpoint(b, engine, optimizer, 1, {"cursor": 2 if change == "data" else 1}, {"tbptt": 2})
    if change:
        with pytest.raises(AssertionError): compare(a, b)
    else:
        assert compare(a, b)["status"] == "exact_match"
    report = diagnose(a, b)
    assert report["status"] == "diagnostic_only"
    assert all(value["equal"] for value in report["metadata"].values())
    rank = report["ranks"][0]
    assert report["model"]["equal"] == (change != "model")
    assert rank["data"]["equal"] == (change != "data")
    assert rank["optimizer"]["equal"] == (change != "optimizer")
    assert all(value["equal"] for value in rank["rng"].values()) == (change != "rng")
    assert rank["optimizer_param_groups"]["equal"] and rank["optimizer_parameter_names"]["equal"]
    if change == "model":
        assert report["model_scale_example"]["max_abs_difference"] > 0
