import copy
from types import SimpleNamespace
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint, read_checkpoint
from worldttn.performance import ExecutionOptions, configure_execution, execution_report
from worldttn.training import train_clip, linear_flow_loss


def model(local=True, persistent=True):
    torch.manual_seed(31)
    m = TinyWorldModel(local_update=local, persistent_meta=persistent)
    return configure_train_scope(m, "dit")


def update(m, optimizer, offload="none", *, audit=False):
    clean, noise, t, camera = inputs()
    return train_clip(m, clean, torch.zeros(1, 1, 2, 8), camera, optimizer,
                      linear_flow_loss, t, noise, width=100, height=100,
                      tbptt=2, activation_offload=offload, audit_update=audit)


@pytest.mark.parametrize("offload", ["none", "cpu"])
def test_meta_train_clip_updates_eta_and_records_detached_local_persistent_metrics(offload):
    m = model()
    optimizer = make_optimizer(m)
    eta = m.ttn_system.local_eta_logits
    initial = eta.detach().clone()
    result = update(m, optimizer, offload, audit=True)
    assert eta.grad is not None and torch.isfinite(eta.grad).all() and eta.grad.norm() > 0
    assert not torch.equal(initial, eta)
    origins = result["optimizer_updates"]["by_origin"]
    assert origins["ttn_new"]["delta_norm"] > 0 and origins["sana_inherited"]["delta_norm"] > 0
    assert result["exposure"]["predicted_chunks"] == 4
    assert result["exposure"]["predicted_latent_frames"] == 12
    assert result["runtime"].commit_count == 5
    assert not result["runtime"].world_state.requires_grad and not result["runtime"].transition_fast.requires_grad
    for chunk in result["chunks"]:
        for anchor in chunk["anchors"]:
            local, persistent = anchor["local"], anchor["persistent"]
            assert local["enabled"] and persistent["meta_gradient_enabled"]
            assert len(local["per_head"]["raw_grad_norm"][0]) == 2
            expected_sigma = [[[0., .5, .5, .5]]] if chunk["chunk"] == 0 else [[[.5]*3]]
            assert local["noise_sigma"] == expected_sigma
            assert "loss_after_correct" in local["support"] and "loss_after_transport" in local["query"]
    def no_graph(value):
        if isinstance(value, dict): return all(no_graph(v) for v in value.values())
        if isinstance(value, (list, tuple)): return all(no_graph(v) for v in value)
        return not isinstance(value, torch.Tensor)
    assert no_graph(result["chunks"])
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert any(p is eta for p in groups["ttn_new"]["params"])
    assert groups["ttn_new"]["lr"] == 1e-5 and groups["sana_inherited"]["lr"] == 1e-6


def test_meta_offload_preserves_parameters_gradients_states_and_updates():
    normal = model()
    offloaded = copy.deepcopy(normal)
    a = update(normal, make_optimizer(normal), "none")
    b = update(offloaded, make_optimizer(offloaded), "cpu")
    assert a["loss"] == b["loss"]
    for (name, p), (other, q) in zip(normal.named_parameters(), offloaded.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    for key in ("world_state", "transition_fast"):
        torch.testing.assert_close(getattr(a["runtime"], key), getattr(b["runtime"], key), atol=0, rtol=0)


def test_meta_checkpoint_resume_matches_continuous_updates_and_rejects_other_math(tmp_path):
    m, resumed = model(), model()
    optimizer = make_optimizer(m)
    update(m, optimizer)
    path = tmp_path / "last.pt"
    save_checkpoint(path, m, optimizer, 1)
    ropt = make_optimizer(resumed)
    assert load_checkpoint(path, resumed, ropt, resume=True) == 1
    a, b = update(m, optimizer), update(resumed, ropt)
    assert a["loss"] == b["loss"]
    for name, tensor in m.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="architecture"):
        read_checkpoint(path, model(persistent=False), resume=True)


def test_old_checkpoint_without_meta_fields_still_loads_in_legacy_mode(tmp_path):
    old = model(local=False, persistent=False)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, old, make_optimizer(old), 3)
    payload = torch.load(path, weights_only=False)
    for flag in ("local_update", "persistent_meta", "persistent_update", "sink_gain", "sink_position",
                 "replay_strength", "replay_budget", "memory_start_chunk"):
        payload["config"].pop(flag)
    torch.save(payload, path)
    assert read_checkpoint(path, model(False, False), resume=True)["step"] == 3
    with pytest.raises(ValueError, match="architecture"):
        read_checkpoint(path, model(), resume=True)


def test_meta_rejects_detached_execution_before_forward_and_reports_live_implementation():
    m = model()
    assert execution_report(m)["psi_implementation"] == "live_projected_meta"
    for core, psi in (("reuse", "reference"), ("reuse", "projected"), ("compiled", "projected")):
        with pytest.raises(ValueError, match="Meta-TTT"):
            configure_execution(m, ExecutionOptions(core, psi))


def test_training_identity_records_meta_math_but_does_not_rewrite_legacy_identity():
    from worldttn.cli import _training_identity
    from test_parallel_cli import CPUFlowConfig
    args = SimpleNamespace(seed=3407, batch_file="fixed-batch", train_scope="dit", optimizer_policy="origin")
    cfg = SimpleNamespace(scheduler=CPUFlowConfig())
    old = _training_identity(args, cfg, {"ttn": {"stage": "C"}}, 2)
    meta = _training_identity(args, cfg, {"ttn": {"stage": "C", "local_update": True, "persistent_meta": True}}, 2)
    assert "meta_ttt" not in old and meta["meta_ttt"]["local_update"] and meta["meta_ttt"]["persistent_meta"]
    assert old != meta


@pytest.mark.skipif(not hasattr(torch.cpu, "Stream"), reason="CPU FSDP2 not available")
def test_meta_fsdp2_cpu_offload_resume_preserves_adam_rng_and_cursor(tmp_path):
    from test_parallel import _resume_worker, _init_uri
    torch.multiprocessing.spawn(_resume_worker, args=(2, _init_uri(), str(tmp_path), "fsdp2", True, "cpu", True), nprocs=2)


@pytest.mark.parametrize("proximal", [False, True])
def test_production_flow_records_exact_scheduler_sigma_separately_from_mapped_timestep(proximal):
    from worldttn.training import SANAFlowLoss
    flow = object.__new__(SANAFlowLoss)
    captures = {}
    sigma = torch.full((1000,), .73137)
    def training_losses(fn, clean, t, **kw):
        fn(clean, torch.full_like(t, 731))
        return {"loss": torch.ones(1)}
    flow.scheduler = SimpleNamespace(sigmas=sigma.numpy(), training_losses=training_losses)
    class Session:
        model = SimpleNamespace(ttn_system=SimpleNamespace(config=SimpleNamespace(
            local_update=not proximal, persistent_meta=not proximal, memory_update="proximal" if proximal else "delta")))
        def forward(self, *args, **kw):
            captures.update(kw)
            return args[0], None
    clean = torch.ones(1, 16, 2, 1, 1)
    t = torch.full((1, 1, 2), 500)
    ctx = SimpleNamespace(system=Session.model.ttn_system, collect_local_stats=True)
    flow(Session(), clean, t, clean, None, ctx, None, 0, 2, None, None, clean)
    torch.testing.assert_close(captures["noise_sigma"], torch.full_like(t, .73137, dtype=torch.float32), rtol=0, atol=0)
    assert torch.equal(captures["sampled_timestep"], t)


@pytest.mark.parametrize("meta", [False, True])
def test_production_flow_alignment_accepts_teacher_without_ttn_system(meta):
    from test_alignment import models_and_batch
    from worldttn.alignment import probe_chunk
    from worldttn.anchor import install_ttn
    from worldttn.core import TTNConfig
    from worldttn.training import SANAFlowLoss
    teacher, student, batch = models_and_batch()
    assert not hasattr(teacher, "ttn_system")
    if meta:
        student = install_ttn(copy.deepcopy(teacher), TTNConfig(
            heads=2, head_dim=8, generators=3, stage="C", local_update=True, persistent_meta=True))
        student.eval().requires_grad_(False)
    flow = object.__new__(SANAFlowLoss)
    def training_losses(fn, clean, t, noise, loss_mask, **kw):
        sigma = t.float()/1000
        output = fn(clean*(1-sigma[..., None, None]) + noise*sigma[..., None, None], t)
        return {"loss": ((output-(noise-clean)).square()*loss_mask).flatten(1).mean(-1)}
    flow.scheduler = SimpleNamespace(sigmas=torch.arange(1000).float().numpy()/1000,
                                     training_losses=training_losses)
    result = probe_chunk(teacher, student, batch, flow, [500], torch.randn_like(batch["clean_latents"]))
    assert len(result["probes"]) == 1
    assert torch.isfinite(torch.tensor(result["probes"][0]["native_flow"]["sana_loss"]))
