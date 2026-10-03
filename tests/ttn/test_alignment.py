"""Same-input isolation and GT-only probe lifecycle, using real TTN anchors."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_anchor import ProjectionContract
from worldttn.alignment import comparison_metrics, replay_anchors, probe_chunk
from worldttn.anchor import install_ttn
from worldttn.core import ANCHORS, TTNConfig
from worldttn.session import TTNSession
from worldttn.training import linear_flow_loss


class SoftmaxProjection(ProjectionContract):
    def forward(self, x, kv_cache=None, **kwargs):
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = self.q_norm(q).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        k = self.k_norm(k).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.dim).transpose(1, 2)
        z = ((q @ k.transpose(-1, -2) / self.dim**.5).softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(z * torch.nn.functional.silu(self.output_gate(x)))
        return (out, list(kv_cache)) if kv_cache is not None else out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
        for block in self.blocks: block.attn = SoftmaxProjection()

    def forward(self, x, t, y, kv_cache, ttn_chunk_context=None, **kwargs):
        b, c, f, h, w = x.shape
        z = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        cache = []
        for i, block in enumerate(self.blocks):
            out, new_cache = block.attn(z, kv_cache=kv_cache[i], HW=(f, h, w),
                                        camera_conditions=kwargs.get("camera_conditions"),
                                        prope_fns=(lambda v: v, lambda v: v, lambda v: v),
                                        ttn_chunk_context=ttn_chunk_context)
            z = z + .02 * out
            cache.append(new_cache)
        return z.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3), cache


def models_and_batch():
    torch.manual_seed(21)
    teacher = Model()
    student = copy.deepcopy(teacher)
    install_ttn(student, TTNConfig(heads=2, head_dim=8, generators=3))
    teacher.eval().requires_grad_(False)
    student.eval().requires_grad_(False)
    cam = torch.cat((torch.eye(4).flatten().expand(1, 4, 16),
                     torch.tensor([10., 10., 5., 5.]).expand(1, 4, 4)), -1)
    batch = {"clean_latents": torch.randn(1, 16, 4, 1, 1), "camera_conditions": cam,
             "y": torch.zeros(1, 1, 3, 2), "mask": torch.tensor([[[[1, 1, 0]]]])}
    return teacher, student, batch


def test_metrics_measure_scale_and_keep_zero_norm_ratios_undefined():
    a = torch.tensor([[[1., 2.], [99., 99.]]])
    b = 2 * a
    result = comparison_metrics(a, b, torch.tensor([[True, False]]))
    assert result["mse"] == 2.5 and result["relative_l2"] == 1.
    assert result["rms_ratio"] == 2. and result["cosine"] == pytest.approx(1.)
    assert comparison_metrics(torch.zeros(1), torch.ones(1))["cosine"] is None
    with pytest.raises(ValueError): comparison_metrics(a, b, torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError): comparison_metrics(a, b[:, :1])
    with pytest.raises(ValueError): comparison_metrics(a, b * float("nan"))


def test_replay_preserves_teacher_downstream_and_receives_identical_inputs():
    teacher, student, batch = models_and_batch()
    session = TTNSession(student, batch["camera_conditions"], 1, 1)
    session.reset(1)
    ctx = session.begin_chunk(0, 4)
    kwargs = dict(kv_cache=[[None] * 10 for _ in range(20)], ttn_chunk_context=ctx)
    expected, _ = teacher(batch["clean_latents"], torch.zeros(1), batch["y"], **kwargs)
    inputs, matched, rows, handles = {}, {}, [], []
    for index in ANCHORS:
        handles.append(teacher.blocks[index].attn.register_forward_pre_hook(
            lambda module, args, i=index: inputs.update({i: args[0]})))
        handles.append(student.blocks[index].attn.register_forward_pre_hook(
            lambda module, args, i=index: matched.update({i: args[0]})))
    with replay_anchors(teacher, student, ctx, rows):
        actual, _ = teacher(batch["clean_latents"], torch.zeros(1), batch["y"], **kwargs)
    for handle in handles: handle.remove()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert all(inputs[i] is matched[i] for i in ANCHORS)
    assert [row["block"] for row in rows] == list(ANCHORS)
    assert all(row["metric_tokens"] == 3 and row["elements"] == 48 for row in rows)
    assert all(not teacher.blocks[i].attn._forward_hooks for i in ANCHORS)


def test_gt_probe_uses_identical_noisy_inputs_and_keeps_only_initial_frame_committed():
    teacher, student, batch = models_and_batch()
    before = {k: v.clone() for k, v in student.state_dict().items()}
    noise, calls = torch.randn_like(batch["clean_latents"]), []
    def loss(session, clean, t, z, y, ctx, cache, start, end, mask, info, lm, on_prediction):
        assert ctx.write_mask.tolist() == [[False, True, True, True]]
        assert mask.tolist() == [[1, 1, 0]], "real text padding must be kept"
        assert lm.flatten().tolist() == [False, True, True, True]
        assert clean is batch["clean_latents"] and z is noise
        def capture(x, mapped_t, *args):
            calls.append((x.clone(), mapped_t.clone()))
            return session.forward(x, mapped_t, *args)
        return linear_flow_loss(SimpleNamespace(forward=capture), clean, t, z, y, ctx, cache,
                                start, end, mask, info, lm, on_prediction)
    result = probe_chunk(teacher, student, batch, loss, [0, 500, 999], noise)
    for a, b in zip(calls[::2], calls[1::2]):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        assert a[1][0, 0, 0] == 0
        torch.testing.assert_close(a[0][:, :, :1], batch["clean_latents"][:, :, :1])
    assert [p["timestep"] for p in result["probes"]] == [0, 500, 999]
    for stats in result["states"].values():
        assert stats["commits"] == 1 and stats["predictions"] == 2 and stats["committed_frames"] == [0]
    for k, v in student.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(p.grad is None for p in student.parameters())
    assert all(not teacher.blocks[i].attn._forward_hooks for i in ANCHORS)


def test_replay_failure_removes_hooks_and_cannot_commit_partial_candidates():
    teacher, student, batch = models_and_batch()
    session = TTNSession(student, batch["camera_conditions"], 1, 1)
    session.reset(1)
    ctx = session.begin_chunk(0, 1, prefill=True).for_clean()
    original = student.blocks[7].attn.forward
    student.blocks[7].attn.forward = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("probe failed"))
    with pytest.raises(RuntimeError, match="probe failed"), replay_anchors(teacher, student, ctx):
        teacher(batch["clean_latents"][:, :, :1], torch.zeros(1), batch["y"],
                kv_cache=[[None] * 10 for _ in range(20)], ttn_chunk_context=ctx)
    student.blocks[7].attn.forward = original
    assert session.runtime.commit_count == 0 and session.runtime.world_state.count_nonzero() == 0
    assert len(ctx.candidates) == 1
    assert all(not teacher.blocks[i].attn._forward_hooks for i in ANCHORS)


@pytest.mark.parametrize("args, message", [
    (["align-chunk"], "--training-run"),
    (["align-chunk", "--training-run", "run", "--frames", "4", "--steps", "20"], "without a sampler"),
    (["align-chunk", "--training-run", "run", "--frames", "4", "--alignment-timesteps", "1", "1"], "distinct"),
])
def test_alignment_cli_checks_inputs_before_cuda(args, message, monkeypatch, capsys):
    import sys
    from worldttn.cli import main
    monkeypatch.setattr(sys, "argv", ["worldttn", *args])
    with pytest.raises(SystemExit): main()
    assert message in capsys.readouterr().err


def test_alignment_command_loads_real_adapter_identity_and_writes_probe_report(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import dataclass
    from types import ModuleType
    from worldttn import alignment as al, evaluation as ev, cli, sana
    from worldttn.checkpoint import save_checkpoint

    @dataclass
    class Section:
        text_encoder_name: str = "fake-text"
        train_sampling_steps: int = 1000
    @dataclass
    class Data:
        num_frames: int = 97
        vae_ratio: tuple = (8, 32)
    config = SimpleNamespace(model=Section(), scheduler=Section(), text_encoder=Section(), data=Data())
    teacher, trained, batch = models_and_batch()
    trained.base_load_report = {"sha256": "base-sha"}
    with torch.no_grad(): trained.blocks[3].attn.proj.bias.add_(.2)
    run = tmp_path / "run"
    run.mkdir()
    save_checkpoint(run / "last.pt", trained, None, 40)
    (run / "run_config.json").write_text(json.dumps({"arguments": {}, "training": {},
                                                    "base": {"sha256": "base-sha", "source": "weights"}}))
    (run / "train.jsonl").write_text(json.dumps({"stage": "A", "step": 40}))
    monkeypatch.setattr(ev, "evaluation_config", lambda *args: config)
    case = dict(case_id="scene/key", seed=3407, reference=batch["clean_latents"],
                camera=batch["camera_conditions"], plucker=None, prompt="scene", info={})
    monkeypatch.setattr(al, "load_evaluation_cases", lambda *args: ([case], [], {}))
    def inject(name, **symbols):
        module = ModuleType(name)
        for k, v in symbols.items(): setattr(module, k, v)
        monkeypatch.setitem(sys.modules, name, module)
    inject("diffusion.model.builder", get_tokenizer_and_text_encoder=lambda *args: (None, nn.Linear(1, 1)))
    inject("train_video_scripts.train_sana_wm_stage1", _encode_prompts=lambda *args: (batch["y"], batch["mask"]))
    builds = []
    def build(*args, install_adapter=True):
        builds.append(install_adapter)
        model = copy.deepcopy(teacher)
        model.base_load_report = {"sha256": "base-sha"}
        if install_adapter: install_ttn(model, trained.ttn_system.config)
        return model
    monkeypatch.setattr(sana, "build_sana", build)
    monkeypatch.setattr(sana, "configure_cross_attention", lambda model, backend: {"backend": backend})
    monkeypatch.setattr(al, "SANAFlowLoss", lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, "timed_cuda", lambda f: (f(), {"seconds": 1., "peak_allocated_bytes": 10}))
    args = SimpleNamespace(output=str(tmp_path / "alignment"), training_run=str(run), adapter=None, stage=None,
                           config=str(cli.ROOT / "configs/worldttn/reference.json"), sana_config=None,
                           dataset_root=None, data_dir=None, vae_cache_dir=None, frames=4,
                           seed=3407, device="cpu", base_weights=None, cross_attn_backend="math", launch={},
                           alignment_timesteps=[0, 500])
    al.alignment_command(args)
    report = json.loads((tmp_path / "alignment/summary.json").read_text())
    assert builds == [False, True]
    assert report["protocol"]["step"] == 40 and report["protocol"]["cfg"] is False
    assert report["protocol"]["training_changed"] is False
    assert len(report["episodes"][0]["probes"]) == 2
    assert report["episodes"][0]["case_id"] == "scene/key"
    assert (tmp_path / "alignment/episodes.jsonl").is_file()
    assert report["episodes"][0]["probes"][0]["anchors"][0]["rms_ratio"] > 0
    with pytest.raises(ValueError, match="new alignment output"):
        al.alignment_command(args)
