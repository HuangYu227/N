"""Paired latent diagnostics: real horizons, causal inputs and honest revisit selection."""
import json
import sys
from types import SimpleNamespace, ModuleType

import pytest
import torch

from worldttn.evaluation import latent_metrics, find_revisits, paired_summary


def camera(xs):
    pose = torch.eye(4).repeat(len(xs), 1, 1)
    pose[:, 0, 3] = torch.tensor(xs)
    return torch.cat((pose.flatten(1), torch.tensor([10., 10., 5., 5.]).repeat(len(xs), 1)), -1)


def test_latent_metrics_exclude_observed_frame_and_report_pair_truth():
    gt = torch.zeros(1, 2, 7, 1, 1)
    pred = gt.clone()
    pred[:, :, 0] = 99  # never part of a generation-quality aggregate
    pred[:, :, 1:] = torch.arange(1, 7).reshape(1, 1, 6, 1, 1)
    result = latent_metrics(pred, gt, [{"frame_a": 1, "frame_b": 6}])
    assert result["mean_future_latent_mse"] == pytest.approx(91 / 6)
    assert result["final_chunk_latent_mse"] == pytest.approx((16 + 25 + 36) / 3)
    assert result["chunks"][0]["metric_start"] == 1
    assert result["revisits"][0]["return_gt_latent_mse"] == 36
    assert result["revisits"][0]["generated_pair_latent_mse"] == 25
    assert result["revisits"][0]["gt_pair_latent_mse"] == 0
    assert result["revisit_return_gt_latent_mse"] == 36
    with pytest.raises(ValueError): latent_metrics(pred, gt[:, :, :-1], [])
    with pytest.raises(ValueError): latent_metrics(pred, gt, [{"frame_a": 1, "frame_b": 7}])
    pred[:, :, 2] = float("nan")
    with pytest.raises(ValueError): latent_metrics(pred, gt, [])


def test_no_returns_are_missing_not_zero_and_summary_requires_complete_pairs():
    metrics = latent_metrics(torch.zeros(1, 2, 7, 1, 1), torch.ones(1, 2, 7, 1, 1), [])
    assert metrics["revisit_return_gt_latent_mse"] is None
    rows = [{"case_id": "clip", "seed": 1, "method": method, "metrics": metrics}
            for method in ("sana", "ttn")]
    summary = paired_summary(rows)
    assert summary["common_case_seed_count"] == 1
    assert summary["ttn_minus_sana"]["final_chunk_latent_mse"] == 0
    assert summary["ttn_minus_sana"]["revisit_return_gt_latent_mse"] is None
    with pytest.raises(ValueError): paired_summary(rows[:1])
    with pytest.raises(ValueError): paired_summary(rows + rows[:1])


def test_revisit_requires_leave_and_return_time_pose_and_intrinsics():
    kwargs = dict(min_gap=4, distance_fraction=.02, angle_deg=5., max_pairs=5)
    assert find_revisits(camera([0.] * 10), **kwargs) == []
    assert find_revisits(camera(list(range(10))), **kwargs) == []
    cam = camera([0., 0., 1., 3., 5., 3., 1., 0., 0., 0.])
    pairs = find_revisits(cam, **kwargs)
    assert pairs and all(0 < p["frame_a"] < p["frame_b"] for p in pairs)
    assert all(p["frame_b"] - p["frame_a"] >= 4 for p in pairs)
    assert all(p["excursion_fraction"] >= .1 for p in pairs)
    cam[6:, 16] *= 2
    assert find_revisits(cam, **kwargs) == []


def test_paired_rollout_never_receives_gt_future_preserves_noise_and_cfg_extras(monkeypatch):
    from worldttn import cli
    observed = torch.ones(1, 2, 1, 1, 1)
    seed_noise = torch.randn(1, 2, 7, 1, 1)
    original_noise = seed_noise.clone()
    captures = []

    class Sampler:
        def __init__(self, model, y, uncondition, **kw):
            assert "clean_latents" not in kw["model_kwargs"]
            assert kw["model_kwargs"]["camera_conditions"].shape[0] == 2
            assert kw["model_kwargs"]["chunk_plucker"].shape[0] == 2
            assert kw["model_kwargs"]["mask"].shape == (1, 3)
            assert kw["model_kwargs"]["mask"].tolist() == [[1, 1, 0]], "never drop text padding"
            assert kw["model_kwargs"]["data_info"]["condition_frame_info"] == {0: 0.}
            captures.append(kw)
            model.forward_long = lambda: None

        def sample_chunks(self, noise, steps):
            torch.testing.assert_close(noise[:, :, 1:], original_noise[:, :, 1:])
            torch.testing.assert_close(noise[:, :, :1], observed)
            noise[:, :, 1:] += 1
            for index, (a, b) in enumerate([(0, 4), (4, 7)]):
                yield index, noise[:, :, a:b], a, b

    mod = ModuleType("diffusion.scheduler.self_forcing_flow_euler_sampler")
    mod.SelfForcingFlowEulerCamCtrl = Sampler
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    model = SimpleNamespace(forward_long=lambda: "original")
    original_forward = model.forward_long
    batch = dict(initial_latent=observed, y=torch.zeros(1, 1, 3, 2),
                 mask=torch.tensor([[[[1, 1, 0]]]]), camera_conditions=camera(list(range(7)))[None],
                 chunk_plucker=torch.zeros(1, 48, 7, 1, 1), width=1, height=1,
                 data_info={})
    config = SimpleNamespace(scheduler=SimpleNamespace(inference_flow_shift=9.8))
    for _ in range(2):
        out, runtime, chunks = cli.rollout(model, config, batch, 20, 4.5, 2, initial_noise=seed_noise)
        assert runtime is None and len(chunks) == 2
        assert model.forward_long is original_forward
        torch.testing.assert_close(out[:, :, 1:], original_noise[:, :, 1:] + 1)
    torch.testing.assert_close(seed_noise, original_noise)
    assert len(captures) == 2


def test_evaluate_cli_requires_explicit_quality_steps_and_training_run():
    import subprocess
    for args in (["--steps", "20"], ["--training-run", "not-used"]):
        result = subprocess.run([sys.executable, "-m", "worldttn.cli", "evaluate", *args],
                                capture_output=True, text=True)
        assert result.returncode == 2
        assert "--training-run and explicit --steps" in result.stderr


def test_selector_rejects_camera_fallback_and_short_horizons():
    from worldttn.evaluation import select_cases
    class Dataset:
        vae_time_stride = 8
        return_chunk_plucker = False
        dataset = [dict(raw_zip="archive", key=k, camera_npz="sidecar", dataset_name="scene", cache_zip="cache")
                   for k in ("no-camera", "short-camera", "short-latent", "valid", "valid")]
        def load_camera_sidecar(self, path):
            import numpy as np
            return {"ids": np.array(["short-camera", "short-latent", "valid"]),
                    "ranges": [[0, 8], [8, 49], [57, 49]],
                    "pose": torch.zeros(106, 4, 4), "intrinsics": torch.zeros(106, 4)}
        def getdata(self, index):
            assert index >= 2, "missing/short cameras must be rejected before fallback"
            f = 4 if index == 2 else 7
            return (torch.zeros(2, f, 1, 1), "scene", None, {}, index, "scene", camera(list(range(f))))
    cases, rejected = select_cases(Dataset(), 7, 1, 3407, {})
    assert cases[0]["case_id"] == "scene/valid" and cases[0]["reference"].shape[2] == 7
    assert len(rejected) == 3 and cases[0]["revisit_pairs"] == []
    with pytest.raises(ValueError, match="found 1"):
        select_cases(Dataset(), 7, 2, 3407, {})


@pytest.mark.parametrize("camera_mode", [None, "sana"])
@pytest.mark.parametrize("compare_reference", [False, True])
def test_evaluate_runs_a_complete_identical_pair_and_never_passes_future_gt(tmp_path, monkeypatch, camera_mode, compare_reference):
    from argparse import Namespace
    from dataclasses import dataclass
    from worldttn import evaluation as ev, cli, sana, checkpoint
    from worldttn.core import TTNConfig, BASE_REVISION
    @dataclass
    class Data:
        type: str = "SanaWMZipLatentDataset"
        load_text_feat: bool = False
        image_size: int = 720
        num_frames: int = 25
        vae_ratio: tuple = (8, 32)
    @dataclass
    class Section:
        inference_flow_shift: float = 9.8
        text_encoder_name: str = "fake-text"
    config = SimpleNamespace(data=Data(), model=Section(), scheduler=Section(), text_encoder=Section(), task="df")
    def restore(run, source):
        assert run["training"]["data"]["num_frames"] == 25
        return config
    monkeypatch.setattr(ev, "evaluation_config", restore)
    def inject(name, **symbols):
        module = ModuleType(name)
        for k, v in symbols.items(): setattr(module, k, v)
        monkeypatch.setitem(sys.modules, name, module)
    class Dataset:
        def __init__(self, **kw):
            assert kw["num_frames"] == 49 and kw["data_repeat"] == 1
            assert kw["sort_dataset"] and not kw["shuffle_dataset"]
    inject("diffusion.data.datasets.video.sana_wm_zip_latent_data", SanaWMZipLatentDataset=Dataset)
    inject("diffusion.model.builder", get_tokenizer_and_text_encoder=lambda name, device: (None, torch.nn.Linear(1, 1)))
    inject("train_video_scripts.train_sana_wm_stage1", _encode_prompts=lambda prompts, *args:
           (torch.full((1, 1, 3, 2), 2. if prompts == [""] else 0.), torch.ones(1, 1, 1, 3)))
    gt = torch.arange(7.).reshape(1, 1, 7, 1, 1)
    case = dict(case_id="scene/key", seed=3407, reference=gt, camera=camera(list(range(7)))[None],
                plucker=None, prompt="scene", info={}, raw_camera_frames=49, revisit_pairs=[])
    monkeypatch.setattr(ev, "select_cases", lambda *args: ([case], []))
    run = tmp_path / "train"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"arguments": {}, "training": {"data": {"num_frames": 25}},
                                                     "base": {"sha256": "base-sha", "source": "weights"}}))
    stage = "C" if compare_reference else "A"
    (run / "train.jsonl").write_text(json.dumps({"step": 40, "stage": stage}))
    torch.save({"format": "TTN-SANA-WM-v0.1", "base_revision": BASE_REVISION,
                "base_sha256": "base-sha", "stage": stage, "step": 40, "config": TTNConfig(stage=stage).to_dict()}, run / "last.pt")
    builds, captures = [], []
    class Model(torch.nn.Module):
        def __init__(self, adapted):
            super().__init__()
            self.adapted = adapted
            self.base_load_report = {"sha256": "base-sha"}
            if adapted:
                self.ttn_system = torch.nn.Module()
                self.ttn_system.config = TTNConfig(stage=stage)
                self.blocks = []
    def build(*args, install_adapter):
        builds.append(install_adapter)
        return Model(install_adapter)
    monkeypatch.setattr(sana, "build_sana", build)
    monkeypatch.setattr(sana, "configure_cross_attention", lambda model, backend: {"backend": backend})
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda path, model: None)
    camera_overrides = []
    def configure_camera(model, mode):
        assert model.adapted, "the SANA baseline must retain its native camera"
        camera_overrides.append(mode)
    from worldttn import anchor
    monkeypatch.setattr(anchor, "configure_camera_attention", configure_camera)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cli, "timed_cuda", lambda call: (call(), {"seconds": 1., "peak_allocated_bytes": 10}))
    def rollout(model, config, batch, steps, cfg, cache, *, initial_noise, on_chunk, **kwargs):
        assert "clean_latents" not in batch and batch["initial_latent"].shape[2] == 1
        assert torch.equal(batch["initial_latent"], gt[:, :, :1])
        assert (batch["uncondition"] == 2).all(), "CFG needs encoded empty text, not smoke's zero placeholder"
        captures.append(initial_noise.clone())
        initial_noise += 5  # cannot contaminate the other method's noise
        chunks = [{"chunk": 0, "start": 0, "end": 4}, {"chunk": 1, "start": 4, "end": 7}]
        state = SimpleNamespace(commit_count=3, predict_count=3, world_state=torch.ones(1, 5, 2, 8, 8),
                                transition_fast=torch.zeros(1, 5, 2, 3)) if model.adapted else None
        for index, chunk in enumerate(chunks):
            on_chunk(chunk)
            if model.adapted and "on_state" in kwargs:
                state.commit_count = state.predict_count = index + 2
                kwargs["on_state"](index, state)
        return gt + 1, state, chunks
    monkeypatch.setattr(cli, "rollout", rollout)
    args = Namespace(output=str(tmp_path / "eval"), training_run=str(run), adapter=None, stage=None,
                     config=str(cli.ROOT / "configs/worldttn/reference.json"), sana_config=None,
                     dataset_root=None, data_dir=None, vae_cache_dir=None, frames=7, eval_cases=1,
                     revisit_min_gap=3, revisit_distance_fraction=.02, revisit_angle_deg=5., revisit_max_pairs=5,
                     seed=3407, device="cpu", base_weights=None, cross_attn_backend="math", launch={},
                     steps=20, cfg_scale=4.5, cached_blocks=2, camera_attention=camera_mode,
                     camera_ablation=camera_mode is not None, ttn_compare_reference=compare_reference,
                     ttn_core_backend="reuse" if compare_reference else "reference", ttn_psi_backend="projected" if compare_reference else "reference")
    ev.evaluate_command(args)
    summary = json.loads((tmp_path / "eval/summary.json").read_text())
    assert builds == ([False, True, True] if compare_reference else [False, True])
    torch.testing.assert_close(captures[0], captures[1])
    if compare_reference:
        torch.testing.assert_close(captures[1], captures[2])
        assert summary["backend_comparison"][0]["generated_latents"]["exactly_equal"]
        assert summary["backend_comparison"][0]["first_nonzero_state_difference"] is None
    assert summary["common_case_seed_count"] == 1 and summary["ttn_minus_sana"]["final_chunk_latent_mse"] == 0
    assert summary["protocol"]["training_latent_frames"] == 4
    assert summary["protocol"]["stage"] == stage and summary["protocol"]["step"] == 40
    assert camera_overrides == ([] if camera_mode is None else [camera_mode] * (2 if compare_reference else 1))
    assert summary["protocol"]["checkpoint_camera_attention"] == "linear"
    assert summary["protocol"]["ttn_camera_attention"] == (camera_mode or "linear")
    assert summary["episodes"][0]["input_sha256"] == summary["episodes"][1]["input_sha256"]
    with pytest.raises(ValueError, match="new evaluation output directory"):
        ev.evaluate_command(args)


def test_native_sana_builder_keeps_original_anchors_for_baseline(tmp_path, monkeypatch):
    from test_anchor import ProjectionContract
    from worldttn.sana import build_sana
    from worldttn.core import TTNConfig, ANCHORS
    from worldttn.anchor import TTNAnchor
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(20)])
            for block in self.blocks: block.attn = ProjectionContract()
    weights = tmp_path / "teacher.pt"
    torch.save(Model().state_dict(), weights)
    def inject(name, **symbols):
        module = ModuleType(name)
        for k, v in symbols.items(): setattr(module, k, v)
        monkeypatch.setitem(sys.modules, name, module)
    inject("diffusion.model.builder", build_model=lambda *args, **kw: Model())
    inject("diffusion.utils.camctrl_config", model_video_camctrl_init_config=lambda *args, **kw: {})
    inject("tools.download", find_model=lambda path: torch.load(path, weights_only=True))
    inject("sana.tools", hf_download_or_fpath=lambda path: path)
    config = SimpleNamespace(model=SimpleNamespace(image_size=720, fp32_attention=True),
                             vae=SimpleNamespace(vae_stride=(8, 32, 32)))
    ttn = TTNConfig(heads=2, head_dim=8, generators=3)
    base = build_sana(config, ttn, str(weights), "cpu", install_adapter=False)
    adapted = build_sana(config, ttn, str(weights), "cpu")
    assert not hasattr(base, "ttn_system") and not any(p.requires_grad for p in base.parameters())
    assert base.base_load_report == adapted.base_load_report
    assert [i for i, b in enumerate(adapted.blocks) if isinstance(b.attn, TTNAnchor)] == list(ANCHORS)
    for i in range(20):
        if i in ANCHORS: continue
        for key, value in base.blocks[i].state_dict().items():
            torch.testing.assert_close(value, adapted.blocks[i].state_dict()[key], atol=0, rtol=0)
