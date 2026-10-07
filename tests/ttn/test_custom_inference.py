import ast
from dataclasses import make_dataclass
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tools import ttn_custom_inference as custom
from tools.ttn_decode_comparison import load_pair


def native_cpu_helpers(monkeypatch):
    """Execute the original CPU helpers without importing optional DiT/CUDA deps."""
    namespace = dict(torch=torch, np=np, Image=Image, TARGET_HEIGHT=704, TARGET_WIDTH=1280)
    for filename, names in (
        ("diffusion/utils/cam_utils.py", {"get_pose_inverse", "compute_raymap"}),
        ("inference_video_scripts/wm/inference_sana_wm.py", {"resize_and_center_crop", "_pack_camera_conditions", "prepare_camera"}),
    ):
        tree = ast.parse((custom.REPO / filename).read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    module = ModuleType("inference_video_scripts.wm.inference_sana_wm")
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_case_native_geometry_and_input_validation(monkeypatch, tmp_path):
    native = native_cpu_helpers(monkeypatch)
    case, image_path, prompt = custom.load_case(custom.REPO / "assets/worldttn/study_static/case.json")
    assert "stationary" in prompt and case["raw_frames"] == 481
    with Image.open(image_path) as image:
        cropped, _, _, _ = native.resize_and_center_crop(image)
        assert cropped.size == (1280, 704)
    batch = custom.make_geometry(case, [8, 32, 32])
    camera = batch["camera_conditions"]
    assert camera.shape == (1, 61, 20) and batch["chunk_plucker"].shape == (1, 48, 61, 22, 40)
    torch.testing.assert_close(camera[0, :, :16], torch.eye(4).flatten().expand(61, -1))
    torch.testing.assert_close(camera[0, :, 16:], torch.tensor([900., 900., 640., 352.]).expand(61, -1))
    # Match native raymaps after rollout's single pixel -> latent scaling.
    latent_intr = camera[0, :, 16:] / 32
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 481, axis=0)
    intr = np.repeat(np.array([[900., 900., 640., 352.]], dtype=np.float32), 481, axis=0)
    native_packed = native.prepare_camera(poses, intr, target_size=(704, 1280), vae_stride=[8, 32, 32])
    torch.testing.assert_close(latent_intr, native_packed["raymap"][:, 16:], rtol=0, atol=0)
    with Image.open(image_path) as image:
        image.save(tmp_path / "first_frame.png")
    (tmp_path / "prompt.txt").write_text(prompt)
    case["image_sha256"] = "wrong"
    (tmp_path / "case.json").write_text(json.dumps(case))
    with pytest.raises(ValueError, match="SHA256"):
        custom.load_case(tmp_path / "case.json")


def test_closed_orbit_translation_orientation_and_native_return(monkeypatch):
    native_cpu_helpers(monkeypatch)
    case, _, prompt = custom.load_case(custom.REPO / "assets/worldttn/study_static/case_orbit.json")
    assert "360-degree" in prompt and "stationary" not in prompt
    poses = custom.camera_poses(case)
    assert poses.shape == (481, 4, 4) and np.isfinite(poses).all()
    np.testing.assert_array_equal(poses[:25], np.broadcast_to(np.eye(4), (25, 4, 4)))
    np.testing.assert_array_equal(poses[432:], np.broadcast_to(np.eye(4), (49, 4, 4)))
    rotation, centres = poses[:, :3, :3], poses[:, :3, 3]
    np.testing.assert_allclose(rotation.transpose(0, 2, 1) @ rotation,
                               np.broadcast_to(np.eye(3), rotation.shape), atol=2e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=2e-7)
    target = np.array([0, 0, case["camera"]["radius"]])
    # Translation and rotation must both occur; all optical axes look at the centre.
    np.testing.assert_allclose(centres + case["camera"]["radius"] * rotation[:, :, 2],
                               np.broadcast_to(target, centres.shape), atol=2e-7)
    assert centres[:, 0].ptp() > 1 and centres[:, 2].max() == pytest.approx(1.2)
    np.testing.assert_allclose(poses[228, :3, :3], np.diag([-1, 1, -1]), atol=1e-7)
    batch = custom.make_geometry(case, [8, 32, 32])
    generated = torch.zeros(1, 2, 61, 2, 2)
    generated[:, :, 55:] = 2
    metrics = custom.return_latent_metrics(generated, batch, case)
    assert metrics["return_latent_ids"] == list(range(55, 61))
    assert metrics["mean_return_to_observed_latent_mse"] == 4
    batch["chunk_plucker"][:, :, -1] += .01
    with pytest.raises(ValueError, match="camera/rays"):
        custom.return_latent_metrics(generated, batch, case)
    for key, invalid in (("radius", 0), ("radius", float("nan")),
                         ("start_hold_raw_frames", 7), ("end_hold_raw_frames", 480)):
        invalid_case = {**case, "camera": {**case["camera"], key: invalid}}
        with pytest.raises(ValueError, match="closed orbit"):
            custom.camera_poses(invalid_case)


@pytest.mark.parametrize("case_name", ["case.json", "case_orbit.json"])
def test_custom_pair_reuses_inputs_and_noise_without_future_gt(monkeypatch, tmp_path, case_name):
    native_cpu_helpers(monkeypatch)
    from worldttn import cli, sana, checkpoint, performance
    from worldttn.core import TTNConfig
    import tools.ttn_decode_comparison as decoder
    def inject(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    part = make_dataclass("Part", [("value", int, 0)])()
    vae = make_dataclass("VAE", [("vae_stride", list), ("vae_latent_dim", int)])([8, 32, 32], 128)
    config = SimpleNamespace(vae=vae, model=part, text_encoder=SimpleNamespace(text_encoder_name="test"),
                             scheduler=SimpleNamespace(inference_flow_shift=9.8))
    config.text_encoder = make_dataclass("Text", [("text_encoder_name", str)])("test")
    config.scheduler = make_dataclass("Scheduler", [("inference_flow_shift", float)])(9.8)
    adapter = tmp_path / "checkpoint.pt"
    adapter.write_bytes(b"immutable")
    run = {"base": {"source": "original", "sha256": "base"}, "arguments": {"sana_config": None}}
    monkeypatch.setattr(custom, "load_evaluation_run", lambda args: (
        run, config, TTNConfig(stage="C", camera_attention="sana", local_update=True, persistent_meta=True),
        adapter, custom.file_sha256(adapter), {"step": 100, "weight_scope": "dit"}))
    monkeypatch.setattr(custom, "encode_first_frame", lambda *args: torch.zeros(1, 128, 1, 22, 40))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "cuda": args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(custom, "diagnostic_noise", lambda shape, device, seed: torch.randn(
        shape, generator=torch.Generator().manual_seed(seed)))
    inject("diffusion.model.builder", get_tokenizer_and_text_encoder=lambda *a: (object(), torch.nn.Identity()))
    inject("train_video_scripts.train_sana_wm_stage1", _encode_prompts=lambda *a: (
        torch.zeros(1, 1, 300, 8), torch.ones(1, 1, 1, 300, dtype=torch.long)))
    built, hashes, decoded = [], [], []
    def build(*args, **kwargs):
        model = torch.nn.Identity()
        model.adapted = kwargs["install_adapter"]
        model.base_load_report = {"sha256": "base"}
        built.append(kwargs)
        return model
    monkeypatch.setattr(sana, "build_sana", build)
    monkeypatch.setattr(sana, "configure_cross_attention", lambda *a: None)
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a: None)
    monkeypatch.setattr(performance, "configure_execution", lambda *a: None)
    monkeypatch.setattr(cli, "timed_cuda", lambda call: (call(), {"seconds": 0}))
    def rollout(model, config, batch, *args, initial_noise, on_chunk):
        assert "clean_latents" not in batch and batch["initial_latent"].shape[2] == 1
        hashes.append(custom.tensor_sha256(initial_noise))
        result = initial_noise.clone()
        result[:, :, :1] = batch["initial_latent"]
        chunks = [{"chunk": 0, "start": 0, "end": 4, "seconds": 0}]
        on_chunk(chunks[0])
        return result, SimpleNamespace(commit_count=21) if model.adapted else None, chunks
    monkeypatch.setattr(cli, "rollout", rollout)
    monkeypatch.setattr(decoder, "main", lambda argv: decoded.append(argv))
    output = tmp_path / "custom"
    custom.main(["--training-run", str(tmp_path), "--output", str(output),
                 "--case", str(custom.REPO / "assets/worldttn/study_static" / case_name)])
    assert hashes[0] == hashes[1]
    assert [row["install_adapter"] for row in built] == [False, True]
    assert built[1]["dtype"] == torch.float32
    summary, _, latents = load_pair(output, 0, 20)
    assert summary["metrics"] is None and summary["protocol"]["no_training_dataset_read"]
    assert summary["protocol"]["step"] == 100 and len(decoded) == 1
    assert [tensor.shape[2] for tensor in latents] == [61, 61]
    assert np.load(output / "camera_poses.npy").shape == (481, 4, 4)
    assert all((row["return_view"] is None) == (case_name == "case.json") for row in summary["episodes"])
