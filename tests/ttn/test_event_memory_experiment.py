"""CPU checks for the pinned inference protocol and its reporting."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from tools import ttn_event_memory_experiment as experiment


def test_source_receipt_tracks_new_module_but_ignores_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    package = tmp_path / "worldttn"
    package.mkdir()
    module = package / "event_memory.py"
    module.write_text("value = 1\n")
    first = experiment.source_receipt()
    (tmp_path / "output.json").write_text("{}")
    assert experiment.source_receipt()["tree_sha256"] == first["tree_sha256"]
    module.write_text("value = 2\n")
    assert experiment.source_receipt()["tree_sha256"] != first["tree_sha256"]


def test_group_selectors_share_archive_and_gain_parameters():
    manifest = {"event_memory": {"anchors": [11, 15, 19], "capacity": 8, "topk": 2,
                "temperature": .1, "gain": .1, "max_relative_delta": .05,
                "retention": "coverage", "start_chunk": 5, "max_read_sigma": .5}}
    baseline = experiment.options_for("baseline", manifest)
    assert baseline.mode == "off" and not baseline.active
    expected = asdict(experiment.options_for("content", manifest))
    for group in experiment.GROUPS[1:]:
        actual = asdict(experiment.options_for(group, manifest))
        assert actual.pop("selector") == group
        assert actual == {key: value for key, value in expected.items() if key != "selector"}


def test_decoded_gt_metrics_exclude_observed_frame_and_detect_brightness_shift():
    import numpy as np
    gt = np.full((33, 16, 16, 3), 100, dtype=np.uint8)
    same = gt.copy()
    same[0] = 0  # observed frame must not contribute
    identical = experiment.decoded_metrics(same, gt)
    changed = experiment.decoded_metrics(np.clip(gt.astype(int) + 30, 0, 255).astype(np.uint8), gt)
    assert [row["raw_frame"] for row in identical["frames"]] == [16, 32]
    assert identical["mean_psnr_db"] == 120
    assert identical["mean_ssim"] == pytest.approx(1., abs=1e-6)
    assert changed["mean_psnr_db"] < identical["mean_psnr_db"]
    assert changed["mean_ssim"] < identical["mean_ssim"]


def comparison_fixture(tmp_path, prefix_changed=False, *, start_chunk=5, effect_frame=15):
    original = tmp_path / "original/case"
    original.mkdir(parents=True)
    baseline = torch.zeros(1, 1, 16, 1, 1)
    torch.save({"latents": baseline}, original / "ttn-latents.pt")
    manifest = {"source": {"tree_sha256": "tree"}, "reference_root": str(original.parent),
                "case": {"name": "case"}, "scope": "single case", "frames": 16,
                "event_memory": {"start_chunk": start_chunk}}
    (tmp_path / "experiment.json").write_text(json.dumps(manifest))
    result = {"status": "completed", "source_tree_sha256": "tree", "input_sha256": {"y": "text"},
              "initial_noise_sha256": "noise", "checkpoint_sha256": "checkpoint",
              "case_id": "case", "seed": 7, "timing": {},
              "metrics": {"mean_future_latent_mse": .5, "final_chunk_latent_mse": .7,
                          "error_slope_per_latent_frame": .1}}
    for group in ("baseline", "content"):
        folder = tmp_path / group
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(result))
        latents = baseline.clone()
        if group == "content" and effect_frame is not None:
            latents[:, :, 0 if prefix_changed else effect_frame] = 1
        torch.save({"latents": latents}, folder / "latents.pt")
    return SimpleNamespace(output=tmp_path, no_decode=True)


def test_comparison_accepts_late_effect_and_checks_historical_baseline(tmp_path):
    experiment.compare(comparison_fixture(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["historical_baseline_check"]["exact"]
    assert summary["comparison"]["content"]["prefix_check"]["exact"]


def test_comparison_rejects_memory_effect_before_start_chunk(tmp_path):
    with pytest.raises(AssertionError, match="first13 latent frames"):
        experiment.compare(comparison_fixture(tmp_path, prefix_changed=True))


@pytest.mark.parametrize("start_chunk,effect_frame,expected_prefix", [(1, 1, 1), (2, 4, 4), (100, None, 16)])
def test_comparison_prefix_follows_memory_start_and_rollout_length(
        tmp_path, start_chunk, effect_frame, expected_prefix):
    experiment.compare(comparison_fixture(tmp_path, start_chunk=start_chunk, effect_frame=effect_frame))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["comparison"]["content"]["prefix_check"]["latent_frames"] == expected_prefix


@pytest.mark.parametrize("comparison_fails", [False, True])
def test_launcher_publishes_completed_only_after_successful_comparison(tmp_path, monkeypatch, comparison_fails):
    (tmp_path / "experiment.json").write_text("{}")
    monkeypatch.setattr(experiment, "check_sources", lambda manifest: None)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "50000\n50000\n")
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *args, **kwargs:
                        SimpleNamespace(pid=100, wait=lambda timeout: 0))

    def compare(args):
        state = json.loads((args.output / "launcher.json").read_text(encoding="utf-8"))
        assert state["status"] == "workers-completed"
        if comparison_fails:
            raise ValueError("comparison failed")

    monkeypatch.setattr(experiment, "compare", compare)
    args = SimpleNamespace(output=tmp_path, groups=["baseline", "content"], minimum_free_gib=40,
                           allocator_gib=32, no_decode=True, decoded_gt_metrics=False)
    if comparison_fails:
        with pytest.raises(ValueError, match="comparison failed"):
            experiment.launch(args)
    else:
        experiment.launch(args)
    state = json.loads((tmp_path / "launcher.json").read_text(encoding="utf-8"))
    assert state["status"] == ("failed" if comparison_fails else "completed")
    assert state["phase"] == "comparison"
    assert state["exit_codes"] == {"baseline": 0, "content": 0}
