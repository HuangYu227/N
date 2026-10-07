import json
import numpy as np
import pytest
import torch
from PIL import ImageFont

from tools.ttn_decode_comparison import comparison_frame, load_pair, pixels, save_return_views


def test_saved_pair_prefix_and_video_conversion(tmp_path):
    identity = dict(case_id="fixed", seed=3407, input_sha256={"camera": "same"},
                    initial_noise_sha256="same", base_sha256="same")
    episodes = [dict(identity, method=method) for method in ("sana", "ttn")]
    summary = {"protocol": {"history_source": "generated", "ttn_ablation": "full"}, "episodes": episodes}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for method in ("sana", "ttn"):
        torch.save(dict(identity, method=method, latents=torch.zeros(1, 128, 61, 2, 2)),
                   tmp_path / f"case-000-{method}.pt")
    _, _, latents = load_pair(tmp_path, 0, 5)
    assert [x.shape[2] for x in latents] == [16, 16]
    _, _, long_latents = load_pair(tmp_path, 0, 20)
    assert [x.shape[2] for x in long_latents] == [61, 61]
    assert ((long_latents[0].shape[2] - 1) * 8 + 1) / 16 == 30.0625
    with pytest.raises(ValueError, match="T >="):
        load_pair(tmp_path, 0, 21)
    episodes[1]["initial_noise_sha256"] = "different"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="initial_noise_sha256"):
        load_pair(tmp_path, 0, 5)
    decoded = torch.full((1, 3, 121, 2, 4), -1.)
    decoded[:, :, -1] = 1.
    video = pixels(decoded)
    assert video.shape == (121, 2, 4, 3)
    assert video[0].max() == 0 and video[-1].min() == 255
    combined = comparison_frame(video[0], video[-1], 97, 8, 100, ImageFont.load_default())
    assert combined.shape == (66, 8, 3)
    np.testing.assert_array_equal(combined[64:, :4], video[0])
    np.testing.assert_array_equal(combined[64:, 4:], video[-1])


def test_return_views_use_only_completed_orbit_and_observed_reference(tmp_path):
    protocol = {"step": 100, "case": {"raw_frames": 481,
                "camera": {"trajectory": "closed_orbit", "end_hold_raw_frames": 48}}}
    a = np.zeros((481, 2, 4, 3), dtype=np.uint8)
    b = a.copy()
    a[433:] = 255
    b[432] = 255  # Still-moving closing interval must not enter the return window.
    assert save_return_views(tmp_path, [a[:121], b[:121]], protocol, ImageFont.load_default()) is None
    assert not (tmp_path / "return-view.json").exists()
    metrics = save_return_views(tmp_path, [a, b], protocol, ImageFont.load_default())
    assert metrics["return_raw_frame_range"] == [433, 481]
    assert metrics["methods"]["sana"]["mean_return_to_observed_rgb_mse"] == 1
    assert metrics["methods"]["ttn"]["mean_return_to_observed_rgb_mse"] == 0
    assert (tmp_path / "return-comparison.png").is_file()
