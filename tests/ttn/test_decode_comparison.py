import json
import numpy as np
import pytest
import torch
from PIL import ImageFont

from tools.ttn_decode_comparison import comparison_frame, load_pair, pixels, save_return_views


def test_encoded_video_is_decodable_and_has_expected_frames(monkeypatch, tmp_path):
    import imageio.v2 as iio
    from tools.ttn_decode_comparison import verify_video
    path = tmp_path / "comparison.mp4"
    path.write_bytes(b"encoded")
    class Reader:
        frames = 481
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def count_frames(self): return self.frames
        def get_meta_data(self): return {"fps": 16, "size": (32, 24)}
        def get_data(self, index): return np.zeros((24, 32, 3), dtype=np.uint8)
    reader = Reader()
    monkeypatch.setattr(iio, "get_reader", lambda *a, **kw: reader)
    assert verify_video(path, 481, (24, 32), 16)["frames"] == 481
    reader.frames = 17
    with pytest.raises(ValueError, match="frame count"):
        verify_video(path, 481, (24, 32), 16)
    reader.frames = 481
    reader.get_data = lambda i: (_ for _ in ()).throw(OSError("truncated video"))
    with pytest.raises(OSError, match="truncated"):
        verify_video(path, 481, (24, 32), 16)


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


def test_explicit_ttn_baseline_pair_checks_identity_and_active_sink(tmp_path):
    identity = dict(case_id="fixed", seed=3407, input_sha256={"camera": "same"},
                    initial_noise_sha256="same", base_sha256="same")
    baseline, variant = tmp_path / "full", tmp_path / "sink"
    baseline.mkdir(); variant.mkdir()
    protocol = dict(history_source="generated", ttn_ablation="full", step=100,
                    checkpoint_sha256="weights", steps=20, cfg_scale=4.5, cached_blocks=2)
    for directory, gain in ((baseline, 0), (variant, .1)):
        row = dict(identity, method="ttn")
        summary = dict(protocol={**protocol, "tla_sink": {"mode": "protected", "gain": gain}}, episodes=[row])
        (directory / "summary.json").write_text(json.dumps(summary))
        torch.save(dict(row, latents=torch.zeros(1, 2, 61, 2, 2)), directory / "case-000-ttn.pt")
    _, _, latents = load_pair(variant, 0, 20, left_evaluation=baseline, left_method="ttn")
    assert len(latents) == 2
    with pytest.raises(ValueError, match="left-evaluation"):
        load_pair(variant, 0, 20, left_method="ttn")
    protocol["checkpoint_sha256"] = "wrong"
    (baseline / "summary.json").write_text(json.dumps(dict(protocol=protocol, episodes=[dict(identity, method="ttn")])))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_pair(variant, 0, 20, left_evaluation=baseline, left_method="ttn")
