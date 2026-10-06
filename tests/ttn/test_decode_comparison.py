import json
import numpy as np
import pytest
import torch
from PIL import ImageFont

from tools.ttn_decode_comparison import comparison_frame, load_pair, pixels


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
