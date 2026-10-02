"""Keep cached latents and raw-camera frame crops on the same time interval."""
import ast
from functools import lru_cache
from glob import glob
import io
import json
import os.path as osp
from pathlib import Path
import random
import warnings
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


@pytest.mark.parametrize("raw_frames,cached_frames,expected", [(1, 25, 1), (9, 25, 2),
    (97, 25, 13), (97, 13, 13), (None, 25, 25)])
def test_raw_frame_limit_crops_latents_and_camera_consistently(tmp_path, raw_frames, cached_frames, expected):
    source = Path(__file__).resolve().parents[2] / "diffusion/data/datasets/video/sana_wm_zip_latent_data.py"
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if not isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in nodes:
        if isinstance(node, ast.ClassDef): node.decorator_list = []  # Skip only the CUDA registry import.
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    with ZipFile(raw / "sample.zip", "w") as archive:
        archive.writestr("clip.json", json.dumps({"prompt": "scene", "width": 64, "height": 64}))
    z = np.broadcast_to(np.arange(cached_frames, dtype=np.float32)[None, :, None, None],
                        (2, cached_frames, 2, 2)).copy()
    payload = io.BytesIO()
    np.savez(payload, z=z)
    with ZipFile(cache / "sample.zip", "w") as archive:
        archive.writestr("clip.npz", payload.getvalue())
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (193, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(193)
    np.savez(raw / "sample_camera.npz", ids=np.array(["clip"]), ranges=np.array([[0, 193]]),
             pose=poses, intrinsics=np.broadcast_to(np.array([64., 64., 32., 32.]), (193, 4)))
    dataset = namespace["SanaWMZipLatentDataset"](str(raw), str(cache), num_frames=raw_frames,
                                                 vae_ratio=(8, 32), min_latent_file_size=0)
    latent, prompt, mask, info, index, _, camera = dataset.getdata(0)
    assert latent.shape == (2, expected, 2, 2)
    assert camera.shape == (expected, 20)
    torch.testing.assert_close(latent[0, :, 0, 0], torch.arange(expected, dtype=torch.float32))
    # Every latent frame gets the corresponding raw pose, without repeated tail padding.
    torch.testing.assert_close(camera[:, 3], torch.arange(expected, dtype=torch.float32) * 8)
    assert prompt == "scene" and info["key"] == "clip"
