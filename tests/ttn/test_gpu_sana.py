"""Opt-in real SANA integration smoke. No large model or CUDA import during CPU collection."""
import os
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("TTN_GPU_BASE"),
                    reason="requires CUDA and TTN_GPU_BASE teacher path")
def test_actual_sana_abc_update_and_cfg_rollout(tmp_path):
    from argparse import Namespace
    from worldttn.cli import smoke_command, ROOT
    smoke_command(
        Namespace(config=str(ROOT / "configs/worldttn/reference.json"),
                  stage=None,
                  sana_config=None,
                  base_weights=os.environ["TTN_GPU_BASE"],
                  device="cuda",
                  output=str(tmp_path),
                  frames=13,
                  latent_height=22,
                  latent_width=40,
                  stages=["A", "B", "C"],
                  seed=3407,
                  batch_file=None,
                  tbptt=2,
                  steps=4,
                  cfg_scale=4.5,
                  cached_blocks=-1))
