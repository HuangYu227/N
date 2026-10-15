from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldttn import cli, sana


@pytest.mark.parametrize('command', ['infer', 'train'])
def test_empty_text_sampling_validation_is_training_only(monkeypatch, command):
    config = SimpleNamespace(data=SimpleNamespace(load_text_feat=False, vae_ratio=(8, 32, 32)),
                             text_encoder=SimpleNamespace(model_max_length=300))
    monkeypatch.setattr(sana, 'load_sana_config', lambda _: config)
    monkeypatch.setattr(sana, 'build_sana', lambda *args, **kwargs: SimpleNamespace(blocks=[]))
    monkeypatch.setattr(sana, 'configure_activation_checkpointing', lambda *args: {})
    monkeypatch.setattr(sana, 'configure_cross_attention', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, 'configure_from_args', lambda *args: None)
    args = Namespace(config=str(Path(__file__).resolve().parents[2] / 'configs/worldttn/inplace_ttt_fp8.json'),
                     stage=None, sana_config=None, command=command, batch_file='one-case.pt',
                     base_weights=None, device='cpu', output='unused', tbptt=None)
    if command == 'train':
        with pytest.raises(ValueError, match='empty text sampling'):
            cli.build(args)
    else:
        model, _, settings = cli.build(args)
        assert model.ttn_training_protocol == 'inplace-ffn-clean-fullgrad-v1'
        assert settings['training_latent_frames'] == 121
