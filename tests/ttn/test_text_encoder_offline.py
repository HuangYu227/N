"""Gemma offline loads share one cached snapshot without optional model imports."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def gemma_loader(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "diffusion/model/builder.py"
    function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer_and_text_encoder")
    tokenizer = SimpleNamespace(padding_side="left")
    decoder = Mock()
    decoder.to.return_value = decoder
    model = Mock()
    model.get_decoder.return_value = decoder
    tokenizers = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    models = SimpleNamespace(from_pretrained=Mock(return_value=model))
    hub = ModuleType("huggingface_hub")
    hub.constants = SimpleNamespace(HF_HUB_OFFLINE=True)
    hub.snapshot_download = Mock(return_value="/cached/gemma/snapshots/commit")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    namespace = {"torch": torch, "AutoTokenizer": tokenizers, "AutoModelForCausalLM": models}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name], hub, tokenizers, models, tokenizer, decoder


@pytest.mark.parametrize("offline", [False, True])
def test_gemma_tokenizer_and_model_share_repo_or_offline_snapshot(gemma_loader, offline):
    load, hub, tokenizers, models, tokenizer, decoder = gemma_loader
    hub.constants.HF_HUB_OFFLINE = offline
    repo = "Efficient-Large-Model/gemma-2-2b-it"
    expected = hub.snapshot_download.return_value if offline else repo
    assert load("gemma-2-2b-it", device="cpu") == (tokenizer, decoder)
    if offline:
        hub.snapshot_download.assert_called_once_with(repo, local_files_only=True)
    else:
        hub.snapshot_download.assert_not_called()
    tokenizers.from_pretrained.assert_called_once_with(expected)
    models.from_pretrained.assert_called_once_with(expected, torch_dtype=torch.bfloat16)
    decoder.to.assert_called_once_with("cpu")
    assert tokenizer.padding_side == "right"


def test_gemma_missing_offline_snapshot_propagates_before_loading(gemma_loader):
    load, hub, tokenizers, models, _, _ = gemma_loader
    failure = FileNotFoundError("requested Gemma snapshot is absent from local cache")
    hub.snapshot_download.side_effect = failure
    with pytest.raises(FileNotFoundError) as raised:
        load("gemma-2-2b-it", device="cpu")
    assert raised.value is failure
    hub.snapshot_download.assert_called_once_with("Efficient-Large-Model/gemma-2-2b-it", local_files_only=True)
    tokenizers.from_pretrained.assert_not_called()
    models.from_pretrained.assert_not_called()
