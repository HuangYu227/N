"""Exercise dataset construction without optional SANA image/registry packages."""
import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    function = next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('process_rank', [None, 0, 2])
def test_dataset_build_and_batch_without_rank_environment(monkeypatch, process_rank):
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def rank():
        assert process_rank is not None, 'single-process data loading must not query an uninitialized group'
        return process_rank

    get_rank = load_function(ROOT / 'diffusion/utils/dist_utils.py', 'get_rank', {
        'dist': SimpleNamespace(is_available=lambda: True,
                                is_initialized=lambda: process_rank is not None, get_rank=rank)})
    dataset = TensorDataset(torch.arange(3))
    dataset.ori_imgs_nums = len(dataset)

    def construct(cfg, registry, default_args):
        assert cfg == {'type': 'Tiny'} and registry == 'registry'
        assert default_args == {'transform': 'transform:default_train', 'resolution': 720}
        return dataset

    build = load_function(ROOT / 'diffusion/data/builder.py', 'build_dataset', {
        'os': os, 'time': time, 'get_rank': get_rank,
        'get_root_logger': lambda: SimpleNamespace(info=calls.append),
        'get_transform': lambda name, resolution: f'transform:{name}',
        'build_from_cfg': construct, 'DATASETS': 'registry', 'colored': lambda text, *a, **kw: text})
    result = build({'type': 'Tiny'}, resolution=720)
    assert result is dataset
    assert [int(batch[0].item()) for batch in DataLoader(result, batch_size=1)] == [0, 1, 2]
    assert len(calls) == (2 if process_rank in (None, 0) else 0)
