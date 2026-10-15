"""Deterministic video-disjoint split and lazy, cached native conditioning."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random
import re
import torch
import numpy as np


def video_id(item):
    # SANA example keys: original-video_start_end_sNNNNNN. Overlapping clips
    # from that original must stay in the same partition.
    match = re.fullmatch(r'(.+)_\d+_\d+_s\d+', item['key'])
    if match is None:
        raise ValueError('Cannot establish original video ID: '+item['key'])
    return item['dataset_name']+'/'+match[1]


def split(items, seed=3407):
    groups = sorted({video_id(item) for item in items})
    if len(groups) < 2:
        raise ValueError('Distillation needs at least two original video groups')
    random.Random(seed).shuffle(groups)
    heldout = set(groups[:max(1, math.ceil(.1*len(groups)))])
    return {'seed': seed, 'label': 'distillation-heldout; teacher exposure unknown',
            'heldout_videos': sorted(heldout),
            'train': [i for i, row in enumerate(items) if video_id(row) not in heldout],
            'validation': [i for i, row in enumerate(items) if video_id(row) in heldout]}


class Data:
    def __init__(self, config, output, device, seed=3407):
        from diffusion.data.datasets.video.sana_wm_zip_latent_data import SanaWMZipLatentDataset
        self.config, self.device, self.seed = config, device, seed
        options = asdict(config.data)
        options.update(data_repeat=1, sort_dataset=True, shuffle_dataset=False, num_frames=961)
        self.dataset = SanaWMZipLatentDataset(**options)
        self.partition = split(self.dataset.dataset, seed)
        self.cache = Path(output)/'conditioning'
        self.cache.mkdir(exist_ok=True, parents=True)
        self.tokenizer = self.encoder = None
        self.identity = hashlib.sha256(json.dumps({'items': self.dataset.dataset, 'config':options},
            sort_keys=True).encode()).hexdigest()

    def index(self, counter, rank, world):
        indices = self.partition['train']
        epoch, offset = divmod(counter*world+rank, len(indices))
        ordered = list(indices)
        random.Random(self.seed+epoch).shuffle(ordered)
        return ordered[offset]

    def validation(self, count):
        # Fixed representatives from different held-out original videos.
        selected, seen = [], set()
        for index in self.partition['validation']:
            group = video_id(self.dataset.dataset[index])
            if group in seen: continue
            selected.append(index); seen.add(group)
            if len(selected) == count: break
        return selected

    @torch.no_grad()
    def batch(self, index, frames=121):
        from worldttn.cli import seed_everything, to_device
        from diffusion.model.builder import get_tokenizer_and_text_encoder
        from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
        # getdata, never __getitem__: no silent substitution of another clip.
        seed = self.seed+index
        py_state, cpu_state, np_state = random.getstate(), torch.get_rng_state(), np.random.get_state()
        cuda_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        try:
            seed_everything(seed)
            sample = self.dataset.getdata(index)
        finally:
            random.setstate(py_state); torch.set_rng_state(cpu_state); np.random.set_state(np_state)
            if cuda_state is not None: torch.cuda.set_rng_state(cuda_state,self.device)
        clean, prompt, _, info, _, _, camera = sample[:7]
        if clean.shape != (128, 121, 22, 40) or camera.shape != (121, 20):
            raise ValueError(f'Native 704x1280/121-latent sample required: {clean.shape}, {camera.shape}')
        if not torch.isfinite(clean).all() or not torch.isfinite(camera).all():
            raise ValueError('Nonfinite dataset input')
        key = hashlib.sha256((self.identity+prompt).encode()).hexdigest()
        file = self.cache/(key+'.pt')
        if file.exists():
            y, mask, null = torch.load(file, map_location='cpu', weights_only=True)
        else:
            if self.encoder is None:
                self.tokenizer, self.encoder = get_tokenizer_and_text_encoder(
                    self.config.text_encoder.text_encoder_name, 'cpu')
                self.encoder.eval().requires_grad_(False)
            y, mask = _encode_prompts([prompt], self.tokenizer, self.encoder, self.config, 'cpu')
            null, _ = _encode_prompts([''], self.tokenizer, self.encoder, self.config, 'cpu')
            from worldttn.checkpoint import atomic_save
            atomic_save((y.cpu(), mask.cpu(), null.cpu()), file)
        reference = clean[None, :, :frames].float()
        result = {'initial_latent': reference[:, :, :1].clone(),
                  'camera_conditions': camera[None, :frames].float(), 'y':y, 'mask':mask,
                  'uncondition':null, 'width':40, 'height':22,
                  'data_info': {k: v[None] if isinstance(v,torch.Tensor) else v for k,v in info.items()}}
        if self.dataset.return_chunk_plucker:
            result['chunk_plucker'] = sample[7][None, :, :frames]
        return to_device(result, self.device), reference
