"""SANA-call lifecycle without importing SANA, usable by training and its sampler."""
import torch
from .runtime import TTNRuntimeState
from .core import ANCHORS


def repeat_batch(t, batch):
    if t.shape[0] == batch: return t
    if batch % t.shape[0]: raise ValueError("conditioning/CFG batch mismatch")
    return t.repeat(batch // t.shape[0], *([1] * (t.ndim - 1)))


def slice_data_info(data_info, start, end, batch):
    info = dict(data_info or {})
    for name in ("image_vae_embeds", ):
        if isinstance(info.get(name), torch.Tensor):
            info[name] = repeat_batch(info[name][:, :, start:end], batch)
    for name in ("image_embeds", ):
        if isinstance(info.get(name), torch.Tensor): info[name] = repeat_batch(info[name], batch)
    return info


def validate_clean_output(output, reference):
    if not isinstance(output, torch.Tensor) or output.shape != reference.shape:
        raise ValueError("clean model output shape mismatch")
    if not torch.isfinite(output).all():
        raise FloatingPointError("nonfinite clean model output; transaction not committed")


def carry_cache(cache):
    """Native GDN recurrent/FFN states pass through; anchors only carry FFN slot 9."""
    return [[None] * 9 + [slot[9]] if i in ANCHORS else list(slot) for i, slot in enumerate(cache)]


class TTNSession:

    def __init__(self, model, camera_conditions, width, height, valid_mask=None, extras=None):
        if camera_conditions.ndim != 3 or camera_conditions.shape[-1] != 20:
            raise ValueError("camera_conditions must contain C2W(16) + intrinsics(4)")
        self.model = model
        self.camera = camera_conditions
        self.width = width
        self.height = height
        self.valid_mask = valid_mask
        self.extras = extras or {}
        self.runtime = None

    def reset(self, batch_size):
        self.runtime = TTNRuntimeState.create(self.model.ttn_system.config, batch_size, self.camera.device)
        return self.runtime

    def begin_chunk(self, start, end, prefill=False):
        b = self.runtime.world_state.shape[0]
        camera = repeat_batch(self.camera[:, start:end], b)
        valid = torch.ones(b, end - start, dtype=torch.bool, device=camera.device)
        if self.valid_mask is not None: valid = repeat_batch(self.valid_mask[:, start:end], b).bool()
        ids = torch.arange(start, end, device=camera.device).expand(b, -1)
        return self.runtime.begin_chunk(self.model.ttn_system,
                                        camera[..., :16].reshape(b, end - start, 4, 4),
                                        camera[..., 16:],
                                        ids,
                                        valid,
                                        self.width,
                                        self.height,
                                        prefill=prefill)

    def camera_for_model(self, start, end, width, height, batch):
        camera = repeat_batch(self.camera[:, start:end], batch).clone()
        camera[..., 16:] *= camera.new_tensor(
            [width / self.width, height / self.height, width / self.width, height / self.height])
        return camera

    def forward(self, x, t, y, context, cache, start, end, mask=None, data_info=None, save=False):
        b = x.shape[0]
        kw = {}
        for name, tensor in self.extras.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor[:, :, start:end] if tensor.ndim == 5 else tensor[:, start:end]
                tensor = repeat_batch(tensor, b)
            kw[name] = tensor
        return self.model(x,
                          t,
                          y,
                          mask=mask,
                          start_f=start,
                          end_f=end,
                          kv_cache=[list(s) for s in cache],
                          save_kv_cache=save,
                          frame_index=torch.arange(start, end, device=x.device),
                          camera_conditions=self.camera_for_model(start, end, x.shape[-1], x.shape[-2], b),
                          frame_valid_mask=context.read_mask,
                          data_info=slice_data_info(data_info, start, end, b),
                          ttn_chunk_context=context,
                          cam_branch_drop_prob=0.,
                          **kw)

    def prefill(self, initial, y, mask=None, data_info=None):
        if initial.shape[2] != 1: raise ValueError("prefill accepts exactly the initial observed frame")
        if self.runtime is None: self.reset(initial.shape[0])
        c = self.begin_chunk(0, 1, prefill=True).for_clean()
        scratch = [[None] * 10 for _ in self.model.blocks]
        # Original GDN/FFN caches here are scratch-only, discarded immediately.
        out, _ = self.forward(initial,
                              torch.zeros(initial.shape[0], device=initial.device),
                              y,
                              c,
                              scratch,
                              0,
                              1,
                              mask,
                              data_info,
                              save=True)
        validate_clean_output(out, initial)
        self.runtime.prefill(c)
        return c

    def clean_forward(self, clean, y, context, cache, start, end, mask=None, data_info=None):
        c = context.for_clean()
        out, new_cache = self.forward(clean,
                                      torch.zeros(clean.shape[0], device=clean.device),
                                      y,
                                      c,
                                      cache,
                                      start,
                                      end,
                                      mask,
                                      data_info,
                                      save=True)
        validate_clean_output(out, clean)
        self.runtime.commit_chunk(c)
        return out, carry_cache(new_cache)
