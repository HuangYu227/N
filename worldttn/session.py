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


def record_noise(context, timestep, noise_sigma=None, sampled_timestep=None, *, total_calls=None):
    """Detached noisy-call metadata shared by training and the native sampler."""
    config = context.system.config
    if config.memory_update == "proximal" and not context.clean_mode and not context.prefill_mode:
        call = context.memory_call_count
        context.memory_call_count += 1
        context.collect_memory_stats = call in ({0, total_calls//2, total_calls-1} if total_calls is not None else {0})
        if context.collect_memory_stats:
            sigma = timestep.float()/1000 if noise_sigma is None else noise_sigma
            context.memory_trajectory.append({"call": call, "noise_timestep": timestep.detach().float().cpu().tolist(),
                "noise_sigma": sigma.detach().float().cpu().tolist(), "anchors": {}})
    if context.replay_active and not context.clean_mode and not context.prefill_mode:
        call = context.replay_call_count
        context.replay_call_count += 1
        context.collect_replay_stats = call in ({0, total_calls//2, total_calls-1} if total_calls is not None else {0})
        if context.collect_replay_stats:
            sigma = timestep.float()/1000 if noise_sigma is None else noise_sigma
            context.replay_trajectory.append({"call": call,
                "noise_timestep": timestep.detach().float().cpu().tolist(),
                "noise_sigma": sigma.detach().float().cpu().tolist(), "anchors": {}})
    if context.sink_reference is not None and not context.clean_mode and not context.prefill_mode:
        call = context.noise_call_count
        context.noise_call_count += 1
        context.collect_sink_stats = call in ({0, total_calls//2, total_calls-1} if total_calls is not None else {0})
        if context.collect_sink_stats:
            sigma = timestep.float()/1000 if noise_sigma is None else noise_sigma
            context.sink_trajectory.append({"call": call,
                "noise_timestep": timestep.detach().float().cpu().tolist(),
                "noise_sigma": sigma.detach().float().cpu().tolist(), "anchors": {}})
    if (not context.collect_local_stats or context.clean_mode or context.prefill_mode
            or not (config.local_update or config.persistent_meta)): return
    sigma = timestep.float()/1000 if noise_sigma is None else noise_sigma
    context.noise_info = {"noise_timestep": timestep.detach().float().cpu().tolist(),
                          "noise_sigma": sigma.detach().float().cpu().tolist()}
    if sampled_timestep is not None:
        context.noise_info["sampled_timestep"] = sampled_timestep.detach().cpu().tolist()
    context.local_trajectory.append({"call": len(context.local_trajectory), **context.noise_info, "anchors": {}})


def carry_cache(cache, camera_attention="linear", previous=None):
    """Carry native GDN/FFN state and, in SANA mode, detached camera K/V."""
    result = []
    for i, slot in enumerate(cache):
        if i not in ANCHORS:
            result.append(list(slot))
            continue
        carried = [None] * 9 + [slot[9]]
        if camera_attention == "sana":
            carried[6] = slot[6]
            for index in (2, 3):
                current = slot[index]
                old = previous[i][index] if previous is not None else None
                carried[index] = (torch.cat((old, current), dim=2) if old is not None and current is not None
                                  else current if current is not None else old)
        result.append(carried)
    return result


class TTNSession:

    def __init__(self, model, camera_conditions, width, height, valid_mask=None, extras=None,
                 *, ablation="full", diagnostics=False, collect_local_stats=False, sink_options=None, replay_options=None):
        if camera_conditions.ndim != 3 or camera_conditions.shape[-1] != 20:
            raise ValueError("camera_conditions must contain C2W(16) + intrinsics(4)")
        self.model = model
        self.camera = camera_conditions
        self.width = width
        self.height = height
        self.valid_mask = valid_mask
        self.extras = extras or {}
        self.runtime = None
        self.ablation, self.diagnostics = ablation, diagnostics
        self.collect_local_stats = collect_local_stats
        self.sink_options = sink_options
        self.replay_options = replay_options

    def reset(self, batch_size):
        self.camera_cache_lengths = {index: [] for index in ANCHORS}
        self.runtime = TTNRuntimeState.create(self.model.ttn_system.config, batch_size, self.camera.device,
                                             ablation=self.ablation, diagnostics=self.diagnostics,
                                             collect_local_stats=self.collect_local_stats, sink_options=self.sink_options,
                                             replay_options=self.replay_options)
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
                                        prefill=prefill, rope=getattr(self.model, "rope", None))

    def camera_for_model(self, start, end, width, height, batch):
        camera = repeat_batch(self.camera[:, start:end], batch).clone()
        camera[..., 16:] *= camera.new_tensor(
            [width / self.width, height / self.height, width / self.width, height / self.height])
        return camera

    def forward(self, x, t, y, context, cache, start, end, mask=None, data_info=None, save=False,
                *, noise_sigma=None, sampled_timestep=None):
        b = x.shape[0]
        record_noise(context, t, noise_sigma, sampled_timestep)
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

    def carry_clean_cache(self, current, previous, *, cached_chunks=-1):
        """Bound actual camera K/V tokens, including packed/patchified layouts."""
        carried = carry_cache(current, self.model.ttn_system.config.camera_attention, previous=previous)
        if cached_chunks < 0: return carried
        if cached_chunks < 1: raise ValueError("cached_chunks must be positive or -1")
        lengths = {i: list(v) for i, v in self.camera_cache_lengths.items()}
        for anchor in ANCHORS:
            key, value = current[anchor][2:4]
            if key is None and value is None: continue
            if key is None or value is None or key.shape != value.shape or key.shape[2] < 1:
                raise ValueError("native camera cache must supply matching nonempty current-chunk K/V")
            lengths[anchor] = (lengths[anchor]+[key.shape[2]])[-cached_chunks:]
            tokens = sum(lengths[anchor])
            for slot in (2, 3):
                # clone releases the evicted prefix allocation; a sliced view would retain it.
                carried[anchor][slot] = carried[anchor][slot][:, :, -tokens:].detach().clone()
        self.camera_cache_lengths = lengths
        return carried

    def clean_forward(self, clean, y, context, cache, start, end, mask=None, data_info=None, *, cached_chunks=-1):
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
        # Validate/crop before publishing the TTN transaction.
        previous_lengths = self.camera_cache_lengths
        carried = self.carry_clean_cache(new_cache, cache, cached_chunks=cached_chunks)
        try:
            self.runtime.commit_chunk(c)
        except Exception:
            self.camera_cache_lengths = previous_lengths
            raise
        # Detached factors are clean-only; outer predicted/state graphs remain live.
        context.psi_snapshot = c.psi_snapshot = None
        context.live_factors = c.live_factors = None
        context.memory_caches = c.memory_caches = ()
        c.memory_candidates.clear()
        return out, carried
