"""Lazy construction of the cached SANA teacher and the TTN adapter."""
import hashlib
import torch
from .anchor import install_ttn


def load_sana_config(path):
    import pyrallis
    from diffusion.utils.camctrl_config import SanaVideoCamCtrlConfig
    with open(path, encoding="utf-8") as handle:
        config = pyrallis.load(SanaVideoCamCtrlConfig, handle)
    config.model.model = "SanaMSVideoCamCtrlStreaming_1600M_P1_D20"
    config.model.ffn_type = "CachedGLUMBConvTemp"
    config.model.pos_embed_type = "casual_wan_rope"
    config.model.class_dropout_prob = 0.
    config.model.softmax_every_n = 4
    config.train.grad_checkpointing = False
    config.train.cp_size = 1
    config.train.use_fsdp = False
    return config


def build_sana(config, ttn_config, base_weights=None, device="cuda", dtype=torch.bfloat16):
    from diffusion.model.builder import build_model
    from diffusion.utils.camctrl_config import model_video_camctrl_init_config
    from tools.download import find_model
    kwargs = model_video_camctrl_init_config(config, latent_size=config.model.image_size // config.vae.vae_stride[-1])
    kwargs["class_dropout_prob"] = 0.
    kwargs["camctrl_type"] = None  # keep the streaming subclass cached GDN class
    kwargs["use_autograd_kernel"] = True
    model = build_model("SanaMSVideoCamCtrlStreaming_1600M_P1_D20",
                        use_grad_checkpoint=False,
                        use_fp32_attention=config.model.fp32_attention,
                        **kwargs)
    from sana.tools import hf_download_or_fpath
    resolved = hf_download_or_fpath(base_weights or ttn_config.base_id)
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    state = find_model(resolved)
    state = state.get("state_dict", state)
    # Released teacher weights sometimes carry the nonlearned absolute-position buffer.
    state = {k: v for k, v in state.items() if k != "pos_embed"}
    current = model.state_dict()
    for name, value in list(state.items()):
        if name in current and value.ndim < current[name].ndim:
            state[name] = value.reshape(*value.shape, *([1] * (current[name].ndim - value.ndim)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"pos_embed", "y_embedder.y_embedding"}
    if set(missing) - allowed or unexpected:
        raise ValueError(f"base checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=dtype)
    install_ttn(model, ttn_config)
    for index in (3, 7, 11, 15, 19):
        model.blocks[index].attn.float()  # FP32 optimizer weights, BF16 CUDA autocast compute
    model.base_load_report = {
        "source": base_weights or ttn_config.base_id,
        "missing": missing,
        "unexpected": unexpected,
        "sha256": digest.hexdigest()
    }
    return model
