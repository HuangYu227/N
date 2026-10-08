"""Pure SANA recomputation with live recurrent/meta graphs and one-time cache writes."""
import ast
import copy
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from test_alignment_detail import cached_sana
from test_sdpa_policy import text_attention
from tools.ttn_compare_resume import identical
from worldttn.sana import configure_activation_checkpointing


@pytest.fixture
def cached_ffn():
    # Execute the released CPU-capable definitions without optional CUDA imports.
    root = Path(__file__).resolve().parents[2]
    namespace = dict(torch=torch, nn=nn, checkpoint=checkpoint,
                     build_act=runpy.run_path(str(root/"diffusion/model/act.py"))["build_act"],
                     build_norm=runpy.run_path(str(root/"diffusion/model/norms.py"))["build_norm"],
                     _INT32_SAFE_CONV_ELEMENTS=1 << 30)
    for filename, names in [("diffusion/model/utils.py", {"val2list", "val2tuple", "get_same_padding", "checkpoint_preserving_strides"}),
                            ("diffusion/model/nets/basic_modules.py",
                             {"ConvLayer", "GLUMBConv", "GLUMBConvTemp", "CachedGLUMBConvTemp"})]:
        source = root/filename
        nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        for node in nodes: node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["CachedGLUMBConvTemp"]


def pure_model(cached_sana, cached_ffn, text_attention):
    from test_full_training import model
    from test_training import TinyBlock
    from worldttn.anchor import TTNAnchor
    class Block(TinyBlock):
        def forward(self, z, cache, context, hw, save):
            if isinstance(self.attn, TTNAnchor):
                result, cache = self.attn(z, HW=hw, ttn_chunk_context=context, kv_cache=cache)
                z = z + .02*result
            # Same text on both paths; padding is part of the real attention call.
            cond = z.new_ones(z.shape[0], 1, 3, z.shape[-1])
            mask = torch.tensor([[1, 1, 0]], device=z.device).expand(z.shape[0], -1)
            z = z + .01*self.cross_attn(z, cond, mask)
            delta, cache = self.mlp(z, HW=hw, kv_cache=list(cache), save_kv_cache=save)
            return z + .01*delta, cache
    m = model(cached_sana)
    for block in m.blocks:
        block.__class__ = Block
        del block.ffn
        block.mlp = cached_ffn(16, 48)
        nn.init.normal_(block.mlp.t_conv.weight, std=.01)  # Exercise temporal cache contribution.
        block.cross_attn = text_attention(16, 2)
        block.cross_attn.set_sdpa_backend("math")
    return m


@pytest.mark.parametrize("chunked", [False, True])
def test_spatial_checkpoint_preserves_cache_and_higher_derivatives(cached_ffn, text_attention, chunked):
    torch.manual_seed(3407)
    a = cached_ffn(8, 24).double()
    b = copy.deepcopy(a)
    b.ttn_activation_checkpointing = True
    if chunked:
        a._apply_spatial_autochunked.__globals__["_INT32_SAFE_CONV_ELEMENTS"] = 1
    x = torch.randn(1, 12, 8, dtype=torch.float64, requires_grad=True)
    results, calls = [], []
    for module in (a, b):
        count = []
        module.t_conv.register_forward_pre_hook(lambda *args, count=count: count.append(1))
        cache = [None]*10
        out, cache = module(x, HW=(3, 2, 2), kv_cache=cache, save_kv_cache=True)
        saved = cache[-1].clone()
        grad = torch.autograd.grad(out.square().sum(), x, create_graph=True)[0]
        higher = torch.autograd.grad(grad.square().sum(), (x, *module.parameters()))
        assert torch.equal(cache[-1], saved) and not cache[-1].requires_grad
        results.append((out, grad, higher, saved))
        calls.append(len(count))
    identical(results[0], results[1], "spatial output/cache/meta derivatives")
    assert calls == [1, 1], "temporal convolution/cache must not be replayed"


@pytest.mark.parametrize("amp", [False, True])
def test_text_checkpoint_preserves_padding_rng_and_outer_gradients(text_attention, amp):
    torch.manual_seed(3407)
    a = text_attention(16, 2, proj_drop=.2)
    b = copy.deepcopy(a)
    b.ttn_activation_checkpointing = True
    for module in (a, b): module.set_sdpa_backend("math")
    x = torch.randn(1, 7, 16, requires_grad=True)
    cond = torch.randn(1, 1, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    probe = torch.randn_like(x)
    rng = torch.get_rng_state()
    results = []
    for module in (a, b):
        torch.set_rng_state(rng)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            out = module(x, cond, mask)
            loss = (out*probe).sum()
        grads = torch.autograd.grad(loss, (x, cond, *module.parameters()))
        results.append((out, grads, torch.get_rng_state()))
    identical(results[0], results[1], "text output/gradients/RNG")
    assert torch.equal(mask, torch.tensor([[1, 1, 1, 0, 0]]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spatial_recompute_preserves_input_stride_after_cpu_packing(cached_ffn, device):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("requires CUDA activation offload")
    from worldttn.training import activation_storage
    module = cached_ffn(8, 24).to(device=device, dtype=torch.float32 if device == "cuda" else torch.float64)
    module.ttn_activation_checkpointing = True
    strides = []
    module.inverted_conv.register_forward_pre_hook(lambda layer, args: strides.append(args[0].stride()))
    x = torch.randn(1, 12, 8, device=device, dtype=next(module.parameters()).dtype, requires_grad=True)
    # save_on_cpu(pin_memory=True) reconstructs packed tensors by shape. Model
    # that layout change on CPU without requiring a CUDA/pinned allocator.
    storage = (activation_storage("cpu", device=device) if device == "cuda" else
               torch.autograd.graph.saved_tensors_hooks(lambda value: value.detach().contiguous().clone(), lambda value: value))
    with storage, torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False):
        output = module(x, HW=(3, 2, 2))
    output.square().sum().backward()
    assert len(strides) == 2 and strides[0] == strides[1]
    assert torch.isfinite(x.grad).all() and all(torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_text_cpu_packing_preserves_strided_dispatch_and_gradients(text_attention, device, amp, broadcast, qk_norm):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("requires CUDA activation offload")
    from contextlib import nullcontext
    from worldttn.training import activation_storage
    torch.manual_seed(3407)
    dtype = torch.float32 if amp else torch.float64
    a = text_attention(16, 2, proj_drop=.2, qk_norm=qk_norm).to(device=device, dtype=dtype)
    b = copy.deepcopy(a)
    b.ttn_activation_checkpointing = True
    for module in (a, b): module.set_sdpa_backend("math")
    x = torch.randn(2, 16, 7, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
    cond = torch.randn(1 if broadcast else 2, 1, 16, 5, device=device, dtype=dtype).transpose(2, 3).requires_grad_()
    mask = torch.tensor([[1, 1, 1, 0, 0]], device=device)
    if broadcast:
        cond = cond.expand(2, 1, 5, 16)
        mask = mask.expand(2, 5)
    else:
        mask = mask.repeat(2, 1)
    expected_mask = mask.clone()
    probe = torch.randn_like(x)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device == "cuda" else None
    results, strides = [], []
    for module in (a, b):
        torch.set_rng_state(rng)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, device)
        seen = []
        module.q_linear.register_forward_pre_hook(lambda layer, args, seen=seen: seen.append(args[0].stride()))
        storage = (activation_storage("cpu", device=device) if device == "cuda" else
                   torch.autograd.graph.saved_tensors_hooks(lambda value: value.detach().contiguous().clone(), lambda value: value))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            # Reference keeps its graph normally. Candidate offloads the
            # checkpoint's inputs through native CUDA storage or a CPU oracle.
            with storage if module is b else nullcontext():
                out = module(x, cond, mask)
            loss = (out*probe).square().sum()
        parameters = (x, cond, *module.parameters())
        grads = torch.autograd.grad(loss, parameters, create_graph=not amp)
        higher = torch.autograd.grad(sum(g.square().sum() for g in grads), parameters) if not amp else ()
        results.append((out, grads, higher, torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if cuda_rng is not None else None))
        strides.append(seen)
    identical(results[0], results[1], "strided text output/gradients/meta/RNG")
    assert all(stride == x.stride() for stride in strides[1]) and len(strides[1]) >= 2
    assert torch.equal(mask, expected_mask)


def test_combined_checkpoint_future_credit_updates_and_saved_payload(monkeypatch, cached_sana, cached_ffn, text_attention):
    from test_full_training import EulerOracle, update
    from worldttn import training
    from worldttn.checkpoint import make_optimizer, rng_state
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    a, b = [pure_model(cached_sana, cached_ffn, text_attention) for _ in range(2)]
    b.load_state_dict(a.state_dict())
    for block in b.blocks:
        block.mlp.ttn_activation_checkpointing = block.cross_attn.ttn_activation_checkpointing = True
    results = []
    for network in (a, b):
        sizes, grads = [], []
        def capture(module, args, kwargs, out):
            context = kwargs["ttn_chunk_context"]
            if context.clean_mode and not context.prefill_mode:
                s, g = context.candidates[0][:2]
                s.retain_grad(); g.retain_grad(); grads.append((s, g))
        handle = network.register_forward_hook(capture, with_kwargs=True)
        def pack(value):
            sizes.append(value.numel()*value.element_size())
            return value.detach()
        opt = make_optimizer(network)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = update(network, opt)
        handle.remove()
        assert all(value.grad is not None and value.grad.norm() > 0 for value in grads[4])
        assert all(value.grad is None for value in grads[3]+grads[7])
        assert result["runtime"].commit_count == result["runtime"].predict_count == 9
        results.append((result, opt, sum(sizes), rng_state()))
    left, right = results
    assert right[2] < left[2], "checkpoint must reduce outer saved payload"
    assert left[0]["loss"] == right[0]["loss"]
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    identical(left[1].state_dict(), right[1].state_dict(), "checkpoint optimizer")
    identical(left[3], right[3], "checkpoint RNG")
    for field in ("world_state", "transition_fast", "replay_values"):
        identical(getattr(left[0]["runtime"], field), getattr(right[0]["runtime"], field), field)
    assert left[0]["runtime"].sink_reference_sha256 == right[0]["runtime"].sink_reference_sha256
    assert left[0]["runtime"].replay_reference_sha256 == right[0]["runtime"].replay_reference_sha256


def test_policy_scope_legacy_defaults_and_resume_identity(monkeypatch, cached_sana, cached_ffn, text_attention):
    from worldttn.cli import _training_identity
    from worldttn.parallel_checkpoint import validate_training_checkpoint
    from test_parallel_cli import CPUFlowConfig
    m = pure_model(cached_sana, cached_ffn, text_attention)
    for name, cls in [("diffusion.model.nets.basic_modules", cached_ffn),
                      ("diffusion.model.nets.sana_blocks", text_attention)]:
        source = ModuleType(name)
        setattr(source, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, source)
    before = copy.deepcopy(m.state_dict())
    report = configure_activation_checkpointing(m, "ffn-cross-attn")
    assert len(report["modules"]) == 40 and not report["stateful_recomputation"]
    assert not any(hasattr(block.attn, "ttn_activation_checkpointing") for block in m.blocks)
    identical(before, m.state_dict(), "checkpoint policy is not model weights")
    args = SimpleNamespace(seed=3407, batch_file="batch.pt", train_scope="dit")
    cfg = SimpleNamespace(scheduler=CPUFlowConfig())
    old = _training_identity(args, cfg, {}, 4)
    new = _training_identity(args, cfg, {"activation_checkpointing": "ffn-cross-attn"}, 4)
    payload = {"distributed": {"format": "TTN-parallel-resume-v1", "mode": "fsdp2", "world_size": 4,
                              "resume_dir": "shards", "training_config": old}}
    with pytest.raises(ValueError, match="training configuration"):
        validate_training_checkpoint(payload, "fsdp2", 4, new)
    assert validate_training_checkpoint(payload, "fsdp2", 4, new, benchmark=True)["benchmark_override"]
    configure_activation_checkpointing(m)
    assert all(not block.mlp.ttn_activation_checkpointing and not block.cross_attn.ttn_activation_checkpointing
               for block in m.blocks)
    assert "activation_checkpointing" not in old["execution"]
    with pytest.raises(ValueError, match="activation_checkpointing"):
        configure_activation_checkpointing(m, "typo")
    with pytest.raises(ValueError, match="cached SANA"):
        configure_activation_checkpointing(nn.Sequential(nn.Linear(1, 1)), "ffn-cross-attn")
