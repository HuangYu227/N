import pytest
import torch
from worldttn import core


def test_independent_benchmark_warmup_preserves_rng_inputs_and_layout(monkeypatch):
    from worldttn import compiled
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    compiled.helpers.cache_clear()
    compiled.benchmark_warmup(True)
    try:
        q, k, v = [torch.randn(1, 7, 2, 8).transpose(1, 2).requires_grad_() for _ in range(3)]
        beta = torch.rand(1, 2, 7, requires_grad=True)
        predicted = torch.randn(1, 2, 8, 8, requires_grad=True)
        write = torch.ones(1, 7, dtype=torch.bool)
        tensors = (q, k, v, beta, predicted, write)
        before = [x.detach().clone() for x in tensors]
        rng = torch.get_rng_state().clone()
        output = compiled.read_only(*tensors, .5, 1e-6)
        output.square().mean().backward()
        assert compiled.warmup_report()["specializations"] == 1
        compiled.read_only(*tensors, .5, 1e-6)
        assert compiled.warmup_report()["specializations"] == 1
        assert torch.equal(rng, torch.get_rng_state())
        for actual, initial in zip(tensors, before): torch.testing.assert_close(actual, initial, rtol=0, atol=0)
        assert q.grad.norm() > 0 and not q.is_contiguous()
    finally:
        compiled.benchmark_warmup(False)
        compiled.helpers.cache_clear()


def test_compiled_noisy_exports_only_read_and_preserves_backward(monkeypatch):
    from worldttn import compiled
    real_compile = torch.compile
    # Exercise actual AOTAutograd on CPU; CUDA test below exercises Inductor.
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    compiled.helpers.cache_clear()
    torch.manual_seed(31)
    originals = [torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8),
                 torch.rand(1, 2, 7), torch.randn(1, 2, 8, 8)]
    mask = torch.tensor([[True] * 5 + [False] * 2])
    results = []
    for optimized in (False, True):
        q, k, v, beta, p = [a.clone().requires_grad_() for a in originals]
        out = compiled.read_only(q, k, v, beta, p, mask, .5, 1e-6) if optimized else q @ core.correct(p, k, v, beta, mask)[0]
        assert isinstance(out, torch.Tensor)
        # Transpose creates the non-contiguous upstream gradient seen on LTU.
        loss = out.transpose(1, 2).square().mean()
        results.append((out, torch.autograd.grad(loss, (q, k, v, beta, p))))
    torch.testing.assert_close(results[0][0], results[1][0])
    for a, b in zip(results[0][1], results[1][1]): torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    compiled.helpers.cache_clear()


def test_compiled_clean_aux_are_live_and_match_reference(monkeypatch):
    from worldttn import compiled
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    compiled.helpers.cache_clear()
    q, k, v = [torch.randn(1, 2, 7, 8, requires_grad=True) for _ in range(3)]
    beta = torch.rand(1, 2, 7, requires_grad=True)
    p = torch.randn(1, 2, 8, 8, requires_grad=True)
    mask = torch.ones(1, 7, dtype=torch.bool)
    out, *values = compiled.with_aux(q, k, v, beta, p, mask, .5, 1e-6)
    aux = core.CorrectionAux(*values)
    expected = core.correct_with_aux(p, k, v, beta, mask)
    for a, b in zip(aux, expected): torch.testing.assert_close(a, b)
    assert aux.candidate.requires_grad and aux.kt_weighted_residual.requires_grad
    (out.square().mean() + aux.candidate.square().mean()).backward()
    assert k.grad.norm() > 0
    compiled.helpers.cache_clear()


def test_compiled_real_stage_c_tbptt2_matches_reference(monkeypatch):
    import copy
    from worldttn import compiled
    from worldttn.performance import ExecutionOptions, configure_execution
    from worldttn.training import train_clip, linear_flow_loss
    from worldttn.checkpoint import make_optimizer
    from test_training import TinyWorldModel, inputs
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    compiled.helpers.cache_clear()
    torch.manual_seed(12)
    teacher = TinyWorldModel()
    with torch.no_grad():
        for head in teacher.ttn_system.controller.heads: head.weight.normal_(0, .002)
    student = copy.deepcopy(teacher)
    configure_execution(student, ExecutionOptions("compiled", "projected"))
    clean, noise, t, camera = inputs()
    states = []
    for model in (teacher, student):
        result = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, make_optimizer(model),
                            linear_flow_loss, t, noise, width=100, height=100, tbptt=2, activation_offload="cpu")
        states.append(result["runtime"])
    for name in ("world_state", "transition_fast"):
        torch.testing.assert_close(getattr(states[0], name), getattr(states[1], name), atol=1e-6, rtol=1e-4)
    for (name, actual), (_, expected) in zip(student.named_parameters(), teacher.named_parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        if expected.grad is None: assert actual.grad is None, name
        else: torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4)
    assert student.ttn_system.controller.heads[0].weight.grad.norm() > 0
    assert student.ttn_system.generators.u.grad.norm() > 0
    compiled.helpers.cache_clear()


def test_operator_matrix_resets_compiler_between_layout_cases(monkeypatch):
    from worldttn import compiled
    from worldttn.benchmark import reset_operator_compilation
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend="aot_eager", fullgraph=True, dynamic=False))
    # Small cache limit reproduces the standalone B/N/grad matrix exhaustion.
    with torch._dynamo.config.patch(recompile_limit=2):
        for n in (3, 5, 7):
            reset_operator_compilation()
            for grad in (False, True):
                q, k, v = [torch.randn(1, 2, n, 8, requires_grad=grad) for _ in range(3)]
                beta = torch.rand(1, 2, n, requires_grad=grad)
                p = torch.randn(1, 2, 8, 8, requires_grad=grad)
                read, *_ = compiled.with_aux(q, k, v, beta, p, torch.ones(1, n, dtype=torch.bool), .5, 1e-6)
                if grad: read.square().mean().backward()
                assert torch.isfinite(read).all()
    reset_operator_compilation()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Inductor backward needs target CUDA GPU")
def test_cuda_inductor_strided_offload_backward():
    from worldttn import compiled
    from worldttn.training import activation_storage
    compiled.helpers.cache_clear()
    torch.manual_seed(3407)
    base = [torch.randn(1, 3520, 20, 112, device="cuda").transpose(1, 2) for _ in range(3)]
    base += [torch.rand(1, 20, 3520, device="cuda"), torch.randn(1, 20, 112, 112, device="cuda")]
    mask = torch.ones(1, 3520, device="cuda", dtype=torch.bool)
    results = []
    for optimized in (False, True):
        q, k, v, beta, p = [a.detach().clone().requires_grad_() for a in base]
        with activation_storage("cpu"):
            out = compiled.read_only(q, k, v, beta, p, mask, .5, 1e-6) if optimized else q @ core.correct(p, k, v, beta, mask)[0]
            loss = out.transpose(1, 2).square().mean()
        grads = torch.autograd.grad(loss, (q, k, v, beta, p))
        results.append((out.detach().cpu(), [g.cpu() for g in grads]))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-4)
    for a, b in zip(results[0][1], results[1][1]): torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
