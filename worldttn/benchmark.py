"""Isolated measurement/acceptance tools. GPU results are never inferred from CPU."""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import math
from pathlib import Path
import statistics
import torch
from . import core
from .performance import precision_audit, ExecutionOptions

VARIANTS = {"V0": ("reference", "reference"), "V1": ("reuse", "reference"),
            "V2": ("reuse", "projected"), "V3": ("compiled", "projected")}


def reset_operator_compilation():
    """Standalone operator job only; each B/N layout has a fresh cold graph."""
    from .compiled import helpers
    helpers.cache_clear()
    torch.compiler.reset()


def compare_saved_rollouts(directory, case_count):
    """Free-rollout differences are descriptive, not single-op tolerance gates."""
    directory = Path(directory)
    def difference(a, r):
        result = tensor_difference(a, r)
        result["exactly_equal"] = bool(torch.equal(a, r))
        result.pop("within_tolerance")
        return result
    reports = []
    for case in range(case_count):
        prefix = f"case-{case:03d}"
        optimized = torch.load(directory / f"{prefix}-ttn.pt", map_location="cpu", weights_only=False)
        reference = torch.load(directory / f"{prefix}-ttn_reference.pt", map_location="cpu", weights_only=False)
        states = sorted(directory.glob(f"{prefix}-ttn-state-*.pt"))
        refs = sorted(directory.glob(f"{prefix}-ttn_reference-state-*.pt"))
        if not states or len(states) != len(refs): raise ValueError("incomplete backend state trajectory")
        chunks = optimized.get("chunks")
        if not chunks or chunks != reference.get("chunks") or len(chunks) != len(states):
            raise ValueError("incomplete or mismatched rollout chunk boundaries")
        report = {"case": case, "generated_latents": difference(optimized["latents"], reference["latents"]),
                  "output_chunks": [], "first_nonzero_output_difference": None,
                  "state_chunks": [], "first_nonzero_state_difference": None}
        last_end = 0
        for index, boundary in enumerate(chunks):
            start, end = boundary["start"], boundary["end"]
            if boundary["chunk"] != index or start != last_end or not start < end <= optimized["latents"].shape[2]:
                raise ValueError("invalid rollout chunk boundary")
            result = difference(optimized["latents"][:, :, start:end], reference["latents"][:, :, start:end])
            report["output_chunks"].append({**boundary, "latents": result})
            if result["max_abs"] > 0 and report["first_nonzero_output_difference"] is None:
                report["first_nonzero_output_difference"] = boundary
            last_end = end
        if last_end != optimized["latents"].shape[2]: raise ValueError("incomplete rollout chunk coverage")
        for chunk, (a, r) in enumerate(zip(states, refs)):
            a, r = [torch.load(p, map_location="cpu", weights_only=False) for p in (a, r)]
            if (a["commits"], a["predictions"]) != (r["commits"], r["predictions"]):
                raise AssertionError("backend state counters differ")
            anchors = []
            for i, anchor in enumerate(core.ANCHORS):
                s = difference(a["world_state"][:, i], r["world_state"][:, i])
                psi = difference(a["transition_fast"][:, i], r["transition_fast"][:, i])
                anchors.append({"anchor": anchor, "state": s, "psi": psi})
                if report["first_nonzero_state_difference"] is None and (s["max_abs"] > 0 or psi["max_abs"] > 0):
                    report["first_nonzero_state_difference"] = {"chunk": chunk, "anchor": anchor}
            report["state_chunks"].append({"chunk": chunk, "anchors": anchors})
        reports.append(report)
    return reports


def tensor_difference(actual, reference, *, atol=1e-6, rtol=1e-4):
    if actual.shape != reference.shape: raise ValueError("backend tensor shapes differ")
    actual, reference = actual.detach().double().cpu(), reference.detach().double().cpu()
    difference = actual - reference
    denominator = reference.norm().item()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {"max_abs": difference.abs().max().item(), "rms": difference.square().mean().sqrt().item(),
            "relative_l2": difference.norm().item() / denominator if denominator > 0 else None,
            "reference_norm": denominator, "finite": finite,
            "within_tolerance": finite and torch.allclose(actual, reference, atol=atol, rtol=rtol)}


@contextmanager
def precision_protocol(mode, audited=None):
    if mode not in ("strict", "production"): raise ValueError("unknown precision protocol")
    precision, tf32 = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if mode == "strict":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
        elif audited is not None:
            torch.set_float32_matmul_precision(audited["matmul_precision"])
            torch.backends.cuda.matmul.allow_tf32 = audited["cuda_matmul_allow_tf32"]
            torch.backends.cudnn.allow_tf32 = audited["cudnn_allow_tf32"]
        yield
    finally:
        if mode == "strict" or audited is not None:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    a = int(position); b = min(a + 1, len(values) - 1)
    return values[a] + (values[b] - values[a]) * (position - a)


def stable_summary(records, expected_steps=10):
    if len(records) != expected_steps + 1: raise ValueError("incomplete stable measurement window")
    samples = records[1:]
    times = [r.get("iteration_seconds", r["seconds"]) for r in samples]
    updates = [r["seconds"] for r in samples]
    if not all(math.isfinite(t) and t > 0 for t in times): raise ValueError("invalid timing sample")
    ranks = [rank for record in samples for rank in record["ranks"]]
    hosts = [phase.get("host_rss_bytes", 0) for rank in ranks for phase in rank.get("memory_phases", [])]
    return {"cold": records[0], "stable": {"count": len(samples), "median_seconds": statistics.median(times),
            "p25_seconds": _quantile(times, .25), "p75_seconds": _quantile(times, .75),
            "peak_allocated_bytes": max(r.get("peak_allocated_bytes", 0) for r in ranks),
            "peak_reserved_bytes": max(r.get("peak_reserved_bytes", 0) for r in ranks),
            "peak_host_rss_bytes": max(hosts, default=0),
            "median_update_seconds": statistics.median(updates),
            "timing": "slowest-rank full iteration when recorded; includes data/legacy telemetry/log publication; checkpoint excluded"},
            "records": samples, "triton": {"status": "skipped", "reason": "target-GPU critical-path gate pending"}}


def triton_gate(step_ms, psi_ms, reduction_ms):
    if not math.isfinite(step_ms) or step_ms <= 0: raise ValueError("positive step timing required")
    if psi_ms is None or reduction_ms is None:
        return {"decision": "profile_required", "reason": "critical-path timing unavailable"}
    if not all(math.isfinite(t) for t in (psi_ms, reduction_ms)) or not 0 <= reduction_ms <= psi_ms <= step_ms:
        raise ValueError("invalid critical-path timing")
    fraction = reduction_ms / step_ms
    speedup = fraction / (1 - fraction) if fraction < 1 else math.inf
    stop = psi_ms / step_ms < .005 or speedup < .01
    return {"decision": "stop" if stop else "go", "psi_step_fraction": psi_ms / step_ms,
            "ideal_time_reduction_fraction": fraction, "ideal_throughput_gain": speedup,
            "assumption": "critical-path removable latency; not a sum of overlapping CUDA events"}


def select_joint_candidates(medians, correctness):
    base = medians["V0"]
    if len(base) != 2 or not all(math.isfinite(t) and t > 0 for t in base): raise ValueError("two positive baseline rounds required")
    candidates = [v for v in medians if v != "V0" and correctness.get(v, False)]
    if any(len(medians[v]) != 2 or not all(math.isfinite(t) and t > 0 for t in medians[v]) for v in candidates):
        raise ValueError("two positive candidate rounds required")
    candidates.sort(key=lambda v: (statistics.mean(t / b for t, b in zip(medians[v], base)), v))
    selected = candidates[:1]
    for variant in candidates[1:]:
        if all(t <= .97 * b for t, b in zip(medians[variant], base)):
            selected.append(variant); break
    return selected


def profiler_update(directory, rank):
    if directory is None: return nullcontext()
    @contextmanager
    def capture():
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profile:
            yield
        profile.export_chrome_trace(str(path / f"rank-{rank}.json"))
    return capture()


def _operator(q, k, v, beta, predicted, write, previous, u, vg, cbase, psi, snapshot, variant):
    core_backend, psi_backend = VARIANTS[variant]
    if core_backend == "reference":
        candidate, w = core.correct(predicted, k, v, beta, write)
        read = q @ candidate
        gradient = core.analytic_psi_gradient(previous, k, v, w, write, u, vg, cbase, psi, 1.)
    else:
        if core_backend == "compiled":
            from .compiled import with_aux
            read, *values = with_aux(q, k, v, beta, predicted, write, .5, 1e-6)
            aux = core.CorrectionAux(*values)
        else:
            aux = core.correct_with_aux(predicted, k, v, beta, write)
            read = q @ aux.candidate
        candidate, w = aux.candidate, aux.w
        fn = core.analytic_psi_gradient_projected if psi_backend == "projected" else core.analytic_psi_gradient_dense_from_aux
        gradient = fn(previous, aux.kt_weighted_residual, snapshot, write.sum(-1), 1.)
    return read, candidate, gradient


def first_cuda_call(fn):
    import time
    torch.cuda.synchronize()
    begin = time.perf_counter(); result = fn(); torch.cuda.synchronize()
    return result, time.perf_counter() - begin


def _cuda_times(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before, reserved_before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    events = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); events.append((start, end))
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    return {"median_ms": statistics.median(times), "p25_ms": _quantile(times, .25),
            "p75_ms": _quantile(times, .75), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "extra_allocated_bytes": torch.cuda.max_memory_allocated() - allocated_before,
            "extra_reserved_bytes": torch.cuda.max_memory_reserved() - reserved_before, "samples": len(times)}


def _kernel_count(fn):
    # Independent profiled invocation; never part of the measured throughput samples.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
    events = [e for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return {"cuda_events": len(events), "names": sorted({e.name for e in events}),
            "scope": "one separate profiled invocation; CUDA events are not critical-path percentages"}


def operator_benchmark(args):
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("operator benchmark requires CUDA; CPU checks are not GPU timing evidence")
    output = Path(args.output)
    from .provenance import implementation_identity
    from .core import DetachedCayleySnapshot
    from .performance import tensor_signature
    audited = None
    if args.precision == "production":
        if not getattr(args, "production_audit", None):
            raise ValueError("production microbenchmark requires --production-audit from a built reference run")
        record = json.loads(Path(args.production_audit).read_text(encoding="utf-8"))
        audited = record.get("precision", record)
        if audited["torch"] != torch.__version__ or audited["cuda"] != torch.version.cuda:
            raise ValueError("production precision audit uses a different Torch/CUDA runtime")
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "synthetic FP32 TTN operators; excludes telemetry/backbone/camera/FSDP",
              "protocol": args.precision, "provenance": implementation_identity(), "results": [],
              "V4": {"status": "skipped", "reason": "Triton gate requires full-step critical-path profiling"}}
    with precision_protocol(args.precision, audited):
        report["precision"] = precision_audit()
        for b in args.batch_sizes:
            for n in args.tokens:
                # The standalone B/N/grad matrix exceeds Dynamo's per-function
                # cache budget. Never reset compiler caches in real model training.
                reset_operator_compilation()
                torch.manual_seed(args.seed)
                h, d, m = 20, 112, 16
                # Preserve the token-major -> head-major non-contiguous production layout.
                q, k, value = [torch.randn(b, n, h, d, device=args.device).transpose(1, 2) for _ in range(3)]
                beta = torch.rand(b, h, n, device=args.device)
                previous = torch.randn(b, h, d, d, device=args.device) * .1
                u, v = [torch.nn.functional.normalize(torch.randn(h, m, d, device=args.device), dim=-1) for _ in range(2)]
                cbase, psi = [torch.randn(b, h, m, device=args.device) * .1 for _ in range(2)]
                factors = core.CayleyFactors(u, v, cbase + psi.tanh())
                predicted = factors.right(previous)
                mask = torch.ones(b, n, dtype=torch.bool, device=args.device)
                mask[-1, -min(7, n):] = False
                snapshot = DetachedCayleySnapshot.from_live(factors, psi, cbase, (0, 0, 0))
                values = (q, k, value, beta, predicted, mask, previous, u, v, cbase, psi, snapshot)
                expected, reference_cold = first_cuda_call(lambda: _operator(*values, "V0"))
                for variant in args.variants:
                    # Correctness and timing are distinct runs; comparisons synchronize only here.
                    actual, cold = (expected, reference_cold) if variant == "V0" else first_cuda_call(lambda: _operator(*values, variant))
                    differences = {name: tensor_difference(a, r) for name, a, r in zip(("read", "candidate", "g_psi"), actual, expected)}
                    if not all(r["within_tolerance"] for r in differences.values()):
                        raise AssertionError(f"{variant} failed direct operator equivalence: {differences}")
                    timing = _cuda_times(lambda: _operator(*values, variant), args.warmup, args.repeats)
                    timing.update(cold_seconds=cold, cold_scope="first clean forward + psi invocation before validation; excludes later outer-gradient compilation")
                    launches = _kernel_count(lambda: _operator(*values, variant))
                    # Benchmark detached psi separately; Correct auxiliary reuse is outside this timing.
                    aux = core.correct_with_aux(predicted, k, value, beta, mask)
                    if variant == "V0":
                        psi_fn = lambda: core.analytic_psi_gradient(previous, k, value, aux.w, mask, u, v, cbase, psi, 1.)
                    else:
                        fn = core.analytic_psi_gradient_dense_from_aux if variant == "V1" else core.analytic_psi_gradient_projected
                        psi_fn = lambda: fn(previous, aux.kt_weighted_residual, snapshot, mask.sum(-1), 1.)
                    psi_timing = _cuda_times(psi_fn, args.warmup, args.repeats)
                    # Direct forward/backward check with non-contiguous upstream layout.
                    gradient_runs = []
                    for chosen in ("V0", variant):
                        qq, kk, vv, bb, pp = [x.detach().clone().requires_grad_() for x in values[:5]]
                        read, candidate, _ = _operator(qq, kk, vv, bb, pp, *values[5:], chosen)
                        grads = torch.autograd.grad(read.transpose(1, 2).square().mean() + candidate.square().mean(), (qq, kk, vv, bb, pp))
                        gradient_runs.append(grads)
                    outer = {name: tensor_difference(a, r) for name, a, r in zip(("q", "k", "v", "beta", "predicted"), gradient_runs[1], gradient_runs[0])}
                    if not all(r["within_tolerance"] for r in outer.values()): raise AssertionError(f"{variant} outer gradient mismatch")
                    row = {"variant": variant, "batch": b, "tokens": n, "heads": h, "d": d, "M": m,
                           "layout": tensor_signature(q), "timing": timing, "psi_timing": psi_timing, "launches": launches,
                           "direct_difference": differences, "outer_gradients": outer, "operator_correctness_pass": True}
                    report["results"].append(row)
                    (output / "performance.json").write_text(json.dumps(report, indent=2))
                    print(f"[TTN microbench] {variant} B={b} N={n} median={timing['median_ms']:.3f}ms psi={psi_timing['median_ms']:.3f}ms", flush=True)
                del values, q, k, value, beta, previous, u, v, cbase, psi, factors, predicted, snapshot, expected, actual, aux
                del psi_fn, gradient_runs, qq, kk, vv, bb, pp, read, candidate, grads
                gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    op = commands.add_parser("operators")
    op.add_argument("--device", default="cuda")
    op.add_argument("--output", required=True)
    op.add_argument("--precision", choices=("strict", "production"), default="strict")
    op.add_argument("--production-audit", help="reference run_config.json/performance.json after model build")
    op.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    op.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2])
    op.add_argument("--tokens", nargs="+", type=int, default=[880, 2640, 3520])
    op.add_argument("--warmup", type=int, default=10); op.add_argument("--repeats", type=int, default=50)
    op.add_argument("--seed", type=int, default=3407)
    gate = commands.add_parser("triton-gate")
    gate.add_argument("--step-ms", required=True, type=float)
    gate.add_argument("--critical-psi-ms", type=float)
    gate.add_argument("--critical-reduction-ms", type=float)
    args = parser.parse_args()
    if args.command == "operators":
        if min(args.warmup, args.repeats, *args.batch_sizes, *args.tokens) < 1: parser.error("positive counts required")
        operator_benchmark(args)
    else: print(json.dumps(triton_gate(args.step_ms, args.critical_psi_ms, args.critical_reduction_ms), indent=2))


if __name__ == "__main__": main()
