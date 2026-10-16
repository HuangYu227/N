# TTRF Code Review

## Review target

Base: 17e8b1748e3f25f711dfe51e9ac009ed43489613
Branch: codex/inplace-ttt-accel

Reviewed files:
- worldttn/inplace_ttt_triton.py
- tests/ttn/test_inplace_ttt_triton.py
- tools/benchmark_inplace_ttt_triton.py
- docs/INPLACE_TTT_TTRF_ACCELERATION.md

## Findings

### P0: no production wiring by design

The new backend is not silently selected by InPlaceFFNTTT. This is intentional. The current repository must preserve the 17e8b17 reference path until target-GPU numerical and latency gates pass.

### P1: Triton read kernel must preserve FP32 delta semantics

The fast state is FP32 while activations are commonly BF16. The kernel therefore accumulates the read contraction in FP32 rather than silently converting the fast state to BF16. This avoids an algorithmic precision change.

### P1: update kernel is unclipped by itself

The update kernel accepts a scalar scale. The caller must compute the clip scale before invoking it. The prototype therefore does not claim that clipping is fused into the kernel. This is preferable to a two-pass norm implementation that would add complexity before profiling proves the benefit.

### P2: reduction order is different

Triton tiled reductions can use a different summation order from torch.bmm. Exact bitwise equality is not expected. The tests use bounded numerical tolerance. For production acceptance, compare both forward output and committed FP32 state.

### P2: target preparation remains unfused

DWConv5 and dense TTT projection are still outside this prototype. This is deliberate. They are lower-risk to optimize after measuring their actual critical-path contribution.

## Correctness contract

For read:

Y = cast(B + Z ΔW^T)

For update:

ΔW_new = ΔW + s V^T Z

The kernels implement the same elementwise finite sums.

No change is made to:
- five-layer placement;
- clean-only state commit;
- prefill read-only behavior;
- mean valid-token reduction;
- FP32 fast state;
- training/inference clip configuration;
- checkpoint format.

## Performance risk

A hand-written Triton GEMM is not automatically faster than cuBLAS/cuBLASLt. In particular, the read kernel deliberately preserves FP32 delta values while BF16 activations are loaded, which may reduce Tensor Core utilization. Therefore the benchmark is a gate, not a promise.

The correct production decision is empirical:

end-to-end speedup > 0
AND
numerical acceptance passes
AND
no regression on long-video memory.

## Recommended next kernel

If TTRF read is not faster, do not keep forcing Triton. The next candidate should be a CUDA/cuBLASLt fused path using:
- BF16 activation inputs;
- FP32 accumulation;
- tiled FP32 fast-state loads;
- fused residual epilogue.

If TTRF read is faster, profile the update kernel next.

## Important algorithmic opportunity

The current fast projection P is a dense C x C matrix initialized near diagonal. A low-rank or diagonal restriction would reduce FLOPs, but that changes the learning hypothesis. It should be a separate algorithm experiment, not hidden inside a backend optimization.

## Verdict

The branch is safe as an experimental backend prototype.

It is NOT yet justified to call it production-optimized until it is compiled and benchmarked on the actual target GPU. The repository does not expose an executable GPU runtime to this review session, so claiming a measured speedup would be false.
