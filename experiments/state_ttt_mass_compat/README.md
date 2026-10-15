# State-TTT / Mass compatibility investigation

## Status

This branch retains the H100 State-TTT implementation from
`54c2d138a56a37381140c8fcd1ee8cd567649f0f`. Its existing training and inference
entry points still run that implementation. The additions in this directory
are a design proposal and an executable algebra check, **not an integrated
normalized/sparse/Sink model or a new training recipe**.

The proposal preserves value-side Cayley transport of S and direct-S TTT,
while investigating a coupled content/support readout, an independent observed
Sink, and RoPE-compatible sparse addresses. See [design.md](design.md) for the
state definitions, update transactions, mathematical limits, and ablations.
The separate Mass implementation is a reference for this proposal; it is not
silently substituted for the H100 model in this branch.

## Reproduce the available check

From the repository root, using an environment with PyTorch installed:

```bash
python experiments/state_ttt_mass_compat/check_algebra.py
```

The script asserts its checks and writes `check_algebra.json` beside itself.
It uses synthetic small matrices in CPU FP64. It checks value-side transport,
coupled normalization, fixed-budget leave-one-out removal, a local quadratic
adaptation step, outer differentiation, sparse-address interference through a
dense projector, a candidate pair feature map, and compressed Sink rephasing.

Its pair layout is synthetic. It does not establish the real model's RoPE
layout, numerical behavior at production dimensions, or GPU/FSDP correctness.
Its sparsity threshold is a toy setting, not a selected model hyperparameter.

## What remains before a model experiment

- Integrate the new state dimensions and reference transactions with the
  anchor, runtime, full-gradient sequence execution, and checkpoint identity.
- Check the actual RoPE channel layout, exact analytic inner gradients, masks,
  empty support, repeated noisy calls, full episode gradients, and CUDA/FSDP.
- Load a trained H100 State-TTT checkpoint for controlled feature/history and
  rollout comparisons on the authorized L20 GPUs 6 and 7.

The trained checkpoint has not yet been transferred to L20. No compatibility
training or video experiment is claimed here. An old State-TTT checkpoint
cannot exactly resume the proposed changed memory representation. Backbone
initialization, if used, must be explicit and reported as a new experiment.

An inner objective decreasing does not imply improved future prediction or
long-rollout stability. Once TTT corrects S, normalized reads also need not
remain convex combinations of historical values. Sparse features evaluated
with dense GEMMs do not by themselves provide a speedup.
