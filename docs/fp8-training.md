# Selective FP8 training on L20

This adds native FP8 GEMMs to the existing dual-psi full-gradient model. It
does not add QAT, truncate temporal gradients, or change the state update.

For the enhanced rank64 State-TTT architecture, use
`configs/worldttn/state_ttt_fp8.json` instead. Its additive state update is an
explicit algorithm change; the original config still selects Cayley psi.
See [single-H100 training and checkpoint instructions](state-ttt-h100.md).

## Precision policy

- TorchAO 0.14.1, PyTorch 2.9.1+cu128, Ada SM89 or newer.
- Per-tensor dynamic scaling: E4M3 inputs/weights and E5M2 gradient inputs.
- FP8 targets: the three text cross-attention Linear projections and the
  two ordinary 1x1 FFN Conv2d projections in each block.
- Original parameter objects, names, shapes and FP32 master storage stay intact.
- TTN anchors, native recurrent attention, temporal/depthwise convolutions,
  S, psi, beta/controller/generators and loss retain existing precision.
- FSDP2 all-gather and reductions retain existing precision. No FP8 all-gather.
- Scaling is recomputed per call; there is no mutable amax history to replay
  or serialize. This is why the initial integration uses current scaling.

The pointwise-convolution adapter flattens spatial positions into GEMM rows,
uses TorchAO's native training implementation, then restores NCHW. Its weight
stays `[out_channels, in_channels, 1, 1]` for existing checkpoint compatibility.
There are no new learnable parameters.

## Dependencies and tests

Install `requirements-fp8.txt` in a separate environment compatible with the
PyTorch version above. Do not update a live training environment.

```bash
python -m pytest tests/ttn/test_fp8.py tests/ttn/test_legacy_sequence.py \
  tests/ttn/test_fullgradient.py -q

# Set these to two GPUs assigned to this validation.
export CUDA_VISIBLE_DEVICES=6,7
export DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1 OMP_NUM_THREADS=2
export FP8_ACCEPTANCE_OUTPUT="$PWD/output/fp8-tiny-acceptance"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  --max_restarts=0 -m pytest tests/ttn/test_fp8_cuda.py -q -s
```

The CPU backend fixture checks wiring and checkpoint shape only; it does not
claim FP8 numeric accuracy. The CUDA tests require actual `_scaled_mm` calls,
finite gradients and FP32 recurrent states, plus a two-rank full-gradient
update/save/resume comparison. Their tiny model does not establish whether
the full 2.66B model with 121 latents fits two GPUs or preserves video quality.

## Full-model training entry

Use the existing training command with either:

```bash
--config configs/worldttn/dual_psi_fullgrad_fp8.json
```

or add `--train-precision fp8` to the existing BF16-config command. The latter
overrides the config's training precision. The flag is training-only; ordinary
inference can read the exported FP32 weights using its existing mixed precision.

`[TTN training precision]` and `run_config.json.training_precision` list the
actual converted modules. The checkpoint training identity records the FP8
recipe, and exact resume rejects a silent BF16/FP8 change. Existing BF16 resume
identities remain unchanged. Four-rank optimizer bundles also cannot exact-resume
on two ranks in this project's checkpoint format. Use `--adapter` without
`--resume` for an explicitly new weight-initialized experiment in that case.

Before a long two-GPU run, measure complete 121-latent updates after Adam state
exists, with the intended offload setting. FP8 GEMMs do not automatically reduce
FP32 master/optimizer storage, and TorchAO may retain high-precision backward
inputs. Memory and throughput gains require a representative measurement.
