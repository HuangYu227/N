# Frame Direct-S TTT with full sequence gradients

Base: `codex/ttn-proximal-memory`, local base commit `be633dfa092c4e2d246901e3b9102c50e094d07a`.
Branch: `codex/ttn-frame-fullgrad`. The training launcher runs only when explicitly invoked.

## What changes

- Five TLA anchors remain at blocks 3, 7, 11, 15, 19. Their projection names and dimensions are unchanged.
- The shipped SANA UCPE/Plucker camera conditioning is preserved. Full-clip Plucker embeddings are sliced with each native group and remain differentiable.
- Every latent frame fits **all of its spatial tokens jointly** and reads the resulting state. A future frame never supplies the state used to read an earlier frame.
- `phi` remains the inherited Q/K normalization followed by SiLU and RoPE. The small Local/Persistent psi rotation branches remain disabled. The fast weights optimized by TTT are S itself.
- Training uses **one observed frame plus 120 noisy future frames**, 40 native generation groups, and **one backward/optimizer update**. `tbptt=0` means no temporal detach; it is not a one-frame gradient window.
- The observed frame is isolated in every layer. Its clean prefill and the subsequent noisy sequence are fused into one layer-wise schedule, mathematically equivalent to evaluating the isolated prefill then continuing each layer's live state with the same weights. There is no extra clean forward after each training group and no generated-history sampler inside training.
- Native GDN states, camera K/V, short-convolution history and FFN temporal history retain gradients inside the sequence. Selection indices remain discrete; selected K/V/W retain gradients. Finite cache capacity still evicts observations and does not promise unlimited explicit history.
- Non-reentrant checkpoints at block and native-group boundaries replay pure local computations. Recomputations never commit runtime state or duplicate telemetry publication. FSDP enters each DiT block once per model forward.
- Inference still denoises three future frames per group. Each solver call scans temporary frame states from the same incoming clean history. Only the completed group's clean forward commits persistent history. The first group separates its observed frame before the three future frames, matching training isolation.

This preserves native **chunk-level** causality; native within-group attention/convolutions may mix three frames. Frame-level TLA writes do not make the entire network strictly frame-causal.

## Update and numerical meaning

For a frame, normalize positive beta weights over its valid spatial tokens. Use the already-retained prefix, selected-middle and recent observations as disjoint sources. The current frame is appended **after** its solve:

```
G = K.T W K + gamma/m sum_r Kr.T Wr Kr
B = K.T W V + gamma/m sum_r Kr.T Wr Vr
lambda = kappa * trace(G)/d + eps
S_new = S_old + solve(G + lambda I, B - G S_old)
Y_frame = Q_frame S_new
```

`gamma=0.25`, `eps=1e-6`; history activates at absolute latent frame **13**, preserving the old fifth-predicted-group boundary. A 16-frame-equivalent token budget retains a 10-frame prefix and 4 recent frames, with remaining capacity selected by the existing participative rule. No visual Softmax is added.

Observed prefill uses `kappa=16`; future frames use `kappa=49`. The latter approximates the old three-frame retention under an isotropic Gram matrix: `(16/17)^(1/3) ≈ 49/50`. It is an initialization heuristic, not an exact equivalence for arbitrary K. Solves and recurrence use FP32 (FP64 in numerical tests), with gradients through Gram matrices, lambda and selected history.

Beta's absolute scale cancels in source normalization; its useful signal is relative spatial weighting. Nonzero gradients or output deltas establish connectivity, not video-quality gains. Neither fixed-feature contraction nor a passing smoke test proves long-rollout stability. Training on noisy history versus inference on generated clean history remains an experimental question.

## Repairs and acceleration

These engineering changes keep frame updates, source weights, solve precision, the objective and the full temporal graph:

| Change | Scope and gradient contract |
|---|---|
| Protected-prefix Gram/RHS reuse | Once the prefix is sealed, retain its live K/V/normalized weights/G/B in the episode cache. Later frames reuse G/B without detach. Changed prefix membership or temporal realignment invalidates the summary. Nothing survives a new training episode or parameter update. |
| Text K/V reuse | Project the same condition once inside each block forward and pass the live tensors to that block's groups. No global cache; checkpoint replay rebuilds them from the same inputs. |
| Diagnostics outside replay | Publish statistics only on the original forward. State updates, selected observations and their derivatives still execute during recomputation. Diagnostic-only matrix products run without building an autograd graph. |
| Explicit checkpoint inputs | Flatten live state/cache/camera/RoPE tensors into checkpoint arguments, including cached G/B. Preserve aliases and strides when CPU snapshots are restored. |
| Immediate input release | Clear the recursive flatten function's closure cell after building its tensor-free specification. Original GPU inputs then release when ordinary references expire, without waiting for Python cyclic GC. Saved snapshots and live gradient edges remain intact. |
| L20 backward dispatch | Select native kernel tile sizes using the tensor's actual device; use the existing smaller column tiles for low shared-memory devices at padded D=128. |
| Pageable activation snapshots | Default to independent contiguous CPU snapshots without pinned-memory accumulation. Optional GPU saved-activation budget changes placement only; it does not bound total GPU memory. |
| Bounded anchor replay | Each TLA anchor's outer checkpoint covers at most eight native groups (24 frames). Live S, retained K/V/W, prefix statistics and native caches cross every boundary; no temporal detach or separate backward. This bounds the GPU tensors retained by nested replay. Native blocks keep whole-block replay. |

Text K/V reuse preserves the forward computation, but changes backward accumulation order. BF16 gradients need tolerance-based comparison; they are not promised bitwise identical. Prefix reuse also keeps gradients live, so it reduces repeated arithmetic rather than eliminating all activation history.

The full training path now records observed prefill anchors in the normal logging schema. The metrics tool supports both chained Slurm runs and foreground runs with `run_config.json`; it does not load model weights to read progress.

## Files

| Area | Entry |
|---|---|
| Frame regression | `worldttn/frame_memory.py` |
| Pure layer-wise scan and checkpoints | `worldttn/sequence.py` |
| Full-sequence loss/backward | `worldttn/sequence_training.py` |
| Existing integration | `anchor.py`, `training.py`, `distributed.py`, `cli.py` |
| Native live caches | `fused_streaming.py`, `sana_gdn_camctrl_blocks.py`, `basic_modules.py` |
| SANA block dispatch | `sana_multi_scale_video_camctrl.py` |
| Training recipe | `configs/worldttn/frame_fullgrad.json` |
| Foreground launcher | `tools/ttn_frame_train.sh` |

Old configs retain the old math, cache detaches and resume identity. Frame checkpoints record their own architecture/protocol identity. **Do not resume an old chunk/TBPTT checkpoint into this experiment**: its optimizer/data progression belongs to a different objective. The launcher starts from pretrained SANA; a deliberate old-weight conversion is not included.

## Local verification and server gate

Local tests cover FP64 finite-difference gradients, partitioned frame scans, spatial permutation, empty writes, history activation, no future-frame leakage, early-frame credit from late loss, real SANA block/FFN/camera code on tiny grids, first-group training/inference parity, immutable noisy-call caches, Plucker conditioning gradients, single optimizer updates and CUDA BF16/offload/recomputation parity. Local tests use a small CPU recurrence in place of the fused native GDN; they do not certify the production native kernels or multi-GPU communication. The Windows sandbox blocks local Gloo subprocess communication, so those existing two-process CLI tests are excluded locally.

The opt-in Linux gate replaces those native fixtures with the real cached GDN, then checks FSDP, pageable offload, recomputation and exact optimizer resume. Its 25-frame fixture crosses both the replay-span and history thresholds with prefix=10/capacity=16/recent=4, and checks gradients from late loss to observed and early noisy frames. A separate native backward gate exercises H=20/D=112 column tiling. Run both on the allocated GPUs before the full model. They write only tiny synthetic checkpoints to a fresh directory:

```bash
conda activate wm
cd /home/newuser001/huangyu/WorldTLA-frame-fullgrad
export CUDA_VISIBLE_DEVICES=0,1,2,5   # verify these four GPUs are allocated to this experiment
export DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1 OMP_NUM_THREADS=4
FRAME_NATIVE_CUDA_TEST=1 python -m pytest -q \
  tests/ttn/test_frame_native_memory.py -k native_phase_a_real_head_dimension
export FRAME_DISTRIBUTED_TEST=1
export FRAME_TEST_OUTPUT="$PWD/output/frame-gate-$(date -u +%Y%m%dT%H%M%SZ)"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m pytest -q -s tests/ttn/test_frame_distributed.py
unset FRAME_DISTRIBUTED_TEST FRAME_TEST_OUTPUT
```

The server needs the existing SANA dependencies and a PyTorch version exporting `torch.distributed.fsdp.fully_shard` (the previous server environment used PyTorch 2.9). A Linux four-GPU run and native-resolution RAM/VRAM peaks have **not** been measured locally.

## Full 500-step run

Use a **separate uploaded directory**, preserving the currently running experiment. Execute in your own tmux session. No nested tmux session is created. `DATASET_ROOT` must contain the `data/` tree from the configured SANA dataset; reuse the exact root from the successful previous run. If needed, specify `DATA_DIR` and `VAE_CACHE_DIR` explicitly.

```bash
conda activate wm
cd /home/newuser001/huangyu/WorldTLA-frame-fullgrad
export CUDA_VISIBLE_DEVICES=0,1,2,5
export DATASET_ROOT=/home/newuser001/huangyu/WorldTTT/datasets/sana-wm-example
export OUTPUT="$PWD/output/worldttn/frame-fullgrad-$(date -u +%Y%m%dT%H%M%SZ)"
export MAX_STEPS=500 SAVE_EVERY=25
bash tools/ttn_frame_train.sh
```

Because the previous single-host offload run exhausted RAM, run the launcher inside an available memory-controlled scope for the first full-size run. For a working user systemd manager, replace the last line with:

```bash
systemd-run --user --scope -p MemoryMax=400G -p MemorySwapMax=0 \
  bash tools/ttn_frame_train.sh
```

400 GiB applies to the **entire process group**, not each rank. It is a ceiling, not an estimate of required memory or a guarantee that the host has enough available RAM. If the scope is unavailable, establish another job/cgroup limit before the first full-size run; do not silently retry unbounded. More GPUs shard parameters/optimizer states, not each rank's full temporal activation history.

`MAX_STEPS=500` is one persistent torchrun job, not 500 separate scheduler submissions. Normal completion saves `last.pt` even when the final step is not divisible by 25. The existing checkpoint publisher verifies model/shards, atomically publishes `last.pt` and applies its rolling retention; no new permanent per-step snapshot series is added. There is no automatic every-25-step video evaluation in this foreground launcher.

Checkpoint IO/error coordination uses a separate Gloo CPU group with a bounded one-hour timeout. The model's NCCL tensor collectives keep their normal timeout; waiting for SHA verification or retention no longer enqueues an idle NCCL collective. Full integrity checks remain in place. `[TTN checkpoint phase]` and `[TTN retention begin]` records expose save/verification/cleanup time. This addresses disk-work stalls, not arbitrary mismatched tensor collective order or failed network connections.

To resume **this protocol, same GPU count/backend**:

```bash
export OUTPUT=/absolute/path/to/the/existing/frame-fullgrad-run
export RESUME=1 MAX_STEPS=500
bash tools/ttn_frame_train.sh
```

## Reading progress and metrics

```bash
tail -n 40 "$OUTPUT/train.log"
grep -E '^\[TTN (progress|memory|checkpoint|retention)' "$OUTPUT/train.log" | tail -n 20
python -m tools.ttn_training_metrics "$OUTPUT"
```

`train.jsonl` records flow loss, gradient norm, step duration, allocation/host-memory phases, saved-tensor payload and protocol/exposures. `stability.jsonl` records per-rank/group/anchor S scale, beta/weight distribution, frame solve objectives, gradient alignment, regularization, prefix reuse and output effects. State spectra are sampled at groups beginning at frames 1 and 13 and every four groups. The first update audit records trainable parameter gradients and actual parameter changes. Saved-tensor cumulative payload, per-process RSS and whole-host RAM are different measurements. Kernel-level/per-process measurements remain distinct from whole-host RAM.

Training `commits=0` is intentional: the noisy training sequence does not create clean runtime transactions. Read `exposure.frame_state_updates_per_anchor=121`, `predicted_chunks=40`, `backward_calls=1`, `temporal_detaches=0` instead. Inference still records prefill plus completed clean-group commits.

For generation/evaluation, use the new config and checkpoint with `--cached-blocks 2`; the existing custom video tool already defaults to two. SANA comparison continues to use the original pretrained SANA computation. Validate fixed cases/seeds at matched noise schedules before attributing any improvement to frame updates or full gradients.

## Inference-only spectral reference probe

`tools.ttn_custom_inference` accepts `--spectral-readout baseline|low-zero|low-reference|full-reference`.
This is a FreqForcing-inspired diagnostic, not a reproduction of its method or a new training objective.
All four modes capture an immutable FP32 copy of the five TLA states after observed-frame prefill.
Only noisy future-frame visual reads at solver sigma >= 0.8 can change; clean history writes and native caches use the existing path.

The reference is temporally RoPE-aligned, then projected with the current queries and output gate.
With gain 0.25, `low-zero` subtracts the low-frequency fast contribution, `low-reference` replaces that contribution with the reference, and `full-reference` applies the same mixing at all spatial frequencies.
Low-frequency filtering uses a Gaussian in the per-frame 2D FFT with bandwidth 0.125 in Nyquist units.
The diagnostic `baseline` leaves output values unchanged. Omitting the option bypasses the controller entirely.
Temporal alignment does not implement camera/world-space transport.

`--reuse-conditioning SOURCE` reuses matching first-frame/text/camera inputs, initial noise and pretrained SANA outputs across TTN checkpoints.
It checks the case, solver settings, backend and VAE/text-encoder identity; it does not reuse the old TTN rollout.
Diagnostics record state-reference SHA, selected sigma/readout effects, spectral energies, timing and memory.
Latent drift from the first frame is not future-GT accuracy or a general video-quality metric.

`tools/ttn_spectral_experiment.py` is the fixed step175 static-case runner used by the current experiment.
It requires a prepared snapshot, conditioning inputs, `experiment.json` and `code/files.json` manifests; it is not a general training launcher.
It waits for the specified GPU to be unoccupied, verifies source hashes, runs the five-chunk wiring smoke, then executes the four groups in separate processes.
Its state/output directory is independent of the main training run. Full-model smoke and quality conclusions require actual GPU results.
