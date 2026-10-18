# Train-free event replay for enhanced StateTTT

This experiment extends the H100 rank64 StateTTT implementation at upstream
`54c2d138a56a37381140c8fcd1ee8cd567649f0f`, retaining the QR/text-cache acceleration
used for step325-to1000. It is an independent inference experiment, not a new
training recipe or a reproduction of Deep Forcing's Softmax KV reader.

## Data flow and boundaries

Each clean anchor constructs post-RoPE sufficient statistics using exactly the
normalized weights of the existing Correct operation:

`G = K^T W K`, `B = K^T W V`, `D = eps + sum(W * ||K||^2)`.

After all five anchors and the clean model output pass validation, the runtime
atomically publishes detached FP32 CPU records alongside the original S/psi
commit. Frame0 is labelled observed; subsequent records are generated history,
not additional ground-truth observations. CFG branches have separate payloads
and selections. Archive tensors are runtime state, absent from model weights.

At the next chunk's Predict, old B records follow the same Cayley right transport
as S, cumulatively: `B' = B R`, `S' = S R`, with G unchanged. This preserves
the covariance `B' - G S' = (B - G S) R` for that coordinate operation; it is
not a proof that transported memories model camera motion correctly.

The original StateTTT and current-token Correct run first. For a selected event:

`delta_j = alpha_s / D_j * (B_j - G_j S_current)`.

Retrieval-weighted deltas are multiplied by gain and then capped per head at
`max_relative_delta * ||S_current||`. Only `Q @ S_read` uses the result.
The clean S candidate, Cayley controller and StateTTT parameters are unchanged
for identical inputs. Changed generated frames can still change future states
indirectly during an autoregressive rollout.

The first noisy call with sigma at most 0.5 chooses events; that chunk reuses its
selection on later eligible solver calls. Clean calls never use the extra read.
Descriptors use the existing pretrained/adapted K0/V features before RoPE, with
spatial pooling where available. They have no new MLP or learned adapter; noise
and timestep dependence remains a hypothesis to inspect, not an invariance.

## Choices and controls

The JSON under `configs/worldttn/event_memory_state_replay.json` is an exploratory
configuration, not a tuned optimum. Omission or mode off/gain zero follows the
original path and allocates no archive. Active memory requires no-grad, Stage C
rank64 StateTTT, full TTT and the reference Correct/Read backend; it is rejected
in training/recomputation. The inference CLI accepts `--event-memory-config`.

- Physical anchor indices: 11/15/19 (internal anchor slots are 2/3/4).
- Capacity eight events; coverage retention preserves first/newest and removes
  the most temporally redundant interior record. This keeps distant events.
- Top two cosine matches, temperature 0.1, gain 0.1, final per-head cap 5%.
- Reads start at prediction chunk five; no age decay or entropy gate is hidden
  in the first comparison.
- Absolute post-RoPE statistics are preserved. Reapplying current RoPE to old K
  is not supported; the earlier proposed relative-position formula was too
  underspecified for the actual three-axis RoPE/Cayley combination.

At B=2 (CFG), H=20, D=112, three anchors and eight events, the two FP32 matrices
alone use about 92 MiB of CPU storage. Metadata/descriptors and a transaction
snapshot add storage. Matrix statistics are exact for this quadratic replay
objective; they do not preserve every detail that token-level Softmax could read.

Selectors share the same archive/retention: content, recent, initial, and shuffled.
Shuffled redirects the selected descriptor to a different complete event from
the same CFG branch, preserving K/V correspondence. It is a mismatched-memory
control; it does not guarantee that two forest events depict unrelated content.
Initial is initial-event replay, not the old convex blend with prefill S.

## Fixed forest experiment

`tools/ttn_event_memory_experiment.py` pins the existing step750 checkpoint,
forest-road input packet and original FP32 noise, then generates five groups
on physical GPUs 4/5 with one worker per GPU. It does not use training GPUs 6/7.
Every group records source/input/checkpoint hashes, per-chunk events, output
effects, latency, memory and GT latent MSE. Future GT stays out of rollout.

The case is `sekai_game/00100100001_0004650_0006450_s000000`, 121 latents,
40 predicted chunks, 961 decoded frames at 16fps, 20 solver steps, CFG4.5,
two cached chunks, FP32 weights and BF16 autocast. Its saved static-camera text
conflicts with the actual moving camera trajectory; all groups retain the same
text to avoid changing another variable. This is a training-seen case/one seed,
not a generalization benchmark or an exact closed-loop return test.

First validate baseline vs content, including the unaffected first four chunks
and the baseline against prior stored latents. Then run recent, shuffled and
initial controls. A visible state/output perturbation alone is not a quality
benefit; content should improve relevant outcomes beyond the other controls.
Optional PSNR/SSIM use VAE-decoded GT with explicit frame sampling, not original
RGB. A turning camera without the same pose/intrinsics is not a same-view target.

No new offline training, external encoder, token router, EMA, RoPE rewrite,
Triton kernel or checkpoint-format change is required by this first version.
