# State-TTT + selective FP8 on one H100

Use `configs/worldttn/state_ttt_fp8.json` for the enhanced model. The existing
`dual_psi_fullgrad_fp8.json` keeps the old dual-psi model for comparison.

The enhanced path keeps camera-conditioned Cayley transport and Correct/Read.
Between them, each anchor applies a rank64 additive state update. The learned
basis is orthogonalized with QR; separate Local/clean per-head gains start at
0.5 and use the curvature step `eta = gain / (curvature_trace + eps)`.
The old Local/Persistent psi updates are disabled. Local state is temporary;
the clean update is absorbed into one committed S, without persisting a second
copy of AB. This is a different TTT parameterization, not a bigger Cayley psi.

All DiT slow weights remain trainable. FFN and text projections use the existing
FP8 recipe; S, State-TTT, camera transport, master parameters and Adam keep their
existing high precision. The training episode has 121 latent frames / 40 predicted
chunks, TBPTT=0 and one full episode backward. The clean history remains GT after
the current noisy loss. This update does not add a new Sink or distillation stage.

## Start from SANA pretrained weights

Run inside your existing tmux pane after activating `worldttn-fp8` and updating
the `codex/ttn-dual-psi-fp8` checkout. GPU **index 3** is the requested device:

```bash
ROOT="${WORLDTTN_WORKSPACE:?Set WORLDTTN_WORKSPACE to your isolated workspace root}"
source "$ROOT/activate-worldttn-fp8.sh"
git fetch origin refs/heads/codex/ttn-dual-psi-fp8:refs/remotes/origin/codex/ttn-dual-psi-fp8
test "$(git branch --show-current)" = codex/ttn-dual-psi-fp8
git merge --ff-only origin/codex/ttn-dual-psi-fp8
DATASET_ROOT="$ROOT/datasets/sana-wm-example"
RUN="$PWD/output/state-ttt-h100-gpu3-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/ttn_train_state_ttt_h100.sh 3 "$DATASET_ROOT" "$RUN"
```

The launcher starts a **single process** (`--parallel single`); the existing
FSDP2 implementation requires at least two processes. A fresh run inherits SANA
pretrained slow weights and learns new State-TTT parameters. An old dual-psi TTN
checkpoint is a different architecture and cannot be used for exact resume.
Batch size is one clip / optimizer step; it is not the same effective batch as
the earlier two- or four-GPU jobs. The inherited/new parameter learning rates
remain 1e-6/1e-5 rather than changing silently with GPU count.

To resume this new architecture, keep its own output directory and pass:

```bash
bash tools/ttn_train_state_ttt_h100.sh 3 "$DATASET_ROOT" "$RUN" --resume
```

## Metrics and checkpoints

- `train.log`: progress, errors and checkpoint publication.
- `train.jsonl`: losses, gradient norms, timings, sampled GPU/RAM and state diagnostics.
- `first_update.json`: each new basis/gain parameter's gradient, actual change and Adam state.
- `stability.jsonl`: detailed S and update diagnostics.
- `run_config.json`: model, precision, training identity and source fingerprint.
- `train.exit-code`: foreground launch outcome, including a nonzero Python failure.

Checkpoint every 25 steps, also at 5/10; keep evaluation models at
25/50/100/250/500. Verified publication precedes retention of eligible old files.

Inspect the first update for ten State-TTT basis/gain tensors with finite,
nonzero gradients and parameter changes. Check that run identity says
`additive-state-ttt-v1`, old psi is disabled, and temporal_detaches=0.
Read the new `state_ttt` diagnostic objects for effective eta, curvature,
S increments, objective changes and paired output effects. The old Cayley-psi
norms can be zero by design; they do not measure this new update's strength.
For performance and capacity, inspect the **second** update after Adam is resident,
not only the initialization GPU snapshot.

The optional independent GPU activation cache defaults to 2 GiB and a 60 GiB
packing allocation threshold; these control tensor placement, not the algorithm.
Host RAM and free GPU memory still need to fit the complete run. Single-H100
full-model capacity and H100 FP8 kernel behavior have not been validated locally;
the launcher prints the actual device/free-memory information before loading.
