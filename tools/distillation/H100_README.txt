H100 GPU5 / enhanced StateTTT step300 / LongLive DMD
==================================================

Branch: codex/ttn-state-ttt-longlive-dmd
This is a new distillation run, not a continuation of the original Adam state.
It trains an eight-solver-step student for 500 student optimizer updates and
2500 fake-score updates. Physical GPU5 only. Microbatch1, accumulation2,
effective batch2. Each episode retains all 121 latents / 40 prediction chunks
and all eight denoising steps in its gradient path; no TBPTT.

Run in your own tmux pane with the existing worldttn-fp8 H100 environment active.
The first distillation recipe uses BF16 autocast and FP32 state/master weights,
not FP8/NVFP4. Teacher, student and fake-score have independent state and caches.
All roles use FSDP2 with a one-rank CUDA mesh for CPU offload, not sharding savings.
The inherited TLA/StateTTT equations, LoRA rank128 and CFG4.5 remain unchanged.

Fresh deployment (do not overwrite an existing directory):

  WORK=/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/yuhuang
  git clone --single-branch --branch codex/ttn-state-ttt-longlive-dmd \
    https://github.com/HuangYu227/N.git "$WORK/WorldTTN-state-ttt-longlive-dmd"
  cd "$WORK/WorldTTN-state-ttt-longlive-dmd"
  source "$WORK/activate-worldttn-fp8.sh"

For an existing distillation checkout, use git status first and git pull
--ff-only in that checkout. Do not change code while its run is active.
Do not update the original training checkout to this branch.

Dependencies: the existing SANA environment plus PEFT, OmegaConf, LPIPS,
W&B, pytest, and a working PyTorch FSDP2 CPUOffloadPolicy (H100 baseline uses
PyTorch2.9.1+cu128). Missing imports stop preflight; the launcher does not
install packages or alter public login settings. If needed, install these
in that environment before running:

  python -m pip install peft==0.18.0 omegaconf==2.3.0 lpips==0.1.4 wandb pytest

Keep SANA's pyproject.toml PEFT pin (0.18.0); do not downgrade it to0.17.1.
If an earlier command already downgraded it, restore only that package:

  python -m pip install --no-deps peft==0.18.0

TorchAO0.14.1 may warn that its C++ extensions are incompatible with Torch2.9.1.
This BF16 distillation path does not call TorchAO FP8/quantization kernels;
the warning alone does not establish a failed run. Do not upgrade/downgrade
Torch or CUDA just to suppress it; inspect any subsequent traceback/exit code.

Default start:

  bash tools/distillation/run_h100.sh

Defaults search the original H100 run:
  $WORK/WorldTTN-dual-psi-fp8/output/state-ttt-h100-gpu5-20261014T084949Z
for last-resume-00000300-*/model.pt, step-00000300.pt, or last.pt ONLY if its
internal step is 300. A later last.pt is never substituted. If step300 was
already removed, provide an independently preserved step300 checkpoint.
Default dataset: $WORK/datasets/sana-wm-example. Default base: original SANA HF
checkpoint in the current user's cache, checked against the teacher base SHA.

Explicit start when paths differ:

  bash tools/distillation/run_h100.sh \
    --teacher /absolute/path/to/step300/model.pt \
    --dataset-root /absolute/path/to/sana-wm-example \
    --base-weights /absolute/path/to/sana_wm_chunk_causal_1600m_720p.safetensors

The launcher checks step300/full-DiT/rank64 curvature StateTTT, independently
copies and hashes the teacher, and writes config.yaml and launch.json. It prints
RUN=/.../output/step300-h100-gpu5-<timestamp>. No existing model is overwritten.
There must be enough GPU5 space for its budget plus4GiB; the original GPU5
training must already have released that memory. No other job is stopped.

RAM and GPU bounds:
  - PyTorch allocator: min(70GiB, actual device total minus8GiB).
  - Task RAM: min(360GiB, MemAvailable minus64GiB) by default; minimum128GiB.
    Override with --ram-limit-gib N only if N+64GiB is available at startup.
  - Each phase uses a named systemd user scope, MemoryMax and MemorySwapMax=0.
    A working systemd user manager/cgroup delegation is required; unavailable
    enforcement stops before model launch. No global swap/GPU policy is changed.
  - CPU pageable saved activations; GPU activation snapshot budget0 initially.

Execution order:
  1. Single-rank CUDA FSDP2/checkpoint/accumulation/resume regression tests.
  2. Capacity-only 25 ->61 ->121 latent real fake and student updates.
     121 includes two full five-fake/one-student cycles with warm optimizer state.
  3. Fresh formal initialization from pinned step300, next-update restore audit,
     then 500/2500 updates. Capacity/audit steps do not count toward that budget.
Failure never automatically reduces sequence length, rank or gradient reach.
H100 capacity has not yet been measured by this code delivery.

Monitor (set RUN to the path printed by the launcher):

  cat "$RUN/status.json"
  tail -n 40 "$RUN/cuda-tests.log"
  tail -n 40 "$RUN/capacity.log"
  tail -F "$RUN/training.log"

status.json names the phase, PID and exit code. Training metrics live in
$RUN/training/metrics-rank0.jsonl, with student and critic counters, per-update
time, losses, DMD direction and denominator, group gradients, GPU/RAM, and
every25 student updates S spectra/output effects plus actual parameter changes.
Camera Q/K/V and GDN beta/decay/output gate projections call adapter wrappers;
capacity rejects any selected LoRA tensor without a gradient. Zero-initialized
LoRA A gradients may legitimately be zero while B gradients are nonzero.
W&B uses WANDB_API_KEY from this process environment; without it, offline logs
are retained. Do not use shared wandb login. It logs separate student/critic axes.

Resume with the same code/environment and pinned identity:

  bash tools/distillation/run_h100.sh --resume "$RUN"

An existing valid capacity proof skips repeated capacity work. Recovery resumes
at the most recent complete25-student-update bundle. Two recent adapter-only
bundles and best-adapters.pt are retained under $RUN/training; the teacher/base
are referenced, not duplicated in every checkpoint. Ctrl+C stops only this
run's named scope and records interrupted status. If the supervisor itself was
killed or the host rebooted, inspect its PID/scope before clearing a stale
runner.lock; it is never removed on assumption alone.

Validation compares teacher20, undistilled8 and EMA student8 on a fixed
video-disjoint distillation validation group. It reports early/middle/tail
latent MSE and decoded PSNR/SSIM/LPIPS plus speed. Lowest LPIPS selects best;
tail latent MSE breaks ties. Teacher exposure to these videos is unknown.
Evaluation restores RNG so it does not alter the next training noise sequence.

After formal completion, generate the fixed30/60s qualitative and TTT-off review:

  CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --nnodes=1 \
    --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
    -m tools.distillation.review_videos --run "$RUN"

The videos and manifest are in $RUN/training/review-videos. Four-step training is never
automatic: use the existing prepare_stage4 tool after actual video review and
the numerical/speed gates (its --eight-run is $RUN/training), then pass its prepared file with --config to a NEW
run_h100 invocation. The teacher stays step300; optimizer/EMA/counts reset.
