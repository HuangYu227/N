# Step25 static State-TTT on/off video

Use the immutable `models/step-00000025.pt` checkpoint and its verified JSON
receipt, even while training continues beyond step25. The mutable `last.pt`
and the latest training record are not used to select weights. Receipt ID,
checkpoint format/stage/architecture, base SHA and parameter names/shapes must
match. The milestone file is SHA256-checked again after inference.

This generates two **TTN** videos sequentially: `no-ttt` and `full`. Local and
clean additive State-TTT updates are both disabled in `no-ttt`; Correct/Read,
the camera-conditioned Cayley transport, inherited GDN and native camera cache
remain. Both methods use the same saved weights, first image, encoded prompt,
camera, initial noise, CFG4.5, 20 solver steps and two cached camera groups.
The runtime is reset per rollout. No SANA video, new training, refiner, Sink,
parameter cast to BF16 or inference FP8 conversion is added. Saved model/core
weights remain FP32 with the existing BF16 inference autocast.

The original study image/static prompt and virtual intrinsics are loaded from
`assets/worldttn/study_static/case.json`, including its image SHA audit. Length:
61 latent time positions / 481 raw frames / 16 FPS, approximately 30 seconds,
704x1280. Decoder tiling uses the native VAE implementation. Decode happens
after the DiT is freed, never with two models resident together.

## GPU5 budget

The user's ceiling is **23,000 MiB for this generation PID**, separate from
the already-running training PID on the same physical GPU5.

- Native PyTorch caching allocator cap: **18,000 MiB**, set before VAE/model work.
- Remaining **5,000 MiB** is conservatively left for CUDA context/library usage.
- Before starting: device free memory must exceed **25,048 MiB** (23,000 plus
  another 2,048 MiB of headroom for the training process).
- `nvidia-smi` samples only this generation PID every 0.25 seconds. At **21,000
  MiB**, or if the query fails, the generation process exits with code75. It
  never signals another PID. Query errors are recorded in `memory.json`.
- PyTorch allocation failure exits the run; the tool does not increase the cap,
  retry with a different algorithm, shorten the video or change resolution.

**This is not a driver-level memory partition.** The hard allocator cap only
covers PyTorch allocations; sampled PID monitoring can miss a transient spike.
CUDA library allocations or another job's growth cannot be guaranteed away by
this script. The reserve and lower stop threshold reduce the risk but do not
prove that `nvidia-smi` usage can never briefly exceed23,000 MiB. A strict total
hardware-enforced ceiling requires an appropriate isolated GPU/MIG resource;
this tool does not reconfigure a live training GPU. If strict isolation is
mandatory, do not run concurrently on the training card.

## Run in another tmux window

```bash
bash <<'BASH'
set -euo pipefail
ROOT=/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/yuhuang
source "$ROOT/activate-worldttn-fp8.sh"
cd "$ROOT/WorldTTN-dual-psi-fp8"
git fetch origin refs/heads/codex/ttn-dual-psi-fp8:refs/remotes/origin/codex/ttn-dual-psi-fp8
git merge --ff-only origin/codex/ttn-dual-psi-fp8

RUN="$PWD/output/state-ttt-h100-gpu5-20261014T084949Z"
OUT="$PWD/Videos/state-ttt-step25-static-30s-$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES=5
export DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=backend:native
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export CUDA_MODULE_LOADING=LAZY
mkdir -p "$(dirname "$OUT")"
printf 'Video output: %s\n' "$OUT"
"$ROOT/envs/worldttn-fp8/bin/python" -u -m tools.ttn_state_ttt_video \
  --training-run "$RUN" --step 25 --output "$OUT" \
  --steps 20 --cfg-scale 4.5 2>&1 | tee "$OUT.launch.log"
BASH
```

The source update adds only inference tools/tests/documentation, not training
code. The foreground command runs in your tmux pane; its pipe returns a failure
if Python fails. A repeated run requires a new output directory. If admission
fails because free GPU memory is too low, preserve the training run and try
later or on another isolated GPU; do not raise this script's budget.

Outputs:

- `videos/no-ttt.mp4`: TTT disabled, left half of comparison.
- `videos/full.mp4`: trained State-TTT enabled, right half of comparison.
- `videos/comparison.mp4`: labeled side-by-side video.
- `no-ttt.pt`, `full.pt`: generated latents.
- `summary.json`: checkpoint SHA, protocol, completion/error and episode timing.
- `episodes.jsonl`: paired conditioning/noise hashes, chunk timings and commits.
- `memory.json`: allocator limit, PID threshold and **sampled** process peak.
- `input-bundle.pt`, `prompt.txt`, `first_frame_used.png`: reproducible inputs.

The script has CPU contract/math regression coverage. Complete H100 GPU/VAE
execution and memory peaks have not been tested locally. If a budget stops the
run, it is not a completed video-quality experiment. A single on/off case can
probe update influence at these fixed weights, not establish general quality
or long-horizon superiority. It is an inference ablation of a model trained
with State-TTT, not a separately trained no-TTT baseline.
