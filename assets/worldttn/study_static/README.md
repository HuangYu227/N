# Custom study: stationary-camera 30-second comparison

`first_frame.png` is the user's newly AI-generated RGB image (1348x742, 1.7 MB).
`prompt.txt` describes that image. `case.json` specifies the image SHA256, seed
3407, a stationary trajectory, 61 latent frames / 481 raw frames / 16 fps.
The virtual pinhole intrinsics `[900, 900, 640, 352]` are specified on the
704x1280 cropped image grid. They are assumed, not measured from the image.

Run original SANA and the trained TTN sequentially with the same image latent,
text embeddings/mask, camera/Plucker tensors, and noise. The command restores
the checkpoint's model/text/scheduler settings, uses reference/reference,
20 Euler steps, CFG 4.5, and the native two-chunk cache window. It reads no
training video or future GT. No optimizer or model checkpoint is created.

Use the existing immutable step100 snapshot, recorded in the long evaluation's
`summary.json` at `protocol.training_run`; do not point this test at the live
training run's changing `last.pt`.

```bash
export ROOT=/data/group/zhaolab/home/z2zhang/huangyu
export PROJECT_ROOT="$ROOT/WorldTTN-dual-psi-meta"
cd "$PROJECT_ROOT"
RUN="$PROJECT_ROOT/output/worldttn/dual-meta-C-423191-to500"
export CUSTOM_TRAINING_RUN="$("$ROOT/envs/worldttn/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["protocol"]["training_run"])' \
  "$RUN/evaluations/step-000100/long/summary.json")"
export CUSTOM_OUTPUT="$RUN/custom-study-step100-$(date -u +%Y%m%dT%H%M%SZ)"
CUSTOM_JOB=$(sbatch --parsable tools/ttn_slurm_custom_video.sbatch)
CUSTOM_JOB="${CUSTOM_JOB%%;*}"
echo "CUSTOM_JOB=$CUSTOM_JOB"
echo "CUSTOM_OUTPUT=$CUSTOM_OUTPUT"
```

The single-GPU job requests the `day` partition, 128 GB RAM, and a two-hour
maximum; actual runtime depends on the allocated node. Permanent inputs and
outputs stay on NFS; compiler scratch uses the existing worker's private `/tmp`.

Outputs: the actual cropped first frame, prompt, input bundle, manifest,
per-method/per-chunk sampling logs and latent files, and `videos/sana.mp4`,
`videos/ttn.mp4`, `videos/comparison.mp4` plus decode metadata. The comparison
labels original SANA on the left and the trained TTN/checkpoint step on the right.

There is no ground-truth future video: no GT latent MSE, PSNR, SSIM, or LPIPS
is reported. This is a qualitative synthetic-input test, not a held-out video
benchmark. The training mathematical implementation and history protocol are
unchanged.
