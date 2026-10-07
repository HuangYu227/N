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

## Closed orbit returning to the observed view

`case_orbit.json` reuses the same first frame and seed, with `prompt_orbit.txt`
describing static objects and a moving camera. The explicit OpenCV C2W path
translates around a horizontal circle of radius 0.6 virtual scene units while
looking inward at its fixed centre, initially `[0,0,0.6]`. It holds the initial
view for 24 raw-frame intervals (1.5 s), completes a smooth 360-degree orbit
at raw frame 432 (27 s), and holds the identical position/orientation/intrinsics
for the final 48 intervals (3 s). Duration is 481/16 = 30.0625 s, with 61 latents
and 20 predicted chunks. This is a synthetic trajectory, not a reconstruction
of the room's dimensions or a guarantee that the circle clears its furniture.

After the same step100 snapshot setup above, submit a separate output:

```bash
export CUSTOM_CASE=assets/worldttn/study_static/case_orbit.json
export CUSTOM_OUTPUT="$RUN/custom-study-orbit-step100-$(date -u +%Y%m%dT%H%M%SZ)"
ORBIT_JOB=$(sbatch --parsable tools/ttn_slurm_custom_video.sbatch)
ORBIT_JOB="${ORBIT_JOB%%;*}"
echo "ORBIT_JOB=$ORBIT_JOB"
echo "ORBIT_OUTPUT=$CUSTOM_OUTPUT"
```

`camera_poses.npy` records all 481 C2W matrices. Episodes include initial-view
latent MSE over returned latents 55..60: their complete eight-frame Plucker
intervals match the initial view exactly; the still-moving closing interval
of latent 54 is excluded. `videos/return-view.json` records decoded RGB MSE
over raw frames 433..480 against the decoded observed frame 0, before MP4
compression. `videos/return-comparison.png` shows initial/last views, SANA on
the upper row and TTN on the lower row. Existing paired MP4s are also produced.
These are consistency scores, not future-GT quality metrics. A video ignoring
the camera can also obtain low return error: inspect the intervening motion
and scene geometry alongside these scores. No model weights or training
settings are changed.
