# H100 checkpoint diagnostics under a 25,000 MiB request

`tools/ttn_checkpoint_eval.py` evaluates existing immutable milestone models, without changing training or its algorithm. The default is steps 100 and 250. Preparation, each checkpoint, and VAE decoding run in separate sequential subprocesses. One parent job spans all phases.

Every CUDA subprocess uses the existing native-allocator/PID guard: a 20,000 MiB PyTorch allocator limit, a 22,500 MiB own-process stop threshold sampled every 0.25 s, and a startup requirement of 27,048 MiB free. It also exits when global free GPU memory falls below 4,096 MiB. Missing PID accounting causes refusal or exit. Only the evaluation subprocess terminates. These software checks are not a driver-level 25,000 MiB memory partition; non-PyTorch allocations and transient spikes can escape sampling. Concurrent unrelated GPU workloads can grow. No automatic limit relaxation or retry is performed.

The two fixed dataset cases are forest road `00100100001_0004650_0006450_s000000` and Japanese street `00300200002_0048306_0050106_s000000`. Their future GT latent sequences remain on CPU and outside model inputs. The original static user study is a third case, with no future GT. Text, empty-text embedding, mask, first latent, camera, Plucker and CUDA noise are prepared once, saved and hash checked for every checkpoint. Original prompts are retained, including the static-camera wording in the dataset captions.

Each checkpoint generates three 481-frame/16fps videos with 61 latents, 20 predicted chunks, 20 solver steps, CFG4.5 and two cached blocks. The saved FP32 DiT parameters are used with native BF16 inference autocast; inference is not converted to FP8. No sink, refiner, training, optimizer or checkpoint selection policy is changed. Hook counts verify full enhanced StateTTT execution.

Outputs:

- `step-XXXXXXXX/summary.json`: latent MSE and its time windows, StateTTT call counts and sampler timing. Static study is explicitly labeled observed-frame drift rather than GT prediction error.
- `step-XXXXXXXX/case-XXX.mp4` and `.pt`: videos and generated latents.
- `decoded-metrics.json`: RGB MSE, PSNR and SSIM on raw frames 8,16,...,480 (2fps, excluding the observation), evaluated at 352x640 after antialiased bilinear resizing. Frames are decoded with the same native training VAE. Dataset pixel reference is a GT-latent VAE reconstruction, not raw original RGB. SSIM uses an 11x11 Gaussian with sigma1.5 and population covariance, valid spatial support; PSNR uses a 1e-12 MSE floor (120dB maximum). Windows are (0,10], (10,20], (20,30] seconds.
- `report.md`: the final 20–30 s metric table.
- `memory-*/memory.json`: per-phase sampled process peaks and stop/failure state.
- `status.json`: parent success/failure, limits and scope.

LPIPS, VBench and generated-camera-pose accuracy are not implemented in this utility and are explicitly marked missing. It adds no dependencies and does not auto-download additional scoring models. MSE/PSNR/SSIM can reward blur or lack of motion, so they do not establish camera adherence or overall perceptual quality. These are known training-data cases with one seed per case, not held-out generalization or independent per-frame statistical samples. No `best.pt` is created and no weights are deleted. Step200 is not a retained H100 milestone; its L20 copy must be evaluated using the same inputs before claiming a paired 200/250 result.

Run inside a chosen tmux window, with only physical GPU4 exposed and the native allocator selected. Use `flock -n /path/to/.eval-gpu4.lock` to prevent accidentally starting this evaluator twice. An occupied GPU in Exclusive_Process mode cannot be shared; do not automatically change its compute mode. A memory refusal or CUDA OOM leaves a failed status and should be inspected rather than worked around by raising the limit.

The script can live outside the training checkout: set `TTN_EVAL_REPO` to the existing repository. This permits fetching and extracting only the evaluator without checking out a branch, merging, or overwriting training source files. `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1` and `SANA_CP_SIZE=1` refer only to this evaluation shell.
