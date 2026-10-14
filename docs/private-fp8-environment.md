# Private dual-psi FP8 workspace

This branch preserves Local/Persistent psi and the existing TLA update. It does
not enable the separate rank64 State-TTT architecture. FP8 is selective GEMM
training; the recurrent core, master parameters and Adam state retain their
existing precision. The full-gradient recipe uses 121 latents and TBPTT=0.

Clone this branch inside a workspace you may write, then run:

```bash
bash WorldTTN-dual-psi-fp8/tools/ttn_setup_private_env.sh /path/to/private/workspace
source /path/to/private/workspace/activate-worldttn-fp8.sh
```

The Linux x86_64 script installs a checksum-verified Miniforge25.3.1-0 under
`toolchain/miniforge3`, and Python3.11 under `envs/worldttn-fp8`. It never runs
sudo, conda init, or installs into a shared environment. Existing target
environments are rejected rather than reconciled. Package/model/compiler caches
and temporary build files stay under the supplied workspace; build thread counts
default to2. The installer hides CUDA devices during imports and does not launch
training or download pretrained weights/datasets.

PyTorch2.9.1+cu128, TorchAO0.14.1 and the project's dependencies are installed
into that prefix. Existing FLA/MMCV model/data imports and the FP8 backend are
checked on CPU. Setup logs and pip freeze are saved under `logs`.

Installing the environment does not reserve GPUs or guarantee host RAM safety.
Before training, inspect the GPU model/driver, permitted GPU IDs, host RAM and
other workloads. Native FP8 requires a supported GPU; do not treat an A100 as an
FP8 device. Full-episode offloaded activations still need substantial host RAM.
Do not concurrently start training merely because model imports passed.

For hardware inspection without a training run:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free --format=csv
free -h
```
