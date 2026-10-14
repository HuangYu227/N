#!/usr/bin/env bash
# Run in an existing tmux pane; touch only the requested GPU and output.
set -euo pipefail
gpu="${1:?Usage: bash tools/ttn_train_state_ttt_h100.sh GPU_INDEX DATASET_ROOT OUTPUT_DIR [--resume]}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX must be one physical GPU index.' >&2; exit 2; }
dataset_root="$(realpath -e "${2:?Dataset root is required}")"
output="$(realpath -m "${3:?A new output directory is required}")"
mode="${4:-}"
[[ $# -le 4 && ( -z "$mode" || "$mode" == --resume ) ]] || { echo 'Only --resume is supported as the fourth argument.' >&2; exit 2; }
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
args=()
if [[ "$mode" == --resume ]]; then
  [[ -f "$output/last.pt" ]] || { echo 'Resume requires OUTPUT_DIR/last.pt.' >&2; exit 2; }
  args+=(--adapter "$output/last.pt" --resume)
else
  [[ ! -e "$output" ]] || { echo 'Fresh training requires a new output directory.' >&2; exit 2; }
fi
[[ -d "$dataset_root/data/sekai_game_train_961frames_16fps_ovl640" &&
   -d "$dataset_root/data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640" ]] || {
  echo 'Dataset root must contain the raw ZIP dataset and its LTX-2 latent cache.' >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="$gpu" DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TTN_ACTIVATION_GPU_CACHE_GIB="${TTN_ACTIVATION_GPU_CACHE_GIB:-2}"
export TTN_ACTIVATION_GPU_ALLOCATION_LIMIT_GIB="${TTN_ACTIVATION_GPU_ALLOCATION_LIMIT_GIB:-60}"
python_bin="${PYTHON:-python}"
"$python_bin" - <<'PY'
import torch
from worldttn.fp8 import _load_backend
if torch.cuda.device_count() != 1:
    raise RuntimeError('Exactly one selected GPU must be visible.')
name = torch.cuda.get_device_name(0)
if 'H100' not in name:
    raise RuntimeError(f'This launcher is for H100; selected GPU is {name}.')
backend = _load_backend()
free, total = torch.cuda.mem_get_info()
print(f'H100 preflight: {name}; free={free/2**30:.2f} GiB; total={total/2**30:.2f} GiB; TorchAO={backend.version}', flush=True)
print('Full-model single-H100 capacity still requires measurement during this run.', flush=True)
PY
mkdir -p "$output"
set +e
"$python_bin" -u -m worldttn.cli train \
  --config configs/worldttn/state_ttt_fp8.json \
  --dataset-root "$dataset_root" --output "$output" \
  --train-scope dit --optimizer-policy origin --backbone-lr 1e-6 \
  --parallel single --train-precision fp8 --activation-offload cpu-pageable \
  --ttn-core-backend reference --ttn-psi-backend reference \
  --cross-attn-backend math --text-encoder-device cpu \
  --tbptt 0 --frames 121 --max-steps 500 --save-every 25 \
  --memory-trace "${args[@]}" 2>&1 | tee -a "$output/train.log"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[0]}" > "$output/train.exit-code"
[[ "${codes[0]}" -eq 0 ]] || exit "${codes[0]}"
exit "${codes[1]}"
