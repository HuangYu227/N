#!/usr/bin/env bash
# Install into an explicit workspace; no sudo, shell-profile edits or GPU work.
set -euo pipefail
umask 077
workspace_root="$(realpath -m "${1:?Usage: bash tools/ttn_setup_private_env.sh /absolute/workspace}")"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || {
  echo 'This validated installer requires Linux x86_64.' >&2; exit 2;
}
[[ "$workspace_root" != / && "$project_root" == "$workspace_root/"* ]] || {
  echo 'Clone the project inside the specified workspace first.' >&2; exit 2;
}
env_prefix="$workspace_root/envs/worldttn-fp8"
[[ ! -e "$env_prefix" ]] || {
  echo "Environment already exists; leaving it untouched: $env_prefix" >&2; exit 2;
}
mkdir -p "$workspace_root"/{envs,toolchain,logs,tmp} "$workspace_root/.cache"/{conda,pip,huggingface,torch,triton,cuda,extensions,inductor}
export XDG_CACHE_HOME="$workspace_root/.cache"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda"
export CONDA_ENVS_PATH="$workspace_root/envs"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_CONFIG_FILE=/dev/null PIP_USER=false
unset PIP_TARGET PIP_PREFIX PYTHONPATH PYTHONHOME
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TMPDIR="$workspace_root/tmp"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
log="$workspace_root/logs/setup-fp8-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
trap 'code=$?; echo "Setup failed (exit $code); keep the log: $log" >&2; exit "$code"' ERR

# A workspace-owned distribution avoids changing another person's Conda base.
forge="$workspace_root/toolchain/miniforge3"
if [[ ! -x "$forge/bin/conda" ]]; then
  [[ ! -e "$forge" ]] || { echo "Incomplete toolchain exists: $forge" >&2; exit 2; }
  installer="$workspace_root/toolchain/Miniforge3-25.3.1-0-Linux-x86_64.sh"
  curl --fail --location --retry 3 --connect-timeout 20 \
    'https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh' \
    --output "$installer"
  printf '%s  %s\n' '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768' "$installer" | sha256sum --check -
  bash "$installer" -b -p "$forge"
fi
"$forge/bin/conda" create --yes --prefix "$env_prefix" --override-channels -c conda-forge python=3.11 pip
source "$forge/etc/profile.d/conda.sh"
conda activate "$env_prefix"
python -c 'import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(); print("Private Python:", sys.executable)' "$env_prefix"
python -m pip install --upgrade pip 'setuptools<80' wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 xformers==0.0.33.post2
python -m pip install --no-build-isolation mmcv==1.7.2
cd "$project_root"
python -m pip install --editable . --requirement requirements-fp8.txt pytest
python -m pip check
python - <<'PY'
import json
from pathlib import Path
import torch, torchao, pyrallis, fla, mmcv
from torch.distributed.fsdp import fully_shard
from worldttn.sana import load_sana_config
from worldttn.fp8 import _load_backend
from worldttn.core import TTNConfig
from worldttn.state_ttt import StateTTT
from train_video_scripts.train_sana_wm_stage1 import _build_dataloader
settings = json.loads(Path('configs/worldttn/dual_psi_fullgrad_fp8.json').read_text())
assert settings['training_precision'] == 'fp8' and settings['tbptt'] == 0
assert settings['ttn']['local_update'] and settings['ttn']['persistent_meta']
assert 'state_ttt_rank' not in settings['ttn']
enhanced = json.loads(Path('configs/worldttn/state_ttt_fp8.json').read_text())
assert enhanced['training_precision'] == 'fp8' and enhanced['tbptt'] == 0
state_config = TTNConfig(**enhanced['ttn'])
assert state_config.state_ttt_rank == 64 and state_config.state_ttt_policy == 'curvature'
assert state_config.state_ttt_gain == 0.5 and state_config.eta_psi == 0
load_sana_config(settings['sana_config'])
backend = _load_backend()
assert backend.version == '0.14.1'
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '12.8'
assert not torch.cuda.is_initialized()
print('PASS: legacy dual-psi and enhanced State-TTT FP8 configs, model/data imports, TorchAO and FSDP2.')
print('No GPU training or model download was started.')
PY
activate="$workspace_root/activate-worldttn-fp8.sh"
[[ ! -e "$activate" ]] || { echo "Activation file exists; leaving it untouched: $activate" >&2; exit 2; }
{
  printf '#!/usr/bin/env bash\nsource %q\nconda activate %q\n' "$forge/etc/profile.d/conda.sh" "$env_prefix"
  for name in XDG_CACHE_HOME CONDA_PKGS_DIRS CONDA_ENVS_PATH PIP_CACHE_DIR HF_HOME TORCH_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH TORCH_EXTENSIONS_DIR TORCHINDUCTOR_CACHE_DIR TMPDIR; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
  printf 'unset PIP_TARGET PIP_PREFIX PYTHONPATH PYTHONHOME\n'
  printf 'export CONDA_REGISTER_ENVS=false PIP_CONFIG_FILE=/dev/null PIP_USER=false\n'
  printf 'export PYTHONNOUSERSITE=1 DISABLE_XFORMERS=1 GDN_DISABLE_COMPILE=1\n'
  printf 'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2\n'
  printf 'cd %q\n' "$project_root"
} > "$activate"
python -m pip freeze > "$workspace_root/logs/environment-fp8-freeze.txt"
printf '\nReady. Activate with:\nsource %q\nLog: %s\n' "$activate" "$log"
