#!/usr/bin/env bash
# Source after activating the existing prefix environment. ROOT defaults to
# the shared parent of the checkout; inherited home-cache overrides are reset.
export ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT="${ROOT%/}"
export XDG_CACHE_HOME="$ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
if [[ -v TRANSFORMERS_CACHE ]]; then export TRANSFORMERS_CACHE="$HF_HUB_CACHE"; fi
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" \
  "$HF_ASSETS_CACHE" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
