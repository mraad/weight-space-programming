#!/usr/bin/env bash
# Run once on gc1 inside tmux, from the repo copy. The NVMe disk is ephemeral: a stop of the instance erases all of this.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HUB_CACHE=/opt/dlami/nvme/hf-hub HF_XET_HIGH_PERFORMANCE=1
WSP=/opt/dlami/nvme/wsp
cd "$WSP/weight-space-programming"

(
  set -e  # a failed step must not reach SETUP-DONE
  uv venv --allow-existing --python 3.12 "$WSP/vllm-venv"
  VIRTUAL_ENV="$WSP/vllm-venv" uv pip install 'vllm>=0.25' ninja  # FlashInfer 0.7 has no prebuilt kernel package, so it compiles them with ninja and nvcc
  "$WSP/vllm-venv/bin/vllm" --version
  uv sync
  .venv/bin/python wsp.py fetch | tail -1
  echo SETUP-DONE
) 2>&1 | tee "$WSP/logs/setup.log" &

(
  hf download Qwen/Qwen3.8-27B
  echo DOWNLOAD-DONE
) 2>&1 | tee "$WSP/logs/download.log" &

wait
