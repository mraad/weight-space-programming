#!/usr/bin/env bash
# Run once on gc1. The NVMe disk is ephemeral: a stop of the instance erases this venv and all data.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV=/opt/dlami/nvme/wsp/geo-venv
uv venv --allow-existing --python 3.12 "$VENV"
# The README results used these torch and smp versions. They fix the network's layout. The other versions were not recorded.
# --no-config: pyproject.toml overrides remove scipy and pandas for MinAtar, and this venv needs them.
VIRTUAL_ENV="$VENV" uv pip install --no-config --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision
VIRTUAL_ENV="$VENV" uv pip install --no-config segmentation-models-pytorch==0.6.0 rasterio geopandas pyogrio scipy
"$VENV/bin/python" -c "import torch, segmentation_models_pytorch as s; print(torch.__version__, torch.cuda.device_count(), s.__version__)"
echo SETUP-DONE
