#!/usr/bin/env bash
# One-time (and after dependency changes): install uv and sync the venv on
# project storage. Runs on the login node; jobs use the venv from inside the
# container, which sees the same /engrfs paths.
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$ROOT/bin" "$FBRL"/{runs,logs,wandb}
if ! command -v uv >/dev/null; then
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
cd "$FBRL/code"
uv sync --frozen --python 3.12
uv run --frozen python -c "import trl, vllm, torch; print('trl', trl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__)"
