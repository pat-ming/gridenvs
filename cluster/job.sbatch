#!/usr/bin/env bash
#SBATCH --partition=general-gpu-long
#SBATCH --account=engr-lab-netanel.raviv
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Usage (via remote.sh submit/sweep): sbatch [overrides] job.sbatch <python module> [args...]
# In an array job, line SLURM_ARRAY_TASK_ID of $SWEEP_FILE is appended to the args.
set -euo pipefail
source /engrfs/project/netanel.raviv/motti_k/fbrl/code/cluster/env.sh
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

extra=()
if [[ -n "${SWEEP_FILE:-}" ]]; then
	eval "extra=($(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SWEEP_FILE"))"
fi
set -- "$@" "${extra[@]}"

echo "job $SLURM_JOB_ID on $SLURMD_NODENAME: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The image's LD_LIBRARY_PATH points at its own torch build; the venv brings
# its own, so drop those entries to keep the two from mixing.
apptainer exec --nv --bind /engrfs:/engrfs,/scratch:/scratch "$IMAGE" bash -c '
	export LD_LIBRARY_PATH=$(tr ":" "\n" <<<"${LD_LIBRARY_PATH:-}" | grep -v -e torch -e dist-packages | paste -sd:)
	cd "$FBRL/code" && exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m "$@"
' _ "$@"
