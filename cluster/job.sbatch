#!/usr/bin/env bash
#SBATCH --partition=general-gpu-long
#SBATCH --account=engr-lab-netanel.raviv
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Usage (via remote.sh submit/sweep): sbatch [overrides] job.sbatch <python module> [args...]
# In an array job, line SLURM_ARRAY_TASK_ID of $SWEEP_FILE is appended to the args. With
# PER_GPU=k, task i instead runs lines i*k .. i*k+k-1 side by side on its one GPU, each
# logging to $FBRL/logs/<job name>-<job id>-<slot>.log.
set -euo pipefail
source /engrfs/project/netanel.raviv/motti_k/fbrl/code/cluster/env.sh
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

echo "job $SLURM_JOB_ID on $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The image's LD_LIBRARY_PATH points at its own torch build; the venv brings its own, so
# drop those entries to keep the two from mixing. The image also sets MASTER_PORT=53394,
# which collides when two runs share a node.
run() {  # run <slot> <python module> [args...]
	SLOT=$1 apptainer exec --nv --bind /engrfs:/engrfs,/scratch:/scratch "$IMAGE" bash -c '
		export LD_LIBRARY_PATH=$(tr ":" "\n" <<<"${LD_LIBRARY_PATH:-}" | grep -v -e torch -e dist-packages | paste -sd:)
		export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000) * 4 + SLOT))
		cd "$FBRL/code" && exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m "$@"
	' _ "${@:2}"
}

if [[ -z "${SWEEP_FILE:-}" ]]; then
	echo "$*"
	run 0 "$@"
	exit
fi

per_gpu=${PER_GPU:-1}
pids=()
for ((slot = 0; slot < per_gpu; slot++)); do
	line=$((SLURM_ARRAY_TASK_ID * per_gpu + slot + 1))
	[[ $line -le $(wc -l <"$SWEEP_FILE") ]] || break
	eval "extra=($(sed -n "${line}p" "$SWEEP_FILE"))"
	echo "slot $slot: $* ${extra[*]}"
	if ((per_gpu == 1)); then
		run 0 "$@" "${extra[@]}"
		exit
	fi
	# vLLM sizes its memory from what's free at startup, so let each run settle first
	(sleep $((slot * 180)); run "$slot" "$@" "${extra[@]}") \
		>"$FBRL/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID-$slot.log" 2>&1 &
	pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
