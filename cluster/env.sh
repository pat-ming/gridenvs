# Sourced by setup.sh and job.sbatch. $HOME is quota-limited (30G), so every
# cache and output goes to project storage; ~/.netrc still supplies W&B auth.

export ROOT=/engrfs/project/netanel.raviv/motti_k
export FBRL=$ROOT/fbrl
export IMAGE=/engrfs/cluster/containers/nvcr.pytorch-25.11.sif

export PATH=$ROOT/bin:$PATH
export UV_CACHE_DIR=$ROOT/uv_cache
export UV_PYTHON_INSTALL_DIR=$ROOT/uv_python
export UV_PROJECT_ENVIRONMENT=$FBRL/venv
export UV_LINK_MODE=copy

export HARNESS_RUNS_DIR=$FBRL/runs
export HF_HOME=$ROOT/hf_cache
# $ROOT is readable by the lab group, so the token lives in $HOME (mode 600).
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export XDG_CACHE_HOME=$ROOT/xdg_cache
export WANDB_DIR=$FBRL/wandb
export WANDB_CACHE_DIR=$ROOT/xdg_cache/wandb
export WANDB_DATA_DIR=$ROOT/xdg_cache/wandb-data
export WANDB_ENTITY=${WANDB_ENTITY:-cse-5100-fall-2026}
export WANDB_PROJECT=${WANDB_PROJECT:-fbrl}

# JIT/compile caches race on BeeGFS when several workers write at once
# (EBUSY on rename), so they live on node-local /scratch (/tmp on the login node).
if mkdir -p "/scratch/$USER" 2>/dev/null; then LOCAL_CACHE=/scratch/$USER/fbrl_cache; else LOCAL_CACHE=/tmp/$USER-fbrl_cache; fi
export TRITON_CACHE_DIR=$LOCAL_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_CACHE/inductor
export VLLM_CACHE_ROOT=$LOCAL_CACHE/vllm
export FLASHINFER_WORKSPACE_BASE=$LOCAL_CACHE/flashinfer

# The image's ldconfig cache points libcuda at a compat dir that doesn't exist;
# apptainer --nv mounts the host driver here instead.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

export SLURM_CONF=/project/compute/slurm/etc/slurm.conf
