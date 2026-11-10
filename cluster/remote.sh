#!/usr/bin/env bash
# Drive shell.engr.wustl.edu from the laptop: sync code, submit batch jobs,
# watch runs, and hold interactive GPU sessions in remote tmux (no SSH
# connection needs to stay open). Adapted from raviv_lab/tools/remote_gpu.sh.
#
#   cluster/remote.sh sync                       # rsync project/ to $FBRL/code (+ GIT_STATE)
#   cluster/remote.sh setup                      # install uv, sync the venv (after dep changes)
#   cluster/remote.sh submit [sbatch opts] -- <module> [args...]
#                                                # e.g. submit -- fbrl.train_grpo --set env.condition=C
#                                                #      submit --time=2-00:00:00 -- fbrl.train_grpo ...
#   cluster/remote.sh sweep [sbatch opts] -- <module> [args...] --grid k=v1,v2 [--grid ...]
#                                                # Slurm array over the grid (see `harness sweep`); the
#                                                # sweep name ($SWEEP_NAME or module_time) is the W&B group
#   cluster/remote.sh pull                       # copy run dirs to results/runs/, minus checkpoints,
#                                                # W&B dirs and per-episode records (those stay on the cluster)
#   cluster/remote.sh logs <jobid> [-f]          # job stdout/stderr
#   cluster/remote.sh harness <args...>          # harness CLI on the cluster runs dir (ls, tail RUN -f, compare ...)
#   cluster/remote.sh sh '<command>'             # one-off command with env.sh sourced, in $FBRL/code
#   cluster/remote.sh jobs | cancel <jobid>
#
#   cluster/remote.sh gpu <session> [gres] [time] [partition] [account]
#                                                # interactive container shell on a GPU, in tmux;
#                                                # defaults: a100-sxm4:1, 8h, general-gpu-long, lab account
#   cluster/remote.sh send <session> <command>   # type into it, wait, print pane
#   cluster/remote.sh cap <session> | ls | kill <session>
#
# REMOTE_HOST overrides shell.engr.wustl.edu.

set -euo pipefail

HOST="${REMOTE_HOST:-shell.engr.wustl.edu}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o RemoteCommand=none -o RequestTTY=no)
FBRL=/engrfs/project/netanel.raviv/motti_k/fbrl
ENV="source $FBRL/code/cluster/env.sh"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

remote() {
	ssh "${SSH_OPTS[@]}" "$HOST" "$1"
}

cmd="${1:-}"
shift || true
case "$cmd" in
sync)
	(cd "$HERE" && git describe --always --dirty 2>/dev/null || echo "no-git $(date -u +%FT%TZ)") >"$HERE/GIT_STATE"
	remote "mkdir -p $FBRL/code"
	rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
		--exclude .git --exclude context.txt --exclude .venv --exclude runs --exclude wandb --exclude __pycache__ \
		--exclude .latex_aux --exclude '*.pdf' --exclude '*.tex' --exclude .pytest_cache --exclude .ruff_cache \
		"$HERE/" "$HOST:$FBRL/code/"
	echo "synced $(cat "$HERE/GIT_STATE") -> $HOST:$FBRL/code"
	;;
setup)
	remote "bash $FBRL/code/cluster/setup.sh"
	;;
submit)
	opts=()
	while [[ $# -gt 0 && "$1" != "--" ]]; do opts+=("$1"); shift; done
	shift || { echo "usage: submit [sbatch opts] -- <module> [args...]" >&2; exit 1; }
	ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "${opts[@]}" -- "$@" <<REMOTE
$ENV
opts=(); while [[ \$1 != "--" ]]; do opts+=("\$1"); shift; done; shift
sbatch --parsable --job-name="\$1" --output="$FBRL/logs/%x-%j.out" "\${opts[@]}" $FBRL/code/cluster/job.sbatch "\$@"
REMOTE
	;;
sweep)
	opts=()
	while [[ $# -gt 0 && "$1" != "--" ]]; do opts+=("$1"); shift; done
	shift || { echo "usage: sweep [sbatch opts] -- <module> [args...] --grid k=v1,v2" >&2; exit 1; }
	module="$1"; shift
	fixed=(); grids=()
	while [[ $# -gt 0 ]]; do
		if [[ "$1" == --grid ]]; then grids+=(--grid "$2"); shift 2; else fixed+=("$1"); shift; fi
	done
	name="${SWEEP_NAME:-${module##*.}_$(date +%m%d_%H%M)}"
	lines="$(cd "$HERE" && uv run --quiet harness sweep "${grids[@]}" 2>/dev/null)"
	n=$(wc -l <<<"$lines" | tr -d ' ')
	file="$FBRL/sweeps/$name.txt"
	remote "mkdir -p $FBRL/sweeps && cat > $file" <<<"$lines"
	echo "$name: $n runs ($file)"
	ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "${opts[@]}" -- "$module" "${fixed[@]}" --set "tracking.wandb.group=$name" <<REMOTE
$ENV
opts=(); while [[ \$1 != "--" ]]; do opts+=("\$1"); shift; done; shift
sbatch --parsable --array=0-$((n - 1)) --job-name="$name" --output="$FBRL/logs/%x-%A_%a.out" \\
	--export=ALL,SWEEP_FILE="$file" "\${opts[@]}" $FBRL/code/cluster/job.sbatch "\$@"
REMOTE
	;;
pull)
	rsync -az -e "ssh ${SSH_OPTS[*]}" \
		--exclude checkpoints --exclude wandb --exclude profiles \
		--exclude eval_episodes.jsonl --exclude rollouts.jsonl \
		"$HOST:$FBRL/runs/" "$HERE/results/runs/"
	echo "pulled runs -> $HERE/results/runs"
	;;
logs)
	jobid="${1:?job id required}"
	follow="${2:-}"
	remote "f=\$(ls $FBRL/logs/*-$jobid.out); if [ '$follow' = -f ]; then tail -n 50 -f \$f; else tail -n 80 \$f; fi"
	;;
harness)
	remote "$ENV; cd $FBRL/code && \$UV_PROJECT_ENVIRONMENT/bin/harness $(printf '%q ' "$@")"
	;;
sh)
	remote "$ENV; cd $FBRL/code && $1"
	;;
jobs)
	remote "$ENV; squeue -u \$USER -o '%.10i %.18j %.16P %.8T %.10M %.10l %.20R'"
	;;
cancel)
	remote "$ENV; scancel ${1:?job id required}"
	;;
gpu)
	session="${1:?session name required}"
	gres="${2:-gpu:a100-sxm4:1}"
	walltime="${3:-08:00:00}"
	partition="${4:-general-gpu-long}"
	account="${5:-engr-lab-netanel.raviv}"
	# Heredoc + positional args so nothing has to survive two layers of quoting.
	ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "$session" "$gres" "$walltime" "$partition" "$account" <<REMOTE
$ENV
session="\$1"; gres="\$2"; walltime="\$3"; partition="\$4"; account="\$5"
tmux kill-session -t "\$session" 2>/dev/null
tmux new-session -d -s "\$session" -x 220 -y 50
tmux send-keys -t "\$session" "srun -p \$partition -A \$account -n 1 -N 1 -c 8 --mem=64G --gres=\$gres --time=\$walltime --pty apptainer exec --nv --bind /engrfs:/engrfs,/scratch:/scratch \$IMAGE bash" Enter
# Poll until the pane settles, dismissing the container's ssh-add passphrase
# prompt once if it appears (its text stays on screen, so only act once).
prev=""; dismissed=0
for i in \$(seq 1 90); do
	sleep 1
	pane=\$(tmux capture-pane -t "\$session" -p)
	if [ \$dismissed -eq 0 ] && grep -q "Enter passphrase" <<<"\$pane"; then
		tmux send-keys -t "\$session" C-c; dismissed=1; prev=""; continue
	fi
	if [ \$i -gt 15 ] && [ "\$pane" = "\$prev" ] && [ -n "\$pane" ]; then break; fi
	prev="\$pane"
done
tmux send-keys -t "\$session" "source $FBRL/code/cluster/env.sh && cd \\\$FBRL/code && source \\\$UV_PROJECT_ENVIRONMENT/bin/activate" Enter
sleep 2
tmux capture-pane -t "\$session" -p
REMOTE
	;;
send)
	session="${1:?session name required}"
	shift
	remote "tmux send-keys -t $session $(printf '%q' "$*") Enter; sleep 3; tmux capture-pane -t $session -p"
	;;
cap)
	remote "tmux capture-pane -t ${1:?session name required} -p"
	;;
ls)
	remote "tmux ls"
	;;
kill)
	remote "tmux kill-session -t ${1:?session name required}"
	;;
*)
	sed -n '2,28p' "$0" >&2
	exit 1
	;;
esac
