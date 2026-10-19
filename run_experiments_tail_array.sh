#!/bin/bash
#SBATCH --job-name=pso-tail
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 72:00:00
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Finishes the active plan: the last 7 configs of 240.  They are the stragglers
# from the three array tasks killed at the 72 h wall on 2026-10-06
# (pso-designC_21147249_1, pso-geo_21147271_0 and _1).
#
#   1. PSO-Torus     @ pr1002  --neighbours-radius 25
#   2. PSO-Mesh      @ gr666   --neighbours-radius 25
#   3. PSO-Torus     @ gr666   --neighbours-radius 25
#   4. PSO-FreeScale @ gr666   --neighbours-radius 5
#   5. PSO-FreeScale @ gr666   --neighbours-radius 25
#   6. PSO-FreeScale @ ali535  --neighbours-radius 5
#   7. PSO-FreeScale @ ali535  --neighbours-radius 25
#
# All seven run at 5 repetitions (repetitions_for(), dim > 500) and all are in
# the STAT group, so none of them touches APV or SWU — the only operators that
# use the Numba kernels.  Every process here is single-threaded pure Python.
#
# ---------------------------------------------------------------------------
# ONE CONFIG PER TASK, ONE CPU PER TASK
# ---------------------------------------------------------------------------
# A config is N sequential repetitions inside ONE process and cannot be split,
# so one config per task is the maximum parallelism available: all seven run at
# once and the whole plan completes in the time of the slowest single config
# (~13 h estimated for the pr1002 one).
#
# Because each task runs exactly one single-threaded process, --cpus-per-task=1
# and --workers 1 are the right sizing.  SLURM bills elapsed x cpus per task, so
# this shape bills only the work actually done — roughly 54 core-h in total, at
# essentially 100 % efficiency.  Requesting 8 CPUs here would bill 8x for seven
# idle cores per task.
#
# Walltime is the full 72 h even though the slowest config is estimated at ~13 h:
# billing follows elapsed rather than requested time, so the margin is free,
# whereas a wall kill costs the whole reservation and produces nothing.  That is
# exactly how the previous three tasks were lost.
#
# On sizing, for next time: the earlier jobs were sized from MEDIAN per-config
# cost, which underestimates badly because the distribution has a long tail
# (measured maxima: gr666 47.0 h, ali535 57.0 h, pr1002 58.5 h).  Size from p90
# or the maximum.  Also note --dyn-factor is no longer needed: the dynamic arm
# is now measured and costs the same as the static one (median 6.4 h vs 6.6 h,
# ratio 0.98).
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING
# ---------------------------------------------------------------------------
#   1. hpc-grants — keep the 25 000 h reserve intact.
#   2. Regenerate the pending list and its split, at submit time:
#
#        uv run experiments/find_missing.py --all-problems --tasks 7 \
#            --out results/pending_tail.log \
#            --split-out results/pending_tail.split
#
#      If it reports a different count, update --array above to 0-(N-1).
#
#   Then:  sbatch run_experiments_tail_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PENDING_LOG=results/pending_tail.log
SPLIT_TABLE=results/pending_tail.split

if [[ ! -f $PENDING_LOG || ! -f $SPLIT_TABLE ]]; then
    echo "ERROR: missing $PENDING_LOG or $SPLIT_TABLE." >&2
    echo "Run experiments/find_missing.py first — see the header of this file." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$SPLIT_TABLE"       # defines N_TASKS, TASK_START[], TASK_END[]

TASK=$SLURM_ARRAY_TASK_ID

if (( TASK >= N_TASKS )); then
    echo "ERROR: array task $TASK exceeds the $N_TASKS slices in $SPLIT_TABLE." >&2
    echo "Update the #SBATCH --array directive to 0-$(( N_TASKS - 1 ))." >&2
    exit 1
fi

START=${TASK_START[$TASK]}
END=${TASK_END[$TASK]}

LINES=$(wc -l < "$PENDING_LOG")
if (( END > LINES )); then
    echo "ERROR: slice $START-$END exceeds $LINES entries in $PENDING_LOG." >&2
    echo "The split table is stale — regenerate both files together." >&2
    exit 1
fi

echo "Task $TASK/$((N_TASKS - 1)): config $START of $LINES in $PENDING_LOG"

uv run experiments/run_parallel.py \
    --retry-failed "$PENDING_LOG" \
    --start "$START" \
    --end "$END" \
    --workers 1 \
    --numba-threads 1
