#!/bin/bash
#SBATCH --job-name=pso-dim500
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 48:00:00
#SBATCH --array=0-11
#SBATCH --cpus-per-task=16
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Runs every experiment config still pending in the reduced problem scope
# (dimension <= 500: berlin52, bier127, a280).  242 configs as of 2026-09-28.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING -- two steps, in this order
# ---------------------------------------------------------------------------
#   1. Check the allocation actually has hours.  The previous grant was
#      exhausted, and while it is, sbatch fails with the misleading
#      "Invalid account or account/partition combination specified" and
#      sacctmgr/sacct return empty:
#
#          hpc-grants
#
#   2. Regenerate the pending list and the matching cost-balanced split.  Do
#      this at submit time, not once and forever -- it re-derives what is
#      actually missing from results/logs/, so a partial previous job never
#      causes re-runs:
#
#          uv run experiments/find_missing.py --tasks 12 --dyn-factor 2 \
#              --out results/pending_dim500.log \
#              --split-out results/pending_dim500.split
#
#      If it reports a different task count, update --array above to match the
#      "Set the array directive to:" line it prints.
#
#   Then:  sbatch run_experiments_dim500_array.sh
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT LOOKS DIFFERENT FROM THE EARLIER ONES
# ---------------------------------------------------------------------------
# The first three submissions sliced the plan into fixed-size index blocks (50,
# then 25 configs per task).  Config cost spans three orders of magnitude, so
# each task held all 16 reserved CPUs while one straggler finished, or was
# killed at the wall with that straggler unfinished.  Measured: ~26 583 core-h
# billed for ~9 212 core-h of real work (35 % efficiency), 15 of 38 tasks dead
# at the wall.  That exhausted the first 25 000 h grant.
#
# Here the tasks are balanced by *estimated cost* instead of config count, so
# every task carries ~84 core-h of work and they finish together.  The slices
# come from find_missing.py's greedy bin packing (spread 0.2 core-h).
#
# On the walltime: 48 h is deliberately generous.  SLURM bills elapsed time,
# not requested time, so an over-long limit costs nothing if the task finishes
# early -- whereas being killed at the wall costs the full reservation AND
# produces no result.  Asking for more than needed is the cheap direction.
# The slowest single pending config is estimated at ~15 h; with 8 workers a
# task's ~84 core-h packs into roughly 11 h, so 48 h leaves wide margin even if
# the dynamic-topology estimates prove badly low.
#
# CAVEAT: 216 of the 242 pending configs (89 %) are dynamic topologies (groups
# H-M), which have no completed runs anywhere -- their cost is extrapolated from
# static-topology runs on the same problem, and --dyn-factor 2 above doubles it
# for safety.  Once some finish, re-run find_missing.py for a measured estimate.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PENDING_LOG=results/pending_dim500.log
SPLIT_TABLE=results/pending_dim500.split

if [[ ! -f $PENDING_LOG || ! -f $SPLIT_TABLE ]]; then
    echo "ERROR: missing $PENDING_LOG or $SPLIT_TABLE." >&2
    echo "Run experiments/find_missing.py first -- see the header of this file." >&2
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

# Guard against a split table that no longer matches the pending log.
LINES=$(wc -l < "$PENDING_LOG")
if (( END > LINES )); then
    echo "ERROR: slice $START-$END exceeds $LINES entries in $PENDING_LOG." >&2
    echo "The split table is stale -- regenerate both files together." >&2
    exit 1
fi

echo "Task $TASK/$((N_TASKS - 1)): configs $START-$END of $LINES in $PENDING_LOG"

uv run experiments/run_parallel.py \
    --retry-failed "$PENDING_LOG" \
    --start "$START" \
    --end "$END" \
    --workers 8 \
    --numba-threads 2
