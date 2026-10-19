#!/bin/bash
#SBATCH --job-name=pso-designC
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 72:00:00
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Runs the configs still pending in the active plan ("design C"): 160 configs
# over berlin52, bier127, a280 and pr1002, of which 84 are done and 76 remain.
# See experiments/commands.sh for what design C excludes and why.
#
# pr1002 uses 5 repetitions per config rather than 10 (repetitions_for() in
# run_parallel.py), which is why all 40 of its configs count as pending: the
# 13 that completed at 10 repetitions are a different command and are left as
# historical data rather than reused.  Its results are therefore not directly
# poolable with the other three instances -- report its n separately.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING
# ---------------------------------------------------------------------------
#   1. hpc-grants
#      Keep the user's 25 000 h reserve intact.  While an allocation is
#      exhausted, sbatch fails with the misleading "Invalid account or
#      account/partition combination specified" and sacct returns nothing.
#
#   2. Regenerate the pending list and its split — at submit time, so a partial
#      earlier job never causes re-runs:
#
#        uv run experiments/find_missing.py --tasks 4 --dyn-factor 2 \
#            --out results/pending_designC.log \
#            --split-out results/pending_designC.split
#
#      If it reports a different task count, update --array above to match.
#
#   Then:  sbatch run_experiments_designC_array.sh
#
# ---------------------------------------------------------------------------
# WHY 4 TASKS x 8 CPUs, AND WHY 72 h
# ---------------------------------------------------------------------------
# A config is N *sequential* repetitions inside ONE process, so it cannot be
# split across tasks: if one config needs more than the walltime, no repacking
# saves it.  A task's elapsed time is therefore
#
#     max( slowest config in the task , task's total work / workers )
#
# and SLURM bills elapsed x cpus-per-task for EVERY task.  Spreading the same
# work over more tasks does not make it cheaper -- it makes it dearer, because
# each extra task reserves its CPUs for the full elapsed while its workers sit
# idle.  Measured trade-off for this plan (1 193 core-h of work, slowest config
# ~44.9 h):
#
#     tasks  CPU/task  work/task  elapsed  billed    efficiency
#        10        16       119h    44.9h   7 184h         17 %
#         6         8       199h    44.9h   2 155h         55 %
#         4         8       298h    44.9h   1 437h         83 %
#         3         8       398h    49.7h   1 193h        100 %
#
# 4 tasks is chosen over 3: it is within 20 % of optimal billing while leaving
# headroom if a config runs slower than estimated.  The earlier dim500 array
# used 10 tasks x 16 CPUs and would have billed ~6x the useful work.
#
# --cpus-per-task=8 with --workers 8 --numba-threads 1: design C contains no APV
# or SWU, and those were the only operators using the Numba kernels (verified --
# numba appears nowhere in the MRO of PSO, PSO-Cognitive, PSO-Mesh,
# PSO-Cognitive-DynSim or PSO-Explore).  Everything here is pure single-threaded
# Python, so one CPU per worker is enough; reserving 2 would double the bill for
# nothing.
#
# Walltime 72 h is the cluster maximum.  Over the 113 completed pr1002 configs at
# 10 repetitions: median 21.6 h, max 58.5 h, none above 72 h.  At 5 repetitions
# the slowest family (Cognitive) is estimated at ~22 h, and find_missing.py's
# doubled "44.9 h" figure already carries the --dyn-factor 2 margin for the
# never-yet-run dynamic configs.  SLURM bills elapsed rather than requested
# time, so asking for the full 72 h is free when tasks finish early, whereas a
# wall kill costs the whole reservation and yields no result.
#
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PENDING_LOG=results/pending_designC.log
SPLIT_TABLE=results/pending_designC.split

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

echo "Task $TASK/$((N_TASKS - 1)): configs $START-$END of $LINES in $PENDING_LOG"

uv run experiments/run_parallel.py \
    --retry-failed "$PENDING_LOG" \
    --start "$START" \
    --end "$END" \
    --workers 8 \
    --numba-threads 1
