#!/bin/bash
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Runs the pending configs for ONE instance, one config per array task.
#
# Submit one job per instance, setting the array size and name explicitly --
# the array directive cannot be computed inside the script:
#
#   uv run experiments/find_missing.py --problems <prob> --tasks 44 \
#       --out results/pending_topo_<prob>.log \
#       --split-out results/pending_topo_<prob>.split
#
#   sbatch --array=0-43 --job-name=pso-topo-<prob> \
#          run_experiments_topo_array.sh <prob>
#
# ---------------------------------------------------------------------------
# WHAT IS PENDING AND WHY
# ---------------------------------------------------------------------------
# After commit 0b2f1e5 the plan is 528 configs (88 per instance): BASE 4, STAT
# 4 algos x 5 static topologies x 3 radii, DYN 4 algos x 3 dynamic topologies x
# 2 percentages.  Exactly half of each instance's 88 are already on disk and are
# reused -- all the PSO and PSO-Cognitive configs, INCLUDING Ring, whose legacy
# results stay valid because no topology file was touched.
#
# The 44 pending per instance are precisely the Explore and Cognitive-Explore
# configs, which have to be recomputed because both were broken until 0b2f1e5:
#
#   * StagnationExploreMixin accepted its mutation unconditionally while gating
#     its crossover, so every particle took a random swap every iteration and a
#     local neighbourhood had nothing to anchor it.  Bare PSO-Explore limped
#     (the global guide is an all-time elite) but every PSO-Explore-<topo> sat
#     at exactly the greedy start.
#   * CognitiveMixin's whole contribution lived inside its own run(), which
#     StagnationExploreMixin overrides, so the cognitive step vanished and
#     PSO-Cognitive-Explore was behaviourally identical to PSO-Explore.
#
# The commands therefore carry an explicit --param stagnation_trigger=50.  That
# is the value the mixin already derives by default, so it changes nothing, but
# it makes the command string differ from the pre-fix Explore logs, which
# find_missing.py matches on and would otherwise count as already done.
#
# ---------------------------------------------------------------------------
# SIZING
# ---------------------------------------------------------------------------
# One config per task, one CPU per task.  A config is 10 sequential repetitions
# in ONE process, so this is the maximum parallelism available, and SLURM bills
# elapsed x cpus per task -- this shape bills only the work actually done.
# These operators are pure Python (no APV/SWU, hence no Numba), so one core per
# process is the right size.
#
# Expected ~33 core-h for berlin52, ~63 for bier127, ~154 for a280 at the
# measured medians; ~250 core-h for all three.  Walltime is the full 72 h even
# though the median slowest config is 0.8-3.6 h: the per-config MAXIMA measured
# on these instances are 33.1 / 36.6 / 42.2 h, billing follows elapsed rather
# than requested time, and a wall kill costs the whole reservation and yields
# nothing.  Three earlier tasks were lost exactly that way on 2026-10-06.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PROBLEM=${1:?usage: sbatch --array=0-N --job-name=pso-topo-<prob> $0 <problem>}

PENDING_LOG=results/pending_topo_${PROBLEM}.log
SPLIT_TABLE=results/pending_topo_${PROBLEM}.split

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
    echo "Submit with --array=0-$(( N_TASKS - 1 ))." >&2
    exit 1
fi

START=${TASK_START[$TASK]}
END=${TASK_END[$TASK]}

LINES=$(wc -l < "$PENDING_LOG")
if (( END > LINES )); then
    echo "ERROR: slice $START-$END exceeds $LINES entries in $PENDING_LOG." >&2
    echo "The split table is stale -- regenerate both files together." >&2
    exit 1
fi

echo "[$PROBLEM] task $TASK/$((N_TASKS - 1)): configs $START-$END of $LINES"

uv run experiments/run_parallel.py \
    --retry-failed "$PENDING_LOG" \
    --start "$START" \
    --end "$END" \
    --workers 1 \
    --numba-threads 1
