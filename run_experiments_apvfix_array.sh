#!/bin/bash
#SBATCH --job-name=pso-apvfix
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 72:00:00
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Parameter probe for the repaired APV and SWU operators (plan group APVX):
#
#   2 algorithms x 3 instances x 3 v_explore values = 18 configs, 10 repetitions
#   PSO-APV / PSO-SWU   @   berlin52 / bier127 / a280   @   v_explore 0 / 0.5 / 1
#
# v_explore = 0.0 is the control: it reproduces the operator exactly as
# published, so those six configs are expected to return the greedy input tour.
# Their value is that they do so WITH the other two fixes applied, which
# separates "the step was unbounded" from "there is no exploration term".
#
# ---------------------------------------------------------------------------
# WHAT WAS WRONG, AND WHY THIS IS A PROBE RATHER THAN THE FULL ARM
# ---------------------------------------------------------------------------
# Every pre-2026-10-19 APV/SWU result is the greedy initial tour, unchanged:
# best_path_length == init_path_length in 1758 of 1760 recorded APV runs and
# 1231 of 1750 SWU runs, median gap to optimum identical to the greedy start on
# every instance (berlin52 8.49 %, bier127 13.26 %, a280 19.98 %, pr1002
# 20.53 %), std 0.0 across repetitions, and iterations_run pinned to the 2601
# early-stop floor.  That cost 2 138 wall-h, 21 % of all compute spent so far.
#
# Three separate defects, all now fixed in pso/operators/_numba_kernels.py:
#
#   1. No acceptance test.  The kernel did new_particles[i] = new_p
#      unconditionally, while PSOBase.run() accepts a candidate only if it is
#      shorter, or with probability 0.1.  APV/SWU dropped that when they
#      overrode run().
#   2. No bound on ||v||_1.  Transpositions are applied independently with
#      probability v each, so sum(v) IS the expected step in transpositions.
#      The per-entry clip to [0,1] bounds the components, not the norm, and
#      |P_p| + |P_g| ~ 2(n-1) entries are reinforced per iteration, so the step
#      reached ~n swaps on an n-city tour.  Measured on berlin52: mean tour
#      length went 8182 -> 18083 after ONE iteration and settled near 23500,
#      against 30040 for a uniformly random tour.
#   3. No exploration term -- the structural one.  The velocity's support is
#      exactly the set of transpositions leading to pbest/gbest.  The swarm's
#      best particle IS gbest, so both sets are empty, its ||v||_1 was 0.0000
#      from iteration 1 and it never moved once in 300 iterations.  No particle
#      could propose anything outside the convex hull of the initial swarm:
#      0 of 12 000 candidates beat the greedy start.  Fixing 1 and 2 alone
#      changed nothing, at any v_max down to 0.5 -- that is what this probe's
#      v_explore=0 control re-confirms at full settings.
#
# Why a probe: with the exploration term the operators do finally optimise, but
# a 2-seed, 40-particle, 1500-iteration local run put the gain at only 0.3-0.6 %
# over the greedy start (best gap 7.86 % vs 8.49 %), where plain PSO reaches
# 0.03 % on the same instance.  That is too thin a result to justify the full
# six-instance arm, and the cost is now unknown in a way it was not before: the
# old runs stopped at 2601 iterations precisely BECAUSE nothing ever improved,
# so a working operator may run towards --max-iter 30000, up to ~11x longer.
# This job measures that before anything larger is submitted.
#
# ---------------------------------------------------------------------------
# SIZING
# ---------------------------------------------------------------------------
# One config per task, one CPU per task: a config is 10 sequential repetitions
# in ONE single-threaded process, so this is the maximum parallelism available
# and SLURM bills only the work done.  Extrapolating the old a280 cost (329 s
# per repetition at 2601 iterations) to the 30 000-iteration cap gives ~10.5 h
# for the worst config, so the expected bill is ~100-200 core-h and the hard
# ceiling is 18 x 72 = 1 296 core-h.  Against 44 966 h remaining and the
# 25 000 h reserve, that leaves the reserve untouched either way.
#
# No Numba threading: --numba-threads 1.  These ARE the Numba operators, but
# each task runs a single process and the kernels parallelise over particles
# within it; giving one task several cores would bill them for one worker.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING
# ---------------------------------------------------------------------------
#   1. hpc-grants -- keep the 25 000 h reserve intact.
#   2. Regenerate the pending list and its split, at submit time:
#
#        uv run experiments/find_missing.py --groups APVX \
#            --problems berlin52 bier127 a280 --tasks 18 \
#            --out results/pending_apvfix.log \
#            --split-out results/pending_apvfix.split
#
#      If it reports a different count, update --array above to 0-(N-1).
#
#   Then:  sbatch run_experiments_apvfix_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PENDING_LOG=results/pending_apvfix.log
SPLIT_TABLE=results/pending_apvfix.split

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
    --workers 1 \
    --numba-threads 1
