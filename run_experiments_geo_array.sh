#!/bin/bash
#SBATCH --job-name=pso-geo
#SBATCH -A plgmmastalerczyk3-cpu
#SBATCH -p plgrid
#SBATCH -t 72:00:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --output=results/slurm_logs/%x_%A_%a.out
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#
# Runs the two GEO instances — ali535 and gr666 — under the active plan
# ("design C"): 40 configs each, 80 in total, none previously computed.
#
# Companion to run_experiments_designC_array.sh, which covers the four EUC_2D
# instances.  Same plan shape, same exclusions (no APV, no SWU, no Ring, Explore
# and Cognitive-Explore only bare); see experiments/commands.sh for the numbers
# behind each exclusion.
#
# ---------------------------------------------------------------------------
# WHY THESE TWO WERE NOT IN THE EARLIER JOB
# ---------------------------------------------------------------------------
# Both declare EDGE_WEIGHT_TYPE: GEO, while data/tsplib/reader.py used to apply
# plain Euclidean distance to every instance regardless of its header.  On
# DDD.MM coordinates that understates GEO distances by about two orders of
# magnitude — the first node pair of ali535 came out as 20.67 against a true
# 2264 — so tours measured over that matrix were ~98.8 % "shorter" than the
# published optimum, which is impossible.
#
# reader.py now dispatches on EDGE_WEIGHT_TYPE and implements the TSPLIB GEO
# formula (DDD.MM coordinates, PI = 3.141592, RRR = 6378.388, integer
# truncation).  It was verified to agree EXACTLY — max error 0.0 over sampled
# pairs — with tsplib95's own get_weight() on both instances, and a greedy tour
# now sits 25.1 % (ali535) and 24.7 % (gr666) above the published optima, in
# line with the EUC_2D instances at 19-22 %.  Unknown weight types are now
# rejected at load time rather than silently measured with the wrong metric.
#
# WARNING: the ali535 and gr666 JSONs already in results/ predate that fix and
# are meaningless.  This job does not reuse them — the plan runs these instances
# at 5 repetitions, so the command strings differ and find_missing.py counts all
# 80 configs as pending — but the stale files are still on disk.  Exclude
# anything matching results/*ali535* or results/*gr666* with a timestamp before
# 2026-10-03 from any analysis.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING
# ---------------------------------------------------------------------------
#   1. hpc-grants — keep the 25 000 h reserve intact.
#   2. Regenerate the pending list and its split, at submit time:
#
#        uv run experiments/find_missing.py --problems ali535 gr666 \
#            --tasks 3 --dyn-factor 2 \
#            --out results/pending_geo.log \
#            --split-out results/pending_geo.split
#
#      If it reports a different task count, update --array above to match.
#
#   Then:  sbatch run_experiments_geo_array.sh
#
# ---------------------------------------------------------------------------
# WHY 3 TASKS x 8 CPUs
# ---------------------------------------------------------------------------
# A config is N sequential repetitions inside ONE process and cannot be split
# across tasks, so a task's elapsed time is
#
#     max( slowest config in the task , task's work / workers )
#
# and SLURM bills elapsed x cpus for EVERY task — spreading the same work over
# more tasks costs more, not less.  Here: ~874 core-h of work, slowest config
# ~24.6 h, so 3 tasks hold ~291 core-h each and run for max(24.6, 291/8) ≈ 36 h,
# billing ≈ 3 x 36 x 8 ≈ 874 core-h, i.e. near-full utilisation.
#
# 8 CPUs with --workers 8 --numba-threads 1: this plan contains no APV or SWU,
# which were the only operators using the Numba kernels, so every process here
# is single-threaded pure Python and one CPU per worker is enough.
#
# Walltime 72 h is the cluster maximum and clears the ~24.6 h worst case — which
# already includes the --dyn-factor 2 margin — by a wide margin.  SLURM bills
# elapsed rather than requested time, so over-requesting is free while a wall
# kill costs the whole reservation and yields nothing.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PENDING_LOG=results/pending_geo.log
SPLIT_TABLE=results/pending_geo.split

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
