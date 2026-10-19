#!/usr/bin/env bash
# PSO-TSP Experiment Suite — active plan ("design C")
#
# 160 experiment configs — 10 runs each, except pr1002 at 5 (see reps_for)
#
#   BASE  4 problems × 4 bare algorithms                   =  16
#   STAT  4 problems × 2 algos × 4 static topos × 3 radii  =  96
#   DYN   4 problems × 2 algos × 3 dyn topos  × 2 pcts     =  48
#
# Repetitions: 10 for berlin52/bier127/a280, 5 for pr1002 — so pr1002 results
# are NOT directly poolable with the rest; report its n separately.
#
# Sequential usage:   bash experiments/commands.sh
# Parallel usage:     uv run experiments/run_parallel.py [--workers N]
# What still needs running:
#                     uv run experiments/find_missing.py
#
# Keep this file in step with build_experiments() in run_parallel.py — they are
# meant to generate the same commands.  Trust the TOTAL arithmetic below over
# any prose: an earlier revision of this header claimed 1782 configs, a figure
# that was never implemented.
#
# ─── WHAT WAS REMOVED, AND WHY ───────────────────────────────────────────────
# Measured on berlin52 (optimum 7542; the deterministic greedy start of
# pso/base.py::_greedy_init sits at a 8.49 % gap).  Median gap to optimum:
#
#   works    PSO 0.03 %, PSO-Cognitive 0.03 %, PSO-Mesh 0.03 %, PSO-Torus 0.03 %,
#            PSO-FreeScale 0.03 %, PSO-Tree 0.68 %, PSO-Cognitive-FreeScale 0.03 %,
#            PSO-Cognitive-Mesh 0.32 %, PSO-Cognitive-Tree 0.92 %
#   broken   every APV and SWU variant — 8.49 %, i.e. exactly the greedy start,
#            with iterations_run pinned to the 2601 early-stop floor;
#            every Explore-with-topology and Cognitive-Explore-with-topology —
#            8.49 %, even though BARE PSO-Explore reaches 3.26 % and bare
#            PSO-Cognitive-Explore 4.42 %: the mixin does not compose with the
#            topology mixins (README already notes it cannot compose with
#            APV/SWU, since both override run());
#            PSO-Ring 8.08 %, while Mesh/Torus/FreeScale reach 0.03 % — the Ring
#            topology is broken on its own.
#
# RESOLVED 2026-10-19 (commit 0b2f1e5): Ring was never broken (that median
# aggregated radii 1-50, and at equal radius a ring has 2r neighbours where a
# mesh has (2r+1)^2-1; at r=25 Ring reaches 3.98-5.08 %), and Explore failed with
# a topology because StagnationExploreMixin accepted its mutation
# unconditionally while gating its crossover.  CognitiveMixin was separately
# inert under Explore, its whole contribution living inside the run() that
# Explore overrides.  Both are fixed, so Ring and the Explore variants are back
# in the plan below; Explore configs carry an explicit stagnation_trigger=50 (the
# value already derived by default) so their command strings differ from the
# pre-fix logs.  APV and SWU remain out of this script; see run_parallel.py's
# opt-in APVX group.
#
# Superseded text follows, kept because the numbers above refer to it:
# So APV and SWU are gone, Ring is gone, and Explore / Cognitive-Explore survive
# only as bare baselines.  Restore them here and in run_parallel.py once fixed.
#
# ali535 and gr666 are also gone, on data validity rather than cost: both are
# EDGE_WEIGHT_TYPE: GEO, while data/tsplib/reader.py::_build_distance_matrix
# computes plain Euclidean sqrt(dx²+dy²) unconditionally.  Applied to lat/long
# degrees that yields tours ~98.8 % SHORTER than the known optimum, which is
# impossible.  Re-add them once the reader handles GEO.
#
# ─── PARAMETER SWEEP ─────────────────────────────────────────────────────────
# Static radius: 1, 5, 25.  The effect is real but saturates — median gap to
# optimum over the working combinations:
#
#   problem   r=1     r=2     r=5     r=10    r=25    r=50
#   berlin52  7.22 %  5.08 %  0.03 %  0.03 %  0.03 %  0.03 %
#   bier127  10.16 %  9.07 %  8.00 %  8.15 %  8.24 %  8.06 %
#   a280     19.98 % 19.98 % 18.92 % 17.54 % 17.41 % 17.62 %
#   pr1002   20.53 % 20.53 % 20.53 % 20.53 % 20.53 % 20.42 %
#
# r=1 is worst, the curve is flat from r=5 on, so {1, 5, 25} spans it: one tight
# neighbourhood, one at the knee, one on the plateau.  10 and 50 are redundant.
#
# Dynamic neighbour fraction: 0.05 → 50 neighbours, 0.25 → 250 (out of 1000).
# Halved to two values for the same reason; the dynamic arm has no measurements
# yet, having never been run.
#
# NOTE on pr1002: its gap is flat at 20.53 % across every radius — plain PSO does
# not improve on it at all — and it costs ~21.6 h per config against 0.76-3.72 h
# for the others.  It is included deliberately; drop it from PROBLEMS if that
# turns out to be wasted compute.
#
# Early stopping: default stagnation_window=100.  Note it cannot trigger before
# ~2600 iterations (pso/base.py requires len(recorded) > 130, recorded every 20),
# and 80.7 % of runs stop exactly there.  To disable: --stagnation-window 0

set -uo pipefail   # no -e: log failures and keep going

COMMON_TAIL="--particles 1000 --max-iter 30000 --quiet"

# Repetitions per config: halved for instances above 500 cities.  A config is N
# SEQUENTIAL repetitions inside one process, so it cannot be split across SLURM
# tasks and its wall time is hard-capped by the 72 h cluster timelimit.  At 10
# repetitions pr1002's slowest family (Cognitive) already takes a 44.9 h median
# and 58.5 h worst case, leaving too little margin for the never-yet-run dynamic
# configs.  Keep this rule identical to repetitions_for() in run_parallel.py.
REPS_DEFAULT=10
REPS_LARGE=5
LARGE_PROBLEM_DIM=500
declare -A PROBLEM_DIMS=( [berlin52]=52 [bier127]=127 [a280]=280 [ali535]=535 [gr666]=666 [pr1002]=1002 )

reps_for() {
    local prob=$1
    if (( ${PROBLEM_DIMS[$prob]:-0} > LARGE_PROBLEM_DIM )); then
        echo "$REPS_LARGE"
    else
        echo "$REPS_DEFAULT"
    fi
}

# All six instances.  ali535 and gr666 were excluded here while the reader
# applied Euclidean distance to their GEO coordinates; data/tsplib/reader.py
# implements the TSPLIB GEO formula since commit 0aad143, so they are back.
PROBLEMS=(berlin52 bier127 a280 ali535 gr666 pr1002)
BASELINE_ALGOS=(PSO PSO-Cognitive PSO-Explore PSO-Cognitive-Explore)
TOPO_ALGOS=(PSO PSO-Cognitive PSO-Explore PSO-Cognitive-Explore)
STATIC_TOPOS=(Ring Tree Mesh Torus FreeScale)   # excluded: nothing
# Explore configs carry this explicitly: it equals the default the mixin derives
# (stagnation_window / 2), so behaviour is unchanged, but the command string then
# differs from the pre-0b2f1e5 Explore logs that find_missing.py matches on.
EXPLORE_PARAMS=(--param stagnation_trigger=50)

# Echo the extra Explore parameters when $1 names an Explore variant.
explore_params_for() {
  case "$1" in *Explore*) printf '%s\n' "${EXPLORE_PARAMS[@]}" ;; esac
}
DYN_TOPOS=(DynSim DynOpp DynMix)
RADII=(1 5 25)
PCTS=(0.05 0.25)

NP=${#PROBLEMS[@]}
NB=${#BASELINE_ALGOS[@]}
NA=${#TOPO_ALGOS[@]}
NS=${#STATIC_TOPOS[@]}
ND=${#DYN_TOPOS[@]}
NR=${#RADII[@]}
NC=${#PCTS[@]}

# BASE:4×4=16   STAT:4×2×4×3=96   DYN:4×2×3×2=48
TOTAL=$(( NP*NB + NP*NA*NS*NR + NP*NA*ND*NC ))   # = 528 for the full plan
COUNTER=0
START_TS=$(date +%s)
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

# run <algo> <problem> [extra...]
#
# Argument ORDER MATTERS and must match Experiment.to_cmd() in run_parallel.py:
#   -a <algo> -p tsplib-<problem> <COMMON> <extra>
# find_missing.py decides what is already done by matching the log's `CMD:` line
# as an exact string, so a different ordering here would make every log written
# by this script invisible to it.  Do not move $COMMON.
run() {
    local algo=$1 prob=$2
    shift 2

    COUNTER=$((COUNTER + 1))
    local n=$(printf '%04d' "$COUNTER")
    local logfile="$LOG_DIR/${n}.log"

    local -a argv=(-a "$algo" -p "tsplib-$prob" --repetitions "$(reps_for "$prob")" $COMMON_TAIL "$@")

    # write command header so the log is self-contained
    printf 'CMD: uv run main.py %s\n\n' "${argv[*]}" > "$logfile"
    uv run main.py "${argv[@]}" >> "$logfile" 2>&1
    local rc=$?

    local now=$(date +%s)
    local elapsed=$(( now - START_TS ))
    local eta=$(( elapsed * (TOTAL - COUNTER) / (COUNTER) ))
    local status="OK"
    [ $rc -ne 0 ] && status="FAIL"

    printf '[%d/%d] %s  elapsed=%ds  eta=%ds  log=%s  %s\n' \
        "$COUNTER" "$TOTAL" "$status" "$elapsed" "$eta" "${n}.log" "${argv[*]}"

    if [ $rc -ne 0 ]; then
        printf '  ^^^ FAILED (rc=%d) — see %s\n' "$rc" "$logfile" >&2
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# BASE — bare algorithms, Global topology                            (16 × 10)
#
# ─────────────────────────────────────────────────────────────────────────────
echo "=== BASE: bare algorithms (Global topology) ==="
for p in "${PROBLEMS[@]}"; do
  for algo in "${BASELINE_ALGOS[@]}"; do
    mapfile -t ep < <(explore_params_for "$algo")
    run "$algo" "$p" "${ep[@]}"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STAT — working algorithms × static topologies × radius             (96 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== STAT: static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for algo in "${TOPO_ALGOS[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      for r in "${RADII[@]}"; do
        mapfile -t ep < <(explore_params_for "$algo")
        run "${algo}-${topo}" "$p" --neighbours-radius "$r" "${ep[@]}"
      done
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# DYN — working algorithms × dynamic topologies × neighbor_pct       (48 × 10)
#
# This is the only part of the study with no results at all so far.
# ─────────────────────────────────────────────────────────────────────────────
echo "=== DYN: dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for algo in "${TOPO_ALGOS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      for pct in "${PCTS[@]}"; do
        mapfile -t ep < <(explore_params_for "$algo")
        run "${algo}-${topo}" "$p" --param "neighbor_pct=$pct" "${ep[@]}"
      done
    done
  done
done

echo "=== All experiments complete ==="
