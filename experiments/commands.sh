#!/usr/bin/env bash
# PSO-TSP Experiment Suite
#
# 1782 experiment configs × 10 runs = 17 820 total runs
# Problems ordered smallest → biggest city count
#
# Sequential usage:   bash experiments/commands.sh
# Parallel usage:     python experiments/run_parallel.py [--workers N]
#
# Operator variants tested:
#   Default, Cognitive, APV, SWU        — all topologies
#   Explore (enhancer on Default)       — all topologies
#   Explore+Cognitive                   — all topologies
#   (Explore does NOT compose with APV/SWU — they both override run())
#
# Static topology neighbourhood radius: 1, 2, 5, 10
# Dynamic topology neighbourhood fraction (neighbor_pct):
#   0.05 → 10 neighbours | 0.10 → 20 | 0.25 → 50 | 0.50 → 100  (out of 200)
#
# Early stopping: default stagnation_window=100 is used.
# To disable early stopping (full 30k iterations), add: --stagnation-window 0

set -uo pipefail   # no -e: log failures and keep going

COMMON="--repetitions 10 --particles 1000 --max-iter 30000 --quiet"
PROBLEMS=(berlin52 bier127 a280 ali535 gr666 pr1002)
RADII=(1 2 5 10 25 50)
PCTS=(0.05 0.10 0.25 0.50)
STATIC_TOPOS=(Ring Tree Mesh Torus FreeScale)
DYN_TOPOS=(DynSim DynOpp DynMix)

# A:36  B-G:6×180=1080  H-M:6×72=432
TOTAL=$(( 6*6 + 6*6*6*5 + 6*6*4*3 ))   # = 1548
COUNTER=0
START_TS=$(date +%s)
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

run() {
    COUNTER=$((COUNTER + 1))
    local n=$(printf '%04d' "$COUNTER")
    local logfile="$LOG_DIR/${n}.log"

    # write command header so the log is self-contained
    printf 'CMD: uv run main.py %s %s\n\n' "$*" "$COMMON" > "$logfile"
    uv run main.py "$@" $COMMON >> "$logfile" 2>&1
    local rc=$?

    local now=$(date +%s)
    local elapsed=$(( now - START_TS ))
    local eta=$(( elapsed * (TOTAL - COUNTER) / (COUNTER) ))
    local status="OK"
    [ $rc -ne 0 ] && status="FAIL"

    printf '[%d/%d] %s  elapsed=%ds  eta=%ds  log=%s  %s\n' \
        "$COUNTER" "$TOTAL" "$status" "$elapsed" "$eta" "${n}.log" "$*"

    if [ $rc -ne 0 ]; then
        printf '  ^^^ FAILED (rc=%d) — see %s\n' "$rc" "$logfile" >&2
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# A. BASELINES — Global topology, all operator variants              (54 runs × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== A: Baselines (Global topology) ==="
for p in "${PROBLEMS[@]}"; do
  run -a PSO                   -p tsplib-$p
  run -a PSO-Cognitive         -p tsplib-$p
  run -a PSO-APV               -p tsplib-$p
  run -a PSO-SWU               -p tsplib-$p
  run -a PSO-Explore           -p tsplib-$p
  run -a PSO-Cognitive-Explore -p tsplib-$p
done

# ─────────────────────────────────────────────────────────────────────────────
# B. DEFAULT × Static topologies × radius                           (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== B: Default × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# C. COGNITIVE × Static topologies × radius                         (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== C: Cognitive × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-Cognitive-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# D. APV × Static topologies × radius                               (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== D: APV × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-APV-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# E. SWU × Static topologies × radius                               (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== E: SWU × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-SWU-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# F. EXPLORE × Static topologies × radius                           (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== F: Explore × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-Explore-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# G. EXPLORE+COGNITIVE × Static topologies × radius                 (180 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== G: Explore+Cognitive × Static topologies ==="
for p in "${PROBLEMS[@]}"; do
  for r in "${RADII[@]}"; do
    for topo in "${STATIC_TOPOS[@]}"; do
      run -a PSO-Cognitive-Explore-$topo -p tsplib-$p --neighbours-radius $r
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# H. DEFAULT × Dynamic topologies × neighbor_pct                    (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== H: Default × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# I. COGNITIVE × Dynamic topologies × neighbor_pct                  (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== I: Cognitive × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-Cognitive-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# J. APV × Dynamic topologies × neighbor_pct                        (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== J: APV × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-APV-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# K. SWU × Dynamic topologies × neighbor_pct                        (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== K: SWU × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-SWU-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# L. EXPLORE × Dynamic topologies × neighbor_pct                    (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== L: Explore × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-Explore-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# M. EXPLORE+COGNITIVE × Dynamic topologies × neighbor_pct          (108 × 10)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== M: Explore+Cognitive × Dynamic topologies ==="
for p in "${PROBLEMS[@]}"; do
  for pct in "${PCTS[@]}"; do
    for topo in "${DYN_TOPOS[@]}"; do
      run -a PSO-Cognitive-Explore-$topo -p tsplib-$p --param neighbor_pct=$pct
    done
  done
done

echo "=== All experiments complete ==="
