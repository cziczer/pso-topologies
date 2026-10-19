#!/usr/bin/env python3
"""Parallel PSO-TSP experiment runner.

Generates the same experiment commands as commands.sh and executes them
concurrently using subprocess workers.  Results land in results/ as usual.

Plan size
---------
The active plan ("design C") is 160 configs x 10 repetitions = 1 600 runs, over
the four instances whose distances are computed correctly:

    BASE   4 problems x 4 bare algorithms                   =  16
    STAT   4 problems x 2 algos x 4 static topos x 3 radii  =  96
    DYN    4 problems x 2 algos x 3 dyn topos x 2 pcts      =  48

It deliberately omits everything measured to be broken -- APV, SWU, Explore and
Cognitive-Explore when composed with a topology, the Ring topology, and the two
GEO instances.  See the scope comments below for the numbers behind each
exclusion.  build_legacy_experiments() reproduces the retired 1548-config plan
that the 941 existing results came from.

Count build_experiments() rather than trusting a comment: a previous revision of
this docstring claimed 1 782 configs, a figure that was never implemented.

Usage
-----
    uv run experiments/run_parallel.py                    # auto-detect workers
    uv run experiments/run_parallel.py --workers 4        # explicit parallelism
    uv run experiments/run_parallel.py --dry-run          # print commands only
    uv run experiments/run_parallel.py --groups A B C     # run selected groups
    uv run experiments/run_parallel.py --problems berlin52 bier127
    uv run experiments/run_parallel.py --retry-failed results/failed_<ts>.log

Failure handling
----------------
Every failed experiment is appended to a structured log file
(results/failed_<timestamp>.log).  The runner never aborts on failure —
all remaining experiments continue.  At the end a summary is printed and the
exit code is 1 only if any experiment failed.

To retry only the failed experiments from a previous run:
    uv run experiments/run_parallel.py --retry-failed results/failed_<ts>.log

Worker / thread guidance
------------------------
Numba's parallel kernels (APV, SWU) spawn one OS thread per logical CPU by
default.  Running W workers each with T Numba threads uses W×T threads total.
The env var NUMBA_NUM_THREADS caps Numba threads per process.

Recommended starting point on an N-core machine:
    --workers ceil(N/2)  with NUMBA_NUM_THREADS=2   (set automatically below)

Override: NUMBA_NUM_THREADS=1 lets you use --workers N for maximum throughput
on Default/Cognitive/Explore experiments (pure Python, single-threaded).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Experiment specification
# ─────────────────────────────────────────────────────────────────────────────

ALL_PROBLEMS = [
    "berlin52", "bier127", "a280", "ali535", "gr666", "pr1002",
]

# TSPLIB instance dimension (city count) per problem.
PROBLEM_DIMS = {
    "berlin52": 52, "bier127": 127, "a280": 280,
    "ali535": 535, "gr666": 666, "pr1002": 1002,
}

# ── Problem scope ────────────────────────────────────────────────────────────
#
# All six instances are in scope.
#
# ali535 and gr666 were excluded until 2026-10-03: both declare
# EDGE_WEIGHT_TYPE: GEO, while data/tsplib/reader.py applied plain Euclidean
# distance to every instance regardless of its header.  On DDD.MM coordinates
# that understates GEO distances by ~2 orders of magnitude, and tours came out
# ~98.8 % "shorter" than the published optimum.  reader.py now dispatches on
# EDGE_WEIGHT_TYPE and implements the TSPLIB GEO formula; it was verified to
# agree exactly (max error 0.0) with tsplib95's own get_weight() on both
# instances, and a greedy tour now sits 25.1 % / 24.7 % above the published
# optima, in line with the EUC_2D instances at 19-22 %.
#
# WARNING: every ali535 and gr666 result produced BEFORE that fix was measured
# with the wrong metric and must not be pooled with anything.  They are not
# silently reused -- the fixed plan runs those instances at 5 repetitions
# (see repetitions_for) so the command strings differ and find_missing.py
# treats them as pending -- but the old JSONs are still sitting in results/.
PROBLEMS = list(ALL_PROBLEMS)
EXCLUDED_PROBLEMS: list[str] = []

# ── Operator scope ───────────────────────────────────────────────────────────
#
# Measured on berlin52 (optimum 7542; the deterministic greedy start sits at a
# 8.49 % gap).  Median gap to optimum per exact algorithm name:
#
#   works:   PSO 0.03 %, PSO-Cognitive 0.03 %, PSO-Mesh 0.03 %, PSO-Torus 0.03 %,
#            PSO-FreeScale 0.03 %, PSO-Tree 0.68 %, PSO-Cognitive-FreeScale 0.03 %,
#            PSO-Cognitive-Mesh 0.32 %, PSO-Cognitive-Tree 0.92 %,
#            PSO-Cognitive-Torus 1.29 %
#   broken:  every APV and every SWU variant, 8.49 % -- i.e. exactly the greedy
#            start, with iterations_run pinned to the 2601 early-stop floor;
#            every Explore-with-topology and Cognitive-Explore-with-topology,
#            8.49 %, although BARE PSO-Explore reaches 3.26 % and bare
#            PSO-Cognitive-Explore 4.42 % -- the mixin does not compose with the
#            topology mixins;
#            PSO-Ring 8.08 %, while Mesh/Torus/FreeScale all reach 0.03 % -- the
#            Ring topology is broken independently of any operator.
#
# So: APV and SWU are dropped entirely, Ring is dropped, and Explore /
# Cognitive-Explore survive only as bare baselines.  Restore them here once the
# three defects are fixed.
TOPO_ALGOS     = ["PSO", "PSO-Cognitive"]
BASELINE_ALGOS = ["PSO", "PSO-Cognitive", "PSO-Explore", "PSO-Cognitive-Explore"]

BROKEN_ALGOS  = ["PSO-APV", "PSO-SWU"]
BROKEN_TOPOS  = ["Ring"]

# ── Fixed APV/SWU probe (group "APVX", opt-in) ───────────────────────────────
#
# APV and SWU returned their greedy input tour because of three separate
# defects, all fixed on 2026-10-19 (see pso/operators/_numba_kernels.py):
#
#   1. no acceptance test -- the particle took every candidate unconditionally,
#      where PSOBase.run() accepts only an improvement, or 10 % of the time;
#   2. no bound on ||v||_1, which IS the expected number of transpositions per
#      iteration, so the step reached ~n swaps on an n-city tour and resampled
#      the tour instead of moving it;
#   3. no exploration term.  This one is structural: the velocity's support is
#      exactly the transpositions leading to pbest/gbest, so the swarm's best
#      particle (whose position IS gbest) had ||v||_1 = 0.0000 and never moved,
#      and no particle could propose a tour outside the convex hull of the
#      initial swarm.  Measured: 0 of 12 000 candidates beat the greedy start.
#
# Fixes 1 and 2 are unconditional.  Fix 3 is the v_explore parameter, default
# 0.0 -- i.e. the operator as published -- so it must be passed explicitly here.
# Passing it explicitly also keeps these command strings distinct from the 372
# pre-fix APV/SWU logs in results/logs/, which find_missing.py would otherwise
# match and skip.
#
# This group is a PARAMETER PROBE, not the final arm: a 2-seed, 40-particle,
# 1500-iteration local run put the improvement over the greedy start at only
# 0.3-0.6 % (best gap 7.86 % against 8.49 % for the untouched start), while
# plain PSO reaches 0.03 % on the same instance.  Measure v_explore at real
# settings on the three cheap instances before committing to all six.
APVX_ALGOS = ["PSO-APV", "PSO-SWU"]
APVX_PROBLEMS = ["berlin52", "bier127", "a280"]
APVX_V_MAX = 2.0
APVX_V_EXPLORE = [0.0, 0.5, 1.0]        # 0.0 is the unchanged-operator control

# Groups that are never part of the default plan; request them with --groups.
OPT_IN_GROUPS = {"APVX"}

STATIC_TOPOS = ["Tree", "Mesh", "Torus", "FreeScale"]
DYN_TOPOS    = ["DynSim", "DynOpp", "DynMix"]

# ── Parameter sweep (design "C") ─────────────────────────────────────────────
#
# The neighbourhood-radius effect is real but saturates.  Median gap to optimum
# over the working combinations:
#
#   problem   r=1     r=2     r=5     r=10    r=25    r=50
#   berlin52  7.22 %  5.08 %  0.03 %  0.03 %  0.03 %  0.03 %
#   bier127  10.16 %  9.07 %  8.00 %  8.15 %  8.24 %  8.06 %
#   a280     19.98 % 19.98 % 18.92 % 17.54 % 17.41 % 17.62 %
#   pr1002   20.53 % 20.53 % 20.53 % 20.53 % 20.53 % 20.42 %
#
# r=1 is clearly worst, the curve is flat from r=5 on, so {1, 5, 25} spans it:
# one tight neighbourhood, one at the knee, one on the plateau.  10 and 50 are
# redundant with 5 and 25.  PCTS is halved to its endpoints-plus-middle for the
# same reason; it has no measurements yet, the dynamic arm having never run.
RADII = [1, 5, 25]
PCTS  = [0.05, 0.25]

# ── Repetitions per config ───────────────────────────────────────────────────
#
# Large instances get half the repetitions.  A config is N *sequential*
# repetitions inside ONE process, so it cannot be split across SLURM tasks:
# per-config wall time is therefore hard-capped by the cluster's 72 h timelimit,
# and no amount of repacking helps a config that needs longer.
#
# Measured over the 113 completed pr1002 configs at 10 repetitions: median
# 21.6 h, max 58.5 h, and the slowest family (Cognitive) a 44.9 h median.  That
# leaves only ~13 h of margin under 72 h — too thin for the dynamic-topology
# configs, which have never run and could be slower.  Halving to 5 repetitions
# brings the Cognitive median to ~22 h and the worst case to ~29 h.
#
# Note the trade-off: 5 repetitions is less statistical power exactly where
# variance matters most, and it makes pr1002 results not directly poolable with
# the 10-repetition results from the other three instances.  Report pr1002's n
# separately in any aggregate.
REPS_DEFAULT = 10
REPS_LARGE = 5
LARGE_PROBLEM_DIM = 500


def repetitions_for(problem: str) -> int:
    """Repetition count for *problem*: fewer for instances above 500 cities."""
    return REPS_LARGE if PROBLEM_DIMS.get(problem, 0) > LARGE_PROBLEM_DIM else REPS_DEFAULT


# Everything after --repetitions.  Kept separate so --repetitions stays in its
# historical position in the command string: for the instances that still use 10
# repetitions the generated command is byte-identical to the one in the existing
# results/logs/*.log, which is what lets find_missing.py recognise them as done.
COMMON_TAIL = ["--particles", "1000", "--max-iter", "30000", "--quiet"]


@dataclass
class Experiment:
    group: str
    algo: str
    problem: str
    extra: list[str] = field(default_factory=list)
    reps: int | None = None          # None -> repetitions_for(problem)

    def repetitions(self) -> int:
        return self.reps if self.reps is not None else repetitions_for(self.problem)

    def to_cmd(self) -> list[str]:
        return (
            ["uv", "run", "main.py", "-a", self.algo, "-p", f"tsplib-{self.problem}"]
            + ["--repetitions", str(self.repetitions())]
            + COMMON_TAIL
            + self.extra
        )

    def label(self) -> str:
        extra_str = " ".join(self.extra) if self.extra else ""
        reps = f"n={self.repetitions()}"
        return f"[{self.group}] {self.algo} @ {self.problem} {reps} {extra_str}".strip()


def build_experiments(
    problems: list[str] | None = None,
    groups: list[str] | None = None,
) -> list[Experiment]:
    """Build the active experiment plan: 160 configs over 4 problems.

    Three groups, which replace the old A-M lettering:

        BASE  bare algorithms on the Global topology, including Explore and
              Cognitive-Explore -- which work bare but not with a topology
        STAT  TOPO_ALGOS x STATIC_TOPOS x RADII
        DYN   TOPO_ALGOS x DYN_TOPOS x PCTS
        APVX  the fixed APV/SWU operators, swept over v_explore.  Opt-in: it is
              excluded unless named in *groups*, so the default plan is
              unchanged.

    Per problem that is 4 + (2x4x3) + (2x3x2) = 40 configs, so 160 over the four
    valid instances.  The retired 1548-config plan that produced the 941 existing
    results is still reproducible via build_legacy_experiments().
    """
    probs = problems or PROBLEMS
    all_groups: dict[str, list[Experiment]] = {
        "BASE": [Experiment("BASE", algo, prob)
                 for prob in probs for algo in BASELINE_ALGOS],
        "STAT": [Experiment("STAT", f"{algo}-{topo}", prob,
                            ["--neighbours-radius", str(r)])
                 for prob, algo, topo, r
                 in product(probs, TOPO_ALGOS, STATIC_TOPOS, RADII)],
        "DYN":  [Experiment("DYN", f"{algo}-{topo}", prob,
                            ["--param", f"neighbor_pct={pct}"])
                 for prob, algo, topo, pct
                 in product(probs, TOPO_ALGOS, DYN_TOPOS, PCTS)],
        "APVX": [Experiment("APVX", algo, prob,
                            ["--param", f"v_max={APVX_V_MAX}",
                             "--param", f"v_explore={ve}"])
                 for prob, algo, ve
                 in product([p for p in probs if p in APVX_PROBLEMS],
                            APVX_ALGOS, APVX_V_EXPLORE)],
    }

    selected = groups or [g for g in all_groups if g not in OPT_IN_GROUPS]
    result = []
    for g in selected:
        if g not in all_groups:
            print(f"Warning: unknown group '{g}', skipping", file=sys.stderr)
            continue
        result.extend(all_groups[g])
    return result


def build_legacy_experiments(
    problems: list[str] | None = None,
    groups: list[str] | None = None,
) -> list[Experiment]:
    """The retired 1548-config plan (groups A-M), kept for reproducibility.

    This is the plan the 941 completed results came from.  It sweeps all six
    operator families over all six problems, so it includes every combination
    later shown to be broken (APV, SWU, Explore-with-topology, Ring) and the two
    GEO instances whose distances are computed wrongly.  Do not submit it; it is
    here so existing result files remain explicable.
    """
    probs = problems or ALL_PROBLEMS
    legacy_algos = ["PSO", "PSO-Cognitive", "PSO-APV", "PSO-SWU",
                    "PSO-Explore", "PSO-Cognitive-Explore"]
    legacy_static = ["Ring", "Tree", "Mesh", "Torus", "FreeScale"]
    legacy_radii = [1, 2, 5, 10, 25, 50]
    legacy_pcts = [0.05, 0.10, 0.25, 0.50]

    # reps=REPS_DEFAULT explicitly: the historical plan used 10 repetitions for
    # every instance, before the large-instance reduction existed.  Without this
    # the legacy commands would no longer match the logs they produced.
    all_groups: dict[str, list[Experiment]] = {
        "A": [Experiment("A", algo, prob, reps=REPS_DEFAULT)
              for prob in probs for algo in legacy_algos],
    }
    for letter, algo in zip("BCDEFG", legacy_algos):
        all_groups[letter] = [
            Experiment(letter, f"{algo}-{topo}", prob,
                       ["--neighbours-radius", str(r)], reps=REPS_DEFAULT)
            for prob, r, topo in product(probs, legacy_radii, legacy_static)
        ]
    for letter, algo in zip("HIJKLM", legacy_algos):
        all_groups[letter] = [
            Experiment(letter, f"{algo}-{topo}", prob,
                       ["--param", f"neighbor_pct={pct}"], reps=REPS_DEFAULT)
            for prob, pct, topo in product(probs, legacy_pcts, DYN_TOPOS)
        ]

    selected = groups or list(all_groups.keys())
    result = []
    for g in selected:
        if g in all_groups:
            result.extend(all_groups[g])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunOutcome:
    exp: Experiment
    returncode: int       # -1 = runner-level exception (not subprocess failure)
    elapsed: float
    stdout: str
    stderr: str
    exception: str = ""   # set when the worker itself raised


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def run_experiment(exp: Experiment, log_dir: Path | None = None) -> RunOutcome:
    cmd = exp.to_cmd()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        outcome = RunOutcome(
            exp=exp,
            returncode=proc.returncode,
            elapsed=time.monotonic() - t0,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
    except Exception:
        outcome = RunOutcome(
            exp=exp,
            returncode=-1,
            elapsed=time.monotonic() - t0,
            stdout="",
            stderr="",
            exception=traceback.format_exc(),
        )

    if log_dir is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = _safe_filename(f"{exp.algo}_{exp.problem}_{ts}")
        log_path = log_dir / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(f"CMD: {' '.join(cmd)}\n")
            fh.write(f"exit_code: {outcome.returncode}\n")
            fh.write(f"elapsed_s: {outcome.elapsed:.2f}\n\n")
            if outcome.stdout:
                fh.write("--- stdout ---\n")
                fh.write(outcome.stdout + "\n")
            if outcome.stderr:
                fh.write("--- stderr ---\n")
                fh.write(outcome.stderr + "\n")
            if outcome.exception:
                fh.write("--- exception ---\n")
                fh.write(outcome.exception + "\n")

    return outcome


def _write_failure(log_path: Path, outcome: RunOutcome, seq: int) -> None:
    entry = {
        "seq": seq,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": outcome.exp.label(),
        "command": " ".join(outcome.exp.to_cmd()),
        "returncode": outcome.returncode,
        "elapsed_s": round(outcome.elapsed, 2),
        "stdout": outcome.stdout,
        "stderr": outcome.stderr,
        "exception": outcome.exception,
        # fields needed by --retry-failed to reconstruct the Experiment
        "_retry": {
            "group": outcome.exp.group,
            "algo": outcome.exp.algo,
            "problem": outcome.exp.problem,
            "extra": outcome.exp.extra,
        },
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_retry(log_path: Path) -> list[Experiment]:
    exps = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            r = entry["_retry"]
            exps.append(Experiment(
                group=r["group"],
                algo=r["algo"],
                problem=r["problem"],
                extra=r["extra"],
                reps=r.get("reps"),      # absent in older logs -> per-problem rule
            ))
    return exps


def main() -> None:
    import math

    cpu_count = os.cpu_count() or 2
    default_workers = max(1, math.ceil(cpu_count / 2))

    parser = argparse.ArgumentParser(description="Run PSO-TSP experiments in parallel")
    parser.add_argument("--workers", "-j", type=int, default=default_workers,
                        help=f"Parallel worker count (default: {default_workers} = ceil(cpus/2))")
    parser.add_argument("--numba-threads", type=int, default=2,
                        help="NUMBA_NUM_THREADS per worker process (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--groups", nargs="+", metavar="G",
                        help="Run only specific groups, e.g. --groups A B H")
    parser.add_argument("--problems", nargs="+", metavar="P",
                        help="Override problem list, e.g. --problems berlin52 bier127 "
                             "(bypasses the validity exclusions)")
    parser.add_argument("--all-problems", action="store_true",
                        help=f"Also include the GEO instances whose distances are "
                             f"computed wrongly: {', '.join(EXCLUDED_PROBLEMS)}. "
                             f"Their results are meaningless until "
                             f"data/tsplib/reader.py implements GEO distance.")
    parser.add_argument("--retry-failed", metavar="LOG",
                        help="Re-run only the experiments listed in a failure log file")
    parser.add_argument("--start", type=int, default=0, metavar="N",
                        help="First experiment index to run, 1-based (default: 1)")
    parser.add_argument("--end", type=int, default=None, metavar="N",
                        help="Last experiment index to run, 1-based inclusive (default: last)")
    args = parser.parse_args()

    if args.retry_failed:
        all_experiments = _load_retry(Path(args.retry_failed))
        print(f"Retrying {len(all_experiments)} failed experiments from {args.retry_failed}")
    else:
        problems = args.problems or (ALL_PROBLEMS if args.all_problems else PROBLEMS)
        all_experiments = build_experiments(problems=problems, groups=args.groups)
        if not args.problems and not args.all_problems and EXCLUDED_PROBLEMS:
            print(f"Problem scope: {', '.join(problems)}  "
                  f"(excluding {', '.join(EXCLUDED_PROBLEMS)} — GEO instances, "
                  f"distances computed wrongly; --all-problems overrides)")

    # 1-based inclusive slice: --start 1 --end 100 runs experiments 1..100
    start_idx = max(0, args.start - 1) if args.start else 0
    end_idx   = args.end if args.end else len(all_experiments)
    experiments = all_experiments[start_idx:end_idx]
    total_all   = len(all_experiments)

    if args.dry_run:
        for i, exp in enumerate(experiments, start=start_idx + 1):
            print(f"[{i}/{total_all}] {' '.join(exp.to_cmd())}")
        print(f"\n# Showing {len(experiments)} of {total_all} experiments", file=sys.stderr)
        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fail_log = results_dir / f"failed_{run_ts}.log"
    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    range_note = f"  (experiments {start_idx+1}–{end_idx} of {total_all})" if (start_idx or end_idx != total_all) else ""
    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.numba_threads))
    print(
        f"Running {len(experiments)} experiments{range_note} "
        f"with {args.workers} workers "
        f"(NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']})\n"
        f"Experiment logs: {log_dir}/\n"
        f"Failure log:     {fail_log}"
    )

    done = 0
    n_ok = 0
    n_fail = 0
    t_start = time.monotonic()

    # map future → (stable 1-based index in the full list, experiment)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_experiment, exp, log_dir): (start_idx + i + 1, exp)
            for i, exp in enumerate(experiments)
        }
        for future in as_completed(futures):
            stable_idx, exp = futures[future]
            try:
                outcome = future.result()
            except Exception:
                # Should not happen — run_experiment catches everything,
                # but guard against executor-level failures (e.g. worker OOM).
                exc = traceback.format_exc()
                outcome = RunOutcome(
                    exp=exp,
                    returncode=-1,
                    elapsed=0.0,
                    stdout="",
                    stderr="",
                    exception=exc,
                )
                log_path = log_dir / f"{_safe_filename(exp.algo + '_' + exp.problem)}_executor_err.log"
                log_path.write_text(f"CMD: {' '.join(exp.to_cmd())}\n\n{exc}", encoding="utf-8")

            done += 1
            ok = outcome.returncode == 0
            n_ok += ok
            n_fail += not ok

            elapsed_total = time.monotonic() - t_start
            eta_sec = (elapsed_total / done) * (len(experiments) - done)
            eta = str(timedelta(seconds=int(eta_sec)))

            status = "OK  " if ok else "FAIL"
            print(
                f"[{done:4d}/{len(experiments)}  #{stable_idx}/{total_all}] {status} "
                f"{outcome.elapsed:6.1f}s  ETA {eta}  {outcome.exp.label()}"
            )

            if not ok:
                _write_failure(fail_log, outcome, seq=done)
                # one-line hint so the user notices without scrolling to the log
                detail = outcome.exception or outcome.stderr
                print(f"  └─ logged to {fail_log.name}: {detail[:120].splitlines()[0]}")

    wall = timedelta(seconds=int(time.monotonic() - t_start))
    print(f"\nFinished in {wall}  —  {n_ok} OK, {n_fail} FAILED.")
    if n_fail:
        print(f"Failure details: {fail_log}")
        print(f"To retry:  uv run experiments/run_parallel.py --retry-failed {fail_log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
