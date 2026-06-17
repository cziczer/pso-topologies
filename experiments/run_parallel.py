#!/usr/bin/env python3
"""Parallel PSO-TSP experiment runner.

Generates the same 1 782 experiment commands as commands.sh and executes them
concurrently using subprocess workers.  Results land in results/ as usual.

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

PROBLEMS = [
    "berlin52", "bier127", "a280", "ali535", "gr666", "pr1002",
]

STATIC_TOPOS = ["Ring", "Tree", "Mesh", "Torus", "FreeScale"]
DYN_TOPOS    = ["DynSim", "DynOpp", "DynMix"]
RADII        = [1, 2, 5, 10, 25, 50]
PCTS         = [0.05, 0.10, 0.25, 0.50]

COMMON = ["--repetitions", "10", "--particles", "1000", "--max-iter", "30000", "--quiet"]


@dataclass
class Experiment:
    group: str
    algo: str
    problem: str
    extra: list[str] = field(default_factory=list)

    def to_cmd(self) -> list[str]:
        return (
            ["uv", "run", "main.py", "-a", self.algo, "-p", f"tsplib-{self.problem}"]
            + COMMON
            + self.extra
        )

    def label(self) -> str:
        extra_str = " ".join(self.extra) if self.extra else ""
        return f"[{self.group}] {self.algo} @ {self.problem} {extra_str}".strip()


def _static_block(group: str, algo_prefix: str, problems: list[str]) -> list[Experiment]:
    exps = []
    for prob, r, topo in product(problems, RADII, STATIC_TOPOS):
        exps.append(Experiment(
            group=group,
            algo=f"{algo_prefix}-{topo}",
            problem=prob,
            extra=["--neighbours-radius", str(r)],
        ))
    return exps


def _dyn_block(group: str, algo_prefix: str, problems: list[str]) -> list[Experiment]:
    exps = []
    for prob, pct, topo in product(problems, PCTS, DYN_TOPOS):
        exps.append(Experiment(
            group=group,
            algo=f"{algo_prefix}-{topo}",
            problem=prob,
            extra=["--param", f"neighbor_pct={pct}"],
        ))
    return exps


def build_experiments(
    problems: list[str] | None = None,
    groups: list[str] | None = None,
) -> list[Experiment]:
    probs = problems or PROBLEMS
    all_groups: dict[str, list[Experiment]] = {}

    # A — Baselines (Global topology)
    a = []
    for prob in probs:
        for algo in ["PSO", "PSO-Cognitive", "PSO-APV", "PSO-SWU",
                     "PSO-Explore", "PSO-Cognitive-Explore"]:
            a.append(Experiment("A", algo, prob))
    all_groups["A"] = a

    all_groups["B"] = _static_block("B", "PSO",                  probs)
    all_groups["C"] = _static_block("C", "PSO-Cognitive",        probs)
    all_groups["D"] = _static_block("D", "PSO-APV",              probs)
    all_groups["E"] = _static_block("E", "PSO-SWU",              probs)
    all_groups["F"] = _static_block("F", "PSO-Explore",          probs)
    all_groups["G"] = _static_block("G", "PSO-Cognitive-Explore",probs)

    all_groups["H"] = _dyn_block("H", "PSO",                   probs)
    all_groups["I"] = _dyn_block("I", "PSO-Cognitive",         probs)
    all_groups["J"] = _dyn_block("J", "PSO-APV",               probs)
    all_groups["K"] = _dyn_block("K", "PSO-SWU",               probs)
    all_groups["L"] = _dyn_block("L", "PSO-Explore",           probs)
    all_groups["M"] = _dyn_block("M", "PSO-Cognitive-Explore",  probs)

    selected = groups or list(all_groups.keys())
    result = []
    for g in selected:
        if g not in all_groups:
            print(f"Warning: unknown group '{g}', skipping", file=sys.stderr)
            continue
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
                        help="Override problem list, e.g. --problems berlin52 bier127")
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
        all_experiments = build_experiments(problems=args.problems, groups=args.groups)

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
