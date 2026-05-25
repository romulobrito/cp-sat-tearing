#!/usr/bin/env python3
"""Timing benchmark for the current CP-SAT formulation.

This experiment replicates the historical combinations benchmark on Narasimhan
while calling the closed-tearing formulation. It records both full
wrapper time and summed internal phase times.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from case_registry import build_cases  # noqa: E402
from cpsat_tearing_core import (  # noqa: E402
    TearingConfig,
    TearingResult,
    all_variables,
    classify_tearing,
)

OUT_ROOT = PROJECT_ROOT / "results_time_benchmark"


def _case_by_slug(slug: str):
    for case in build_cases():
        if case.slug == slug:
            return case
    valid = ", ".join(case.slug for case in build_cases())
    raise SystemExit(f"Unknown case slug: {slug}. Valid slugs: {valid}")


def _sample_combinations(
    variables: Sequence[str],
    k: int,
    n_samples: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    """Sample unique combinations without materializing the full space."""

    total = math.comb(len(variables), k)
    if n_samples > total:
        raise ValueError(f"n_samples={n_samples} exceeds total combinations ({total}).")
    rng = random.Random(seed)
    combinations = set()
    while len(combinations) < n_samples:
        combinations.add(tuple(sorted(rng.sample(list(variables), k))))
    return sorted(combinations)


def _iter_combinations(
    variables: Sequence[str],
    k: int,
    *,
    full: bool,
    n_samples: int,
    seed: int,
) -> Iterable[Tuple[str, ...]]:
    if full:
        return itertools.combinations(variables, k)
    return _sample_combinations(variables, k, n_samples, seed)


def _status_global(result: TearingResult) -> str:
    statuses = set(result.status_by_phase.values())
    return "OPTIMAL" if statuses == {"OPTIMAL"} else ",".join(sorted(statuses))


def _result_row(
    idx: int,
    combo: Tuple[str, ...],
    result: TearingResult | None,
    error: str | None,
    elapsed_wrapper: float,
) -> Dict[str, object]:
    base: Dict[str, object] = {
        "CombinationID": idx,
        "MeasuredVariables": ", ".join(combo),
        "NumMeasurements": len(combo),
        "ClassificationTime_s": round(elapsed_wrapper, 6),
        "Error": error or "",
    }
    if result is None:
        base.update(
            {
                "SolverStatus": "ERROR",
                "ClosedCoverage": "0/0",
                "ExternalReach": "0/0",
                "DirectCoverage": "0/0",
                "PctClosedCoverage": 0.0,
                "PctExternalReach": 0.0,
                "PctDirectCoverage": 0.0,
                "EffectiveIndeterminate": "",
                "OpenTears": "",
                "RawTears": "",
                "CyclicSCCs": "",
                "ExecutionGraphDAG": "",
                "SolverPhaseTime_s": 0.0,
            }
        )
        return base

    total = result.total_variables
    base.update(
        {
            "SolverStatus": _status_global(result),
            "ClosedCoverage": f"{result.n_closed_coverage}/{total}",
            "ExternalReach": f"{result.n_external_reach}/{total}",
            "DirectCoverage": f"{result.n_direct}/{total}",
            "PctClosedCoverage": round(100 * result.n_closed_coverage / total, 4),
            "PctExternalReach": round(100 * result.n_external_reach / total, 4),
            "PctDirectCoverage": round(100 * result.n_direct / total, 4),
            "EffectiveIndeterminate": len(result.effective_indeterminate),
            "OpenTears": len(result.tears_open),
            "RawTears": len(result.tear_pairs),
            "CyclicSCCs": len(result.cyclic_sccs_full_graph),
            "ExecutionGraphDAG": "Yes" if result.execution_graph_is_dag else "No",
            "SolverPhaseTime_s": round(sum(result.time_by_phase_s.values()), 6),
        }
    )
    return base


def _summary_rows(
    *,
    case_name: str,
    n_variables: int,
    n_equations: int,
    k: int,
    total_space: int,
    n_tested: int,
    n_closed_full: int,
    n_external_full: int,
    n_errors: int,
    total_classification_time: float,
    total_solver_phase_time: float,
    total_wall_time: float,
    seed: int,
    full: bool,
    time_limit: float,
) -> pd.DataFrame:
    overhead = max(0.0, total_wall_time - total_classification_time)
    rows = [
        ("System", case_name),
        ("TotalVariables", n_variables),
        ("TotalEquations", n_equations),
        ("SetSizeK", k),
        ("TotalCombinationSpace", total_space),
        ("Mode", "exhaustive" if full else "sampling"),
        ("Seed", seed),
        ("PhaseTimeLimit_s", time_limit),
        ("TestedCombinations", n_tested),
        ("FullClosedCoverageCount", n_closed_full),
        ("FullClosedCoveragePct", f"{100*n_closed_full/n_tested:.2f}%"),
        ("FullExternalReachCount", n_external_full),
        ("FullExternalReachPct", f"{100*n_external_full/n_tested:.2f}%"),
        ("ErrorCases", n_errors),
        ("TotalClassificationTime_s", f"{total_classification_time:.6f}"),
        ("TotalSolverPhaseTime_s", f"{total_solver_phase_time:.6f}"),
        ("TotalWallTime_s", f"{total_wall_time:.6f}"),
        ("OutsideClassificationOverhead_s", f"{overhead:.6f}"),
        ("AvgClassificationTimePerCombination_s", f"{total_classification_time/n_tested:.6f}"),
        ("AvgSolverTimePerCombination_s", f"{total_solver_phase_time/n_tested:.6f}"),
        ("ClassificationThroughput_comb_per_s", f"{n_tested/total_classification_time:.6f}"),
        ("WallThroughput_comb_per_s", f"{n_tested/total_wall_time:.6f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combinations benchmark for the current CP-SAT formulation."
    )
    parser.add_argument("--case", default="18_narasimhan", help="Case slug.")
    parser.add_argument("--k", type=int, default=17, help="Number of sensors per combination.")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of sampled combinations.")
    parser.add_argument("--full", action="store_true", help="Run full combinatorial space.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=60.0, help="Per-phase time limit.")
    parser.add_argument("--allow-feasible", action="store_true", help="Accept FEASIBLE in addition to OPTIMAL.")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable structural preprocessing.")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    case = _case_by_slug(args.case)
    eqs = case.equations_fn()
    variables = all_variables(eqs)
    total_space = math.comb(len(variables), args.k)
    n_expected = total_space if args.full else min(args.n_samples, total_space)
    combinations = _iter_combinations(
        variables,
        args.k,
        full=args.full,
        n_samples=n_expected,
        seed=args.seed,
    )
    config = TearingConfig(
        time_limit_s=args.time_limit,
        require_optimal=not args.allow_feasible,
        random_seed=args.seed,
        use_structural_preprocessing=not args.no_preprocessing,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"System: {case.name}")
    print(f"Equations: {len(eqs)} | Variables: {len(variables)} | k={args.k}")
    print(f"Total space: {total_space}")
    print(f"Mode: {'exhaustive' if args.full else f'sampling {n_expected} combinations'}")
    print(f"Output: {run_dir}")

    rows: List[Dict[str, object]] = []
    total_classification_time = 0.0
    total_solver_time = 0.0
    wall_start = time.perf_counter()

    for idx, combo in enumerate(combinations, 1):
        start = time.perf_counter()
        result = None
        error = None
        try:
            result = classify_tearing(
                eqs,
                set(combo),
                known_constants=set(case.known_constants),
                allowed_outputs=case.allowed_outputs,
                case_name=case.name,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:300]
        elapsed = time.perf_counter() - start
        total_classification_time += elapsed
        if result is not None:
            total_solver_time += sum(result.time_by_phase_s.values())
        rows.append(_result_row(idx, combo, result, error, elapsed))

        if args.progress_every and (idx % args.progress_every == 0 or idx == n_expected):
            elapsed_wall = time.perf_counter() - wall_start
            rate = idx / elapsed_wall if elapsed_wall > 0 else 0.0
            remaining = n_expected - idx
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[PROGRESS] {idx}/{n_expected} ({100*idx/n_expected:.1f}%) | "
                f"classification_time={total_classification_time:.2f}s | "
                f"wall_time={elapsed_wall:.2f}s | ETA={eta:.1f}s"
            )

    total_wall_time = time.perf_counter() - wall_start
    df = pd.DataFrame(rows)
    n_errors = int((df["SolverStatus"] == "ERROR").sum())
    n_closed_full = int((df["ClosedCoverage"] == f"{len(variables)}/{len(variables)}").sum())
    n_external_full = int((df["ExternalReach"] == f"{len(variables)}/{len(variables)}").sum())
    df_resumo = _summary_rows(
        case_name=case.name,
        n_variables=len(variables),
        n_equations=len(eqs),
        k=args.k,
        total_space=total_space,
        n_tested=len(df),
        n_closed_full=n_closed_full,
        n_external_full=n_external_full,
        n_errors=n_errors,
        total_classification_time=total_classification_time,
        total_solver_phase_time=total_solver_time,
        total_wall_time=total_wall_time,
        seed=args.seed,
        full=args.full,
        time_limit=args.time_limit,
    )

    csv_path = run_dir / "benchmark_combinations.csv"
    summary_csv_path = run_dir / "benchmark_summary.csv"
    xlsx_path = run_dir / "benchmark_combinations.xlsx"
    json_path = run_dir / "benchmark_audit.json"
    df.to_csv(csv_path, index=False)
    df_resumo.to_csv(summary_csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_resumo.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="AllCombinations", index=False)
        stats = df["ClosedCoverage"].value_counts().rename_axis("ClosedCoverage").reset_index(name="Count")
        stats.to_excel(writer, sheet_name="CoverageStats", index=False)

    audit = {
        "executed_at": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "case": case.name,
        "case_slug": case.slug,
        "k": args.k,
        "seed": args.seed,
        "full": args.full,
        "n_tested": len(df),
        "total_space": total_space,
        "config": config.__dict__,
        "total_classification_time_s": total_classification_time,
        "total_solver_phase_time_s": total_solver_time,
        "total_wall_time_s": total_wall_time,
        "n_closed_full": n_closed_full,
        "n_external_full": n_external_full,
        "n_errors": n_errors,
    }
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary")
    for _, row in df_resumo.iterrows():
        print(f"{row['Metric']}: {row['Value']}")
    print(f"\nArtifacts generated in: {run_dir}")


if __name__ == "__main__":
    main()
