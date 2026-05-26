#!/usr/bin/env python3
"""Run all registered cases using final closed-tearing semantics."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from case_registry import CaseSpec, build_cases  # type: ignore[reportMissingImports]
from cpsat_tearing_core import TearingConfig, TearingResult, classify_tearing  # type: ignore[reportMissingImports]

OUT_ROOT = PROJECT_ROOT / "results"


def _global_status(result: TearingResult) -> str:
    statuses = set(result.status_by_phase.values())
    return "OPTIMAL" if statuses == {"OPTIMAL"} else ",".join(sorted(statuses))


def _summary_row(case: CaseSpec, result: TearingResult) -> Dict[str, object]:
    return {
        "Order": case.order,
        "Group": case.group,
        "Case": case.name,
        "Q": result.total_equations,
        "V": result.total_variables,
        "A": result.total_incidences,
        "ModelQ": result.model_equations,
        "ModelA": result.model_incidences,
        "StructuralRank": result.structural_rank,
        "PreRemovedEqCount": len(result.removed_equations),
        "PreRemovedEqList": ", ".join(result.removed_equations),
        "Sensors": len(result.measured),
        "KnownConstants": len(result.known_constants),
        "Direct": len(result.inferred_direct),
        "ClosedLoop": len(result.inferred_closed_loop),
        "ClosedTears": len(result.tears_closed),
        "OpenConditioned": len(result.inferred_conditioned_open),
        "ClosedExternalTears": len(result.tears_closed_external),
        "OpenTears": len(result.tears_open),
        "PureIndeterminate": len(result.indeterminate_pure),
        "EffectiveIndeterminate": len(result.effective_indeterminate),
        "ClosedCoverage": f"{result.n_closed_coverage}/{result.total_variables}",
        "ExternalReach": f"{result.n_external_reach}/{result.total_variables}",
        "DirectCoverage": f"{result.n_direct}/{result.total_variables}",
        "PctClosedCoverage": round(100 * result.n_closed_coverage / result.total_variables, 4),
        "PctExternalReach": round(100 * result.n_external_reach / result.total_variables, 4),
        "PctDirectCoverage": round(100 * result.n_direct / result.total_variables, 4),
        "RawTears": len(result.tear_pairs),
        "CyclicSCCs": len(result.cyclic_sccs_full_graph),
        "CyclicSCCList": "; ".join(",".join(c) for c in result.cyclic_sccs_full_graph),
        "ExecutionGraphDAG": "Yes" if result.execution_graph_is_dag else "No",
        "SolverStatus": _global_status(result),
        "TotalPhaseTime_s": round(sum(result.time_by_phase_s.values()), 6),
    }


def _execution_order_rows(case: CaseSpec, result: TearingResult) -> List[Dict[str, object]]:
    """Build execution-order rows grouped by topological level.

    Only variables with a selected output equation are included.
    """
    rows: List[Dict[str, object]] = []
    grouped: Dict[int, List[str]] = {}
    grouped_eqs: Dict[int, set[str]] = {}

    for variable, equation in result.output_equation.items():
        if not equation:
            continue
        level_raw = result.levels.get(variable)
        if level_raw is None or str(level_raw) == "":
            continue
        level = int(level_raw)
        grouped.setdefault(level, []).append(variable)
        grouped_eqs.setdefault(level, set()).add(equation)

    for level in sorted(grouped):
        variables = sorted(grouped[level])
        equations = sorted(grouped_eqs[level])
        rows.append(
            {
                "CaseSlug": case.slug,
                "Case": case.name,
                "Level": level,
                "Variables": ", ".join(variables),
                "OutputEquations": ", ".join(equations),
                "VariableCount": len(variables),
            }
        )
    return rows


def _write_case(case_dir: Path, case: CaseSpec, result: TearingResult, row: Dict[str, object]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "resultado.json").write_text(
        json.dumps({"case": case.name, "result": result.to_dict()}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    classes = [
        ("measured_sensor", result.measured),
        ("known_constant", result.known_constants),
        ("inferred_direct", result.inferred_direct),
        ("inferred_closed_loop", result.inferred_closed_loop),
        ("closed_tear", result.tears_closed),
        ("inferred_conditioned_open_tear", result.inferred_conditioned_open),
        ("closed_tear_external_dependency", result.tears_closed_external),
        ("open_tear", result.tears_open),
        ("pure_indeterminate", result.indeterminate_pure),
    ]
    variable_rows: List[Dict[str, object]] = []
    for kind, variables in classes:
        for var in variables:
            variable_rows.append({
                "Variable": var,
                "Class": kind,
                "OutputEquation": result.output_equation.get(var, ""),
                "UsesLocalTears": ", ".join(result.output_uses_tears.get(var, [])),
                "Level": result.levels.get(var, ""),
            })
    tears_rows = [{"Equation": q, "Variable": v, "Status": (
        "open" if v in result.tears_open else
        "closed_external" if v in result.tears_closed_external else "closed"
    )} for q, v in result.tear_pairs]
    pd.DataFrame(variable_rows).to_csv(case_dir / "variables.csv", index=False)
    pd.DataFrame(tears_rows).to_csv(case_dir / "tears.csv", index=False)
    execution_order_rows = _execution_order_rows(case, result)
    pd.DataFrame(execution_order_rows).to_csv(case_dir / "execution_order.csv", index=False)
    with pd.ExcelWriter(case_dir / "resultado.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([row]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(variable_rows).to_excel(writer, sheet_name="Variables", index=False)
        pd.DataFrame(tears_rows).to_excel(writer, sheet_name="Tears", index=False)
        pd.DataFrame(execution_order_rows).to_excel(writer, sheet_name="ExecutionOrder", index=False)
        pd.DataFrame(result.cut_graph_edges, columns=["Source", "Target"]).to_excel(writer, sheet_name="ExecutionGraph", index=False)
        pd.DataFrame(result.full_graph_edges, columns=["Source", "Target"]).to_excel(writer, sheet_name="FullGraph", index=False)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{CP-SAT classification with closed \emph{tearing}. $C_{cl}$ denotes closed/autonomous coverage, $C_{ext}$ denotes conditional external reach, and $C_{dir}$ denotes direct acyclic propagation coverage.}",
        r"\label{tab:tearing-main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrlllrr}",
        r"\toprule",
        r"Case & Sens. & $K$ & Dir. & Loop & Closed T. & Closed T. ext. & Open cond. & Open T. & Eff. indet. & $C_{cl}$ & $C_{ext}$ & $C_{dir}$ & Raw T. & Cyclic SCC \\",
        r"\midrule",
    ]
    current_group = None
    for _, row in df.iterrows():
        group = str(row["Group"])
        if group != current_group:
            if current_group is not None:
                lines.append(r"\addlinespace")
            lines.append(rf"\multicolumn{{15}}{{l}}{{\textit{{{_latex_escape(group)}}}}} \\")
            current_group = group
        values = [
            _latex_escape(row["Case"]),
            row["Sensors"],
            row["KnownConstants"],
            row["Direct"],
            row["ClosedLoop"],
            row["ClosedTears"],
            row["ClosedExternalTears"],
            row["OpenConditioned"],
            row["OpenTears"],
            row["EffectiveIndeterminate"],
            row["ClosedCoverage"],
            row["ExternalReach"],
            row["DirectCoverage"],
            row["RawTears"],
            row["CyclicSCCs"],
        ]
        lines.append(" & ".join(str(v) for v in values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final closed-tearing analysis for registered cases.")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--allow-feasible", action="store_true")
    parser.add_argument("--case", action="append", help="Case slug (can be repeated).")
    args = parser.parse_args()
    config = TearingConfig(time_limit_s=args.time_limit, require_optimal=not args.allow_feasible)
    cases = build_cases()
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.slug in selected]
        missing = selected - {case.slug for case in cases}
        if missing:
            raise SystemExit(f"Unknown case slugs: {sorted(missing)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    execution_order_rows_all: List[Dict[str, object]] = []
    print(f"Output: {run_dir}")
    for case in cases:
        result = classify_tearing(
            case.equations_fn(), set(case.measured), known_constants=set(case.known_constants),
            allowed_outputs=case.allowed_outputs, case_name=case.name, config=config,
        )
        row = _summary_row(case, result)
        rows.append(row)
        execution_order_rows_all.extend(_execution_order_rows(case, result))
        _write_case(run_dir / case.slug, case, result, row)
        print(
            f"[{case.order:02d}] {case.name}: closed={row['ClosedCoverage']} "
            f"external={row['ExternalReach']} direct={row['DirectCoverage']} "
            f"indet={row['EffectiveIndeterminate']} open={row['OpenTears']} "
            f"SCC={row['CyclicSCCs']} status={row['SolverStatus']}"
        )
    df = pd.DataFrame(rows).sort_values("Order")
    consol = run_dir / "consolidated"
    consol.mkdir(exist_ok=True)
    df.to_csv(consol / "tearing_results_table.csv", index=False)
    pd.DataFrame(execution_order_rows_all).to_csv(consol / "execution_order_summary.csv", index=False)
    with pd.ExcelWriter(consol / "tearing_results_table.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(execution_order_rows_all).to_excel(writer, sheet_name="ExecutionOrder", index=False)
    _write_latex_table(df, consol / "tearing_results_table.tex")
    _write_latex_table(df, PROJECT_ROOT / "tearing_results_table.tex")
    audit = {
        "semantics_version": "tearing_closed",
        "executed_at": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config.__dict__,
        "cases": len(rows),
        "all_optimal": all(row["SolverStatus"] == "OPTIMAL" for row in rows),
        "all_execution_graphs_dag": all(row["ExecutionGraphDAG"] == "Yes" for row in rows),
        "fully_autonomous_cases": sum(row["ClosedCoverage"].split("/")[0] == row["ClosedCoverage"].split("/")[1] for row in rows),
        "cyclic_sccs_total": int(df["CyclicSCCs"].sum()),
    }
    (consol / "auditoria.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Consolidated: {consol}")


if __name__ == "__main__":
    main()
