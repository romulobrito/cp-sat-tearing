# CP-SAT Tearing

Final, reproducible CP-SAT pipeline for structural variable classification with closed tearing semantics.

## Motivation

Process monitoring and reconciliation workflows need to separate:
- what is directly measured;
- what is autonomously inferable in closed structure;
- what is only conditionally reachable through open tears.

This repository provides a deterministic and auditable CP-SAT implementation for that purpose.

## Problem formulation (high level)

Given equations and variables, the model:
- assigns at most one output variable per equation;
- allows local tears to break cyclic dependencies;
- classifies variables by structural status;
- reports complementary coverage metrics: `C_dir`, `C_cl`, `C_ext`.

## Where it was applied

Studied cases included in this package:
- URS mass-balance scenarios (ideal and real variants);
- URS stage KPI and bank KPI scenarios;
- Narasimhan steam plant;
- Sanchez and Romagnoli olefins plant.

## General result summary

At a high level, runs show:
- ideal URS case reaches full closed coverage (`C_cl = 43/43`);
- real URS variants expose conditional reach and open-tear dependence;
- selected added-measurement scenarios recover autonomous coverage;
- literature cases illustrate both fully closed and partially open structural regimes.

## Reproduce

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 tests/test_tearing_semantics.py
python3 scripts/run_cases.py
```

Optional:

```bash
python3 scripts/run_matching_ablation.py
python3 scripts/run_time_benchmark.py --case 18_narasimhan --k 17 --n-samples 1000
```

## Outputs

Main audit artifacts are generated at:

- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.csv`
- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.xlsx`
- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.tex`
- `results/run_YYYYMMDD_HHMMSS/consolidated/auditoria.json`

## About paper LaTeX results

The script `scripts/run_cases.py` generates the CP-SAT consolidated LaTeX table used as the reproducible source for the CP-SAT result block:
- `tearing_results_table.tex`

Historical Incidence/QR comparison numbers are reference baselines in the case definitions, not recomputed by this solver.

## Full technical documentation

See `docs/README.tex` for the full formulation, semantics, execution guide, and audit interpretation.
