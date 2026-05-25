# CP-SAT Tearing

Final, reproducible CP-SAT pipeline for structural variable classification with closed tearing semantics.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 tests/test_tearing_semantics.py
python3 scripts/run_cases.py
```

## Outputs

Main audit artifacts are generated at:

- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.csv`
- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.xlsx`
- `results/run_YYYYMMDD_HHMMSS/consolidated/tearing_results_table.tex`
- `results/run_YYYYMMDD_HHMMSS/consolidated/auditoria.json`

## Full technical documentation

See `docs/README.tex` for the full formulation, semantics, execution guide, and audit interpretation.
