PYTHON ?= python3

.PHONY: setup test run-cases run-ablation run-benchmark audit

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) tests/test_tearing_semantics.py

run-cases:
	$(PYTHON) scripts/run_cases.py

run-ablation:
	$(PYTHON) scripts/run_matching_ablation.py

run-benchmark:
	$(PYTHON) scripts/run_time_benchmark.py

audit: run-cases
	@echo "Audit generated under results/run_*/consolidated/auditoria.json"
