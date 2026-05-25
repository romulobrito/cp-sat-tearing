#!/usr/bin/env python3
"""Tests for closed coverage and external reach semantics."""
from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpsat_tearing_core import (
    TearingConfig,
    classify_tearing,
    preprocess_equations_structural,
)
from case_registry import narasimhan_measured
import narasimhan_steam_plant as narasimhan


def cfg() -> TearingConfig:
    return TearingConfig(time_limit_s=30, require_optimal=True)


def test_inferencia_direta() -> None:
    r = classify_tearing({"q": ["m", "x"]}, {"m"}, config=cfg(), case_name="direto")
    assert r.n_closed_coverage == 2
    assert r.n_external_reach == 2
    assert r.n_direct == 2
    assert r.inferred_direct == ["x"]
    assert r.effective_indeterminate == []


def test_tear_aberto_nao_e_cobertura_fechada() -> None:
    r = classify_tearing({"q": ["x", "y"]}, set(), config=cfg(), case_name="aberto")
    assert r.n_closed_coverage == 0
    assert r.n_external_reach == 1
    assert len(r.inferred_conditioned_open) == 1
    assert len(r.tears_open) == 1
    assert len(r.effective_indeterminate) == 2


def test_dependencia_transitiva_de_tear_aberto() -> None:
    eqs = {"q1": ["x", "y"], "q2": ["z", "x"]}
    r = classify_tearing(eqs, set(), config=cfg(), case_name="propagacao_open")
    assert r.n_closed_coverage == 0
    assert r.n_external_reach == 2
    assert len(r.tears_open) == 1
    assert set(r.inferred_conditioned_open) == {"x", "z"}
    assert set(r.effective_indeterminate) == {"x", "y", "z"}


def test_tear_fechado_cobre_bloco_sem_afirmar_unicidade_numerica() -> None:
    eqs = {"q1": ["m", "x", "y"], "q2": ["x", "y"]}
    r = classify_tearing(eqs, {"m"}, config=cfg(), case_name="fechado")
    assert r.n_closed_coverage == 3
    assert r.n_external_reach == 3
    assert r.tears_open == []
    assert len(r.tears_closed) == 1
    assert len(r.inferred_closed_loop) == 1
    assert r.cyclic_sccs_full_graph == [["x", "y"]]
    assert r.execution_graph_is_dag


def test_parametro_conhecido_nao_e_sensor_nem_saida() -> None:
    r = classify_tearing(
        {"q": ["KPI", "P", "A"]}, {"P"}, known_constants={"A"},
        allowed_outputs={"q": {"KPI"}}, config=cfg(), case_name="kpi",
    )
    assert r.measured == ["P"]
    assert r.known_constants == ["A"]
    assert r.inferred_direct == ["KPI"]
    assert "A" not in r.output_equation
    assert r.n_closed_coverage == 3


def test_preprocessamento_matching_remove_equacao_sem_incidencia() -> None:
    eqs = {"q1": ["x"], "q_vazia": []}
    reduced, removed, structural_rank = preprocess_equations_structural(eqs, ["x"])
    assert structural_rank == 1
    assert reduced == {"q1": ["x"]}
    assert removed == ["q_vazia"]


def test_classificador_registra_preprocessamento() -> None:
    r = classify_tearing({"q1": ["m", "x"], "q_vazia": []}, {"m"}, config=cfg(), case_name="pre")
    assert r.n_closed_coverage == 2
    assert r.model_equations == 1
    assert r.total_equations == 2
    assert r.structural_rank == 1
    assert r.removed_equations == ["q_vazia"]


def test_narasimhan_sem_dependencia_aberta() -> None:
    r = classify_tearing(
        narasimhan.equations_narasimhan(), narasimhan_measured(),
        config=cfg(), case_name="narasimhan",
    )
    assert r.n_closed_coverage == 28
    assert r.n_external_reach == 28
    assert r.effective_indeterminate == []
    assert r.tears_open == []
    assert r.execution_graph_is_dag


def main() -> None:
    tests = [
        test_inferencia_direta,
        test_tear_aberto_nao_e_cobertura_fechada,
        test_dependencia_transitiva_de_tear_aberto,
        test_tear_fechado_cobre_bloco_sem_afirmar_unicidade_numerica,
        test_parametro_conhecido_nao_e_sensor_nem_saida,
        test_preprocessamento_matching_remove_equacao_sem_incidencia,
        test_classificador_registra_preprocessamento,
        test_narasimhan_sem_dependencia_aberta,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("ALL_TESTS_OK")


if __name__ == "__main__":
    main()
