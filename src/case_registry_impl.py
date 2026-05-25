#!/usr/bin/env python3
"""Case registry for the final tearing pipeline.

Key semantics:
- ``measured`` contains only truly measured process signals;
- ``known_constants`` contains known physical parameters (not sensors);
- KPI cases use ``allowed_outputs`` to restrict each equation to meaningful KPI
  outputs, avoiding non-physical inversions (e.g., solving for membrane area);
- incidence/QR/legacy-CP values are historical references, not recomputed by
  this pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import narasimhan_steam_plant as narasimhan  # noqa: E402
import classification_Sanchez_and_Romagnoli as olefinas  # noqa: E402
import URS_classification_cases_ideal_real_real_min_inst_kpi_bancos as urs_bancos  # noqa: E402
import URS_classification_real_min_inst_kpis_estagios as urs_estagios  # noqa: E402

Equations = Dict[str, List[str]]


@dataclass(frozen=True)
class HistoricalBaseline:
    obs: str
    tears: int
    inf: int


@dataclass(frozen=True)
class CaseSpec:
    order: int
    group: str
    name: str
    slug: str
    equations_fn: Callable[[], Equations]
    measured: Set[str]
    known_constants: Set[str]
    allowed_outputs: Optional[Mapping[str, Set[str]]] = None
    incidence_ref: Optional[HistoricalBaseline] = None
    qr_ref: Optional[HistoricalBaseline] = None
    cp_old_ref: Optional[HistoricalBaseline] = None


def urs_ideal_measured() -> Set[str]:
    return {
        "Pa_A", "Pa_B", "Pa_C", "Pa_D", "Pa_E", "Pb_A", "Pb_B", "Pb_C", "Pb_D", "Pb_E",
        "Pc_A", "Pc_B", "Pc_C", "Pc_D", "Pc_E", "Rc_A", "Rc_B", "Rc_C", "Rc_D", "Rc_E",
        "Ra_A", "Ra_B", "Ra_C", "Ra_D", "Ra_E", "R",
    }


def urs_real_measured() -> Set[str]:
    return urs_ideal_measured() - {"R", "Ra_C", "Ra_D", "Ra_E"}


def narasimhan_measured_variables() -> Set[str]:
    return {"x1", "x3", "x5", "x6", "x7", "x9", "x11", "x12", "x13", "x14", "x15", "x16", "x18", "x19", "x20", "x26", "x27"}


def olefins_measured_variables() -> Set[str]:
    return {"x1", "x2", "x7", "x8", "x9", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x37", "x46", "x53"}


def stage_kpi_measured_sensors() -> Set[str]:
    return {"PT1_F", "PT1_R", "PT2_F", "PT2_R", "Pa", "Pb", "Pc"}


def stage_kpi_known_constants() -> Set[str]:
    return {"A", "Lp", "Dpi"}


def bank_kpi_known_constants() -> Set[str]:
    return {"A_a", "A_b", "A_c", "Lp_a", "Lp_b", "Lp_c", "Dpi_a", "Dpi_b", "Dpi_c"}


def stage_kpi_allowed_outputs() -> Dict[str, Set[str]]:
    return {
        "Eq20": {"S1"}, "Eq21": {"S2"}, "Eq22": {"IDF1"},
        "Eq23": {"IDF2"}, "Eq24": {"PDI1"}, "Eq25": {"PDI2"},
    }


def bank_kpi_allowed_outputs() -> Dict[str, Set[str]]:
    return {
        "Eq26": {"Sa"}, "Eq27": {"Sb"}, "Eq28": {"Sc"},
        "Eq29": {"IDFa"}, "Eq30": {"IDFb"}, "Eq31": {"IDFc"},
        "Eq32": {"PDIa"}, "Eq33": {"PDIb"}, "Eq34": {"PDIc"},
    }


def build_cases() -> List[CaseSpec]:
    ideal = urs_ideal_measured()
    real = urs_real_measured()
    stage_m = stage_kpi_measured_sensors()
    stage_k = stage_kpi_known_constants()
    return [
        CaseSpec(1, "URS - Balancos de massa", "Caso URS ideal", "01_urs_ideal", urs_bancos.equations_documento, ideal, set()),
        CaseSpec(2, "URS - Balancos de massa", "Caso URS Real", "02_urs_real", urs_bancos.equations_documento, real, set()),
        CaseSpec(3, "URS - Balancos de massa", "Caso URS Real - R medida", "03_urs_real_R", urs_bancos.equations_documento, real | {"R"}, set()),
        CaseSpec(4, "URS - Balancos de massa", "Caso URS Real - RaC medida", "04_urs_real_RaC", urs_bancos.equations_documento, real | {"Ra_C"}, set()),
        CaseSpec(5, "URS - Balancos de massa", "Caso URS Real - RaD medida", "05_urs_real_RaD", urs_bancos.equations_documento, real | {"Ra_D"}, set()),
        CaseSpec(6, "URS - Balancos de massa", "Caso URS Real - RaE medida", "06_urs_real_RaE", urs_bancos.equations_documento, real | {"Ra_E"}, set()),
        CaseSpec(7, "URS - Balancos de massa", "Caso URS Real - R e RaC medidas", "07_urs_real_R_RaC", urs_bancos.equations_documento, real | {"R", "Ra_C"}, set()),
        CaseSpec(8, "URS - Balancos de massa", "Caso URS Real - R e RaD medidas", "08_urs_real_R_RaD", urs_bancos.equations_documento, real | {"R", "Ra_D"}, set()),
        CaseSpec(9, "URS - Balancos de massa", "Caso URS Real - R e RaE medidas", "09_urs_real_R_RaE", urs_bancos.equations_documento, real | {"R", "Ra_E"}, set()),
        CaseSpec(10, "URS - Balancos de massa", "Caso URS Real - RaC e RaD medidas", "10_urs_real_RaC_RaD", urs_bancos.equations_documento, real | {"Ra_C", "Ra_D"}, set()),
        CaseSpec(11, "URS - Balancos de massa", "Caso URS Real - RaC e RaE medidas", "11_urs_real_RaC_RaE", urs_bancos.equations_documento, real | {"Ra_C", "Ra_E"}, set()),
        CaseSpec(12, "URS - Balancos de massa", "Caso URS Real - RaD e RaE medidas", "12_urs_real_RaD_RaE", urs_bancos.equations_documento, real | {"Ra_D", "Ra_E"}, set()),
        CaseSpec(13, "URS - KPIs por estagio", "Caso URS Estagio", "13_urs_estagio", urs_estagios.equations_kpi_estagios, stage_m, stage_k, stage_kpi_allowed_outputs()),
        CaseSpec(14, "URS - KPIs por estagio", "Caso URS Estagio - PT1p medida", "14_urs_estagio_PT1p", urs_estagios.equations_kpi_estagios, stage_m | {"PT1_P"}, stage_k, stage_kpi_allowed_outputs()),
        CaseSpec(15, "URS - KPIs por estagio", "Caso URS Estagio - PT2p medida", "15_urs_estagio_PT2p", urs_estagios.equations_kpi_estagios, stage_m | {"PT2_P"}, stage_k, stage_kpi_allowed_outputs()),
        CaseSpec(16, "URS - KPIs por estagio", "Caso URS Estagio - PT1p e PT2p medidas", "16_urs_estagio_PT1p_PT2p", urs_estagios.equations_kpi_estagios, stage_m | {"PT1_P", "PT2_P"}, stage_k, stage_kpi_allowed_outputs()),
        CaseSpec(17, "URS - KPIs por bancos", "Caso URS Banco", "17_urs_banco", urs_bancos.equations_kpi_bancos, {"Pa", "Pb", "Pc"}, bank_kpi_known_constants(), bank_kpi_allowed_outputs()),
        CaseSpec(18, "Literatura", "Caso Narasimhan (Steam Plant)", "18_narasimhan", narasimhan.equations_narasimhan, narasimhan_measured_variables(), set()),
        CaseSpec(19, "Literatura", "Caso Planta de Olefinas (Sanchez e Romagnoli)", "19_olefinas", olefinas.equations_v2, olefins_measured_variables(), set()),
    ]


# Backward-compatible aliases used in previous drafts.
sistema_ideal_urs = urs_ideal_measured
sistema_real_urs = urs_real_measured
narasimhan_measured = narasimhan_measured_variables
olefinas_measured = olefins_measured_variables
kpi_estagios_sensores_reais = stage_kpi_measured_sensors
kpi_estagios_parametros = stage_kpi_known_constants
kpi_bancos_parametros = bank_kpi_known_constants
outputs_kpi_estagios = stage_kpi_allowed_outputs
outputs_kpi_bancos = bank_kpi_allowed_outputs
