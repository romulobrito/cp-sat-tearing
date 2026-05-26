#!/usr/bin/env python3
"""Compact equation definitions for URS document and stage-KPI cases."""

from __future__ import annotations

from typing import Dict, List, Set

Equations = Dict[str, List[str]]


def equations_documento() -> Equations:
    """Return URS document mass-balance equations (Eq1..Eq19)."""
    return {
        "Eq1": ["F", "R", "P"],
        "Eq2": ["P", "PA", "PB", "PC", "PD", "PE"],
        "Eq3": ["PA", "Pa_A", "Pb_A", "Pc_A"],
        "Eq4": ["PB", "Pa_B", "Pb_B", "Pc_B"],
        "Eq5": ["PC", "Pa_C", "Pb_C", "Pc_C"],
        "Eq6": ["PD", "Pa_D", "Pb_D", "Pc_D"],
        "Eq7": ["PE", "Pa_E", "Pb_E", "Pc_E"],
        "Eq8": ["R", "Rc_A", "Rc_B", "Rc_C", "Rc_D", "Rc_E"],
        "Eq9": ["F", "FA", "FB", "FC", "FD", "FE"],
        "Eq10": ["FA", "Ra_A", "Pa_A", "Rb_A", "Pb_A"],
        "Eq11": ["FB", "Ra_B", "Pa_B", "Rb_B", "Pb_B"],
        "Eq12": ["FC", "Ra_C", "Pa_C", "Rb_C", "Pb_C"],
        "Eq13": ["FD", "Ra_D", "Pa_D", "Rb_D", "Pb_D"],
        "Eq14": ["FE", "Ra_E", "Pa_E", "Rb_E", "Pb_E"],
        "Eq15": ["Rb_A", "Rc_A", "Pc_A", "Ra_A"],
        "Eq16": ["Rb_B", "Rc_B", "Pc_B", "Ra_B"],
        "Eq17": ["Rb_C", "Rc_C", "Pc_C", "Ra_C"],
        "Eq18": ["Rb_D", "Rc_D", "Pc_D", "Ra_D"],
        "Eq19": ["Rb_E", "Rc_E", "Pc_E", "Ra_E"],
    }


def equations_kpi_estagios() -> Equations:
    """Return URS stage-KPI equations (Eq20..Eq25)."""
    return {
        "Eq20": ["S1", "PT1_R", "PT1_P", "Pa", "Pb", "Dpi", "A"],
        "Eq21": ["S2", "PT2_R", "PT2_P", "Pc", "Dpi", "A"],
        "Eq22": ["IDF1", "PT1_R", "PT1_P", "Pa", "Pb", "Lp", "Dpi", "A"],
        "Eq23": ["IDF2", "PT2_R", "PT2_P", "Pc", "Lp", "Dpi", "A"],
        "Eq24": ["PDI1", "PT1_F", "PT1_R"],
        "Eq25": ["PDI2", "PT2_F", "PT2_R"],
    }


def equations_total() -> Equations:
    """Return merged equations for URS document plus stage KPIs."""
    equations: Equations = {}
    equations.update(equations_documento())
    equations.update(equations_kpi_estagios())
    return equations


def all_vars_documento(eqs: Equations) -> Set[str]:
    """Return all variables in a URS equations dictionary."""
    return {var for variables in eqs.values() for var in variables}
