#!/usr/bin/env python3
"""Compact equation definitions for URS document and bank-KPI cases."""

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


def equations_kpi_bancos() -> Equations:
    """Return URS bank-KPI equations (Eq26..Eq34)."""
    return {
        "Eq26": ["Sa", "PTa_R", "PTa_P", "Pa", "Dpi_a", "A_a"],
        "Eq27": ["Sb", "PTb_R", "PTb_P", "Pb", "Dpi_b", "A_b"],
        "Eq28": ["Sc", "PTc_R", "PTc_P", "Pc", "Dpi_c", "A_c"],
        "Eq29": ["IDFa", "PTa_R", "PTa_P", "Pa", "Lp_a", "Dpi_a", "A_a"],
        "Eq30": ["IDFb", "PTb_R", "PTb_P", "Pb", "Lp_b", "Dpi_b", "A_b"],
        "Eq31": ["IDFc", "PTc_R", "PTc_P", "Pc", "Lp_c", "Dpi_c", "A_c"],
        "Eq32": ["PDIa", "PTa_F", "PTa_R"],
        "Eq33": ["PDIb", "PTb_F", "PTb_R"],
        "Eq34": ["PDIc", "PTc_F", "PTc_R"],
    }


def all_vars_documento(eqs: Equations) -> Set[str]:
    """Return all variables in a URS equations dictionary."""
    return {var for variables in eqs.values() for var in variables}
