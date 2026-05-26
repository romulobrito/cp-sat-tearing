#!/usr/bin/env python3
"""Compact equation definitions for the Narasimhan steam-plant case."""

from __future__ import annotations

from typing import Dict, List, Set

Equations = Dict[str, List[str]]


def equations_narasimhan() -> Equations:
    """Return structural equations for the Narasimhan benchmark."""
    return {
        "Eq1": ["x1", "x2", "x4", "x3"],
        "Eq2": ["x7", "x8", "x5", "x6", "x9"],
        "Eq3": ["x5", "x1", "x10"],
        "Eq4": ["x10", "x11", "x12"],
        "Eq5": ["x3", "x13", "x11", "x14", "x15", "x16", "x17"],
        "Eq6": ["x6", "x2", "x13"],
        "Eq7": ["x14", "x18", "x7", "x19", "x20", "x21"],
        "Eq8": ["x15", "x22", "x18", "x23", "x24"],
        "Eq9": ["x12", "x16", "x22", "x25"],
        "Eq10": ["x19", "x23", "x27", "x26"],
        "Eq11": ["x20", "x26", "x28", "x8"],
        "Eq12": ["x4", "x27", "x28", "x9", "x17", "x21", "x24", "x25"],
    }


def all_variables(eqs: Equations) -> Set[str]:
    """Return all variables present in the provided equations dictionary."""
    return {var for variables in eqs.values() for var in variables}
