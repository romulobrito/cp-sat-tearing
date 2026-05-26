#!/usr/bin/env python3
"""Compact equation definitions for the olefins benchmark case."""

from __future__ import annotations

from typing import Dict, List, Set

Equations = Dict[str, List[str]]


def equations_v2() -> Equations:
    """Return structural equations for Sanchez and Romagnoli olefins system."""
    return {
        "Eq1": ["x1", "x3", "x4"],
        "Eq2": ["x3", "x5", "x6"],
        "Eq3": ["x2", "x4", "x5", "x6", "x7", "x8", "x9"],
        "Eq4": ["x8", "x10", "x11", "x12"],
        "Eq5": ["x11", "x10", "x7"],
        "Eq6": ["x9", "x13", "x14", "x15"],
        "Eq7": ["x15", "x16", "x34"],
        "Eq8": ["x34", "x17", "x35"],
        "Eq9": ["x35", "x18", "x20", "x36"],
        "Eq10": ["x36", "x23", "x24", "x25"],
        "Eq11": ["x24", "x21", "x22", "x26"],
        "Eq12": ["x20", "x22", "x23", "x19"],
        "Eq13": ["x16", "x17", "x18", "x25", "x26", "x27", "x28", "x29", "x30"],
        "Eq14": ["x30", "x31", "x32", "x33"],
        "Eq15": ["x28", "x29", "x31", "x32", "x33", "x63", "x37", "x61"],
        "Eq16": ["x38", "x61", "x55"],
        "Eq17": ["x54", "x55", "x48"],
        "Eq18": ["x56", "x54", "x57", "x58"],
        "Eq19": ["x60", "x59"],
        "Eq20": ["x57", "x59", "x62", "x56"],
        "Eq21": ["x48", "x47", "x49"],
        "Eq22": ["x47", "x44", "x45"],
        "Eq23": ["x51", "x50"],
        "Eq24": ["x50", "x52"],
        "Eq25": ["x52", "x51", "x53"],
        "Eq26": ["x44", "x42", "x63"],
        "Eq27": ["x45", "x46", "x43"],
        "Eq28": ["x42", "x43", "x41"],
        "Eq29": ["x41", "x38", "x40"],
        "Eq30": ["x40", "x39", "x62"],
        "Eq31": ["x39", "x58", "x60"],
        "Eq32": ["x1", "x2", "x12", "x13", "x14", "x19", "x21", "x27", "x37", "x46", "x53"],
    }


def all_vars_v2(eqs: Equations) -> Set[str]:
    """Return all variables present in an equations dictionary."""
    return {var for variables in eqs.values() for var in variables}
