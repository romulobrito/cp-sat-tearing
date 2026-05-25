#!/usr/bin/env python3
"""
Teste com as EXATAS condições do documento "Instrumentação Mínima da URS"
=========================================================================
Para comparação justa: mesmas 43 variáveis, mesmas 19 equações
"""

from ortools.sat.python import cp_model
from typing import Dict, List, Set
from collections import defaultdict, deque
import math
import pandas as pd
import os
from datetime import datetime

# metodo inferencia por variavel da ultima execucao ('z' ou 'p')
_LAST_INFERENCE_METHOD: Dict[str, str] = {}


def _build_inference_graph(inference_eqs: Dict[str, List[str]], eqs: Dict[str, List[str]]):
    G = {v: set() for vs in eqs.values() for v in vs}
    for v, qs in inference_eqs.items():
        for q in qs:
            preds = [u for u in eqs[q] if u != v]
            for u in preds:
                G.setdefault(u, set()).add(v)
                G.setdefault(v, set())
    return G


def _has_path(G: Dict[str, Set[str]], src: str, tgt: str) -> bool:
    if src == tgt:
        return True
    vis, dq = set([src]), deque([src])
    while dq:
        u = dq.popleft()
        for w in G.get(u, ()):        
            if w == tgt:
                return True
            if w not in vis:
                vis.add(w); dq.append(w)
    return False


def equations_documento() -> Dict[str, List[str]]:
    """
    Equações EXATAS do documento (SEM as variáveis Fc_A, Fc_B, Fc_C, Fc_D, Fc_E)
    """
    return {
        "Eq1":  ["F", "R", "P"],
        "Eq2":  ["P", "PA", "PB", "PC", "PD", "PE"],
        "Eq3":  ["PA", "Pa_A", "Pb_A", "Pc_A"],
        "Eq4":  ["PB", "Pa_B", "Pb_B", "Pc_B"],
        "Eq5":  ["PC", "Pa_C", "Pb_C", "Pc_C"],
        "Eq6":  ["PD", "Pa_D", "Pb_D", "Pc_D"],
        "Eq7":  ["PE", "Pa_E", "Pb_E", "Pc_E"],
        "Eq8":  ["R", "Rc_A", "Rc_B", "Rc_C", "Rc_D", "Rc_E"],
        "Eq9":  ["F", "FA", "FB", "FC", "FD", "FE"],
        "Eq10": ["FA", "Ra_A", "Pa_A", "Rb_A", "Pb_A"],
        "Eq11": ["FB", "Ra_B", "Pa_B", "Rb_B", "Pb_B"],
        "Eq12": ["FC", "Ra_C", "Pa_C", "Rb_C", "Pb_C"],
        "Eq13": ["FD", "Ra_D", "Pa_D", "Rb_D", "Pb_D"],
        "Eq14": ["FE", "Ra_E", "Pa_E", "Rb_E", "Pb_E"],
        # Igualdades internas (Eq. 15-19)
        "Eq15": ["Rb_A", "Rc_A", "Pc_A", "Ra_A"],
        "Eq16": ["Rb_B", "Rc_B", "Pc_B", "Ra_B"],
        "Eq17": ["Rb_C", "Rc_C", "Pc_C", "Ra_C"],
        "Eq18": ["Rb_D", "Rc_D", "Pc_D", "Ra_D"],
        "Eq19": ["Rb_E", "Rc_E", "Pc_E", "Ra_E"],
    }

def all_vars_documento(eqs: Dict[str, List[str]]) -> Set[str]:
    vars_set: Set[str] = set()
    for vs in eqs.values():
        vars_set.update(vs)
    return vars_set

def equations_kpi_estagios() -> Dict[str, List[str]]:
    """
    Equacoes de KPI por estagios (capitulo 5.1 do documento)
    Variaveis usadas (ASCII) - total 18 variaveis:
      - S1, S2, IDF1, IDF2, PDI1, PDI2
      - PT1_F, PT1_R, PT1_P, PT2_F, PT2_R, PT2_P
      - Pa, Pb, Pc (vazoes de permeado por banco agregadas por estagio)
      - Parametros contabilizados como variaveis (conforme documento): A, Lp, Dpi
    """
    return {
        # S por estagio: inclui Dpi e A
        "Eq20": ["S1", "PT1_R", "PT1_P", "Pa", "Pb", "Dpi", "A"],
        "Eq21": ["S2", "PT2_R", "PT2_P", "Pc", "Dpi", "A"],
        # IDF por estagio: inclui Lp, Dpi e area A no termo de vazao especifica
        "Eq22": ["IDF1", "PT1_R", "PT1_P", "Pa", "Pb", "Lp", "Dpi", "A"],
        "Eq23": ["IDF2", "PT2_R", "PT2_P", "Pc", "Lp", "Dpi", "A"],
        # PDI por estagio
        "Eq24": ["PDI1", "PT1_F", "PT1_R"],
        "Eq25": ["PDI2", "PT2_F", "PT2_R"],
    }

def equations_total() -> Dict[str, List[str]]:
    """
    Merge das equacoes de balancos (Eq1..Eq19) com KPIs por estagios (Eq20..Eq25).
    """
    eqs = {}
    eqs.update(equations_documento())
    eqs.update(equations_kpi_estagios())
    return eqs

def classify_total_v2(
    measured: Set[str], *,
    W_obs=100, W_tears=1, W_bonus=0, W_promo=10, T_max=None,
    force_heads=None, no_raise_on_infeasible=False, W_align=1,
    repair_tears_slack: int | None = 0,
    enable_repair_3bc: bool = False,
    solver_time_limit_s: int = 60
):
    """
    Classificador para o SISTEMA TOTAL (balancos + KPIs por estagios).
    Reaproveita a mesma logica (o/x/z/p/t/L) com objetivo em duas fases.
    """
    eqs = equations_total()
    V = sorted(all_vars_documento(eqs))
    invalid = measured - set(V)
    if invalid:
        raise ValueError(f"Invalid measured variables: {sorted(invalid)}")

    uses = {v: [q for q, vs in eqs.items() if v in vs] for v in V}
    M = len(V)

    mdl = cp_model.CpModel()

    y = {v: mdl.NewBoolVar(f"y_{v}") for v in V}
    x = {v: mdl.NewBoolVar(f"x_{v}") for v in V}
    o = {v: mdl.NewBoolVar(f"o_{v}") for v in V}
    z = {(q, v): mdl.NewBoolVar(f"z_{q}_{v}") for q, vs in eqs.items() for v in vs}
    t = {(q, v): mdl.NewBoolVar(f"t_{q}_{v}") for q, vs in eqs.items() for v in vs}
    p = {(q, v): mdl.NewBoolVar(f"p_{q}_{v}") for q, vs in eqs.items() for v in vs}
    L = {v: mdl.NewIntVar(0, M, f"L_{v}") for v in V}

    s_align = []
    if W_align != 0:
        for q, vs in eqs.items():
            for v in vs:
                for u in vs:
                    if u == v:
                        continue
                    sp = mdl.NewBoolVar(f"s_p_{q}_{u}_head_{v}")
                    mdl.Add(sp <= p[q, v])
                    mdl.Add(sp <= t[q, u])
                    mdl.Add(sp >= p[q, v] + t[q, u] - 1)
                    s_align.append(sp)
                    sz = mdl.NewBoolVar(f"s_z_{q}_{u}_head_{v}")
                    mdl.Add(sz <= z[q, v])
                    mdl.Add(sz <= t[q, u])
                    mdl.Add(sz >= z[q, v] + t[q, u] - 1)
                    s_align.append(sz)

    for v in V:
        z_sum = sum(z[q, v] for q in uses[v])
        t_sum = sum(t[q, v] for q in uses[v])
        p_sum = sum(p[q, v] for q in uses[v])

        mdl.Add(y[v] == 1) if v in measured else mdl.Add(y[v] == 0)

        mdl.Add(z_sum <= 1)
        mdl.Add(t_sum <= 1)
        mdl.Add(p_sum <= 1)
        mdl.Add(z_sum + p_sum <= 1)

        mdl.Add(x[v] >= y[v]); mdl.Add(x[v] >= z_sum); mdl.Add(x[v] >= t_sum); mdl.Add(x[v] >= p_sum)
        mdl.Add(o[v] >= y[v]); mdl.Add(o[v] >= z_sum); mdl.Add(o[v] >= p_sum)

        mdl.Add(x[v] <= y[v] + z_sum + t_sum + p_sum)
        mdl.Add(o[v] <= y[v] + z_sum + p_sum)

    for v in measured:
        mdl.Add(x[v] == 1)

    for q, vs in eqs.items():
        for v in vs:
            mdl.Add(z[q, v] <= 1 - y[v])
            mdl.Add(z[q, v] <= 1 - t[q, v])
            mdl.Add(t[q, v] <= 1 - y[v])

            for u in vs:
                if u != v:
                    mdl.Add(z[q, v] <= o[u] + t[q, u])

            mdl.Add(p[q, v] <= 1 - y[v])
            mdl.Add(p[q, v] <= 1 - t[q, v])
            for u in vs:
                if u != v:
                    mdl.Add(p[q, v] <= o[u])

            for u in vs:
                if u != v:
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([z[q, v], t[q, u].Not()])
                    mdl.Add(L[u] >= L[v] + 1).OnlyEnforceIf([z[q, u], t[q, v].Not()])
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([p[q, v], t[q, u].Not()])

        mdl.Add(sum(z[q, v] for v in vs) + sum(p[q, v] for v in vs) <= 1)

    if T_max is not None:
        mdl.Add(sum(t.values()) <= T_max)

    bonus_terms = []
    if W_bonus:
        for q, vs in eqs.items():
            for v in vs:
                others = [u for u in vs if u != v]
                if all(u in measured for u in others):
                    bonus_terms.append(z[(q, v)])

    primary = (
        W_obs * sum(o.values())
        - W_tears * sum(t.values())
        + (W_bonus * sum(bonus_terms) if W_bonus else 0)
    )

    exclude_eqs_phase1 = set()
    if force_heads and "exclude_phase1" in force_heads:
        exclude_eqs_phase1 = force_heads["exclude_phase1"]
        for eq in exclude_eqs_phase1:
            if eq in eqs:
                for v in eqs[eq]:
                    if (eq, v) in z:
                        mdl.Add(z[eq, v] == 0)
                    if (eq, v) in p:
                        mdl.Add(p[eq, v] == 0)

    mdl.Maximize(primary)

    solver = cp_model.CpSolver()
    params = solver.parameters
    params.max_time_in_seconds = solver_time_limit_s
    params.random_seed = 42
    params.num_search_workers = 1
    params.randomize_search = False
    params.log_search_progress = False

    status1 = solver.Solve(mdl)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if no_raise_on_infeasible:
            return [], [], [], [], [], [], {}, {}
        raise RuntimeError(f"Phase 1 failed: {solver.StatusName(status1)}")

    best_primary = solver.ObjectiveValue()
    p1 = [(q, v) for (q, v), var in p.items() if solver.Value(var) == 1]
    z1 = [(q, v) for (q, v), var in z.items() if solver.Value(var) == 1]
    o1 = sum(solver.Value(var) for var in o.values())
    t1 = sum(solver.Value(var) for var in t.values())
    tear_vars_f1 = {v for v in V if any(solver.Value(t[q, v]) for q in uses[v])}

    if status1 == cp_model.OPTIMAL:
        mdl.Add(primary >= math.ceil(best_primary))
    else:
        mdl.Add(primary >= math.floor(best_primary))
    # Promocao so pode atuar em variaveis que eram tears na Fase 1.
    for v in V:
        if v not in tear_vars_f1:
            mdl.Add(sum(p[q, v] for q in uses[v]) == 0)
    align_term = sum(s_align) if s_align else 0
    mdl.Maximize(primary + (W_promo * sum(p.values())) + (W_align * align_term))

    status2 = solver.Solve(mdl)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if no_raise_on_infeasible:
            return [], [], [], [], [], [], {}, {}
        raise RuntimeError(f"Phase 2 failed: {solver.StatusName(status2)}")

    p2 = [(q, v) for (q, v), var in p.items() if solver.Value(var) == 1]
    z2 = [(q, v) for (q, v), var in z.items() if solver.Value(var) == 1]
    o2 = sum(solver.Value(var) for var in o.values())
    t2 = sum(solver.Value(var) for var in t.values())
    prom_fix_candidates = [(q, v) for (q, v) in p2 if v in tear_vars_f1]

    def _extract_current_solution():
        med_loc = [v for v in V if solver.Value(y[v])]
        inf_loc = [v for v in V if (solver.Value(o[v]) == 1 and solver.Value(y[v]) == 0)]
        indet_loc = []
        for vv in V:
            if solver.Value(y[vv]) == 0:
                z_on = any(solver.Value(z[q, vv]) for q in uses[vv])
                p_on = any(solver.Value(p[q, vv]) for q in uses[vv])
                t_on = any(solver.Value(t[q, vv]) for q in uses[vv])
                if (not z_on) and (not p_on) and (not t_on):
                    indet_loc.append(vv)
        tears_list_loc, tears_dict_loc = [], defaultdict(list)
        for (q, vv), var in t.items():
            if solver.Value(var) == 1:
                tears_list_loc.append((q, vv))
                tears_dict_loc[q].append(vv)
        inference_eqs_loc = {}
        inference_method_loc: Dict[str, str] = {}
        for vv in V:
            if solver.Value(y[vv]) == 0 and solver.Value(o[vv]) == 1:
                for q in uses[vv]:
                    if solver.Value(p[q, vv]) == 1:
                        inference_eqs_loc[vv] = [q]
                        inference_method_loc[vv] = "p"
                        break
                    if solver.Value(z[q, vv]) == 1:
                        inference_eqs_loc[vv] = [q]
                        inference_method_loc[vv] = "z"
                        break
        return med_loc, inf_loc, indet_loc, tears_list_loc, inference_eqs_loc, inference_method_loc, tears_dict_loc

    last_solve_feasible = True
    fallback_payload = _extract_current_solution()
    if prom_fix_candidates:
        for (qfix, vfix) in prom_fix_candidates:
            mdl.Add(p[qfix, vfix] == 1)
            for q2 in uses[vfix]:
                mdl.Add(t[q2, vfix] == 0)

        strong = mdl.NewBoolVar("repair_strong")
        weak = mdl.NewBoolVar("repair_weak")
        mdl.Add(strong + weak == 1)
        mdl.Add(sum(o.values()) >= o2 + 1).OnlyEnforceIf(strong)
        if repair_tears_slack is None:
            mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf(strong)
        else:
            mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf(strong)
        mdl.Add(sum(o.values()) >= o2).OnlyEnforceIf(weak)
        if repair_tears_slack is None:
            mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf(weak)
        else:
            mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf(weak)

        mdl.Maximize(primary)
        status3 = solver.Solve(mdl)
        if status3 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            last_solve_feasible = False
        else:
            fallback_payload = _extract_current_solution()

    # FASE 3b/3c opcional (piloto):
    # Estrategia clone-like: cada tentativa extra e isolada por um literal
    # de ativacao e, se inviavel, e desativada por a_try == 0 (rollback logico).
    if enable_repair_3bc and last_solve_feasible:
        tear_vars = {v for (q, v), var in t.items() if solver.Value(var) == 1}

        if (not prom_fix_candidates) and tear_vars:
            repaired_3b = False
            for vtear in sorted(tear_vars):
                cand_qs = [q for q in uses[vtear] if all(solver.Value(o[u]) == 1 for u in eqs[q] if u != vtear)]
                if not cand_qs:
                    continue
                for qstar in cand_qs:
                    a_try_b = mdl.NewBoolVar(f"try3b_{qstar}_{vtear}")
                    mdl.Add(a_try_b == 1)
                    mdl.Add(p[qstar, vtear] == 1).OnlyEnforceIf(a_try_b)
                    for q2 in uses[vtear]:
                        mdl.Add(t[q2, vtear] == 0).OnlyEnforceIf(a_try_b)

                    strong_b = mdl.NewBoolVar(f"repair_strong_b_{qstar}_{vtear}")
                    weak_b = mdl.NewBoolVar(f"repair_weak_b_{qstar}_{vtear}")
                    mdl.Add(strong_b + weak_b == 1).OnlyEnforceIf(a_try_b)
                    mdl.Add(sum(o.values()) >= o2 + 1).OnlyEnforceIf([a_try_b, strong_b])
                    if repair_tears_slack is None:
                        mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf([a_try_b, strong_b])
                    else:
                        mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf([a_try_b, strong_b])
                    mdl.Add(sum(o.values()) >= o2).OnlyEnforceIf([a_try_b, weak_b])
                    if repair_tears_slack is None:
                        mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf([a_try_b, weak_b])
                    else:
                        mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf([a_try_b, weak_b])

                    mdl.Maximize(primary)
                    status3b = solver.Solve(mdl)
                    if status3b in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        o2 = sum(solver.Value(var) for var in o.values())
                        t2 = sum(solver.Value(var) for var in t.values())
                        fallback_payload = _extract_current_solution()
                        repaired_3b = True
                        break

                    # rollback logico da tentativa inviavel
                    mdl.Add(a_try_b == 0)
                    mdl.Maximize(primary)
                    status_restore = solver.Solve(mdl)
                    if status_restore not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        last_solve_feasible = False
                        break

                if repaired_3b:
                    break
                if not last_solve_feasible:
                    break

        if last_solve_feasible:
            current_indet = []
            for v in V:
                if solver.Value(y[v]) == 0:
                    z_on = any(solver.Value(z[q, v]) for q in uses[v])
                    p_on = any(solver.Value(p[q, v]) for q in uses[v])
                    t_on = any(solver.Value(t[q, v]) for q in uses[v])
                    if (not z_on) and (not p_on) and (not t_on):
                        current_indet.append(v)

            for vhard in sorted(current_indet):
                cand_qs = []
                for q in eqs:
                    if vhard not in eqs[q]:
                        continue
                    head_taken = any((solver.Value(z[q, w]) == 1) or (solver.Value(p[q, w]) == 1) for w in eqs[q])
                    if not head_taken:
                        cand_qs.append(q)

                attempts = []
                for q in cand_qs:
                    preds = [u for u in eqs[q] if u != vhard]
                    need_move = []
                    doable = True
                    for u in preds:
                        if solver.Value(o[u]) == 1:
                            continue
                        q_old = None
                        for qq in uses[u]:
                            if (qq != q) and (solver.Value(t[qq, u]) == 1):
                                q_old = qq
                                break
                        if q_old is not None:
                            need_move.append((u, q_old))
                        else:
                            doable = False
                            break
                    if not doable:
                        continue

                    a_try = mdl.NewBoolVar(f"try3c_{q}_{vhard}")
                    for (u, q_old) in need_move:
                        mdl.Add(t[q, u] == 1).OnlyEnforceIf(a_try)
                        for q2 in uses[u]:
                            if q2 != q:
                                mdl.Add(t[q2, u] == 0).OnlyEnforceIf(a_try)

                    mdl.Add(p[q, vhard] == 1).OnlyEnforceIf(a_try)
                    mdl.Add(sum(o.values()) >= o2).OnlyEnforceIf(a_try)
                    mdl.Add(sum(t.values()) <= t2 + (repair_tears_slack or 0)).OnlyEnforceIf(a_try)
                    attempts.append(a_try)

                if attempts:
                    mdl.Add(sum(attempts) <= 1)
                    solved_one = False
                    for a_try in attempts:
                        mdl.Add(a_try == 1)
                        mdl.Maximize(primary)
                        status3c = solver.Solve(mdl)
                        if status3c in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                            o2 = sum(solver.Value(var) for var in o.values())
                            t2 = sum(solver.Value(var) for var in t.values())
                            fallback_payload = _extract_current_solution()
                            solved_one = True
                            break
                        mdl.Add(a_try == 0)
                        mdl.Maximize(primary)
                        status_restore = solver.Solve(mdl)
                        if status_restore not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                            last_solve_feasible = False
                            break
                    if not solved_one:
                        if not last_solve_feasible:
                            break
                        continue

    if last_solve_feasible:
        med, inf, indet, tears_list, inference_eqs, inference_method, tears_dict = _extract_current_solution()
        _sanity_check(med, inf, indet, tears_dict, eqs)
        _LAST_INFERENCE_METHOD.clear()
        _LAST_INFERENCE_METHOD.update(inference_method)
        return med, inf, indet, tears_list, [], [], inference_eqs, tears_dict
    else:
        if fallback_payload is not None:
            med, inf, indet, tears_list, inference_eqs, inference_method, tears_dict = fallback_payload
            _sanity_check(med, inf, indet, tears_dict, eqs)
            _LAST_INFERENCE_METHOD.clear()
            _LAST_INFERENCE_METHOD.update(inference_method)
            return med, inf, indet, tears_list, [], [], inference_eqs, tears_dict
        return [], [], [], [], [], [], {}, {}


def classify_documento_v2(
    measured: Set[str], *,
    W_obs=100, W_tears=1, W_bonus=0, W_promo=10, T_max=None,
    force_heads=None, no_raise_on_infeasible=False, W_align=1,
    repair_tears_slack: int | None = 0,
    enable_repair_3bc: bool = False,
    solver_time_limit_s: int = 60
):
    """
    Wrapper de compatibilidade: classificacao do bloco do documento (Eq1..Eq19).
    Reaproveita a implementacao consolidada no modulo de bancos para manter
    consistencia de regras e resultados.
    """
    from URS_classification_cases_ideal_real_real_min_inst_kpi_bancos import (
        classify_documento_v2 as classify_documento_v2_impl,
        _LAST_INFERENCE_METHOD as source_last_inference_method,
    )

    result = classify_documento_v2_impl(
        measured,
        W_obs=W_obs,
        W_tears=W_tears,
        W_bonus=W_bonus,
        W_promo=W_promo,
        T_max=T_max,
        force_heads=force_heads,
        no_raise_on_infeasible=no_raise_on_infeasible,
        W_align=W_align,
        repair_tears_slack=repair_tears_slack,
        enable_repair_3bc=enable_repair_3bc,
        solver_time_limit_s=solver_time_limit_s,
    )

    _LAST_INFERENCE_METHOD.clear()
    _LAST_INFERENCE_METHOD.update(source_last_inference_method)
    return result


def classify_documento(measured: Set[str], *, W_obs=100, W_tears=1, T_max=None):
    """Wrapper historico que chama a versao v2 com defaults."""
    return classify_documento_v2(measured, W_obs=W_obs, W_tears=W_tears, T_max=T_max)


def _sanity_check(med: List[str], inf: List[str], indet: List[str], tears_dict, eqs) -> None:
    allvars = set().union(*eqs.values())
    nonmeasured = allvars - set(med)
    tears_set_all = {v for vs in tears_dict.values() for v in (vs if isinstance(vs, (list, tuple, set)) else [vs])}
    tears_noninf = tears_set_all - set(inf)
    leftover = nonmeasured - set(inf) - tears_noninf - set(indet)
    assert not leftover, f"Unclassified: {sorted(leftover)}"
    obs = len(med) + len(inf)
    assert 0 <= obs <= len(allvars)


def imprimir_sequencia_reconciliacao(levels, inference_eqs, tears, med, inf, cenario_nome):
    """Imprime a sequência de reconciliação baseada nos níveis topológicos"""
    print(f" SEQUÊNCIA DE RECONCILIAÇÃO - {cenario_nome}")
    print("-" * 70)
    
    if not inf:
        print("  Nenhuma variável para inferir - todas são medidas")
        print()
        return
    
    # Organizar variáveis por nível
    vars_por_nivel = {}
    for var in inf:
        nivel = levels.get(var, 0)
        if nivel not in vars_por_nivel:
            vars_por_nivel[nivel] = []
        vars_por_nivel[nivel].append(var)
    
    # Variáveis tears (nível 0)
    if tears:
        print(" NÍVEL 0 - INICIALIZAÇÃO (TEARS):")
        for eq, vs in sorted(tears.items()):
            vs_list = vs if isinstance(vs, list) else [vs]
            for tear_var in vs_list:
                if tear_var in inf:
                    print(f"  {tear_var} -- TEAR da {eq} (valor inicial assumido)")
        print()
    
    # Sequência de cálculo por níveis
    eqs = equations_documento()
    for nivel in sorted(vars_por_nivel.keys()):
        if nivel == 0 and tears:
            continue  # Já imprimimos os tears
            
        vars_nivel = sorted(vars_por_nivel[nivel])
        print(f" NÍVEL {nivel} - CÁLCULO:")
        
        for var in vars_nivel:
            if var in inference_eqs:
                eq_name = inference_eqs[var]
                eq_vars = eqs[eq_name]
                outras_vars = [v for v in eq_vars if v != var]
                
                # Categorizar outras variáveis
                medidas = [v for v in outras_vars if v in med]
                inferidas = [v for v in outras_vars if v in inf]
                tears_vars = [v for v in outras_vars if v in [t for t in tears.values()]]
                
                print(f"  {var} -- {eq_name}({', '.join(outras_vars)})")
                if medidas:
                    print(f"       Medidas: {', '.join(medidas)}")
                if inferidas:
                    print(f"       Inferidas anteriormente: {', '.join(inferidas)}")
                if tears_vars:
                    print(f"       Tears: {', '.join(tears_vars)}")
        print()

def gerar_codigo_soft_sensor(levels, inference_eqs, tears, med, inf, cenario_nome):
    """Gera código Python automaticamente para soft-sensor"""
    print(f" CÓDIGO PYTHON GERADO - {cenario_nome}")
    print("-" * 70)
    
    if not inf:
        print("# Nenhum código necessário - todas as variáveis são medidas")
        print()
        return
    
    print("# Soft-sensor gerado automaticamente pelo v3")
    print("# Baseado na sequência de reconciliação otimizada")
    print()
    print("def soft_sensor_urs(medidas):")
    print('    """')
    print(f'    Soft-sensor URS - {cenario_nome}')
    print(f'    Medidas necessárias: {sorted(med)}')
    print(f'    Variáveis inferidas: {sorted(inf)}')
    print('    """')
    print()
    
    # Extrair medidas do dicionário
    print("    # Extrair medidas")
    for var in sorted(med):
        print(f"    {var} = medidas['{var}']")
    print()
    
    # Organizar variáveis por nível
    vars_por_nivel = {}
    for var in inf:
        nivel = levels.get(var, 0)
        if nivel not in vars_por_nivel:
            vars_por_nivel[nivel] = []
        vars_por_nivel[nivel].append(var)
    
    # Inicializar tears
    if tears:
        print("    # Inicializar tears (valores assumidos)")
        for eq, vs in sorted(tears.items()):
            vs_list = vs if isinstance(vs, list) else [vs]
            for tear_var in vs_list:
                if tear_var in inf:
                    print(f"    {tear_var} = 0  # TEAR da {eq} - valor inicial")
        print()
    
    # Gerar código por níveis
    eqs_codigo = {
        "Eq1": "F + R + P",
        "Eq2": "P + PA + PB + PC + PD + PE", 
        "Eq3": "PA + Pa_A + Pb_A + Pc_A",
        "Eq4": "PB + Pa_B + Pb_B + Pc_B",
        "Eq5": "PC + Pa_C + Pb_C + Pc_C",
        "Eq6": "PD + Pa_D + Pb_D + Pc_D",
        "Eq7": "PE + Pa_E + Pb_E + Pc_E",
        "Eq8": "R + Rc_A + Rc_B + Rc_C + Rc_D + Rc_E",
        "Eq9": "F + FA + FB + FC + FD + FE",
        "Eq10": "FA + Ra_A + Pa_A + Rb_A + Pb_A",
        "Eq11": "FB + Ra_B + Pa_B + Rb_B + Pb_B",
        "Eq12": "FC + Ra_C + Pa_C + Rb_C + Pb_C",
        "Eq13": "FD + Ra_D + Pa_D + Rb_D + Pb_D",
        "Eq14": "FE + Ra_E + Pa_E + Rb_E + Pb_E",
        "Eq15": "Rb_A + Rc_A + Pc_A + Ra_A",
        "Eq16": "Rb_B + Rc_B + Pc_B + Ra_B",
        "Eq17": "Rb_C + Rc_C + Pc_C + Ra_C",
        "Eq18": "Rb_D + Rc_D + Pc_D + Ra_D",
        "Eq19": "Rb_E + Rc_E + Pc_E + Ra_E",
    }
    
    for nivel in sorted(vars_por_nivel.keys()):
        if nivel == 0 and tears:
            continue  # Tears já inicializados
            
        vars_nivel = sorted(vars_por_nivel[nivel])
        print(f"    # Nível {nivel} - Cálculo")
        
        for var in vars_nivel:
            if var in inference_eqs:
                eq_name = inference_eqs[var]
                eq_formula = eqs_codigo.get(eq_name, "")
                
                if eq_formula:
                    # Resolver para a variável
                    termos = eq_formula.split(" + ")
                    outros_termos = [t for t in termos if t != var]
                    
                    if len(outros_termos) == 1:
                        print(f"    {var} = -{outros_termos[0]}")
                    else:
                        print(f"    {var} = -({' + '.join(outros_termos)})")
                else:
                    print(f"    {var} = calcular_{eq_name}({var})  # Implementar equação")
        print()
    
    print("    # Retornar resultados")
    print("    return {")
    for var in sorted(inf):
        print(f"        '{var}': {var},")
    print("    }")
    print()

def imprimir_detalhes_solucao(cenario_nome, med, inf, indet, tears, total_vars, medidas, levels=None, inference_eqs=None):
    """Imprime detalhes completos da solução"""
    print(f" DETALHES DA SOLUÇÃO - {cenario_nome}")
    print("-" * 60)
    
    # Resumo geral
    obs_percent = (len(med)+len(inf))/total_vars*100
    print(f"Observabilidade: {len(med)+len(inf)}/{total_vars} = {obs_percent:.1f}%")
    print(f"Medidas: {len(med)} | Inferíveis: {len(inf)} | Indetermináveis: {len(indet)}")
    print()
    
    # Variáveis medidas
    print(" VARIÁVEIS MEDIDAS:")
    if med:
        for i, var in enumerate(sorted(med), 1):
            print(f"  {i:2d}. {var}")
    else:
        print("  Nenhuma")
    print()
    
    # Tears por equação
    print(" TEARS UTILIZADOS:")
    if tears:
        eqs = equations_documento()
        for eq, vs in sorted(tears.items()):
            vs_list = vs if isinstance(vs, list) else [vs]
            for tear_var in vs_list:
                eq_vars = eqs[eq]
                outras_vars = [v for v in eq_vars if v != tear_var]
                print(f"  {eq}: {tear_var} (tear)")
                print(f"      Equação: {' + '.join(eq_vars)} = 0")
                print(f"      Outras variáveis: {', '.join(outras_vars)}")
    else:
        print("  Nenhum tear necessário")
    print()
    
    # Variáveis inferíveis
    print(" VARIÁVEIS INFERÍVEIS:")
    if inf:
        for i, var in enumerate(sorted(inf), 1):
            print(f"  {i:2d}. {var}")
    else:
        print("  Nenhuma")
    print()
    
    # Variáveis indetermináveis
    print(" VARIÁVEIS INDETERMINÁVEIS:")
    if indet:
        for i, var in enumerate(sorted(indet), 1):
            print(f"  {i:2d}. {var}")
    else:
        print("  Nenhuma!")
    print()
    
    # Sequência de reconciliação e código
    if levels and inference_eqs and inf:
        imprimir_sequencia_reconciliacao(levels, inference_eqs, tears, med, inf, cenario_nome)
        # gerar_codigo_soft_sensor(levels, inference_eqs, tears, med, inf, cenario_nome)


def save_results_to_excel(
    med: List[str],
    inf: List[str],
    indet: List[str],
    tears_list: List[tuple],
    tears_reclassificados: List[tuple],
    tears_dict: Dict[str, List[str]],
    inference_eqs: Dict[str, List[str]],
    eqs: Dict[str, List[str]],
    measured_vars: Set[str],
    filename: str | None = None,
    inference_method: Dict[str, str] | None = None
):
    """
    Salva os resultados da classificação em um arquivo Excel com 8 abas.
    Mesmo padrão do arquivo URS principal.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resultados_classificacao_{timestamp}.xlsx"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "resultados_classificacao")
    os.makedirs(results_dir, exist_ok=True)
    filepath = filename if os.path.isabs(filename) else os.path.join(results_dir, filename)

    total_vars = len(set().union(*eqs.values()))
    observaveis = len(med) + len(inf)
    observabilidade_pct = (observaveis / total_vars * 100) if total_vars > 0 else 0

    strict_pairs = [(eq, v) for (eq, v) in tears_list if v not in inf]
    strict_vars = sorted({v for _, v in strict_pairs})
    indet_com_tears = sorted(set(indet) | set(strict_vars))

    eqs_utilizadas: Set[str] = set()
    for v, eqs_list in inference_eqs.items():
        eqs_utilizadas.update(eqs_list)
    for v, eq in tears_reclassificados:
        eqs_utilizadas.add(eq)
    grau_sobredeterminacao = len(eqs_utilizadas) - len(inf)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        method_map = inference_method if inference_method is not None else _LAST_INFERENCE_METHOD
        # Aba 1: Resumo
        resumo_data = {
            'Metrica': [
                'Total de Variaveis',
                'Total de Equacoes',
                'Variaveis Medidas',
                'Variaveis Inferidas',
                'Variaveis de Corte (Tears)',
                'Indeterminaveis (excl. tears)',
                'Indeterminaveis (incl. tears)',
                'Variaveis Observaveis',
                'Observabilidade (%)',
                'Equacoes Utilizadas',
                'Grau de Sobredeterminacao',
                'Tipo do Sistema',
                'Parametros_tratados_como_variaveis (medidos)'
            ],
            'Valor': [
                total_vars,
                len(eqs),
                len(med),
                len(inf),
                len(strict_vars),
                len(indet),
                len(indet_com_tears),
                observaveis,
                f"{observabilidade_pct:.1f}%",
                len(eqs_utilizadas),
                grau_sobredeterminacao,
                'SOBREDETERMINADO' if grau_sobredeterminacao > 0 else 'EXATAMENTE DETERMINADO' if grau_sobredeterminacao == 0 else 'SUBDETERMINADO',
                ', '.join(sorted([p for p in ["A","Lp","Dpi"] if p in measured_vars]))
            ]
        }
        pd.DataFrame(resumo_data).to_excel(writer, sheet_name='Resumo', index=False)

        # Aba 2: Variaveis Medidas
        df_medidas = pd.DataFrame({
            'Variavel': sorted(med),
            'Tipo': ['Medida'] * len(med),
            'Observacao': ['Sensor instalado'] * len(med)
        })
        df_medidas.to_excel(writer, sheet_name='Variaveis_Medidas', index=False)

        # Aba 3: Variaveis Inferidas (inclui Metodo)
        inf_data = []
        for v in sorted(inf):
            eqs_list = inference_eqs.get(v, [])
            eq_usada = eqs_list[0] if eqs_list else 'N/A'
            inf_data.append({
                'Variavel': v,
                'Tipo': 'Inferida',
                'Equacao_Utilizada': eq_usada,
                'Variaveis_Necessarias': ', '.join(sorted([u for u in eqs.get(eq_usada, []) if u != v])) if eq_usada != 'N/A' else '',
                'Metodo': method_map.get(v, ''),
            })
        pd.DataFrame(inf_data).to_excel(writer, sheet_name='Variaveis_Inferidas', index=False)

        # Aba 4: Tears (Variaveis de Corte)
        tears_data = []
        for eq, v in strict_pairs:
            tears_data.append({
                'Variavel': v,
                'Equacao': eq,
                'Tipo': 'Tear (Variavel de Corte)',
                'Status': 'Nao Resolvido',
                'Outras_Variaveis_Eq': ', '.join(sorted([u for u in eqs.get(eq, []) if u != v]))
            })
        for v, eq in tears_reclassificados:
            tears_data.append({
                'Variavel': v,
                'Equacao': eq,
                'Tipo': 'Tear Reclassificado',
                'Status': 'Promovido para Inferida',
                'Outras_Variaveis_Eq': ', '.join(sorted([u for u in eqs.get(eq, []) if u != v]))
            })
        pd.DataFrame(tears_data).to_excel(writer, sheet_name='Tears_Variaveis_Corte', index=False)

        # Aba 5: Indeterminaveis
        indet_data = []
        for v in sorted(indet):
            eqs_contem = [q for q, vars_eq in eqs.items() if v in vars_eq]
            indet_data.append({
                'Variavel': v,
                'Tipo': 'Indeterminavel',
                'Equacoes_que_Contem': ', '.join(sorted(eqs_contem)),
                'Razao': 'Nao pode ser determinada com instrumentacao atual'
            })
        pd.DataFrame(indet_data).to_excel(writer, sheet_name='Indeterminaveis', index=False)

        # Aba 6: Estrategia de Inferencia (inclui Metodo)
        todas_inferidas = {}
        for var, eqs_list in inference_eqs.items():
            if eqs_list:
                todas_inferidas[var] = eqs_list[0]
        for var, eq in tears_reclassificados:
            todas_inferidas[var] = eq
        estrategia_data = []
        for var in sorted(todas_inferidas.keys()):
            eq = todas_inferidas[var]
            outras_vars = sorted([u for u in eqs.get(eq, []) if u != var])
            estrategia_data.append({
                'Variavel_Inferida': var,
                'Equacao_Utilizada': eq,
                'Variaveis_Necessarias': ', '.join(outras_vars),
                'Ordem_Inferencia': 'Determinada por dependencias',
                'Metodo': method_map.get(var, ''),
            })
        pd.DataFrame(estrategia_data).to_excel(writer, sheet_name='Estrategia_Inferencia', index=False)

        # Aba 7: Equacoes do Sistema
        eqs_data = []
        for eq, vars_eq in sorted(eqs.items()):
            utilizada = eq in eqs_utilizadas
            tipo_uso = 'Inferencia' if utilizada else 'Nao Utilizada'
            tem_tear = eq in tears_dict
            if tem_tear:
                tears_eq = tears_dict[eq] if isinstance(tears_dict[eq], list) else [tears_dict[eq]]
                tipo_uso += f" (Tears: {', '.join(tears_eq)})"
            eqs_data.append({
                'Equacao': eq,
                'Variaveis': ', '.join(sorted(vars_eq)),
                'Numero_Variaveis': len(vars_eq),
                'Utilizada': 'Sim' if utilizada else 'Nao',
                'Tipo_Uso': tipo_uso,
                'Tem_Tears': 'Sim' if tem_tear else 'Nao'
            })
        pd.DataFrame(eqs_data).to_excel(writer, sheet_name='Equacoes_Sistema', index=False)

        # Aba 8: Todas as Variaveis
        todas_vars = sorted(set().union(*eqs.values()))
        todas_vars_data = []
        for v in todas_vars:
            if v in med:
                tipo = 'Medida'
                detalhes = 'Sensor instalado'
            elif v in inf:
                eq_inf = inference_eqs.get(v, ['N/A'])[0]
                tipo = 'Inferida'
                detalhes = f'Via equacao {eq_inf}'
            elif v in strict_vars:
                eq_tear = next((eq for eq, var in strict_pairs if var == v), 'N/A')
                tipo = 'Tear (Variavel de Corte)'
                detalhes = f'Corte na equacao {eq_tear}'
            elif v in indet:
                tipo = 'Indeterminavel'
                detalhes = 'Nao determinavel com instrumentacao atual'
            else:
                tipo = 'Indefinido'
                detalhes = 'Status nao classificado'
            eqs_contem = [q for q, vars_eq in eqs.items() if v in vars_eq]
            todas_vars_data.append({
                'Variavel': v,
                'Classificacao': tipo,
                'Detalhes': detalhes,
                'Equacoes_que_Contem': ', '.join(sorted(eqs_contem)),
                'Numero_Equacoes': len(eqs_contem),
                'Metodo': (_LAST_INFERENCE_METHOD.get(v, '') if v in inf else ''),
            })
        pd.DataFrame(todas_vars_data).to_excel(writer, sheet_name='Todas_Variaveis', index=False)

    print("\nRESULTADOS SALVOS EM EXCEL:")
    print(f" Arquivo: {filepath}")
    print(f" {len(pd.ExcelFile(filepath).sheet_names)} abas criadas:")
    print("   - Resumo")
    print("   - Variaveis_Medidas")
    print("   - Variaveis_Inferidas")
    print("   - Tears_Variaveis_Corte")
    print("   - Indeterminaveis")
    print("   - Estrategia_Inferencia")
    print("   - Equacoes_Sistema")
    print("   - Todas_Variaveis")
    return filepath


def teste_cenarios_documento():
    """Testa os cenários EXATOS do documento com o v3 corrigido"""
    run_w_align = 1
    align_tag = "align_on" if run_w_align != 0 else "align_off"
    print(" TESTE COM CONDIÇÕES EXATAS DO DOCUMENTO URS")
    print("=" * 60)
    print("Documento: 'Instrumentação Mínima da URS' - Marília Caroline C. de Sá")
    print("Configurações de entrada IDÊNTICAS ao documento original")
    print()
    
    total_vars = len(all_vars_documento(equations_documento()))
    print(f"Sistema: {total_vars} variáveis, 19 equações (conforme documento)")
    print("Equações 15-19 SEM as variáveis Fc_A, Fc_B, Fc_C, Fc_D, Fc_E")
    print()
    
    print(" CONFIGURAÇÕES ESPERADAS DO DOCUMENTO:")
    print("  • Sistema Ideal: 26 medidas → 100% observável")
    print("  • Sistema Real: 22 medidas → ~65% observável")  
    print("  • Instr. Mínima: +2 críticas → 100% observável")
    print()
    
    # CENÁRIO 1: Sistema Ideal do Documento (26 medidas conforme documento)
    print(" CENÁRIO 1: Sistema Ideal do Documento")
    print("=" * 50)
    sistema_ideal = {
        # Pressões de entrada (5)
        "Pa_A", "Pa_B", "Pa_C", "Pa_D", "Pa_E",
        # Pressões intermediárias (5) 
        "Pb_A", "Pb_B", "Pb_C", "Pb_D", "Pb_E",
        # Pressões concentrado (5)
        "Pc_A", "Pc_B", "Pc_C", "Pc_D", "Pc_E",
        # Vazões de reciclo (5)
        "Rc_A", "Rc_B", "Rc_C", "Rc_D", "Rc_E",
        # Vazões entrada (5) - TODAS medidas no ideal
        "Ra_A", "Ra_B", "Ra_C", "Ra_D", "Ra_E",
        # Vazão total de resíduo (1) - redundante segundo documento
        "R"
        # Total: 5+5+5+5+5+1 = 26 variáveis (conforme documento)
    }
    
    print(f"Medidas configuradas: {len(sistema_ideal)} variáveis")
    print(f"Lista: {sorted(sistema_ideal)}")
    print()
    # sistema_ideal |= {"Ra_C", "Ra_D", "Ra_E"}
    med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs, tears_dict = classify_documento_v2(
        sistema_ideal, W_align=run_w_align
    )
    tears = tears_dict
    obs_percent = (len(med)+len(inf))/total_vars*100
    
    print(f"RESULTADO RESUMIDO:")
    print(f"  Observabilidade: {obs_percent:.1f}% | Tears: {len([1 for (q,v) in tears_list if v not in inf])} | Status: {'SUCESSO' if obs_percent == 100 else ' PARCIAL'}")
    print()
    
    imprimir_detalhes_solucao("SISTEMA IDEAL", med, inf, indet, tears, total_vars, sistema_ideal, None, inference_eqs)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_pct = (len(med) + len(inf)) / total_vars * 100 if total_vars > 0 else 0
        sistema_info = f"URS_{len(equations_documento())}eq_{len(all_vars_documento(equations_documento()))}vars_{obs_pct:.1f}pct_{align_tag}"
        nome_arquivo = f"resultados_{sistema_info}_cenario_ideal_{timestamp}.xlsx"
        excel_path = save_results_to_excel(
            med=med,
            inf=inf,
            indet=indet,
            tears_list=tears_list,
            tears_reclassificados=tears_reclassificados,
            tears_dict=tears_dict,
            inference_eqs=inference_eqs,
            eqs=equations_documento(),
            measured_vars=sistema_ideal,
            filename=nome_arquivo,
            inference_method=_LAST_INFERENCE_METHOD,
        )
    except Exception as e:
        print(f" Erro ao salvar arquivo Excel (cenario ideal): {e}")
    
    # CENÁRIO 2: Sistema Real do Documento (22 medidas conforme documento)
    print("\n" + " CENÁRIO 2: Sistema Real do Documento")
    print("=" * 48)
    # Sistema real = ideal MENOS as 4 variáveis críticas em falha (conforme Tabela 1 do documento)
    sistema_real = sistema_ideal - {"R", "Ra_C", "Ra_D", "Ra_E"}
    # Variáveis em falha conforme documento:
    # - R: 23-100% dados faltantes  
    # - Ra_C: 30-100% dados faltantes
    # - Ra_D: 28-100% dados faltantes  
    # - Ra_E: 64-100% dados faltantes
    # Total: 26 - 4 = 22 variáveis (conforme documento)
    
    print(f"Medidas configuradas: {len(sistema_real)} variáveis")
    print(f"Lista: {sorted(sistema_real)}")
    print()
    
    med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs_real, tears_real = classify_documento_v2(
        sistema_real, W_align=run_w_align
    )
    obs_percent = (len(med)+len(inf))/total_vars*100
    
    print(f"RESULTADO RESUMIDO:")
    print(f"  Observabilidade: {obs_percent:.1f}% | Tears: {len([1 for (q,v) in tears_list if v not in inf])} | Status: {'SUPERIOR' if obs_percent > 65 else 'CONFORME'}")
    print(f"  Documento esperava: ~65% | v3 conseguiu: {obs_percent:.1f}%")
    print()
    
    imprimir_detalhes_solucao("SISTEMA REAL", med, inf, indet, tears_real, total_vars, sistema_real, None, inference_eqs_real)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_pct = (len(med) + len(inf)) / total_vars * 100 if total_vars > 0 else 0
        sistema_info = f"URS_{len(equations_documento())}eq_{len(all_vars_documento(equations_documento()))}vars_{obs_pct:.1f}pct_{align_tag}"
        nome_arquivo = f"resultados_{sistema_info}_cenario_real_{timestamp}.xlsx"
        excel_path = save_results_to_excel(
            med=med,
            inf=inf,
            indet=indet,
            tears_list=tears_list,
            tears_reclassificados=tears_reclassificados,
            tears_dict=tears_real,
            inference_eqs=inference_eqs_real,
            eqs=equations_documento(),
            measured_vars=sistema_real,
            filename=nome_arquivo,
            inference_method=_LAST_INFERENCE_METHOD,
        )
    except Exception as e:
        print(f" Erro ao salvar arquivo Excel (cenario real): {e}")
    
    # CENÁRIO 3: Instrumentação Mínima do Documento (24 medidas conforme documento)
    print("\n" + " CENÁRIO 3: Instrumentação Mínima do Documento")
    print("=" * 52)
    print("Testando combinações de 2 variáveis críticas conforme documento")
    print()
    
    # Combinações testadas no documento (seção 4.2.3)
    pares_minimos = [
        ("Ra_C", "Ra_D", "Ra_C e Ra_D medidas"),
        ("Ra_C", "Ra_E", "Ra_C e Ra_E medidas"), 
        ("Ra_D", "Ra_E", "Ra_D e Ra_E medidas"),
    ]
    
    # Testar primeira combinação (Ra_C + Ra_D)
    var1, var2, descricao = pares_minimos[0]
    instrumentacao_minima = sistema_real | {var1, var2}
    # Total: 22 + 2 = 24 variáveis (conforme documento)
    
    print(f"Teste: {descricao}")
    print(f"Medidas configuradas: {len(instrumentacao_minima)} variáveis")
    print(f"Lista: {sorted(instrumentacao_minima)}")
    print()
    
    med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs_min, tears_min = classify_documento_v2(
        instrumentacao_minima, W_align=run_w_align
    )
    obs_percent = (len(med)+len(inf))/total_vars*100
    
    print(f"RESULTADO RESUMIDO:")
    print(f"  Observabilidade: {obs_percent:.1f}% | Tears: {len([1 for (q,v) in tears_list if v not in inf])} | Status: {'SUCESSO' if obs_percent == 100 else ' PARCIAL'}")
    print(f"  Documento esperava: 100% | v3 conseguiu: {obs_percent:.1f}%")
    print()
    
    imprimir_detalhes_solucao("INSTRUMENTAÇÃO MÍNIMA", med, inf, indet, tears_min, total_vars, instrumentacao_minima, None, inference_eqs_min)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_pct = (len(med) + len(inf)) / total_vars * 100 if total_vars > 0 else 0
        sistema_info = f"URS_{len(equations_documento())}eq_{len(all_vars_documento(equations_documento()))}vars_{obs_pct:.1f}pct_{align_tag}"
        nome_arquivo = f"resultados_{sistema_info}_cenario_instr_min_{timestamp}.xlsx"
        excel_path = save_results_to_excel(
            med=med,
            inf=inf,
            indet=indet,
            tears_list=tears_list,
            tears_reclassificados=tears_reclassificados,
            tears_dict=tears_min,
            inference_eqs=inference_eqs_min,
            eqs=equations_documento(),
            measured_vars=instrumentacao_minima,
            filename=nome_arquivo,
            inference_method=_LAST_INFERENCE_METHOD,
        )
    except Exception as e:
        print(f" Erro ao salvar arquivo Excel (instr. minima): {e}")


def teste_cenarios_kpi_estagios():
    """
    Executa cenarios de KPI por estagios:
      - Real: sem PT1_P e PT2_P
      - Incremental 1: Real + PT1_P
      - Incremental 2: Real + PT2_P
      - Incremental 3: Real + PT1_P + PT2_P
    """
    print(" TESTE KPI POR ESTAGIOS (REAL e INCREMENTAIS)")
    print("=" * 60)
    run_w_align = 1
    align_tag = "align_on" if run_w_align != 0 else "align_off"

    eqs_kpi = equations_kpi_estagios()
    total_vars = len(all_vars_documento(eqs_kpi))

    base_real = {"PT1_F", "PT1_R", "PT2_F", "PT2_R", "Pa", "Pb", "Pc", "A", "Lp", "Dpi"}
    cenarios = [
        ("REAL", base_real),
        ("INC_PT1P", base_real | {"PT1_P"}),
        ("INC_PT2P", base_real | {"PT2_P"}),
        ("INC_PT1P_PT2P", base_real | {"PT1_P", "PT2_P"}),
    ]

    for nome, medidas in cenarios:
        print(f"\n CENARIO: {nome}")
        print("-" * 50)
        print(f"Medidas: {len(medidas)} -> {sorted(medidas)}")
        try:
            med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs, tears_dict = classify_kpi_estagios_v2(
                medidas, W_align=run_w_align
            )
        except Exception as e:
            print(f"Erro na classificacao: {e}")
            continue

        obs_pct = (len(med) + len(inf)) / total_vars * 100 if total_vars > 0 else 0
        print(f"Observabilidade: {len(med)+len(inf)}/{total_vars} = {obs_pct:.1f}%")
        print(f"Inferidas: {sorted(inf)}")
        strict_pairs = [(eq, v) for (eq, v) in tears_list if v not in inf]
        print(f"Tears (nao reclassificados): {sorted({v for _, v in strict_pairs})}")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sistema_info = f"KPI_Estagios_{len(eqs_kpi)}eq_{total_vars}vars_{obs_pct:.1f}pct_{align_tag}"
            nome_arquivo = f"resultados_{sistema_info}_{nome}_{timestamp}.xlsx"
            save_results_to_excel(
                med=med,
                inf=inf,
                indet=indet,
                tears_list=tears_list,
                tears_reclassificados=tears_reclassificados,
                tears_dict=tears_dict,
                inference_eqs=inference_eqs,
                eqs=eqs_kpi,
                measured_vars=medidas,
                filename=nome_arquivo,
                inference_method=_LAST_INFERENCE_METHOD,
            )
        except Exception as e:
            print(f"Erro ao salvar Excel ({nome}): {e}")


def classify_kpi_estagios_v2(
    measured: Set[str], *,
    W_obs=100, W_tears=1, W_bonus=0, W_promo=10, T_max=None,
    force_heads=None, no_raise_on_infeasible=False, W_align=1,
    repair_tears_slack: int | None = 0,
    enable_repair_3bc: bool = False,
    solver_time_limit_s: int = 60
):
    """
    Classifica KPIs por estagios com a mesma logica o/x/z/p/t/L (duas fases).
    enable_repair_3bc e aceito para interface uniforme (nao utilizado aqui).
    """
    eqs = equations_kpi_estagios()
    V = sorted(all_vars_documento(eqs))
    invalid = measured - set(V)
    if invalid:
        raise ValueError(f"Invalid measured variables: {sorted(invalid)}")

    uses = {v: [q for q, vs in eqs.items() if v in vs] for v in V}
    M = len(V)

    mdl = cp_model.CpModel()

    y = {v: mdl.NewBoolVar(f"y_{v}") for v in V}
    x = {v: mdl.NewBoolVar(f"x_{v}") for v in V}
    o = {v: mdl.NewBoolVar(f"o_{v}") for v in V}
    z = {(q, v): mdl.NewBoolVar(f"z_{q}_{v}") for q, vs in eqs.items() for v in vs}
    t = {(q, v): mdl.NewBoolVar(f"t_{q}_{v}") for q, vs in eqs.items() for v in vs}
    p = {(q, v): mdl.NewBoolVar(f"p_{q}_{v}") for q, vs in eqs.items() for v in vs}
    L = {v: mdl.NewIntVar(0, M, f"L_{v}") for v in V}

    s_align = []
    if W_align != 0:
        for q, vs in eqs.items():
            for v in vs:
                for u in vs:
                    if u == v:
                        continue
                    sp = mdl.NewBoolVar(f"s_p_{q}_{u}_head_{v}")
                    mdl.Add(sp <= p[q, v])
                    mdl.Add(sp <= t[q, u])
                    mdl.Add(sp >= p[q, v] + t[q, u] - 1)
                    s_align.append(sp)
                    sz = mdl.NewBoolVar(f"s_z_{q}_{u}_head_{v}")
                    mdl.Add(sz <= z[q, v])
                    mdl.Add(sz <= t[q, u])
                    mdl.Add(sz >= z[q, v] + t[q, u] - 1)
                    s_align.append(sz)

    for v in V:
        z_sum = sum(z[q, v] for q in uses[v])
        t_sum = sum(t[q, v] for q in uses[v])
        p_sum = sum(p[q, v] for q in uses[v])

        if v in measured:
            mdl.Add(y[v] == 1)
        else:
            mdl.Add(y[v] == 0)

        mdl.Add(z_sum <= 1)
        mdl.Add(t_sum <= 1)
        mdl.Add(p_sum <= 1)
        mdl.Add(z_sum + p_sum <= 1)

        mdl.Add(x[v] >= y[v]); mdl.Add(x[v] >= z_sum); mdl.Add(x[v] >= t_sum); mdl.Add(x[v] >= p_sum)
        mdl.Add(o[v] >= y[v]); mdl.Add(o[v] >= z_sum); mdl.Add(o[v] >= p_sum)

        mdl.Add(x[v] <= y[v] + z_sum + t_sum + p_sum)
        mdl.Add(o[v] <= y[v] + z_sum + p_sum)

    for v in measured:
        mdl.Add(x[v] == 1)

    for q, vs in eqs.items():
        for v in vs:
            mdl.Add(z[q, v] <= 1 - y[v])
            mdl.Add(z[q, v] <= 1 - t[q, v])
            mdl.Add(t[q, v] <= 1 - y[v])

            for u in vs:
                if u != v:
                    mdl.Add(z[q, v] <= o[u] + t[q, u])

            mdl.Add(p[q, v] <= 1 - y[v])
            mdl.Add(p[q, v] <= 1 - t[q, v])
            for u in vs:
                if u != v:
                    mdl.Add(p[q, v] <= o[u])

            for u in vs:
                if u != v:
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([z[q, v], t[q, u].Not()])
                    mdl.Add(L[u] >= L[v] + 1).OnlyEnforceIf([z[q, u], t[q, v].Not()])
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([p[q, v], t[q, u].Not()])

        mdl.Add(sum(z[q, v] for v in vs) + sum(p[q, v] for v in vs) <= 1)

    if T_max is not None:
        mdl.Add(sum(t.values()) <= T_max)

    bonus_terms = []
    if W_bonus:
        for q, vs in eqs.items():
            for v in vs:
                others = [u for u in vs if u != v]
                if all(u in measured for u in others):
                    bonus_terms.append(z[(q, v)])

    primary = (
        W_obs * sum(o.values())
        - W_tears * sum(t.values())
        + (W_bonus * sum(bonus_terms) if W_bonus else 0)
    )

    exclude_eqs_phase1 = set()
    if force_heads and "exclude_phase1" in force_heads:
        exclude_eqs_phase1 = force_heads["exclude_phase1"]
        for eq in exclude_eqs_phase1:
            if eq in eqs:
                for v in eqs[eq]:
                    if (eq, v) in z:
                        mdl.Add(z[eq, v] == 0)
                    if (eq, v) in p:
                        mdl.Add(p[eq, v] == 0)

    mdl.Maximize(primary)

    solver = cp_model.CpSolver()
    params = solver.parameters
    params.max_time_in_seconds = solver_time_limit_s
    params.random_seed = 42
    params.num_search_workers = 1
    params.randomize_search = False
    params.log_search_progress = False

    status1 = solver.Solve(mdl)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if no_raise_on_infeasible:
            return [], [], [], [], [], [], {}, {}
        raise RuntimeError(f"Phase 1 failed: {solver.StatusName(status1)}")

    best_primary = solver.ObjectiveValue()
    tear_vars_f1 = {v for v in V if any(solver.Value(t[q, v]) for q in uses[v])}

    if status1 == cp_model.OPTIMAL:
        mdl.Add(primary >= math.ceil(best_primary))
    else:
        mdl.Add(primary >= math.floor(best_primary))
    # Promocao so pode atuar em variaveis que eram tears na Fase 1.
    for v in V:
        if v not in tear_vars_f1:
            mdl.Add(sum(p[q, v] for q in uses[v]) == 0)
    align_term = sum(s_align) if s_align else 0
    mdl.Maximize(primary + (W_promo * sum(p.values())) + (W_align * align_term))

    status2 = solver.Solve(mdl)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if no_raise_on_infeasible:
            return [], [], [], [], [], [], {}, {}
        raise RuntimeError(f"Phase 2 failed: {solver.StatusName(status2)}")

    last_solve_feasible = True

    if last_solve_feasible:
        med = [v for v in V if solver.Value(y[v])]
        inf = [v for v in V if (solver.Value(o[v]) == 1 and solver.Value(y[v]) == 0)]
        indet = []
        for v in V:
            if solver.Value(y[v]) == 0:
                z_on = any(solver.Value(z[q, v]) for q in uses[v])
                p_on = any(solver.Value(p[q, v]) for q in uses[v])
                t_on = any(solver.Value(t[q, v]) for q in uses[v])
                if (not z_on) and (not p_on) and (not t_on):
                    indet.append(v)
        tears_list, tears_dict = [], defaultdict(list)
        for (q, v), var in t.items():
            if solver.Value(var) == 1:
                tears_list.append((q, v))
                tears_dict[q].append(v)
        inference_eqs = {}
        inference_method: Dict[str, str] = {}
        for v in V:
            if solver.Value(y[v]) == 0 and solver.Value(o[v]) == 1:
                for q in uses[v]:
                    if solver.Value(p[q, v]) == 1:
                        inference_eqs[v] = [q]
                        inference_method[v] = 'p'
                        break
                    if solver.Value(z[q, v]) == 1:
                        inference_eqs[v] = [q]
                        inference_method[v] = 'z'
                        break
        _sanity_check(med, inf, indet, tears_dict, eqs)
        _LAST_INFERENCE_METHOD.clear()
        _LAST_INFERENCE_METHOD.update(inference_method)
        return med, inf, indet, tears_list, [], [], inference_eqs, tears_dict
    else:
        return [], [], [], [], [], [], {}, {}


if __name__ == "__main__":
    teste_cenarios_kpi_estagios() 
