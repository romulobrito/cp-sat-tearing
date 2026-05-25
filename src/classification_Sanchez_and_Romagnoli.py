#!/usr/bin/env python3
"""
Classificação de variáveis — Sistema v2 (genérico)
Observação: o total de equações/variáveis vem de `equations_v2()` e é reportado dinamicamente em tempo de execução.
Baseado na metodologia validada em teste_simples_classificacao.py
"""
from ortools.sat.python import cp_model
from typing import Dict, List, Set
import math
import pandas as pd
import os
from datetime import datetime

# metodo de inferencia por variavel da ultima execucao ('z' ou 'p')
_LAST_INFERENCE_METHOD: Dict[str, str] = {}


def equations_v2():
    """
    Sistema de equacoes do Sistema v2 (versao estendida)
    Formato: lado esquerdo = lado direito; todas as variaveis da equacao participam do balanco
    """
    return {
        "Eq1":  ["x1", "x3", "x4"],                          # x1 = x3 + x4
        "Eq2":  ["x3", "x5", "x6"],                          # x3 = x5 + x6
        "Eq3":  ["x2", "x4", "x5", "x6", "x7", "x8", "x9"],  # x2 + x4 + x5 + x6 + x7 = x8 + x9
        "Eq4":  ["x8", "x10", "x11", "x12"],                 # x8 + x10 = x11 + x12
        "Eq5":  ["x11", "x10", "x7"],                        # x11 = x10 + x7
        "Eq6":  ["x9", "x13", "x14", "x15"],                 # x9 = x13 + x14 + x15
        "Eq7":  ["x15", "x16", "x34"],                       # x15 = x16 + x34
        "Eq8":  ["x34", "x17", "x35"],                       # x34 = x17 + x35
        "Eq9":  ["x35", "x18", "x20", "x36"],                # x35 = x18 + x20 + x36
        "Eq10": ["x36", "x23", "x24", "x25"],                # x36 = x23 + x24 + x25
        "Eq11": ["x24", "x21", "x22", "x26"],                # x24 = x21 + x22 + x26
        "Eq12": ["x20", "x22", "x23", "x19"],                # x20 + x22 + x23 = x19
        "Eq13": ["x16", "x17", "x18", "x25", "x26", "x27", "x28", "x29", "x30"], # x16 + x17 + x18 + x25 + x26 = x27 + x28 + x29 + x30
        "Eq14": ["x30", "x31", "x32", "x33"],                # x30 = x31 + x32 + x33
        "Eq15": ["x28", "x29", "x31", "x32", "x33", "x63", "x37", "x61"], # x28 + x29 + x31 + x32 + x33 + x63 = x37 + x61
        "Eq16": ["x38", "x61", "x55"],                       # x38 + x61 = x55
        "Eq17": ["x54", "x55", "x48"],                       # x54 + x55 = x48
        "Eq18": ["x56", "x54", "x57", "x58"],                # x56 = x54 + x57 + x58
        "Eq19": ["x60", "x59"],                              # x60 = x59
        "Eq20": ["x57", "x59", "x62", "x56"],                # x57 + x59 + x62 = x56
        "Eq21": ["x48", "x47", "x49"],                       # x48 = x47 + x49
        "Eq22": ["x47", "x44", "x45"],                       # x47 = x44 + x45
        "Eq23": ["x51", "x50"],                              # x51 = x50
        "Eq24": ["x50", "x52"],                              # x50 = x52
        "Eq25": ["x52", "x51", "x53"],                       # x52 = x51 + x53
        "Eq26": ["x44", "x42", "x63"],                       # x44 = x42 + x63
        "Eq27": ["x45", "x46", "x43"],                       # x45 = x46 + x43
        "Eq28": ["x42", "x43", "x41"],                       # x42 + x43 = x41
        "Eq29": ["x41", "x38", "x40"],                       # x41 = x38 + x40
        "Eq30": ["x40", "x39", "x62"],                       # x40 = x39 + x62
        "Eq31": ["x39", "x58", "x60"],                       # x39 + x58 = x60
        "Eq32": ["x1", "x2", "x12", "x13", "x14", "x19", "x21", "x27", "x37", "x46", "x53"]  # x1 + x2 = x12 + x13 + x14 + x19 + x21 + x27 + x37 + x46 + x53
    }

def all_vars_v2(eqs: Dict[str, List[str]]) -> Set[str]:
    """Extrai todas as variáveis do sistema"""
    vars_set = set()
    for vs in eqs.values():
        vars_set.update(vs)
    return vars_set

def classify_v2(measured: Set[str], *, W_obs=100, W_tears=1, W_bonus=0, W_promo=5, T_max=None, force_heads=None, no_raise_on_infeasible=False, W_align=1, repair_tears_slack: int | None = 0):
    """
    Classifica variaveis do sistema v2 usando CP-SAT, incorporando promocao p[q,v]
    diretamente na modelagem.

    Observacao:
    - O pos-processamento cycle-safe continua existindo como etapa complementar
      para reclassificacoes adicionais de tears remanescentes, sem criar ciclos.

    Parâmetros sugeridos (exemplo): W_obs=100, W_tears=1, W_bonus=0..2, W_promo=5..10, T_max=1
    
    force_heads: Dict[str, str] opcional para forçar certas equações como cabeças (teste)
                Ex: {"Eq4": "x1"} força z[Eq4,x1]=1
    """
    eqs = equations_v2()
    V = sorted(all_vars_v2(eqs))
    
    # Validação das medidas
    invalid = measured - set(V)
    if invalid:
        raise ValueError(f"Variáveis medidas inválidas: {invalid}. Válidas: {V}")
    
    uses = {v: [q for q, vs in eqs.items() if v in vs] for v in V}
    M = len(V)
    
    mdl = cp_model.CpModel()
    
    # Variáveis de decisão
    y = {v: mdl.NewBoolVar(f"y_{v}") for v in V}  # medição
    x = {v: mdl.NewBoolVar(f"x_{v}") for v in V}  # disponível (medida/inf/tear/promo)
    o = {v: mdl.NewBoolVar(f"o_{v}") for v in V}  # observável/disponível sem tear (medida/inf/promo)
    z = {(q, v): mdl.NewBoolVar(f"z_{q}_{v}") for q, vs in eqs.items() for v in vs}  # inferência
    t = {(q, v): mdl.NewBoolVar(f"t_{q}_{v}") for q, vs in eqs.items() for v in vs}  # tear
    p = {(q, v): mdl.NewBoolVar(f"p_{q}_{v}") for q, vs in eqs.items() for v in vs}  # promoção
    L = {v: mdl.NewIntVar(0, M, f"L_{v}") for v in V}  # níveis topológicos
    # --- Termos de alinhamento tear-cabeca (para objetivo da Fase 2) ---
    # Se W_align == 0, evita criar variaveis/restricoes de alinhamento para reduzir o modelo.
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

    # --- Vínculos por variável (capacidades/exclusividades; bounds de x e o) ---
    for v in V:
        z_sum = sum(z[q, v] for q in uses[v])
        t_sum = sum(t[q, v] for q in uses[v])
        p_sum = sum(p[q, v] for q in uses[v])

        # fixa y
        mdl.Add(y[v] == 1) if v in measured else mdl.Add(y[v] == 0)

        # capacidades por variável
        mdl.Add(z_sum <= 1)  # no máx. uma equação inferindo v
        mdl.Add(t_sum <= 1)  # no máx. um tear envolvendo v
        mdl.Add(p_sum <= 1)  # no máx. uma promoção escolhida para v

        # exclusividades por variável
        # REMOVIDO: permitir tear em uma equacao e resolver em outra
        # mdl.Add(z_sum + t_sum <= 1)  # não pode inferir e ser tear
        mdl.Add(z_sum + p_sum <= 1)  # não pode ter z e p simultaneamente
        # REMOVIDO: permitir tear em uma equacao e resolver em outra
        # mdl.Add(p_sum + t_sum <= 1)  # nao ter tear e promocao ao mesmo tempo

        # lower bounds (ativadores)
        mdl.Add(x[v] >= y[v]); mdl.Add(x[v] >= z_sum); mdl.Add(x[v] >= t_sum); mdl.Add(x[v] >= p_sum)
        mdl.Add(o[v] >= y[v]); mdl.Add(o[v] >= z_sum); mdl.Add(o[v] >= p_sum)  # (tear não ativa o)

        # upper bounds
        mdl.Add(x[v] <= y[v] + z_sum + t_sum + p_sum)
        mdl.Add(o[v] <= y[v] + z_sum + p_sum)


    # Disponibilidade imediata das medidas
    for v in measured:
        mdl.Add(x[v] == 1)

    # --- Restrições por equação ---
    for q, vs in eqs.items():
        for v in vs:
            # z só em não medido; z e t exclusivos; t só em não medido
            mdl.Add(z[q, v] <= 1 - y[v])
            mdl.Add(z[q, v] <= 1 - t[q, v])
            mdl.Add(t[q, v] <= 1 - y[v])

            # Gating de inferência: todos os outros u devem estar disponiveis sem tear (o[u]=1) OU serem tears NA MESMA eq.
            for u in vs:
                if u != v:
                    mdl.Add(z[q, v] <= o[u] + t[q, u])

            # Promoção p: só em não medido, e não no mesmo lugar marcado como tear
            mdl.Add(p[q, v] <= 1 - y[v])
            mdl.Add(p[q, v] <= 1 - t[q, v])  # p e t locais exclusivos

            # p só pode ligar se TODOS os outros u dessa equação já forem observáveis
            # OU estiverem como tear NA MESMA equação (aliviando o gate local)
            for u in vs:
                if u != v:
                    mdl.Add(p[q, v] <= o[u])

            # Topologia (evita ciclos) tanto para z quanto para p
            for u in vs:
                if u != v:
                    # para z: aresta u->v so se z[q,v] e u nao for tear em q
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([z[q, v], t[q, u].Not()])
                    # simetrico: aresta v->u so se z[q,u] e v nao for tear em q
                    mdl.Add(L[u] >= L[v] + 1).OnlyEnforceIf([z[q, u], t[q, v].Not()])
                    # para p: aresta u->v so se p[q,v] e u nao for tear em q
                    mdl.Add(L[v] >= L[u] + 1).OnlyEnforceIf([p[q, v], t[q, u].Not()])

        # Capacidade por equação: no máx. uma cabeça (z ou p) escolhida nessa equação
        mdl.Add(sum(z[q, v] for v in vs) + sum(p[q, v] for v in vs) <= 1)

    # Limite global de tears (opcional)
    if T_max is not None:
        mdl.Add(sum(t.values()) <= T_max)

    # Bônus opcional: preferir z "fáceis" (todos predecessores medidos)
    bonus_terms = []
    if W_bonus:
        for q, vs in eqs.items():
            for v in vs:
                others = [u for u in vs if u != v]
                if all(u in measured for u in others):
                    bonus_terms.append(z[(q, v)])

   # Objetivo: p torna promoção atrativa; tears penalizados fracamente; observabilidade muito valorizada
    # -------- FASE 1: objetivo principal (sem contar W_promo) --------
    primary = (
        W_obs * sum(o.values())
        - W_tears * sum(t.values())
        + (W_bonus * sum(bonus_terms) if W_bonus else 0)
    )
    # === FASE 1: maximiza objetivo principal ===
    # Criar variável para força exclusão de equações (APENAS PARA TESTE)
    exclude_eqs_phase1 = set()
    if force_heads and "exclude_phase1" in force_heads:
        exclude_eqs_phase1 = force_heads["exclude_phase1"]
        print(f"[TESTE] Excluindo equações da Fase 1: {exclude_eqs_phase1}")
        # Forçar z=0 e p=0 para todas as variáveis das equações excluídas
        for eq in exclude_eqs_phase1:
            if eq in eqs:
                for v in eqs[eq]:
                    if (eq, v) in z:
                        mdl.Add(z[eq, v] == 0)
                    if (eq, v) in p:
                        mdl.Add(p[eq, v] == 0)
    
    mdl.Maximize(primary)

    # Solver determinístico
    solver = cp_model.CpSolver()
    params = solver.parameters
    params.max_time_in_seconds = 600
    params.random_seed = 42
    params.num_search_workers = 1
    params.randomize_search = False
    # params.log_search_progress = True  # desativado para suprimir logs detalhados
    params.log_search_progress = False

    status1 = solver.Solve(mdl)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Fase 1 falhou: {solver.StatusName(status1)}")

    best_primary = solver.ObjectiveValue()
    print("[FASE 1] ", solver.StatusName(status1))
    print("[FASE 1] primary* =", best_primary)
    print("[FASE 1] time      =", solver.WallTime())
    print("[FASE 1] sum(p)  =", sum(solver.Value(v) for v in p.values()))

    # --- INSERIR AQUI (depois da impressão da Fase 1) ---
    p1 = [(q,v) for (q,v), var in p.items() if solver.Value(var) == 1]
    z1 = [(q,v) for (q,v), var in z.items() if solver.Value(var) == 1]
    o1 = sum(solver.Value(var) for var in o.values())
    t1 = sum(solver.Value(var) for var in t.values())
    print(f"[FASE 1] sum(p)  = {len(p1)} | sum(z) = {len(z1)} | sum(o) = {o1} | sum(t) = {t1}")
    print(f"[FASE 1] p ativos: {sorted(p1)}")
    
    # Tears ativos na Fase 1 (por variavel)
    tear_vars_f1 = {v for v in V if any(solver.Value(t[q, v]) for q in uses[v])}


    # -------- FASE 2: fixa o valor ótimo da fase 1 e maximiza promoções --------
    # Remover restrições de exclusão da Fase 1 (APENAS PARA TESTE)
    if exclude_eqs_phase1:
        print(f"[TESTE] Removendo exclusões para Fase 2, liberando: {exclude_eqs_phase1}")
        # Não podemos remover constraints, mas podemos recriar o modelo
        # Por simplicidade, vamos apenas informar que as equações estão liberadas
        # O solver tentará usar as promoções p[q,v] para essas equações
    
    # Como os pesos sao inteiros, podemos usar ceil; com solucao otima, 'primary' nao pode ser maior.
    if status1 == cp_model.OPTIMAL:
        mdl.Add(primary >= math.ceil(best_primary))
    else:
        # Conservador quando apenas FEASIBLE: garante nao piorar a Fase 1
        mdl.Add(primary >= math.floor(best_primary))
    # Promocao so pode atuar em variaveis que eram tears na Fase 1.
    for v in V:
        if v not in tear_vars_f1:
            mdl.Add(sum(p[q, v] for q in uses[v]) == 0)
    align_term = sum(s_align) if s_align else 0
    mdl.Maximize(primary + (W_promo * sum(p.values())) + (W_align * align_term))

    status2 = solver.Solve(mdl)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Fase 2 falhou: {solver.StatusName(status2)}")

    # (opcional) checar o primary atingido na fase 2
    primary_phase2 = (
        W_obs * sum(solver.Value(v) for v in o.values())
        - W_tears * sum(solver.Value(v) for v in t.values())
        + (W_bonus * sum(solver.Value(v) for v in bonus_terms) if W_bonus else 0)
    )

    print("[FASE 2] ", solver.StatusName(status2))
    print("[FASE 2] primary   =", primary_phase2, " (>= fase 1)")
    print("[FASE 2] sum(p)    =", sum(solver.Value(v) for v in p.values()))
    print("[FASE 2] time      =", solver.WallTime())

    # --- INSERIR AQUI (depois da impressão da Fase 2) ---
    p2 = [(q,v) for (q,v), var in p.items() if solver.Value(var) == 1]
    z2 = [(q,v) for (q,v), var in z.items() if solver.Value(var) == 1]
    o2 = sum(solver.Value(var) for var in o.values())
    t2 = sum(solver.Value(var) for var in t.values())

    print(f"[FASE 2] sum(p)  = {len(p2)} | sum(z) = {len(z2)} | sum(o) = {o2} | sum(t) = {t2}")

    novos_p = sorted(set(p2) - set(p1))
    perdeu_z = sorted(set(z1) - set(z2))
    print(f"[FASE 2] Δp (novas promoções): {novos_p}")
    print(f"[FASE 2] z desligados: {perdeu_z}")

    # Candidatos a materializacao: promocoes da Fase 2 que atacam tears presentes na Fase 1
    prom_fix_candidates = [(q, v) for (q, v) in p2 if v in tear_vars_f1]
    print(f"[REPAIR] candidatos a 'promover tear da Fase 1': {sorted(prom_fix_candidates)}")

    # Snapshot da solução da Fase 2 (para fallback em caso de inviabilidade)
    from collections import defaultdict
    med_f2 = [v for v in V if solver.Value(y[v])]
    inf_f2 = [v for v in V if (solver.Value(o[v]) == 1 and solver.Value(y[v]) == 0)]
    indet_f2 = []
    for v in V:
        if solver.Value(y[v]) == 0:
            z_on = any(solver.Value(z[q, v]) for q in uses[v])
            p_on = any(solver.Value(p[q, v]) for q in uses[v])
            t_on = any(solver.Value(t[q, v]) for q in uses[v])
            if (not z_on) and (not p_on) and (not t_on):
                indet_f2.append(v)
    tears_list_f2, tears_dict_f2 = [], defaultdict(list)
    for (q, v), var in t.items():
        if solver.Value(var) == 1:
            tears_list_f2.append((q, v))
            tears_dict_f2[q].append(v)
    inference_eqs_f2 = {}
    for v in V:
        if solver.Value(y[v]) == 0 and solver.Value(o[v]) == 1:
            for q in uses[v]:
                if solver.Value(z[q, v]) == 1 or solver.Value(p[q, v]) == 1:
                    inference_eqs_f2[v] = [q]
                    break

    # Controle de viabilidade do último solve
    last_solve_feasible = True

    # === FASE 3 (REPAIR): materializa promoções que substituem tears ===
    tear_vars = {v for (q, v), var in t.items() if solver.Value(var) == 1}
    # Usar tears da Fase 1 como alvo para materializacao
    prom_fix = list(prom_fix_candidates)

    print(f"[REPAIR] tears atuais: {sorted(tear_vars)}")
    print(f"[REPAIR] promocoes a fixar (z=1; t=0 na variavel): {sorted(prom_fix)}")

    if prom_fix:
        for (qfix, vfix) in prom_fix:
            # materializar substituicao como promocao ativa e sem tears na variavel
            mdl.Add(p[qfix, vfix] == 1)
            for q2 in uses[vfix]:
                mdl.Add(t[q2, vfix] == 0)

        # Duas tentativas: forte (ganhar +1 e nao aumentar tears+slack) e fraca (nao piorar)
        strong = mdl.NewBoolVar("repair_strong")
        weak   = mdl.NewBoolVar("repair_weak")
        mdl.Add(strong + weak == 1)
        # strong: o >= o2+1; t <= t2+slack
        mdl.Add(sum(o.values()) >= o2 + 1).OnlyEnforceIf(strong)
        if repair_tears_slack is None:
            mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf(strong)
        else:
            mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf(strong)
        # weak: o >= o2; t <= t2+slack
        mdl.Add(sum(o.values()) >= o2).OnlyEnforceIf(weak)
        if repair_tears_slack is None:
            mdl.Add(sum(t.values()) <= t2).OnlyEnforceIf(weak)
        else:
            mdl.Add(sum(t.values()) <= t2 + max(0, repair_tears_slack)).OnlyEnforceIf(weak)

        # volta a maximizar o objetivo principal e resolve novamente
        mdl.Maximize(primary)
        status3 = solver.Solve(mdl)
        print("[FASE 3] ", solver.StatusName(status3))
        if status3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            o3 = sum(solver.Value(var) for var in o.values())
            t3 = sum(solver.Value(var) for var in t.values())
            print(f"[FASE 3] sum(o) = {o3} | sum(t) = {t3}")
        else:
            last_solve_feasible = False
            print("[FASE 3] inviavel com salvaguardas; mantendo solucao da Fase 2")
    else:
        print("[REPAIR] Nenhuma promoção aplicável ao tear atual.")
        # === FASE 3b (REPAIR+): tenta promover explicitamente o tear atual ===
        if tear_vars:
            for vtear in sorted(tear_vars):
                cand_qs = [q for q in uses[vtear]
                           if all(solver.Value(o[u]) == 1 for u in eqs[q] if u != vtear)]
                if not cand_qs:
                    continue
                qstar = cand_qs[0]
                # força a promoção do tear e proíbe tears em vtear
                mdl.Add(p[qstar, vtear] == 1)
                for q2 in uses[vtear]:
                    mdl.Add(t[q2, vtear] == 0)

                # Duas tentativas: forte (ganhar +1 e nao aumentar tears+slack) e fraca (nao piorar)
                strong = mdl.NewBoolVar("repair_strong_b")
                weak   = mdl.NewBoolVar("repair_weak_b")
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
                status3b = solver.Solve(mdl)
                print("[FASE 3b] ", solver.StatusName(status3b))
                if status3b in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    o3b = sum(solver.Value(var) for var in o.values())
                    t3b = sum(solver.Value(var) for var in t.values())
                    print(f"[FASE 3b] promoted {(qstar, vtear)} | sum(o)={o3b} | sum(t)={t3b}")
                else:
                    last_solve_feasible = False
                    print("[FASE 3b] inviavel com salvaguardas; mantendo solucao da Fase 2")
                break

    
    # === FASE 3c (REPAIR++): realocar tears para a eq. da cabeça e desbloquear INDET ===
    # Objetivo: para cada indeterminável v, tentar criar cabeça em alguma eq q∋v,
    # movendo tears necessários (existentes) de outras equações para q. Não cria tears novos.
    if last_solve_feasible:
        # Recomputar indetermináveis com a solução atual do solver
        current_indet = []
        for v in V:
            if solver.Value(y[v]) == 0:  # não medido
                z_on = any(solver.Value(z[q, v]) for q in uses[v])
                p_on = any(solver.Value(p[q, v]) for q in uses[v])
                t_on = any(solver.Value(t[q, v]) for q in uses[v])
                if (not z_on) and (not p_on) and (not t_on):
                    current_indet.append(v)

        for vhard in sorted(current_indet):
            # Equações candidatas que contêm vhard e NÃO têm cabeça já escolhida
            cand_qs = []
            for q in eqs:
                if vhard not in eqs[q]:
                    continue
                head_taken = any(
                    (solver.Value(z[q, w]) == 1) or (solver.Value(p[q, w]) == 1)
                    for w in eqs[q]
                )
                if not head_taken:
                    cand_qs.append(q)

            # Tentativas reificadas (no máximo 1 por vhard)
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

                # Realoca tears (somente se a_try==1)
                for (u, q_old) in need_move:
                    mdl.Add(t[q, u] == 1).OnlyEnforceIf(a_try)
                    for q2 in uses[u]:
                        if q2 != q:
                            mdl.Add(t[q2, u] == 0).OnlyEnforceIf(a_try)

                # Cabeça via promoção (somente se a_try==1)
                mdl.Add(p[q, vhard] == 1).OnlyEnforceIf(a_try)

                # Salvaguardas reificadas
                mdl.Add(sum(o.values()) >= o2).OnlyEnforceIf(a_try)
                mdl.Add(sum(t.values()) <= t2 + (repair_tears_slack or 0)).OnlyEnforceIf(a_try)

                attempts.append(a_try)

            if attempts:
                mdl.Add(sum(attempts) <= 1)
                mdl.Maximize(primary)
                status3c = solver.Solve(mdl)
                chosen = [a.Name() for a in attempts if solver.Value(a) == 1]
                print("[FASE 3c] ", solver.StatusName(status3c), "para", vhard, "| escolhida:", chosen)
                if status3c in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    o2 = sum(solver.Value(var) for var in o.values())
                    t2 = sum(solver.Value(var) for var in t.values())


    # --- Extração de resultados ---
    if last_solve_feasible:
        med = [v for v in V if solver.Value(y[v])]
        inf = [v for v in V if (solver.Value(o[v]) == 1 and solver.Value(y[v]) == 0)]

        # Indetermináveis: não medidos e sem z/p/t ativos
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
        # Re-saneamento: garante que todo nao-medido esteja em INF, TEAR, ou INDET
        all_vars_set = set(V)
        tears_set_now_all = {v for vs in tears_dict.values() for v in vs}
        tears_set_now = tears_set_now_all - set(inf)  # tears estritos
        for v in all_vars_set:
            if (v not in med) and (v not in inf) and (v not in tears_set_now) and (v not in indet):
                indet.append(v)
        _LAST_INFERENCE_METHOD.clear()
        _LAST_INFERENCE_METHOD.update(inference_method)
    else:
        # retornar snapshot da Fase 2
        med, inf, indet = med_f2, inf_f2, indet_f2
        tears_list, tears_dict = tears_list_f2, tears_dict_f2
        inference_eqs = inference_eqs_f2
        _LAST_INFERENCE_METHOD.clear()
        _LAST_INFERENCE_METHOD.update({v: '' for v in inf})

    _sanity_check(med, inf, indet, tears_dict, eqs)
    return med, inf, indet, tears_list, [], [], inference_eqs, tears_dict


def poscheck_relatorio_v2(med, inf, indet, tears, inference_eqs, eqs, tears_list=None):
    """
    Pós-processamento oficial cycle-safe.
    Encapsula o fallback com checagem de ciclo para:
    - promover tears para inferidas quando possível sem fechar ciclos;
    - rebaixar tears remanescentes para indetermináveis quando apropriado.

    Mantém assinatura histórica para compatibilidade.
    """
    tears_dict: Dict[str, List[str]] = {}
    if isinstance(tears, dict):
        for eq, vs in tears.items():
            if isinstance(vs, (list, tuple, set)):
                tears_dict[eq] = list(vs)
            else:
                tears_dict[eq] = [vs]

    if tears_list is not None:
        for eq, v in tears_list:
            tears_dict.setdefault(eq, [])
            if v not in tears_dict[eq]:
                tears_dict[eq].append(v)

    med2, inf2, indet2, tears_pairs, tears_reclassificados, tears_rebaixados, _, _ = (
        fallback_cycle_safe_promotions(
            med=list(med),
            inf=list(inf),
            indet=list(indet),
            tears_dict=tears_dict,
            inference_eqs=dict(inference_eqs),
            eqs=eqs,
            demote_leftovers=True,
        )
    )

    inf2_set = set(inf2)
    tears_set_final = sorted({v for _, v in tears_pairs if v not in inf2_set})
    return med2, inf2, indet2, tears_set_final, tears_reclassificados, tears_rebaixados

def print_resultado_v2(med, inf, indet, tears_list, tears_reclassificados, tears, inference_eqs, eqs):
    """Imprime resultados da classificação"""
    total_vars = len(all_vars_v2(eqs))
    
    # CORREÇÃO: Observáveis = medidas + inferidas (NÃO incluir tears)
    observaveis = len(med) + len(inf)
    
    # Calcular grau de sobredeterminação corretamente
    eqs_utilizadas = set()
    for v, eqs_list in inference_eqs.items():
        eqs_utilizadas.update(eqs_list)
    for v, eq in tears_reclassificados:
        eqs_utilizadas.add(eq)
    
    grau_sobredeterminacao = len(eqs_utilizadas) - len(inf)
    
    print(f"\n========= RESULTADOS DA CLASSIFICACAO =========")
    print(f"Medidas         ({len(med):2d}): {med}")
    print(f"Inferidas       ({len(inf):2d}): {inf}")
    # Pares (eq, v) de tears estritos (v nao inferido)
    strict_pairs = [(eq, v) for (eq, v) in tears_list if v not in inf]
    strict_vars  = sorted({v for _, v in strict_pairs})
    
    # Visao 1 (papel): quais variaveis foram usadas como cortes
    print(f"Variaveis de corte ({len(strict_vars):2d}): {strict_vars}")
    
    # Visao 2 (status de identificabilidade): indeterminaveis incluem tears nao reclassificados
    indet_com_tears = sorted(set(indet) | set(strict_vars))
    print(f"Indeterminaveis (inclui tears) ({len(indet_com_tears):2d}): {indet_com_tears}")
    # Opcional: ainda expor a metrica sem tears para auditoria
    print(f"Indeterminaveis (excl. tears) ({len(indet):2d}): {indet}")
    
    if tears_reclassificados:
        print(f"\n========= TEARS RECLASSIFICADOS COMO INFERIDAS =========")
        for v, q in tears_reclassificados:
            print(f"Variavel {v} agora inferida pela equacao {q}")
    
    
    print(f"\n========= TEARS UTILIZADOS =========")
    tears_distintos = set()
    for eq, vs in tears.items():
        if isinstance(vs, (list, tuple, set)):
            for v in vs:
                if v not in inf:
                    print(f"Equacao {eq}: variavel {v}")
                    tears_distintos.add(v)
        else:
            if vs not in inf:
                print(f"Equacao {eq}: variavel {vs}")
                tears_distintos.add(vs)
    print(f"Total de tears distintos: {len(tears_distintos)}")

    
    print(f"\n========= ESTATISTICAS =========")
    print(f"Total de variaveis: {total_vars}")
    print(f"Variaveis observaveis: {observaveis} de {total_vars}")
    print(f"Observabilidade: {observaveis/total_vars*100:.1f}%")

    # Indetermináveis efetivos = indetermináveis "puros" + tears não resolvidos
    strict_pairs = [(eq, v) for (eq, v) in tears_list if v not in inf]
    strict_vars  = sorted({v for _, v in strict_pairs})
    indet_com_tears = sorted(set(indet) | set(strict_vars))
    print(f"Indeterminaveis (incl. tears): {len(indet_com_tears)}")

    # (opcional) checagem de consistência
    print(f"Checagem: {observaveis} + {len(indet_com_tears)} = {observaveis + len(indet_com_tears)} (total={total_vars})")

    print(f"Equacoes utilizadas para inferencia: {len(eqs_utilizadas)}")
    print(f"Variaveis inferidas: {len(inf)}")
    print(f"Grau de sobredeterminacao: {grau_sobredeterminacao}")

    
    # Classificar o sistema baseado no grau de sobredeterminação
    if grau_sobredeterminacao > 0:
        print(f"Tipo do sistema: SOBREDETERMINADO")
    elif grau_sobredeterminacao == 0:
        print(f"Tipo do sistema: EXATAMENTE DETERMINADO")
    else:
        print(f"Tipo do sistema: SUBDETERMINADO")
    
    # Adicionar estratégia detalhada de inferência
    print_estrategia_inferencia(med, inf, indet, tears_reclassificados, inference_eqs, eqs)

def print_estrategia_inferencia(med, inf, indet, tears_reclassificados, inference_eqs, eqs):
    """Imprime a estratégia detalhada de inferência de variáveis"""
    print(f"\n========= ESTRATEGIA DETALHADA DE INFERENCIA =========")
    
    # Criar dicionário com todas as variáveis inferidas e suas equações
    todas_inferidas = {}
    
    # Adicionar variáveis inferidas pelo CP-SAT
    for var, eqs_list in inference_eqs.items():
        if eqs_list:  # Se a variável foi inferida por alguma equação
            todas_inferidas[var] = eqs_list[0]  # Pegar a primeira equação (CP-SAT escolhe uma)
    
    # Adicionar tears reclassificados
    for var, eq in tears_reclassificados:
        todas_inferidas[var] = eq
    
    # Ordenar por variável
    for var in sorted(todas_inferidas.keys()):
        eq = todas_inferidas[var]
        print(f"{var:3s}: inferida via {eq}")
    
    # Mostrar equações não utilizadas
    eqs_utilizadas = set(todas_inferidas.values())
    eqs_nao_utilizadas = set(eqs.keys()) - eqs_utilizadas
    
    if eqs_nao_utilizadas:
        numsort = lambda s: int(''.join(ch for ch in s if ch.isdigit())) if any(ch.isdigit() for ch in s) else 0
        print(f"\nEquacoes nao utilizadas para inferencia: {sorted(eqs_nao_utilizadas, key=numsort)}")
    else:
        print(f"\nTodas as {len(eqs)} equacoes foram utilizadas para inferencia!")
    
    # Mostrar sequência de cálculo
    print(f"\n========= SEQUENCIA DE CALCULO =========")
    print(f"1. Medidas iniciais ({len(med)}): {sorted(med, key=lambda x: x.lower())}")
    
    # Simular sequência de inferência com convergência
    conhecidas = set(med)
    iteracao = 1
    
    while True:
        tamanho_anterior = len(conhecidas)
        novas_inferidas = []
        
        for var, eq in todas_inferidas.items():
            if var not in conhecidas:
                # Verificar se todas as outras variáveis da equação estão disponíveis
                outras_vars = [v for v in eqs[eq] if v != var]
                if all(v in conhecidas for v in outras_vars):
                    novas_inferidas.append((var, eq))
        
        if novas_inferidas:
            print(f"{iteracao + 1}. Iteracao {iteracao}:")
            for var, eq in sorted(novas_inferidas):
                print(f"   {var} via {eq}")
                conhecidas.add(var)
            iteracao += 1
        
        # Verificar convergência: parar quando não há mais crescimento
        if len(conhecidas) == tamanho_anterior:
            break
    
    print(f"\nTotal de iteracoes necessarias: {iteracao}")
    total_vars = len(all_vars_v2(eqs))
    observaveis_sequencial = len(conhecidas)
    observaveis_total = len(med) + len(inf)
    
    if total_vars > 0:
        print(f"Observabilidade sequencial: {observaveis_sequencial}/{total_vars} = {observaveis_sequencial/total_vars*100:.1f}%")
        print(f"Observabilidade total: {observaveis_total}/{total_vars} = {observaveis_total/total_vars*100:.1f}%")
        if observaveis_sequencial != observaveis_total:
            print(f"Nota: Diferença devido a dependencias circulares na sequencia de calculo")
    else:
        print(f"Observabilidade final: {len(conhecidas)}/0 = 0.0%")


# ===== Fallback cycle-safe =====

from collections import deque

def _build_inference_graph(inference_eqs: Dict[str, List[str]], eqs: Dict[str, List[str]]):
    """
    Grafo de dependência G com arestas u->v para cada variável v inferida via eq q:
    para todo u in V_q\{v}, adiciona aresta u->v.
    """
    G = {v: set() for vs in eqs.values() for v in vs}
    for v, qs in inference_eqs.items():
        for q in qs:
            preds = [u for u in eqs[q] if u != v]
            for u in preds:
                G.setdefault(u, set()).add(v)
                G.setdefault(v, set())
    return G

def _has_path(G: Dict[str, Set[str]], src: str, tgt: str) -> bool:
    """BFS: existe caminho src ⇝ tgt em G?"""
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

def measurement_redundancy(
    eqs: Dict[str, List[str]],
    med: List[str],
    inf: List[str],
    inference_eqs: Dict[str, List[str]],
    tears_dict: Dict[str, List[str]] | Dict[str, str] | None = None,
    *,
    ignore_eqs_with_tears: bool = True,
    ensure_cycle_safe: bool = True,
):
    """
    Redundancia estrutural de MEDICOES (sensor-level, cycle-safe):
      Uma medicao v e redundante se existe ao menos uma equacao q NAO usada
      tal que v esta em q, todas as outras variaveis de q ja sao conhecidas (med ∪ inf),
      e, se ensure_cycle_safe=True, q nao fecha ciclo (nao ha caminho v->u no grafo final).
      Opcionalmente ignoramos equacoes com tears ativos.
    """
    used_eqs = {q for qs in inference_eqs.values() for q in qs}
    eqs_with_tears = set(tears_dict.keys()) if tears_dict else set()
    known = set(med) | set(inf)
    G = _build_inference_graph(inference_eqs, eqs) if ensure_cycle_safe else None

    def _cycle_ok(v: str, q: str) -> bool:
        if not ensure_cycle_safe:
            return True
        preds = [u for u in eqs[q] if u != v]
        return not any(_has_path(G, v, u) for u in preds)

    redund_map: Dict[str, List[str]] = {}
    for v in sorted(med):
        witnesses: List[str] = []
        for q, vs in eqs.items():
            if q in used_eqs:
                continue
            if v not in vs:
                continue
            if ignore_eqs_with_tears and (q in eqs_with_tears):
                continue
            if not all((u in known) for u in vs if u != v):
                continue
            if not _cycle_ok(v, q):
                continue
            witnesses.append(q)
        if witnesses:
            redund_map[v] = sorted(witnesses)

    non_redund = sorted([v for v in med if v not in redund_map])
    return redund_map, non_redund


def variable_redundancy(
    eqs: Dict[str, List[str]],
    med: List[str],
    inf: List[str],
    inference_eqs: Dict[str, List[str]],
    tears_dict: Dict[str, List[str]] | Dict[str, str] | None = None,
    *,
    ignore_eqs_with_tears: bool = True,
    ensure_cycle_safe: bool = True,
):
    """
    Redundancia de VARIAVEIS observaveis (equation-level, cycle-safe):
      Uma variavel v observavel (tipicamente em 'inf') e redundante se, alem da(s)
      equacao(oes) ja usada(s) para inferi-la, existe pelo menos UMA equacao NAO usada
      que tambem a determine com as demais variaveis ja conhecidas (med ∪ inf),
      e que nao feche ciclo (se ensure_cycle_safe=True). Util para robustez.
    """
    used_eqs = {q for qs in inference_eqs.values() for q in qs}
    eqs_with_tears = set(tears_dict.keys()) if tears_dict else set()
    known = set(med) | set(inf)
    G = _build_inference_graph(inference_eqs, eqs) if ensure_cycle_safe else None

    def _cycle_ok(v: str, q: str) -> bool:
        if not ensure_cycle_safe:
            return True
        preds = [u for u in eqs[q] if u != v]
        return not any(_has_path(G, v, u) for u in preds)

    redund_map: Dict[str, List[str]] = {}
    for v in sorted(inf):
        witnesses: List[str] = []
        for q, vs in eqs.items():
            if q in used_eqs:
                continue
            if v not in vs:
                continue
            if ignore_eqs_with_tears and (q in eqs_with_tears):
                continue
            if not all((u in known) for u in vs if u != v):
                continue
            if not _cycle_ok(v, q):
                continue
            witnesses.append(q)
        if witnesses:
            redund_map[v] = sorted(witnesses)

    non_redund = sorted([v for v in inf if v not in redund_map])
    return redund_map, non_redund


def print_redundancy_report(
    eqs: Dict[str, List[str]],
    med: List[str],
    inf: List[str],
    inference_eqs: Dict[str, List[str]],
    tears_dict: Dict[str, List[str]] | Dict[str, str] | None = None,
    *,
    ignore_eqs_with_tears: bool = True,
):
    """Relatorio compacto de redundancia estrutural (medicoes) e redundancia por variavel."""
    red_meas, essential_meas = measurement_redundancy(
        eqs, med, inf, inference_eqs, tears_dict,
        ignore_eqs_with_tears=ignore_eqs_with_tears, ensure_cycle_safe=True
    )
    red_vars, nonredund_inf = variable_redundancy(
        eqs, med, inf, inference_eqs, tears_dict,
        ignore_eqs_with_tears=ignore_eqs_with_tears, ensure_cycle_safe=True
    )

    print("\n========= REDUNDANCIA (cycle-safe) =========")
    print(f"Sensores redundantes ({len(red_meas)}):")
    for v in sorted(red_meas.keys()):
        print(f"  {v}: pode ser reconstruido via {red_meas[v]}")
    print(f"Sensores essenciais ({len(essential_meas)}): {essential_meas}")

    print(f"\nVariaveis redundantes por multiplas equacoes ({len(red_vars)}):")
    for v in sorted(red_vars.keys()):
        print(f"  {v}: alternativas {red_vars[v]}")
    print(f"Variaveis observaveis nao redundantes ({len(nonredund_inf)}): {nonredund_inf}")

def _sanity_check(med: List[str], inf: List[str], indet: List[str], tears_dict, eqs) -> None:
    allvars = set().union(*eqs.values())
    nonmeasured = allvars - set(med)
    tears_set_all = {
        v
        for vs in tears_dict.values()
        for v in (vs if isinstance(vs, (list, tuple, set)) else [vs])
    }
    # tears estritos (nao-inferidos)
    tears_noninf = tears_set_all - set(inf)
    # cobertura: todo nao-medido deve cair em INF, TEAR-NAO-INF, ou INDET
    leftover = nonmeasured - set(inf) - tears_noninf - set(indet)
    assert not leftover, f"Faltando classificar: {sorted(leftover)}"
    obs = len(med) + len(inf)
    assert 0 <= obs <= len(allvars)

def fallback_cycle_safe_promotions(
    med: List[str],
    inf: List[str],
    indet: List[str],
    tears_dict: Dict[str, str],          # mapeia eq -> v (tear ativo)
    inference_eqs: Dict[str, List[str]], # mapeia v -> [eqs que inferem v]
    eqs: Dict[str, List[str]],
    demote_leftovers: bool = True,        # rebaixar tears sem caminho para INDET
    repair_tears_slack: int | None = 0
):
    """
    Tenta promover tears a inferidas usando APENAS equacoes NAO usadas e sem fechar ciclo:
      (i) V_q\{v} subsete de conhecidas (med ∪ inf),
      (ii) adicionar arestas u->v (u in V_q\{v}) nao cria ciclo (checa v -> u em G).
    Itera ate nao haver mais promocao possivel.
    """
    # conjuntos auxiliares
    used_eqs = {q for qs in inference_eqs.values() for q in qs}
    known    = set(med) | set(inf)

    # conjunto de tears (variaveis) a partir do dicionario eq->[v1, v2, ...]
    tears_set = {
        v
        for vs in tears_dict.values()
        for v in (vs if isinstance(vs, (list, tuple, set)) else [vs])
    }

    # grafo atual de inferencias
    G = _build_inference_graph(inference_eqs, eqs)

    promoted = []
    changed  = True
    while changed:
        changed = False
        # percorre uma copia para poder remover do set durante loop
        for v in list(tears_set):
            # eqs candidatas: nao utilizadas e que contem v
            cand_qs = [q for q, vs in eqs.items() if (q not in used_eqs) and (v in vs)]
            for q in cand_qs:
                preds = [u for u in eqs[q] if u != v]
                # (i) todos predecessores ja conhecidos (sem usar tears)
                if not all(u in known for u in preds):
                    continue
                # (ii) checagem de ciclo: se ja existe caminho v -> u, entao adicionar u->v fecharia ciclo
                closes_cycle = any(_has_path(G, v, u) for u in preds)
                if closes_cycle:
                    continue

                # --- PROMOCAO SEGURA ---
                inf.append(v)
                known.add(v)
                inference_eqs.setdefault(v, []).append(q)
                used_eqs.add(q)
                tears_set.remove(v)
                promoted.append((v, q))

                # atualiza o grafo: adiciona arestas u->v
                for u in preds:
                    G.setdefault(u, set()).add(v)
                    G.setdefault(v, set())
                changed = True
                break  # passa para o proximo tear

    # Nao rebaixar tears que sao predecessores de inferencias ativas
    required_tears = set()
    for w, qs in inference_eqs.items():
        for q in qs:
            preds_q = [u for u in eqs[q] if u != w]
            for u in preds_q:
                if u in tears_set:
                    required_tears.add(u)

    # opcional: rebaixar os tears remanescentes a indeterminaveis quando
    # NAO existir NENHUMA eq nao usada com todos os predecessores ja conhecidos
    demoted = []
    if demote_leftovers:
        for v in list(tears_set):
            # protege tears necessarios como predecessores de inferencias escolhidas
            if v in required_tears:
                continue
            candid = [
                q for q, vs in eqs.items()
                if (q not in used_eqs) and (v in vs) and all((u in known) for u in vs if u != v)
            ]
            if not candid:
                indet.append(v)
                tears_set.remove(v)
                demoted.append(v)

    # sincroniza o dicionario de tears com o set remanescente (filtra listas por v in tears_set)
    for eq in list(tears_dict.keys()):
        vs = tears_dict[eq]
        if not isinstance(vs, list):
            vs = [vs]
        vs = [v for v in vs if v in tears_set]
        if vs:
            tears_dict[eq] = vs
        else:
            del tears_dict[eq]

    # Validador: remove inferencias inconsistentes (gates violados apos rebaixamentos)
    def _validate_and_prune(med_l: List[str], inf_l: List[str], inference_eqs_l: Dict[str, List[str]], tears_d: Dict[str, List[str]], eqs_l: Dict[str, List[str]]):
        known_l = set(med_l) | set(inf_l)
        tears_by_eq = {eq: (vs if isinstance(vs, list) else [vs]) for eq, vs in tears_d.items()}
        stable = False
        while not stable:
            stable = True
            valid_inf = []
            new_inf_eqs: Dict[str, List[str]] = {}
            for v in inf_l:
                qs = inference_eqs_l.get(v, [])
                ok_eqs: List[str] = []
                for q in qs:
                    preds = [u for u in eqs_l[q] if u != v]
                    if all((u in known_l) or (u in tears_by_eq.get(q, [])) for u in preds):
                        ok_eqs.append(q)
                if ok_eqs:
                    valid_inf.append(v)
                    new_inf_eqs[v] = ok_eqs
                else:
                    # remove v e repete
                    stable = False
            if not stable:
                # atualiza listas e conjuntos
                inf_l = sorted(valid_inf)
                inference_eqs_l = new_inf_eqs
                known_l = set(med_l) | set(inf_l)
        return inf_l, inference_eqs_l

    inf, inference_eqs = _validate_and_prune(med, inf, inference_eqs, tears_dict, eqs)

    # Classificar variáveis restantes como indetermináveis
    all_vars_set = set().union(*eqs.values())
    known = set(med) | set(inf)
    tears_set = {v for vs in tears_dict.values() for v in (vs if isinstance(vs, list) else [vs])}
    
    for v in all_vars_set:
        if v not in known and v not in tears_set and v not in indet:
            indet.append(v)

    inf.sort(); indet.sort()
    # lista de pares (eq, v) apos promo/rebaixamento (util para impressao/relatorio)
    tears_list_pairs = [(eq, v) for eq, vs in tears_dict.items()
                        for v in (vs if isinstance(vs, list) else [vs])]
    return med, inf, indet, tears_list_pairs, promoted, demoted, inference_eqs, tears_dict


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
    filename: str = None,
    inference_method: Dict[str, str] | None = None
):
    """
    Salva os resultados da classificação em um arquivo Excel com múltiplas abas.
    
    Args:
        med: Lista de variáveis medidas
        inf: Lista de variáveis inferidas  
        indet: Lista de variáveis indetermináveis
        tears_list: Lista de pares (equação, variável) para tears
        tears_reclassificados: Lista de tears reclassificados
        tears_dict: Dicionário de tears por equação
        inference_eqs: Dicionário de equações de inferência por variável
        eqs: Dicionário de equações do sistema
        measured_vars: Conjunto de variáveis medidas original
        filename: Nome do arquivo (opcional)
        inference_method: Dicionário de métodos de inferência por variável (opcional)
    
    Abas criadas no Excel:
        1. Resumo: Estatísticas gerais do sistema (total vars, obs%, tipo sistema, etc.)
        2. Variáveis_Medidas: Lista detalhada de todas as variáveis medidas
        3. Variáveis_Inferidas: Variáveis inferidas com suas equações e dependências
        4. Tears_Variáveis_Corte: Tears utilizados e reclassificados com status
        5. Indetermináveis: Variáveis que não podem ser determinadas
        6. Estratégia_Inferência: Sequência e método de inferência por variável
        7. Equações_Sistema: Todas as equações com status de utilização
        8. Todas_Variáveis: Classificação completa de todas as variáveis do sistema
    
    Returns:
        str: Caminho completo do arquivo Excel criado
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resultados_classificacao_{timestamp}.xlsx"
    
    # Caminho do arquivo na mesma pasta do código
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    # Calcular estatísticas
    total_vars = len(set().union(*eqs.values()))
    observaveis = len(med) + len(inf)
    observabilidade_pct = (observaveis / total_vars * 100) if total_vars > 0 else 0
    
    # Tears estritos (não inferidos)
    strict_pairs = [(eq, v) for (eq, v) in tears_list if v not in inf]
    strict_vars = sorted({v for _, v in strict_pairs})
    indet_com_tears = sorted(set(indet) | set(strict_vars))
    
    # Equações utilizadas
    eqs_utilizadas = set()
    for v, eqs_list in inference_eqs.items():
        eqs_utilizadas.update(eqs_list)
    for v, eq in tears_reclassificados:
        eqs_utilizadas.add(eq)
    
    grau_sobredeterminacao = len(eqs_utilizadas) - len(inf)
    
    # Criar DataFrames para cada aba
    method_map = inference_method if inference_method is not None else _LAST_INFERENCE_METHOD
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # Aba 1: Resumo Geral
        resumo_data = {
            'Métrica': [
                'Total de Variáveis',
                'Total de Equações', 
                'Variáveis Medidas',
                'Variáveis Inferidas',
                'Variáveis de Corte (Tears)',
                'Indetermináveis (excl. tears)',
                'Indetermináveis (incl. tears)',
                'Variáveis Observáveis',
                'Observabilidade (%)',
                'Equações Utilizadas',
                'Grau de Sobredeterminação',
                'Tipo do Sistema'
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
                'SOBREDETERMINADO' if grau_sobredeterminacao > 0 else 
                'EXATAMENTE DETERMINADO' if grau_sobredeterminacao == 0 else 
                'SUBDETERMINADO'
            ]
        }
        df_resumo = pd.DataFrame(resumo_data)
        df_resumo.to_excel(writer, sheet_name='Resumo', index=False)
        
        # Aba 2: Variáveis Medidas
        df_medidas = pd.DataFrame({
            'Variável': sorted(med),
            'Tipo': ['Medida'] * len(med),
            'Observação': ['Sensor instalado'] * len(med)
        })
        df_medidas.to_excel(writer, sheet_name='Variáveis_Medidas', index=False)
        
        # Aba 3: Variáveis Inferidas (inclui Metodo)
        inf_data = []
        for v in sorted(inf):
            eqs_list = inference_eqs.get(v, [])
            eq_usada = eqs_list[0] if eqs_list else 'N/A'
            inf_data.append({
                'Variável': v,
                'Tipo': 'Inferida',
                'Equação_Utilizada': eq_usada,
                'Variáveis_Necessárias': ', '.join(sorted([u for u in eqs.get(eq_usada, []) if u != v])) if eq_usada != 'N/A' else '',
                'Metodo': method_map.get(v, ''),
            })
        df_inferidas = pd.DataFrame(inf_data)
        df_inferidas.to_excel(writer, sheet_name='Variáveis_Inferidas', index=False)
        
        # Aba 4: Tears (Variáveis de Corte)
        tears_data = []
        for eq, v in strict_pairs:
            tears_data.append({
                'Variável': v,
                'Equação': eq,
                'Tipo': 'Tear (Variável de Corte)',
                'Status': 'Não Resolvido',
                'Outras_Variáveis_Eq': ', '.join(sorted([u for u in eqs.get(eq, []) if u != v]))
            })
        
        # Adicionar tears reclassificados
        for v, eq in tears_reclassificados:
            tears_data.append({
                'Variável': v,
                'Equação': eq,
                'Tipo': 'Tear Reclassificado',
                'Status': 'Promovido para Inferida',
                'Outras_Variáveis_Eq': ', '.join(sorted([u for u in eqs.get(eq, []) if u != v]))
            })
        
        df_tears = pd.DataFrame(tears_data)
        df_tears.to_excel(writer, sheet_name='Tears_Variáveis_Corte', index=False)
        
        # Aba 5: Indetermináveis
        indet_data = []
        for v in sorted(indet):
            # Encontrar equações que contêm a variável
            eqs_contem = [q for q, vars_eq in eqs.items() if v in vars_eq]
            indet_data.append({
                'Variável': v,
                'Tipo': 'Indeterminável',
                'Equações_que_Contêm': ', '.join(sorted(eqs_contem)),
                'Razão': 'Não pode ser determinada com instrumentação atual'
            })
        df_indet = pd.DataFrame(indet_data)
        df_indet.to_excel(writer, sheet_name='Indetermináveis', index=False)
        
        # Aba 6: Estratégia de Inferência (inclui Metodo)
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
                'Variável_Inferida': var,
                'Equação_Utilizada': eq,
                'Variáveis_Necessárias': ', '.join(outras_vars),
                'Ordem_Inferência': 'Determinada por dependências',
                'Metodo': method_map.get(var, ''),
            })
        df_estrategia = pd.DataFrame(estrategia_data)
        df_estrategia.to_excel(writer, sheet_name='Estratégia_Inferência', index=False)
        
        # Aba 7: Equações do Sistema
        eqs_data = []
        for eq, vars_eq in sorted(eqs.items()):
            utilizada = eq in eqs_utilizadas
            tipo_uso = 'Inferência' if utilizada else 'Não Utilizada'
            
            # Verificar se tem tear
            tem_tear = eq in tears_dict
            if tem_tear:
                tears_eq = tears_dict[eq] if isinstance(tears_dict[eq], list) else [tears_dict[eq]]
                tipo_uso += f' (Tears: {", ".join(tears_eq)})'
            
            eqs_data.append({
                'Equação': eq,
                'Variáveis': ', '.join(sorted(vars_eq)),
                'Número_Variáveis': len(vars_eq),
                'Utilizada': 'Sim' if utilizada else 'Não',
                'Tipo_Uso': tipo_uso,
                'Tem_Tears': 'Sim' if tem_tear else 'Não'
            })
        df_eqs = pd.DataFrame(eqs_data)
        df_eqs.to_excel(writer, sheet_name='Equações_Sistema', index=False)
        
        # Aba 8: Todas as Variáveis (classificação completa)
        todas_vars = sorted(set().union(*eqs.values()))
        todas_vars_data = []
        
        for v in todas_vars:
            if v in med:
                tipo = 'Medida'
                detalhes = 'Sensor instalado'
            elif v in inf:
                eq_inf = inference_eqs.get(v, ['N/A'])[0]
                tipo = 'Inferida'
                detalhes = f'Via equação {eq_inf}'
            elif v in strict_vars:
                # Encontrar a equação do tear
                eq_tear = next((eq for eq, var in strict_pairs if var == v), 'N/A')
                tipo = 'Tear (Variável de Corte)'
                detalhes = f'Corte na equação {eq_tear}'
            elif v in indet:
                tipo = 'Indeterminável'
                detalhes = 'Não determinável com instrumentação atual'
            else:
                tipo = 'Indefinido'
                detalhes = 'Status não classificado'
            
            # Encontrar equações que contêm a variável
            eqs_contem = [q for q, vars_eq in eqs.items() if v in vars_eq]
            
            todas_vars_data.append({
                'Variável': v,
                'Classificação': tipo,
                'Detalhes': detalhes,
                'Equações_que_Contêm': ', '.join(sorted(eqs_contem)),
                'Número_Equações': len(eqs_contem),
                'Metodo': (_LAST_INFERENCE_METHOD.get(v, '') if v in inf else '')
            })
        
        df_todas_vars = pd.DataFrame(todas_vars_data)
        df_todas_vars.to_excel(writer, sheet_name='Todas_Variáveis', index=False)
    
    print(f"\ RESULTADOS SALVOS EM EXCEL:")
    print(f" Arquivo: {filepath}")
    print(f" {len(pd.ExcelFile(filepath).sheet_names)} abas criadas:")
    print("   - Resumo")
    print("   - Variáveis_Medidas") 
    print("   - Variáveis_Inferidas")
    print("   - Tears_Variáveis_Corte")
    print("   - Indetermináveis")
    print("   - Estratégia_Inferência")
    print("   - Equações_Sistema")
    print("   - Todas_Variáveis")
    
    return filepath


if __name__ == "__main__":
    # measured_vars = {"x1","x3","x5","x6","x7","x9","x11","x13","x14","x15","x16","x18","x19","x20","x26","x27"}
    # measured_vars = {"x1","x3","x5","x6","x7","x9","x11","x13","x14","x15","x16","x18","x19","x20","x22","x26","x27"}
    # measured_vars = {"x1","x2","x3","x5","x6","x7","x8","x9","x10","x11","x13","x14","x15","x16","x18","x19","x20","x26","x27"}
    # measured_vars = {"x5","x6","x7","x11","x13","x14","x15","x16","x18","x19","x20","x26","x27"}
    
    # TOY MODEL - Cenário 1: Demonstrar ganho real das Fases 2 e 3
    # Medindo x4 e x5, deixando x6 como tear que pode ser promovido
    # print("=== TOY MODEL SIMPLES - DEMONSTRAÇÃO FASES 2 E 3 ===")
    # print("\nSistema: 4 variáveis, 4 equações")
    # print("Ciclo: x1→x2→x3→x1 (Eq1, Eq2, Eq3)")
    # print("Ponte: x1=x4 (Eq4)")
    # print("\nCenário: Medindo APENAS x4")
    # print("Esperado:")
    # print("- Fase 1: 1 tear no ciclo, não usa Eq4")
    # print("- Fase 2: identifica que pode promover via Eq4")
    # print("- Fase 3: materializa promoção, quebra ciclo → 100% observabilidade\n")
    
    # measured_vars = {"x4"}
    
    # Sistema original - conjunto de medidas
    measured_vars  = {
        "x1","x2","x7","x8","x9","x12",
        "x13","x14","x15","x16","x17","x18","x19","x20",
        "x21","x22","x23","x25","x26","x27","x28","x29","x30",
        "x31","x32","x33","x37","x46","x53"
    }
    run_w_align = 1
    align_tag = "align_on" if run_w_align != 0 else "align_off"

    
    print("========================================")
    print("CLASSIFICACAO DE VARIAVEIS - SISTEMA V2")
    print("========================================")
    eqs = equations_v2()
    all_vars = all_vars_v2(eqs)
    print(f"Sistema: {len(eqs)} equacoes, {len(all_vars)} variaveis")
    print(f"Variaveis medidas: {len(measured_vars)}")
    print(f"Variaveis medidas: {sorted(measured_vars)}")
    
    # TESTE: Forcar exclusao de Eq4 na Fase 1
    # force_exclusion = {"exclude_phase1": {"Eq4"}}
    force_exclusion = None  # Desativado - rodando normalmente
    
    print("\n=== TESTE SEM TEARS (comparacao com Sanchez & Romagnoli) ===")
    try:
        med_no_tears, inf_no_tears, indet_no_tears, tears_list_no_tears, _, _, inference_eqs_no_tears, tears_no_tears = classify_v2(
            measured_vars, 
            W_obs=100, 
            W_tears=1, 
            W_promo=10,
            W_align=run_w_align,
            T_max=0,  # SEM TEARS
            force_heads=force_exclusion
        )
        print(f"SEM TEARS - Observaveis: {len(med_no_tears) + len(inf_no_tears)}/63 = {(len(med_no_tears) + len(inf_no_tears))/63*100:.1f}%")
        print(f"SEM TEARS - Inferidas: {sorted(inf_no_tears)}")
        print(f"SEM TEARS - Indeterminaveis: {len(indet_no_tears)} variaveis")
    except Exception as e:
        print(f"ERRO no teste sem tears: {e}")
    
    print("\n=== RESULTADO PRINCIPAL (com tears) ===")
    try:
        med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs, tears = classify_v2(
            measured_vars, 
            W_obs=100, 
            W_tears=1, 
            W_promo=10,
            W_align=run_w_align,
            force_heads=force_exclusion
        )
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs, tears = [], [], [], [], [], [], {}, {}
    
    # Tenta melhorar com fallback cycle-safe apenas se nao atingiu 100%
    total_vars = len(all_vars)
    obs_total  = len(med) + len(inf)

    if obs_total < total_vars:
        print("\n[Fallback] Observabilidade nao e maxima; executando promocao cycle-safe...")
        med, inf, indet, tears_list, tears_reclassificados, tears_rebaixados, inference_eqs, tears = (
            fallback_cycle_safe_promotions(
                med, inf, indet, tears, inference_eqs, eqs,
                demote_leftovers=True
            )
        )
    else:
        if not tears_reclassificados:
            tears_reclassificados, tears_rebaixados = [], []
    
    print_resultado_v2(med, inf, indet, tears_list, tears_reclassificados, tears, inference_eqs, eqs)

    # === SALVAR RESULTADOS EM EXCEL ===
    try:
        # Nome personalizado baseado no sistema e timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_pct = (len(med) + len(inf)) / len(all_vars) * 100 if len(all_vars) > 0 else 0
        sistema_info = f"Sistema_Sanchez_Romagnoli_{len(eqs)}eq_{len(all_vars)}vars_{obs_pct:.1f}pct_{align_tag}"
        nome_arquivo = f"resultados_{sistema_info}_{timestamp}.xlsx"
        
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(script_dir), "resultados_classificacao")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, nome_arquivo)
        
        excel_path = save_results_to_excel(
            med=med, 
            inf=inf, 
            indet=indet, 
            tears_list=tears_list, 
            tears_reclassificados=tears_reclassificados,
            tears_dict=tears, 
            inference_eqs=inference_eqs, 
            eqs=eqs,
            measured_vars=measured_vars,
            filename=nome_arquivo,
            inference_method=_LAST_INFERENCE_METHOD
        )
        print(f" Arquivo Excel salvo com sucesso!")
    except Exception as e:
        print(f" Erro ao salvar arquivo Excel: {e}")
        print("   Certifique-se de que o pandas e openpyxl estão instalados:")
        print("   pip install pandas openpyxl")

        # === NOVO: relatorio de redundancia ===
        # Por padrao, ignora equacoes com tears ativos como testemunhas de redundancia.
        print_redundancy_report(
            eqs=eqs,
            med=med,
            inf=inf,
            inference_eqs=inference_eqs,
            tears_dict=tears,
            ignore_eqs_with_tears=True
        )

