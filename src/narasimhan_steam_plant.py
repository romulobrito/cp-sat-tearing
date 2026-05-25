#!/usr/bin/env python3
"""
Classificador CP-SAT enxuto para o sistema de Narasimhan.

Objetivo preservado:
    construir um esquema de cálculo com seleção de saídas e tears,
    maximizando a cobertura de variáveis calculáveis e identificando
    explicitamente as malhas fechadas por tearing.

Mudança conceitual importante:
    - observabilidade direta: calculável sem depender de tear;
    - cobertura solucionável: calculável executando o esquema após assumir
      valores iniciais para os tears;
    - tear fechado: variável usada como corte e também calculada por outra
      equação, caracterizando uma malha que exige fechamento/iteração.

Dependência: pip install ortools
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

Equations = Dict[str, List[str]]
Pair = Tuple[str, str]


def equations_narasimhan() -> Equations:
    """Sistema de Narasimhan: 12 equações e 28 variáveis."""
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


def all_variables(eqs: Equations) -> List[str]:
    return sorted({v for values in eqs.values() for v in values})


@dataclass(frozen=True)
class ClassificationResult:
    measured: List[str]
    inferred_direct: List[str]
    inferred_conditioned: List[str]
    indeterminate: List[str]
    tear_pairs: List[Pair]
    tears_closed: List[str]
    tears_open: List[str]
    output_equation: Dict[str, str]
    levels: Dict[str, int]
    status_by_phase: Dict[str, str]
    execution_graph_is_dag: bool
    full_graph_cycles: List[List[str]]

    @property
    def coverage_variables(self) -> Set[str]:
        return set(self.measured) | set(self.inferred_direct) | set(self.inferred_conditioned)

    @property
    def direct_variables(self) -> Set[str]:
        return set(self.measured) | set(self.inferred_direct)

    @property
    def n_coverage(self) -> int:
        return len(self.coverage_variables)

    @property
    def n_direct(self) -> int:
        return len(self.direct_variables)


def _new_solver(time_limit_s: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.randomize_search = False
    solver.parameters.log_search_progress = False
    return solver


def _solve_optimal_or_raise(
    solver: cp_model.CpSolver,
    model: cp_model.CpModel,
    phase_name: str,
) -> int:
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"{phase_name}: o solver retornou {solver.StatusName(status)}. "
            "Para reportar resultados científicos, aumente time_limit_s até obter OPTIMAL."
        )
    return status


def _build_graph(
    eqs: Equations,
    output_equation: Dict[str, str],
    tear_pairs: Set[Pair],
    *,
    remove_cut_arcs: bool,
) -> Dict[str, Set[str]]:
    graph = {v: set() for v in all_variables(eqs)}
    for output, q in output_equation.items():
        for predecessor in eqs[q]:
            if predecessor == output:
                continue
            if remove_cut_arcs and (q, predecessor) in tear_pairs:
                continue
            graph[predecessor].add(output)
    return graph


def _is_dag(graph: Dict[str, Set[str]]) -> bool:
    indegree = {v: 0 for v in graph}
    for source in graph:
        for target in graph[source]:
            indegree[target] += 1
    queue = [v for v, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in graph[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(graph)


def _cyclic_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Retorna componentes fortemente conexas que caracterizam ciclos."""
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    components: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph[node]:
            if target not in indices:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])

        if lowlink[node] == indices[node]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or (len(component) == 1 and component[0] in graph[component[0]]):
                components.append(sorted(component))

    for node in graph:
        if node not in indices:
            visit(node)
    return sorted(components)


def classify_tearing(
    measured: Set[str],
    *,
    eqs: Optional[Equations] = None,
    time_limit_s: float = 30.0,
    max_tears: Optional[int] = None,
) -> ClassificationResult:
    """
    Resolve o esquema de tearing por três critérios lexicográficos:
      1. maximiza a cobertura computável após os cortes;
      2. minimiza o número de tears necessários;
      3. maximiza tears fechados por equações de atualização.

    Nesta versão não existe variável de ``promoção`` independente. Um tear é
    considerado fechado quando a mesma variável, usada como corte em uma
    equação, é saída escolhida de outra equação. Essa definição elimina a
    inconsistência da implementação anterior e mantém o papel operacional.
    """
    eqs = equations_narasimhan() if eqs is None else eqs
    variables = all_variables(eqs)
    invalid = measured - set(variables)
    if invalid:
        raise ValueError(f"Variáveis medidas inválidas: {sorted(invalid)}")

    uses = {v: [q for q, values in eqs.items() if v in values] for v in variables}
    model = cp_model.CpModel()

    output = {(q, v): model.NewBoolVar(f"output_{q}_{v}") for q, values in eqs.items() for v in values}
    tear = {(q, v): model.NewBoolVar(f"tear_{q}_{v}") for q, values in eqs.items() for v in values}
    available = {v: model.NewBoolVar(f"available_{v}") for v in variables}
    closed = {v: model.NewBoolVar(f"closed_tear_{v}") for v in variables}
    level = {v: model.NewIntVar(0, len(variables), f"level_{v}") for v in variables}

    for v in variables:
        output_sum = sum(output[q, v] for q in uses[v])
        tear_sum = sum(tear[q, v] for q in uses[v])
        model.Add(output_sum <= 1)
        model.Add(tear_sum <= 1)

        if v in measured:
            model.Add(available[v] == 1)
            model.Add(output_sum == 0)
            model.Add(tear_sum == 0)
            model.Add(closed[v] == 0)
        else:
            model.Add(available[v] == output_sum)
            model.Add(closed[v] <= output_sum)
            model.Add(closed[v] <= tear_sum)
            model.Add(closed[v] >= output_sum + tear_sum - 1)

    for q, values in eqs.items():
        model.Add(sum(output[q, v] for v in values) <= 1)
        for v in values:
            # Uma variável não pode ser simultaneamente entrada cortada e saída da mesma equação.
            model.Add(output[q, v] + tear[q, v] <= 1)
            # Um tear local só existe se a equação efetivamente calcular outra variável.
            other_outputs = [output[q, w] for w in values if w != v]
            model.Add(tear[q, v] <= sum(other_outputs))

            for u in values:
                if u == v:
                    continue
                # Para calcular v, u deve já estar calculável ou ser o corte local da equação.
                model.Add(output[q, v] <= available[u] + tear[q, u])
                # A ordem de execução ignora apenas o arco explicitamente cortado pelo tear.
                model.Add(level[v] >= level[u] + 1).OnlyEnforceIf(
                    [output[q, v], tear[q, u].Not()]
                )

    if max_tears is not None:
        model.Add(sum(tear.values()) <= max_tears)

    solver = _new_solver(time_limit_s)
    statuses: Dict[str, str] = {}

    coverage_expr = sum(available.values())
    tear_expr = sum(tear.values())
    closed_expr = sum(closed.values())

    model.Maximize(coverage_expr)
    status = _solve_optimal_or_raise(solver, model, "Fase 1 - cobertura")
    statuses["coverage"] = solver.StatusName(status)
    best_coverage = int(round(solver.ObjectiveValue()))
    model.Add(coverage_expr == best_coverage)

    model.Minimize(tear_expr)
    status = _solve_optimal_or_raise(solver, model, "Fase 2 - tears mínimos")
    statuses["tears"] = solver.StatusName(status)
    best_tears = int(round(solver.ObjectiveValue()))
    model.Add(tear_expr == best_tears)

    model.Maximize(closed_expr)
    status = _solve_optimal_or_raise(solver, model, "Fase 3 - fechamento de tears")
    statuses["closure"] = solver.StatusName(status)

    output_equation: Dict[str, str] = {}
    for (q, v), variable in output.items():
        if solver.Value(variable):
            output_equation[v] = q

    tear_pairs = sorted((q, v) for (q, v), variable in tear.items() if solver.Value(variable))
    tear_pair_set = set(tear_pairs)
    tears_closed = sorted(v for v in variables if solver.Value(closed[v]))
    tears_active = {v for _, v in tear_pairs}
    tears_open = sorted(tears_active - set(tears_closed))
    levels = {v: solver.Value(level[v]) for v in variables}

    # Identifica aquilo que é inferível sem usar qualquer tear, propagando apenas a partir das medidas.
    direct_known = set(measured)
    changed = True
    while changed:
        changed = False
        for v, q in output_equation.items():
            if v in direct_known:
                continue
            predecessors = [u for u in eqs[q] if u != v]
            uses_tear = any((q, u) in tear_pair_set for u in predecessors)
            if not uses_tear and all(u in direct_known for u in predecessors):
                direct_known.add(v)
                changed = True

    inferred_direct = sorted(direct_known - measured)
    computed_variables = set(output_equation)
    inferred_conditioned = sorted(computed_variables - set(inferred_direct))
    indeterminate = sorted(set(variables) - set(measured) - computed_variables)

    execution_graph = _build_graph(eqs, output_equation, tear_pair_set, remove_cut_arcs=True)
    full_graph = _build_graph(eqs, output_equation, tear_pair_set, remove_cut_arcs=False)
    execution_is_dag = _is_dag(execution_graph)
    if not execution_is_dag:
        raise AssertionError("Falha interna: o grafo de execução deveria ser acíclico após os cortes.")

    return ClassificationResult(
        measured=sorted(measured),
        inferred_direct=inferred_direct,
        inferred_conditioned=inferred_conditioned,
        indeterminate=indeterminate,
        tear_pairs=tear_pairs,
        tears_closed=tears_closed,
        tears_open=tears_open,
        output_equation=dict(sorted(output_equation.items())),
        levels=levels,
        status_by_phase=statuses,
        execution_graph_is_dag=execution_is_dag,
        full_graph_cycles=_cyclic_components(full_graph),
    )


def print_report(result: ClassificationResult, eqs: Equations) -> None:
    total = len(all_variables(eqs))
    print("\n========= CLASSIFICAÇÃO CP-SAT COM TEARING =========")
    print(f"Status por fase: {result.status_by_phase}")
    print(f"Medidas ({len(result.measured)}): {result.measured}")
    print(f"Inferidas diretamente ({len(result.inferred_direct)}): {result.inferred_direct}")
    print(f"Inferidas condicionadas a tear ({len(result.inferred_conditioned)}): {result.inferred_conditioned}")
    print(f"Indetermináveis ({len(result.indeterminate)}): {result.indeterminate}")
    print(f"Tears ativos ({len(result.tear_pairs)}): {result.tear_pairs}")
    print(f"Tears fechados ({len(result.tears_closed)}): {result.tears_closed}")
    print(f"Tears abertos ({len(result.tears_open)}): {result.tears_open}")
    print("\n--------- MÉTRICAS ---------")
    print(f"Observabilidade direta: {result.n_direct}/{total} = {100 * result.n_direct / total:.1f}%")
    print(f"Cobertura solucionável com tearing: {result.n_coverage}/{total} = {100 * result.n_coverage / total:.1f}%")
    print(f"Grafo de execução após cortes é DAG: {result.execution_graph_is_dag}")
    print(f"Malhas no grafo completo: {result.full_graph_cycles or 'nenhuma'}")
    print("\n--------- SAÍDAS SELECIONADAS ---------")
    for v, q in result.output_equation.items():
        kind = "direta" if v in result.inferred_direct else "condicionada/fechamento"
        print(f"{v:>4} <- {q:<4} ({kind})")


def classify_v2(measured: Set[str], *, eqs: Optional[Equations] = None, time_limit_s: float = 30.0):
    """Wrapper compatível com a forma de retorno dos scripts antigos."""
    result = classify_tearing(measured, eqs=eqs, time_limit_s=time_limit_s)
    inferred = sorted(set(result.inferred_direct) | set(result.inferred_conditioned))
    tears_dict: Dict[str, List[str]] = defaultdict(list)
    for q, v in result.tear_pairs:
        tears_dict[q].append(v)
    inference_eqs = {v: [q] for v, q in result.output_equation.items()}
    return (
        result.measured,
        inferred,
        result.indeterminate,
        result.tear_pairs,
        [],
        [],
        inference_eqs,
        dict(tears_dict),
    )


if __name__ == "__main__":
    measured_variables = {
        "x1", "x3", "x5", "x6", "x7", "x9", "x11", "x12", "x13",
        "x14", "x15", "x16", "x18", "x19", "x20", "x26", "x27",
    }
    equations = equations_narasimhan()
    result = classify_tearing(measured_variables, eqs=equations)
    print_report(result, equations)
