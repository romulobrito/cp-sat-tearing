#!/usr/bin/env python3
"""CP-SAT core for structural classification with closed tearing.

The implemented semantics separate three non-equivalent notions:
1) closed/autonomous coverage (no open-tear dependency),
2) conditional external reach (computable if open tears are externally provided),
3) effective indeterminability (open tears + open-conditioned variables + pure
   non-computable variables).

The model selects equation outputs and local cuts, but does not claim algebraic
uniqueness of closed loops. Algebraic validation must be done separately
(e.g., QR/SVD/Jacobian analysis).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ortools.sat.python import cp_model

Equations = Dict[str, List[str]]
Pair = Tuple[str, str]


@dataclass(frozen=True)
class TearingConfig:
    """Deterministic solver configuration."""

    time_limit_s: float = 60.0
    max_tears: Optional[int] = None
    require_optimal: bool = True
    random_seed: int = 42
    use_structural_preprocessing: bool = True
    use_union_of_maximum_matchings: bool = True


@dataclass(frozen=True)
class TearingResult:
    """Auditable classification result.

    The sets below form a partition of model variables:
      measured, known_constants, inferred_direct, inferred_closed_loop,
      tears_closed, inferred_conditioned_open, tears_closed_external,
      tears_open, indeterminate_pure.
    """

    case_name: str
    total_variables: int
    total_equations: int
    total_incidences: int
    model_equations: int
    model_incidences: int
    structural_rank: int
    preprocessing_enabled: bool
    retained_equations: List[str]
    removed_equations: List[str]
    measured: List[str]
    known_constants: List[str]
    inferred_direct: List[str]
    inferred_closed_loop: List[str]
    tears_closed: List[str]
    inferred_conditioned_open: List[str]
    tears_closed_external: List[str]
    tears_open: List[str]
    indeterminate_pure: List[str]
    effective_indeterminate: List[str]
    tear_pairs: List[Pair]
    output_equation: Dict[str, str]
    output_uses_tears: Dict[str, List[str]]
    levels: Dict[str, int]
    status_by_phase: Dict[str, str]
    objective_by_phase: Dict[str, float]
    time_by_phase_s: Dict[str, float]
    execution_graph_is_dag: bool
    cyclic_sccs_full_graph: List[List[str]]
    cut_graph_edges: List[Pair]
    full_graph_edges: List[Pair]
    externally_conditioned: List[str]

    @property
    def initial_known(self) -> Set[str]:
        return set(self.measured) | set(self.known_constants)

    @property
    def closed_coverage_variables(self) -> Set[str]:
        """Variables covered without open external dependency."""
        return (
            self.initial_known
            | set(self.inferred_direct)
            | set(self.inferred_closed_loop)
            | set(self.tears_closed)
        )

    @property
    def external_reach_variables(self) -> Set[str]:
        """Variables computable if open tears are externally provided."""
        return (
            self.closed_coverage_variables
            | set(self.inferred_conditioned_open)
            | set(self.tears_closed_external)
        )

    @property
    def direct_variables(self) -> Set[str]:
        return self.initial_known | set(self.inferred_direct)

    @property
    def n_closed_coverage(self) -> int:
        return len(self.closed_coverage_variables)

    @property
    def n_external_reach(self) -> int:
        return len(self.external_reach_variables)

    @property
    def n_direct(self) -> int:
        return len(self.direct_variables)

    @property
    def is_autonomously_covered(self) -> bool:
        return self.n_closed_coverage == self.total_variables

    @property
    def is_only_conditionally_reached(self) -> bool:
        return self.n_external_reach == self.total_variables and not self.is_autonomously_covered

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.update(
            {
                "n_closed_coverage": self.n_closed_coverage,
                "n_external_reach": self.n_external_reach,
                "n_direct": self.n_direct,
                "is_autonomously_covered": self.is_autonomously_covered,
                "is_only_conditionally_reached": self.is_only_conditionally_reached,
            }
        )
        return data


def all_variables(eqs: Equations) -> List[str]:
    return sorted({v for values in eqs.values() for v in values})


def preprocess_equations_structural(
    equations: Equations,
    variables: Sequence[str] | None = None,
    *,
    use_union_of_maximum_matchings: bool = True,
) -> Tuple[Equations, List[str], int]:
    """Remove equations outside all maximum matchings.

    The conservative default keeps the union of equations that appear in at
    least one maximum matching. This does not lock a specific matching before
    CP-SAT; it only removes equations that cannot contribute to structural rank.

    Returns:
        ``(equations_reduced, removed_equations, structural_rank)``.
    """

    eq_names = list(equations)
    var_names = list(variables) if variables is not None else all_variables(equations)
    var_index = {v: j for j, v in enumerate(var_names)}
    adj: List[List[int]] = []
    for q in eq_names:
        adj.append([var_index[v] for v in equations[q] if v in var_index])

    def try_match(
        eq_i: int,
        seen: List[bool],
        var_match: List[int],
        *,
        skip_eq: int | None = None,
        blocked_var: int | None = None,
    ) -> bool:
        if skip_eq is not None and eq_i == skip_eq:
            return False
        for var_j in adj[eq_i]:
            if blocked_var is not None and var_j == blocked_var:
                continue
            if seen[var_j]:
                continue
            seen[var_j] = True
            if var_match[var_j] == -1 or try_match(
                var_match[var_j],
                seen,
                var_match,
                skip_eq=skip_eq,
                blocked_var=blocked_var,
            ):
                var_match[var_j] = eq_i
                return True
        return False

    def max_matching_size(
        *,
        skip_eq: int | None = None,
        blocked_var: int | None = None,
    ) -> int:
        var_match = [-1] * len(var_names)
        size = 0
        for eq_i in range(len(eq_names)):
            if skip_eq is not None and eq_i == skip_eq:
                continue
            seen = [False] * len(var_names)
            if try_match(eq_i, seen, var_match, skip_eq=skip_eq, blocked_var=blocked_var):
                size += 1
        return size

    structural_rank = max_matching_size()
    if use_union_of_maximum_matchings:
        retained_indices: Set[int] = set()
        for eq_i, neighbors in enumerate(adj):
            for var_j in neighbors:
                forced_size = 1 + max_matching_size(skip_eq=eq_i, blocked_var=var_j)
                if forced_size == structural_rank:
                    retained_indices.add(eq_i)
                    break
    else:
        var_match = [-1] * len(var_names)
        retained_indices = set()
        for eq_i in range(len(eq_names)):
            seen = [False] * len(var_names)
            if try_match(eq_i, seen, var_match):
                retained_indices.add(eq_i)

    retained = [eq_names[i] for i in range(len(eq_names)) if i in retained_indices]
    removed = [q for q in eq_names if q not in retained]
    reduced = {q: list(equations[q]) for q in retained}
    return reduced, removed, structural_rank


def _new_solver(config: TearingConfig) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_s
    solver.parameters.random_seed = config.random_seed
    solver.parameters.num_search_workers = 1
    solver.parameters.randomize_search = False
    solver.parameters.log_search_progress = False
    return solver


def _solve(solver: cp_model.CpSolver, model: cp_model.CpModel, phase: str, config: TearingConfig) -> int:
    status = solver.Solve(model)
    valid = (cp_model.OPTIMAL,) if config.require_optimal else (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if status not in valid:
        raise RuntimeError(
            f"{phase}: solver retornou {solver.StatusName(status)} "
            f"(require_optimal={config.require_optimal})."
        )
    return status


def _graph_edges(
    eqs: Equations,
    output_equation: Mapping[str, str],
    tear_pairs: Set[Pair],
    *,
    remove_cut_arcs: bool,
) -> List[Pair]:
    edges: Set[Pair] = set()
    for target, q in output_equation.items():
        for source in eqs[q]:
            if source == target:
                continue
            if remove_cut_arcs and (q, source) in tear_pairs:
                continue
            edges.add((source, target))
    return sorted(edges)


def _adjacency(nodes: Iterable[str], edges: Sequence[Pair]) -> Dict[str, Set[str]]:
    graph = {node: set() for node in nodes}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())
    return graph


def _is_dag(nodes: Iterable[str], edges: Sequence[Pair]) -> bool:
    graph = _adjacency(nodes, edges)
    indegree = {node: 0 for node in graph}
    for targets in graph.values():
        for target in targets:
            indegree[target] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in graph[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(graph)


def _cyclic_sccs(nodes: Iterable[str], edges: Sequence[Pair]) -> List[List[str]]:
    """Retorna componentes fortemente conexas ciclicas, nao numero de ciclos."""
    graph = _adjacency(nodes, edges)
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


def _reachable_from_sources(nodes: Iterable[str], edges: Sequence[Pair], sources: Set[str]) -> Set[str]:
    graph = _adjacency(nodes, edges)
    reached = set(sources)
    stack = list(sources)
    while stack:
        node = stack.pop()
        for target in graph.get(node, set()):
            if target not in reached:
                reached.add(target)
                stack.append(target)
    return reached


def _direct_closure(
    eqs: Equations,
    initial_known: Set[str],
    output_equation: Mapping[str, str],
    tear_pairs: Set[Pair],
) -> Set[str]:
    """Propagacao aciclica que nao utiliza qualquer arco local cortado."""
    known = set(initial_known)
    changed = True
    while changed:
        changed = False
        for target, q in output_equation.items():
            if target in known:
                continue
            predecessors = [u for u in eqs[q] if u != target]
            if any((q, u) in tear_pairs for u in predecessors):
                continue
            if all(u in known for u in predecessors):
                known.add(target)
                changed = True
    return known


def classify_tearing(
    eqs: Equations,
    measured: Set[str],
    *,
    known_constants: Optional[Set[str]] = None,
    allowed_outputs: Optional[Mapping[str, Set[str]]] = None,
    case_name: str = "caso",
    config: Optional[TearingConfig] = None,
) -> TearingResult:
    """Solve structural classification with closed tearing and external reach.

    Prioridades lexicograficas:
      1. maximizar cobertura autonoma/fechada;
      2. maximizar alcance calculavel admitindo entradas externas (diagnostico);
      3. minimizar tears abertos para o mesmo alcance;
      4. minimizar numero bruto de cortes;
      5. maximizar saidas localmente diretas.

    ``known_constants`` are available process parameters (not sensors).
    ``allowed_outputs`` can restrict physically inadmissible KPI inversions.
    """
    config = config or TearingConfig()
    original_eqs = {q: list(values) for q, values in eqs.items()}
    original_variables = all_variables(original_eqs)
    if config.use_structural_preprocessing:
        eqs, removed_equations, structural_rank = preprocess_equations_structural(
            original_eqs,
            original_variables,
            use_union_of_maximum_matchings=config.use_union_of_maximum_matchings,
        )
    else:
        eqs = dict(original_eqs)
        removed_equations = []
        _, _, structural_rank = preprocess_equations_structural(
            original_eqs,
            original_variables,
            use_union_of_maximum_matchings=config.use_union_of_maximum_matchings,
        )
    retained_equations = list(eqs)
    constants = set(known_constants or set())
    variables = original_variables
    variable_set = set(variables)
    initial_known = set(measured) | constants
    invalid = initial_known - variable_set
    if invalid:
        raise ValueError(f"Variaveis iniciais invalidas em {case_name}: {sorted(invalid)}")
    overlap = set(measured) & constants
    if overlap:
        raise ValueError(f"Variaveis nao podem ser sensor e parametro conhecido simultaneamente: {sorted(overlap)}")

    uses = {v: [q for q, values in eqs.items() if v in values] for v in variables}
    permitted = {
        q: set(allowed_outputs[q]) if allowed_outputs and q in allowed_outputs else (set(values) - constants)
        for q, values in eqs.items()
    }
    for q, values in eqs.items():
        bad = permitted[q] - set(values)
        if bad:
            raise ValueError(f"Saidas admissiveis invalidas em {q}: {sorted(bad)}")

    model = cp_model.CpModel()
    output = {(q, v): model.NewBoolVar(f"output_{q}_{v}") for q, values in eqs.items() for v in values}
    tear = {(q, v): model.NewBoolVar(f"tear_{q}_{v}") for q, values in eqs.items() for v in values}
    available = {v: model.NewBoolVar(f"available_{v}") for v in variables}
    closed_struct = {v: model.NewBoolVar(f"closed_struct_{v}") for v in variables}
    open_tear = {v: model.NewBoolVar(f"open_tear_{v}") for v in variables}
    external = {v: model.NewBoolVar(f"external_{v}") for v in variables}
    autonomous = {v: model.NewBoolVar(f"autonomous_{v}") for v in variables}
    level = {v: model.NewIntVar(0, len(variables), f"level_{v}") for v in variables}
    direct_pair = {(q, v): model.NewBoolVar(f"direct_{q}_{v}") for q, values in eqs.items() for v in values}

    for v in variables:
        output_sum = sum(output[q, v] for q in uses[v])
        tear_sum = sum(tear[q, v] for q in uses[v])
        model.Add(output_sum <= 1)
        model.Add(tear_sum <= 1)
        if v in initial_known:
            model.Add(available[v] == 1)
            model.Add(output_sum == 0)
            model.Add(tear_sum == 0)
            model.Add(closed_struct[v] == 0)
            model.Add(open_tear[v] == 0)
            model.Add(external[v] == 0)
        else:
            model.Add(available[v] == output_sum)
            # tear fechado estrutural = corte local + equacao de atualizacao
            model.Add(closed_struct[v] <= output_sum)
            model.Add(closed_struct[v] <= tear_sum)
            model.Add(closed_struct[v] >= output_sum + tear_sum - 1)
            # tear aberto = corte sem equacao de atualizacao
            model.Add(open_tear[v] <= tear_sum)
            model.Add(open_tear[v] <= 1 - output_sum)
            model.Add(open_tear[v] >= tear_sum - output_sum)
            model.Add(external[v] >= open_tear[v])
        model.Add(autonomous[v] <= available[v])
        model.Add(autonomous[v] <= 1 - external[v])
        model.Add(autonomous[v] >= available[v] - external[v])

    for q, values in eqs.items():
        model.Add(sum(output[q, v] for v in values) <= 1)
        for v in values:
            if v not in permitted[q]:
                model.Add(output[q, v] == 0)
            model.Add(output[q, v] + tear[q, v] <= 1)
            # Um corte so existe se a equacao calcula uma outra variavel.
            model.Add(tear[q, v] <= sum(output[q, w] for w in values if w != v))
            predecessor_tears = [tear[q, u] for u in values if u != v]
            model.Add(direct_pair[q, v] <= output[q, v])
            for u in values:
                if u == v:
                    continue
                model.Add(output[q, v] <= available[u] + tear[q, u])
                # Ordem apenas no grafo com arestas cortadas removidas.
                model.Add(level[v] >= level[u] + 1).OnlyEnforceIf([output[q, v], tear[q, u].Not()])
                # No grafo completo, dependencia de tear aberto contamina toda saida alcançavel.
                model.Add(external[v] >= external[u] + output[q, v] - 1)
                model.Add(direct_pair[q, v] <= 1 - tear[q, u])
            if predecessor_tears:
                model.Add(direct_pair[q, v] >= output[q, v] - sum(predecessor_tears))
            else:
                model.Add(direct_pair[q, v] == output[q, v])

    if config.max_tears is not None:
        model.Add(sum(tear.values()) <= config.max_tears)

    solver = _new_solver(config)
    statuses: Dict[str, str] = {}
    objectives: Dict[str, float] = {}
    times: Dict[str, float] = {}

    closed_coverage_expr = sum(autonomous.values())
    external_reach_expr = sum(available.values())
    open_expr = sum(open_tear.values())
    tear_expr = sum(tear.values())
    direct_expr = sum(direct_pair.values())

    phases = [
        ("fase_1_max_cobertura_fechada", "max", closed_coverage_expr),
        ("fase_2_max_alcance_condicional", "max", external_reach_expr),
        ("fase_3_min_tears_abertos", "min", open_expr),
        ("fase_4_min_tears_brutos", "min", tear_expr),
        ("fase_5_max_diretas_locais", "max", direct_expr),
    ]
    for index, (phase, sense, expression) in enumerate(phases):
        if sense == "max":
            model.Maximize(expression)
        else:
            model.Minimize(expression)
        status = _solve(solver, model, phase, config)
        best = int(round(solver.ObjectiveValue()))
        statuses[phase] = solver.StatusName(status)
        objectives[phase] = solver.ObjectiveValue()
        times[phase] = solver.WallTime()
        if index < len(phases) - 1:
            model.Add(expression == best)

    output_equation = {
        v: q for v in variables for q in uses[v] if solver.Value(output[q, v]) == 1
    }
    output_vars = set(output_equation)
    tear_pairs = sorted((q, v) for (q, v), var in tear.items() if solver.Value(var) == 1)
    tear_pair_set = set(tear_pairs)
    tear_vars = {v for _, v in tear_pairs}
    open_tears = tear_vars - output_vars
    closed_struct_tears = tear_vars & output_vars

    cut_edges = _graph_edges(eqs, output_equation, tear_pair_set, remove_cut_arcs=True)
    full_edges = _graph_edges(eqs, output_equation, tear_pair_set, remove_cut_arcs=False)
    external_dependency = _reachable_from_sources(variables, full_edges, open_tears)
    externally_computed = output_vars & external_dependency
    closed_external = closed_struct_tears & external_dependency
    closed_internal = closed_struct_tears - external_dependency

    direct_known = _direct_closure(eqs, initial_known, output_equation, tear_pair_set)
    inferred_direct = (direct_known - initial_known) - closed_struct_tears
    autonomous_outputs = output_vars - external_dependency
    inferred_closed_loop = autonomous_outputs - inferred_direct - closed_internal
    inferred_conditioned_open = externally_computed - closed_external
    indeterminate_pure = variable_set - initial_known - output_vars - open_tears
    effective_indeterminate = indeterminate_pure | open_tears | inferred_conditioned_open | closed_external

    output_uses_tears: Dict[str, List[str]] = {}
    for target, q in output_equation.items():
        used = sorted(u for u in eqs[q] if u != target and (q, u) in tear_pair_set)
        if used:
            output_uses_tears[target] = used

    result = TearingResult(
        case_name=case_name,
        total_variables=len(variables),
        total_equations=len(original_eqs),
        total_incidences=sum(len(values) for values in original_eqs.values()),
        model_equations=len(eqs),
        model_incidences=sum(len(values) for values in eqs.values()),
        structural_rank=structural_rank,
        preprocessing_enabled=config.use_structural_preprocessing,
        retained_equations=sorted(retained_equations),
        removed_equations=sorted(removed_equations),
        measured=sorted(measured),
        known_constants=sorted(constants),
        inferred_direct=sorted(inferred_direct),
        inferred_closed_loop=sorted(inferred_closed_loop),
        tears_closed=sorted(closed_internal),
        inferred_conditioned_open=sorted(inferred_conditioned_open),
        tears_closed_external=sorted(closed_external),
        tears_open=sorted(open_tears),
        indeterminate_pure=sorted(indeterminate_pure),
        effective_indeterminate=sorted(effective_indeterminate),
        tear_pairs=tear_pairs,
        output_equation=dict(sorted(output_equation.items())),
        output_uses_tears=dict(sorted(output_uses_tears.items())),
        levels={v: solver.Value(level[v]) for v in variables},
        status_by_phase=statuses,
        objective_by_phase=objectives,
        time_by_phase_s=times,
        execution_graph_is_dag=_is_dag(variables, cut_edges),
        cyclic_sccs_full_graph=_cyclic_sccs(variables, full_edges),
        cut_graph_edges=cut_edges,
        full_graph_edges=full_edges,
        externally_conditioned=sorted(external_dependency),
    )
    validate_result(result)
    return result


def validate_result(result: TearingResult) -> None:
    classes = {
        "medidas": set(result.measured),
        "parametros": set(result.known_constants),
        "diretas": set(result.inferred_direct),
        "loop": set(result.inferred_closed_loop),
        "tears_fechados": set(result.tears_closed),
        "cond_open": set(result.inferred_conditioned_open),
        "closed_external": set(result.tears_closed_external),
        "tears_abertos": set(result.tears_open),
        "indet_puro": set(result.indeterminate_pure),
    }
    names = list(classes)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = classes[left_name] & classes[right_name]
            if overlap:
                raise AssertionError(f"Sobreposicao {left_name} x {right_name}: {sorted(overlap)}")
    union = set().union(*classes.values())
    if len(union) != result.total_variables:
        raise AssertionError(f"Particao incompleta: {len(union)} != {result.total_variables}")
    expected_effective = (
        set(result.indeterminate_pure)
        | set(result.tears_open)
        | set(result.inferred_conditioned_open)
        | set(result.tears_closed_external)
    )
    if expected_effective != set(result.effective_indeterminate):
        raise AssertionError("Indeterminabilidade efetiva nao corresponde a dependencia de tears abertos.")
    if not result.execution_graph_is_dag:
        raise AssertionError("Grafo de execucao apos cortes deve ser DAG.")
    if result.n_closed_coverage + len(result.effective_indeterminate) != result.total_variables:
        raise AssertionError("Cobertura fechada + indeterminabilidade efetiva nao fecha o total.")
