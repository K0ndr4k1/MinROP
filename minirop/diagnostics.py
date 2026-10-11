"""Optional constraint diagnostics using independent subset solver contexts.

Records wrap existing Equality objects without changing their interpretation.
No backend-native cores, hidden input bindings, search feedback, or state edits.
"""

from dataclasses import asdict, dataclass
from enum import Enum

from .constraints import Equality
from .solver import SolveStatus, SolverContext


class ConstraintCategory(str, Enum):
    MEMORY = "memory"
    INITIAL = "initial"
    CONTROL_FLOW = "control_flow"
    CHECKPOINT = "checkpoint"
    FINAL_GOAL = "final_goal"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class ConstraintSource:
    sequence_index: int | None = None
    source_sequence_index: int | None = None
    goal_index: int | None = None
    checkpoint_index: int | None = None
    register: str | None = None
    memory_binding_index: int | None = None
    auxiliary_index: int | None = None


@dataclass(frozen=True)
class ConstraintRecord:
    id: int
    category: ConstraintCategory
    constraint: Equality
    source: ConstraintSource


@dataclass(frozen=True)
class CategoryDiagnostic:
    category: ConstraintCategory
    without_category_status: SolveStatus

    @property
    def required_for_unsat(self) -> bool | None:
        if self.without_category_status is SolveStatus.UNKNOWN:
            return None
        return self.without_category_status is SolveStatus.SAT


@dataclass(frozen=True)
class ReductionCheck:
    removed_id: int
    remaining_ids: tuple[int, ...]
    status: SolveStatus


@dataclass(frozen=True)
class UnsatDiagnostics:
    status: SolveStatus
    records: tuple[ConstraintRecord, ...]
    conflict_ids: tuple[int, ...]
    conflict_categories: tuple[ConstraintCategory, ...]
    category_diagnostics: tuple[CategoryDiagnostic, ...]
    reduction_checks: tuple[ReductionCheck, ...]
    minimize_requested: bool
    irreducible: bool
    diagnostic_solver_calls: int

    @property
    def solver_calls(self) -> int:
        """Diagnostic calls only; excludes the original full-system solve."""
        return self.diagnostic_solver_calls

    @property
    def unknown_solver_calls(self) -> int:
        return sum(item.without_category_status is SolveStatus.UNKNOWN for item in self.category_diagnostics) + sum(
            item.status is SolveStatus.UNKNOWN for item in self.reduction_checks)

    @property
    def partial(self) -> bool:
        return self.unknown_solver_calls != 0

    @property
    def categories_present(self) -> tuple[ConstraintCategory, ...]:
        return tuple(item.category for item in self.category_diagnostics)

    @property
    def categories_required(self) -> tuple[ConstraintCategory, ...]:
        return tuple(item.category for item in self.category_diagnostics if item.required_for_unsat is True)

    def to_dict(self) -> dict:
        """Diagnostic metadata only; omit expressions, solvers and states."""
        return {
            "status": self.status.value,
            "records": [{"id": record.id, "category": record.category.value,
                         "source": asdict(record.source)} for record in self.records],
            "constraints_total": len(self.records),
            "conflict_ids": list(self.conflict_ids),
            "conflict_size": len(self.conflict_ids),
            "conflict_categories": [category.value for category in self.conflict_categories],
            "categories_present": [category.value for category in self.categories_present],
            "categories_required": [category.value for category in self.categories_required],
            "category_diagnostics": [
                {"category": item.category.value, "without_category_status": item.without_category_status.value,
                 "required_for_unsat": item.required_for_unsat} for item in self.category_diagnostics],
            "reduction_checks": [
                {"removed_id": item.removed_id, "remaining_ids": list(item.remaining_ids),
                 "status": item.status.value} for item in self.reduction_checks],
            "minimize_requested": self.minimize_requested,
            "irreducible": self.irreducible,
            "partial": self.partial,
            "diagnostic_solver_calls": self.diagnostic_solver_calls,
            "unknown_solver_calls": self.unknown_solver_calls,
        }


def _solve_subset(records: tuple[ConstraintRecord, ...], *, memory_name: str,
                  timeout_ms: int | None) -> SolveStatus:
    # Crucially, no initial_state: its memory equalities are already explicit
    # records and must remain removable. Only the selected equalities are added.
    problem = SolverContext(memory_name=memory_name, timeout_ms=timeout_ms)
    for record in records:
        problem.add(record.constraint)
    status = problem.solve().status
    if status not in (SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN):
        raise RuntimeError(f"unexpected diagnostic solver status: {status}")
    if not records and status is SolveStatus.UNSAT:
        raise RuntimeError("empty diagnostic constraint set unexpectedly UNSAT")
    return status


def _diagnose_unsat(records: tuple[ConstraintRecord, ...], *, minimize_conflict: bool,
                     memory_name: str, timeout_ms: int | None) -> UnsatDiagnostics:
    """Only called after the original complete set has been proved UNSAT.

    Stable deletion maintains a known UNSAT subset: replace it only after a
    fresh UNSAT result. A SAT deletion stays SAT after later deletions, so one
    complete pass suffices for irreducibility if no deletion returned UNKNOWN.
    Category tests always refer to the ORIGINAL full system, not the reduced one.
    """
    if not records:
        raise RuntimeError("empty original constraint set unexpectedly UNSAT")
    categories = tuple(dict.fromkeys(record.category for record in records))
    category_results = []
    for category in categories:
        subset = tuple(record for record in records if record.category is not category)
        status = _solve_subset(subset, memory_name=memory_name, timeout_ms=timeout_ms)
        category_results.append(CategoryDiagnostic(category, status))
    core = records
    checks = []
    if minimize_conflict:
        for record in records:
            subset = tuple(item for item in core if item.id != record.id)
            status = _solve_subset(subset, memory_name=memory_name, timeout_ms=timeout_ms)
            checks.append(ReductionCheck(record.id, tuple(item.id for item in subset), status))
            if status is SolveStatus.UNSAT:
                core = subset
    irreducible = minimize_conflict and all(item.status is not SolveStatus.UNKNOWN for item in checks)
    return UnsatDiagnostics(
        status=SolveStatus.UNSAT, records=records,
        conflict_ids=tuple(record.id for record in core),
        conflict_categories=tuple(dict.fromkeys(record.category for record in core)),
        category_diagnostics=tuple(category_results), reduction_checks=tuple(checks),
        minimize_requested=minimize_conflict, irreducible=irreducible,
        diagnostic_solver_calls=len(category_results) + len(checks),
    )
