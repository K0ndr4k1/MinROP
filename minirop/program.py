"""Bounded execution of a caller-supplied instruction set; no discovery or ROP search."""
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum

from .constraints import Equality
from .executor import execute
from .flags import evaluate_condition
from .gadget import Instruction
from .ir import Jcc, Ret
from .semantics import UnsupportedInstruction, lift_instruction
from .state import CPUState, make_symbolic_initial_state
from .symbolic import BoolValue, WORD_MASK, bool_binary, bool_not


class ProgramStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    UNSAT = "unsat"


@dataclass(frozen=True)
class ProgramPath:
    """One final or stopped state. constraints includes its path condition."""

    address: int
    state: CPUState
    path_condition: BoolValue
    constraints: tuple[Equality, ...]
    steps: int
    status: ProgramStatus
    reason: str | None = None


@dataclass(frozen=True)
class ProgramResult:
    status: ProgramStatus
    paths: tuple[ProgramPath, ...]


def _feasibility(state, condition, constraints, timeout_ms):
    from .solver import SolverContext

    problem = SolverContext(state, timeout_ms=timeout_ms)
    for constraint in constraints:
        problem.add(constraint)
    problem.add_equal(condition, True)
    return problem.solve()


def execute_program(instructions: Iterable[Instruction], initial_state: CPUState | None = None, *,
                    entry: int | None = None, constraints: Iterable[Equality] = (),
                    max_steps: int = 1000, max_paths: int = 32,
                    timeout_ms: int | None = 5000) -> ProgramResult:
    """Execute only the supplied addresses, with per-path steps and an active-path cap.

    Normal fallthrough past the highest instruction ends a path; other gaps are
    unresolved. Every reached Jcc must target a supplied instruction boundary.
    Other control transfers (including RET) are unsupported in this interface.
    Feasibility uses the existing SolverContext. UNKNOWN and either execution
    limit return INCOMPLETE; only proven-infeasible paths are discarded.
    """
    from .solver import SolveStatus

    for name, value in (("max_steps", max_steps), ("max_paths", max_paths)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError("timeout_ms must be positive or None")
    supplied = tuple(instructions)
    if not supplied:
        raise ValueError("a nonempty fixed instruction set is required")
    for instruction in supplied:
        if (not isinstance(instruction, Instruction) or type(instruction.address) is not int
                or not 0 <= instruction.address <= WORD_MASK
                or type(instruction.size) is not int or not 1 <= instruction.size <= 15
                or not isinstance(instruction.bytes, bytes) or instruction.size != len(instruction.bytes)
                or instruction.address + instruction.size > WORD_MASK + 1):
            raise ValueError("invalid fixed-program instruction layout")
    ordered = sorted(supplied, key=lambda instruction: instruction.address)
    if any(left.address + left.size > right.address for left, right in zip(ordered, ordered[1:])):
        raise ValueError("duplicate or overlapping fixed-program instructions")
    by_address = {instruction.address: instruction for instruction in supplied}
    end = (ordered[-1].address + ordered[-1].size) & WORD_MASK
    entry = ordered[0].address if entry is None else entry
    if type(entry) is not int or not 0 <= entry <= WORD_MASK:
        raise ValueError("entry must be a 64-bit address")
    constraints = tuple(constraints)
    if any(not isinstance(constraint, Equality) for constraint in constraints):
        raise TypeError("constraints must be miniROP Equality instances")
    if initial_state is not None and not isinstance(initial_state, CPUState):
        raise TypeError("initial_state must be a CPUState")
    state = make_symbolic_initial_state() if initial_state is None else initial_state.copy()
    state.set_register("rip", entry)

    def path(address, current, condition, steps, status, reason=None):
        return ProgramPath(address, current, condition,
                           constraints + (Equality(condition, True),), steps, status, reason)

    initial = _feasibility(state, True, constraints, timeout_ms)
    if initial.status is SolveStatus.UNSAT:
        return ProgramResult(ProgramStatus.UNSAT, ())
    if initial.status is SolveStatus.UNKNOWN:
        return ProgramResult(ProgramStatus.INCOMPLETE,
                             (path(entry, state, True, 0, ProgramStatus.INCOMPLETE, "solver_unknown"),))
    active = deque([path(entry, state, True, 0, ProgramStatus.INCOMPLETE)])
    stopped = []
    while active:
        current = active.popleft()
        instruction = by_address.get(current.address)
        if instruction is None:
            stopped.append(replace(current, status=ProgramStatus.UNSUPPORTED, reason="unresolved_target"))
            continue
        if current.steps >= max_steps:
            stopped.append(replace(current, reason="step_limit"))
            continue
        try:
            operation = lift_instruction(instruction)
            if isinstance(operation, Ret):
                raise UnsupportedInstruction(instruction, "RET is unsupported in fixed-program execution")
            if isinstance(operation, Jcc):
                if operation.target not in by_address:
                    stopped.append(replace(current, status=ProgramStatus.UNSUPPORTED,
                                           reason="unresolved_target"))
                    continue
                condition = evaluate_condition(operation.condition, current.state)
                if type(condition) is bool:
                    successors = [(operation.target if condition else operation.fallthrough,
                                   current.path_condition)]
                else:
                    successors = [(operation.target, bool_binary("and", current.path_condition, condition)),
                                  (operation.fallthrough, bool_binary("and", current.path_condition,
                                                                    bool_not(condition)))]
                next_paths = []
                for address, predicate in successors:
                    # Boolean conditions need no additional proof; symbolic successors do.
                    checked = (_feasibility(current.state, predicate, constraints, timeout_ms)
                               if type(condition) is not bool or current.state.execution_conditions else None)
                    if checked is not None and checked.status is SolveStatus.UNSAT:
                        continue
                    child_state = current.state.copy()
                    child_state.set_register("rip", address)
                    child = path(address, child_state, predicate, current.steps + 1, ProgramStatus.INCOMPLETE)
                    if checked is not None and checked.status is SolveStatus.UNKNOWN:
                        stopped.append(replace(child, reason="solver_unknown"))
                    elif address == end and address not in by_address:
                        # Only fallthrough can exit: taken targets were checked above.
                        stopped.append(replace(child, status=ProgramStatus.COMPLETE))
                    else:
                        next_paths.append(child)
            else:
                child_state = execute((operation,), current.state)
                address = (instruction.address + instruction.size) & WORD_MASK
                child_state.set_register("rip", address)
                child = path(address, child_state, current.path_condition,
                             current.steps + 1, ProgramStatus.INCOMPLETE)
                next_paths = []
                checked = (_feasibility(child_state, current.path_condition, constraints, timeout_ms)
                           if child_state.execution_conditions != current.state.execution_conditions or child_state.exception else None)
                if checked is not None and checked.status is SolveStatus.UNSAT:
                    continue
                if checked is not None and checked.status is SolveStatus.UNKNOWN:
                    stopped.append(replace(child, reason="solver_unknown"))
                elif address == end and address not in by_address:
                    stopped.append(replace(child, status=ProgramStatus.COMPLETE))
                else:
                    next_paths.append(child)
        except (ValueError, TypeError, KeyError) as error:
            stopped.append(replace(current, status=ProgramStatus.UNSUPPORTED, reason=str(error)))
            continue
        if len(active) + len(next_paths) > max_paths:
            # Preserve the entire feasible frontier in the result, including the
            # unscheduled children; it is never presented as completed or UNSAT.
            stopped.extend(replace(item, reason="path_limit") for item in (*active, *next_paths))
            active.clear()
            break
        active.extend(next_paths)
    paths = tuple(stopped)
    if any(item.status is ProgramStatus.INCOMPLETE for item in paths):
        status = ProgramStatus.INCOMPLETE
    elif any(item.status is ProgramStatus.UNSUPPORTED for item in paths):
        status = ProgramStatus.UNSUPPORTED
    elif paths:
        status = ProgramStatus.COMPLETE
    else:
        status = ProgramStatus.UNSAT
    return ProgramResult(status, paths)
