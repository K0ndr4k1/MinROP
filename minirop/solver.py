"""Z3 backend for persistent byte memory and equality goals.

The core IR/state/executor remain solver independent. SAT denotes a witness in
this byte-array model, not hardware validity or a unique assignment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

import z3

from .constraints import Equality, RegisterGoal
from .executor import execute_gadget
from .gadget import Gadget
from .sequence import SequenceEffect, StateSnapshot, analyze_sequence
from .state import CPUState, make_symbolic_initial_state
from .symbolic import (BoolExpression, BoolConstant, BoolSymbol, BoolNot, BoolBinary,
                       BoolCompare, BoolValue, GuardedBool, IntegerValue, GuardedValue, value_requirements, Ite, as_bool, Add, Constant, Expression, MemoryRead, Sub, Symbol, Value,
                       Xor, And, Or, Shl, Shr, Sar, ProductPart, BitSlice, BitExtend, InitialMemory, MemoryState, MemoryStore, as_expression)

if TYPE_CHECKING:
    from .diagnostics import UnsatDiagnostics


class SolveStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    assignments: Mapping[str, int | bool] = field(default_factory=lambda: MappingProxyType({}))
    memory_values: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))
    reason_unknown: str | None = None
    raw_model: z3.ModelRef | None = field(default=None, repr=False, compare=False)


class Z3Translator:
    """A symbol table and one configurable input Array(BitVec64, BitVec8).

    MemoryStore versions derive from this input array. Each translator
    owns a distinct Z3 context, so callers cannot accidentally combine snapshots
    or same-named symbols across independent problems.
    """

    def __init__(self, memory_name: str = "memory_0") -> None:
        if not isinstance(memory_name, str) or not memory_name:
            raise ValueError("memory_name must be a nonempty string")
        self.context = z3.Context()
        self.memory_name = memory_name
        word = z3.BitVecSort(64, ctx=self.context)
        self.memory = z3.Array(memory_name, word, z3.BitVecSort(8, ctx=self.context))
        self._symbols: dict[str, z3.BitVecRef | z3.BoolRef] = {}
        self._cache: dict[Expression | BoolExpression | MemoryState, z3.ExprRef] = {InitialMemory(): self.memory}
        self._reads: dict[MemoryRead, z3.BitVecRef] = {}

    @property
    def symbols(self) -> Mapping[str, z3.BitVecRef | z3.BoolRef]:
        return MappingProxyType(self._symbols)

    @property
    def memory_reads(self) -> Mapping[MemoryRead, z3.BitVecRef]:
        return MappingProxyType(self._reads)

    def translate_memory(self, memory: MemoryState) -> z3.ArrayRef:
        if memory in self._cache:
            return self._cache[memory]
        if not isinstance(memory, MemoryStore):
            raise TypeError("invalid memory version")
        result = self.translate_memory(memory.parent)
        address, value = self.translate(memory.address), self.translate(memory.value)
        for i in range(memory.width // 8):
            result = z3.Store(result, address + i, z3.Extract(8 * i + 7, 8 * i, value))
        self._cache[memory] = result
        return result

    @staticmethod
    def read_word(memory: z3.ArrayRef, address: z3.BitVecRef, width: int = 64) -> z3.BitVecRef:
        bytes_ = [z3.Select(memory, address + i) for i in reversed(range(width // 8))]
        value = bytes_[0] if width == 8 else z3.Concat(*bytes_)
        return z3.ZeroExt(64 - width, value) if width < 64 else value

    def translate_bool(self, value: BoolValue) -> z3.BoolRef:
        expression = as_bool(value)
        if expression in self._cache:
            return self._cache[expression]
        if isinstance(expression, BoolConstant):
            result = z3.BoolVal(expression.value, ctx=self.context)
        elif isinstance(expression, BoolSymbol):
            if expression.name not in self._symbols:
                self._symbols[expression.name] = z3.Bool(expression.name, ctx=self.context)
            result = self._symbols[expression.name]
            if not z3.is_bool(result):
                raise ValueError("symbol name used with incompatible sorts")
        elif isinstance(expression, GuardedBool):
            result = self.translate_bool(expression.value)
        elif isinstance(expression, BoolNot):
            result = z3.Not(self.translate_bool(expression.value))
        elif isinstance(expression, BoolBinary):
            left, right = self.translate_bool(expression.left), self.translate_bool(expression.right)
            result = {"and": z3.And, "or": z3.Or, "xor": z3.Xor}[expression.operator](left, right)
        else:
            left, right = self.translate(expression.left), self.translate(expression.right)
            result = left == right if expression.operator == "eq" else z3.ULT(left, right)
        self._cache[expression] = result
        return result

    def translate(self, value: Value | BoolValue) -> z3.BitVecRef | z3.BoolRef:
        if type(value) is bool or isinstance(value, BoolExpression):
            return self.translate_bool(value)
        expression = as_expression(value)
        if expression.bits != 64:
            raise ValueError("translator supports only 64-bit expressions")
        if expression in self._cache:
            return self._cache[expression]
        if isinstance(expression, Constant):
            result = z3.BitVecVal(expression.value, 64, ctx=self.context)
        elif isinstance(expression, Symbol):
            if expression.name not in self._symbols:
                self._symbols[expression.name] = z3.BitVec(expression.name, 64, ctx=self.context)
            result = self._symbols[expression.name]
            if z3.is_bool(result):
                raise ValueError("symbol name used with incompatible sorts")
        elif isinstance(expression, Ite):
            result = z3.If(self.translate_bool(expression.condition),
                           self.translate(expression.when_true), self.translate(expression.when_false))
        elif isinstance(expression, BitSlice):
            result = z3.Extract(expression.offset + expression.width - 1,
                                expression.offset, self.translate(expression.value))
            if expression.width < 64:
                result = z3.ZeroExt(64 - expression.width, result)
        elif isinstance(expression, BitExtend):
            source = z3.Extract(expression.source_width - 1, 0, self.translate(expression.value))
            extend = z3.SignExt if expression.signed else z3.ZeroExt
            result = extend(expression.target_width - expression.source_width, source)
            if expression.target_width < 64:
                result = z3.ZeroExt(64 - expression.target_width, result)
        elif isinstance(expression, ProductPart):
            width = expression.width
            left, right = self.translate(expression.left), self.translate(expression.right)
            if width < 64:
                left, right = z3.Extract(width - 1, 0, left), z3.Extract(width - 1, 0, right)
            extend = z3.SignExt if expression.signed else z3.ZeroExt
            # Multiplication takes place at 2*width, including 128 bits for r64.
            product = extend(width, left) * extend(width, right)
            offset = width if expression.high else 0
            result = z3.Extract(offset + width - 1, offset, product)
            if width < 64:
                result = z3.ZeroExt(64 - width, result)
        elif isinstance(expression, GuardedValue):
            result = self.translate(expression.value)
        elif isinstance(expression, IntegerValue):
            width, op = expression.width, expression.operation
            values = [self.translate(v) for v in expression.values]
            narrow = lambda v: z3.Extract(width - 1, 0, v) if width < 64 else v
            if op.startswith("div"):
                low, high, divisor = map(narrow, values)
                dividend = z3.Concat(high, low)
                extended = (z3.SignExt if expression.signed else z3.ZeroExt)(width, divisor)
                quotient = dividend / extended if expression.signed else z3.UDiv(dividend, extended)
                remainder = z3.SRem(dividend, extended) if expression.signed else z3.URem(dividend, extended)
                truncated = z3.Extract(width - 1, 0, quotient)
                valid = z3.And(divisor != 0, quotient == (z3.SignExt if expression.signed else z3.ZeroExt)(width, truncated))
                if op == "divok":
                    result = z3.If(valid, z3.BitVecVal(1, 64, ctx=self.context), z3.BitVecVal(0, 64, ctx=self.context))
                else:
                    result = truncated if op == "divq" else z3.Extract(width - 1, 0, remainder)
                    if width < 64:
                        result = z3.ZeroExt(64 - width, result)
            else:
                value = narrow(values[0])
                count64 = values[1] & (63 if width == 64 else 31)
                if op in ("shl", "shr", "sar"):
                    count = narrow(count64)
                    result = value << count if op == "shl" else value >> count if op == "sar" else z3.LShR(value, count)
                else:
                    carry = op.startswith(("rcl", "rcr"))
                    size = width + int(carry)
                    ring = z3.Concat(z3.Extract(0, 0, values[2]), value) if carry else value
                    count = z3.Extract(size - 1, 0, count64) if size < 64 else z3.ZeroExt(size - 64, count64) if size > 64 else count64
                    count = z3.URem(count, size)
                    result = ((z3.LShR(ring, count) | (ring << (size - count))) if op.startswith(("ror", "rcr"))
                              else ((ring << count) | z3.LShR(ring, size - count)))
                    result = z3.ZeroExt(63, z3.Extract(width, width, result)) if op.endswith("_cf") else z3.Extract(width - 1, 0, result)
                if width < 64 and not op.endswith("_cf"):
                    result = z3.ZeroExt(64 - width, result)
        elif isinstance(expression, MemoryRead):
            result = self.read_word(self.translate_memory(expression.memory),
                                    self.translate(expression.address), expression.width)
            self._reads[expression] = result
        elif isinstance(expression, (Add, Sub, Xor, And, Or, Shl, Shr, Sar)):
            left, right = self.translate(expression.left), self.translate(expression.right)
            if isinstance(expression, Add):
                result = left + right
            elif isinstance(expression, Sub):
                result = left - right
            elif isinstance(expression, Xor):
                result = left ^ right
            elif isinstance(expression, And):
                result = left & right
            elif isinstance(expression, Or):
                result = left | right
            elif isinstance(expression, Shl):
                result = left << right
            elif isinstance(expression, Sar):
                result = left >> right
            else:
                result = z3.LShR(left, right)
        else:
            raise TypeError(f"unsupported expression: {type(expression).__name__}")
        self._cache[expression] = result
        return result


class SolverContext:
    """Equality constraints over a single input snapshot; no instruction logic.

    Pass the INITIAL state to bind named slots and concrete words to the array.
    Values/addresses are captured at construction; later mutations do not alter
    the problem. timeout_ms=None disables the default 5-second solving timeout.
    """

    def __init__(self, initial_state: CPUState | StateSnapshot | None = None, *,
                 memory_name: str = "memory_0", timeout_ms: int | None = 5000) -> None:
        if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
            raise ValueError("timeout_ms must be positive or None")
        self.translator = Z3Translator(memory_name)
        self._solver = z3.Solver(ctx=self.translator.context)
        self._constraints: list[Equality] = []
        if timeout_ms is not None:
            self._solver.set(timeout=timeout_ms)
        if initial_state is not None:
            # Use the state API, never infer slot identity from names or indices.
            for address, value in initial_state.memory_bindings():
                self.add_equal(MemoryRead(as_expression(address)), value)
            self.add_execution_conditions(initial_state)

    def add_execution_conditions(self, state):
        for condition in state.execution_conditions:
            self.add_equal(condition, True)
        if state.exception is not None:
            self.add_equal(False, True)

    def add_equal(self, left: Value | BoolValue, right: Value | BoolValue) -> None:
        for operand in (left, right):
            for condition in value_requirements(operand):
                self._solver.add(self.translator.translate_bool(condition))
        left_z3 = self.translator.translate(left)
        right_z3 = self.translator.translate(right)
        if left_z3.sort() != right_z3.sort():
            raise TypeError("equality operands require the same sort")
        self._solver.add(left_z3 == right_z3)
        self._constraints.append(Equality(left, right))

    @property
    def constraints(self) -> tuple[Equality, ...]:
        """All asserted equalities, including input-memory consistency."""
        return tuple(self._constraints)

    def add(self, equality: Equality) -> None:
        if not isinstance(equality, Equality):
            raise TypeError("expected miniROP Equality")
        self.add_equal(equality.left, equality.right)

    def solve(self) -> SolveResult:
        checked = self._solver.check()
        if checked == z3.unsat:
            return SolveResult(SolveStatus.UNSAT)
        if checked == z3.unknown:
            return SolveResult(SolveStatus.UNKNOWN, reason_unknown=self._solver.reason_unknown())
        if checked != z3.sat:
            raise RuntimeError(f"unexpected solver status: {checked}")
        model = self._solver.model()
        # Model completion selects a witness for underconstrained values; these
        # are not claimed to be unique or forced by the goals.
        assignments = {name: (z3.is_true(model.eval(symbol, model_completion=True)) if z3.is_bool(symbol)
                              else model.eval(symbol, model_completion=True).as_long())
                       for name, symbol in sorted(self.translator.symbols.items())}
        memory_values = {}
        for read, term in self.translator.memory_reads.items():
            address = model.eval(self.translator.translate(read.address), model_completion=True).as_long()
            # The replay witness always describes M0, not a post-store version.
            initial_word = self.translator.read_word(
                self.translator.memory, self.translator.translate(read.address))
            memory_values[address] = model.eval(initial_word, model_completion=True).as_long()
        return SolveResult(SolveStatus.SAT, MappingProxyType(assignments),
                           MappingProxyType(dict(sorted(memory_values.items()))), raw_model=model)


def solve_effect(effect: CPUState, goals: Mapping[str, Value], *, initial_state: CPUState,
                 constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                 timeout_ms: int | None = 5000) -> SolveResult:
    """Solve an already-executed effect using its corresponding initial snapshot.

    The caller must supply the state used to produce effect. Goal right sides
    may be constants or miniROP expressions referring to input symbols.
    """
    effect = effect.copy()  # Goal reads may add architectural definedness obligations.
    problem = SolverContext(initial_state, memory_name=memory_name, timeout_ms=timeout_ms)
    # Observe all final values, including unconstrained ones. For example leave
    # reads old RBP even when only RIP has a goal; expose that word in the witness.
    for _, value in effect.register_bindings():
        problem.translator.translate(value)
    for register, expected in goals.items():
        problem.add_equal(effect.get_register(register), expected)
    for equality in constraints:
        problem.add(equality)
    problem.add_execution_conditions(effect)
    return problem.solve()


def solve_gadget(gadget: Gadget, goals: Mapping[str, Value], *, initial_state: CPUState | None = None,
                 constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                 timeout_ms: int | None = 5000) -> SolveResult:
    """Convenience composition: shared symbolic execution, then goal solving."""
    initial = make_symbolic_initial_state() if initial_state is None else initial_state.copy()
    effect = execute_gadget(gadget, initial)
    return solve_effect(effect, goals, initial_state=initial, constraints=constraints,
                        memory_name=memory_name, timeout_ms=timeout_ms)


@dataclass(frozen=True)
class SequenceSolveResult:
    effect: SequenceEffect
    solution: SolveResult
    constraints: tuple[Equality, ...]
    final_goals: tuple[RegisterGoal, ...] = ()
    final_goal_constraints: tuple[Equality, ...] = ()
    sequence_constraints: tuple[Equality, ...] = ()
    diagnostics: UnsatDiagnostics | None = None

    @property
    def diagnostic_solver_calls(self) -> int:
        return 0 if self.diagnostics is None else self.diagnostics.diagnostic_solver_calls

    @property
    def status(self) -> SolveStatus:
        return self.solution.status

    @property
    def goals_satisfied(self) -> bool | None:
        """Joint compatibility: SAT=True, UNSAT=False, UNKNOWN=None.

        False does not diagnose any single goal: the base sequence itself may
        be inconsistent. With no goals, this reports base sequence feasibility.
        """
        if self.status is SolveStatus.UNKNOWN:
            return None
        return self.status is SolveStatus.SAT


FinalGoals = Mapping[str, Value] | Iterable[RegisterGoal]


def _normalize_goals(goals: FinalGoals | None) -> tuple[RegisterGoal, ...]:
    if goals is None:
        return ()
    if isinstance(goals, Mapping):
        result = tuple(RegisterGoal(register, expected) for register, expected in goals.items())
    else:
        result = tuple(goals)
    for goal in result:
        if not isinstance(goal, RegisterGoal):
            raise TypeError("goals must be a mapping or an iterable of RegisterGoal")
        if not isinstance(goal.register, str):
            raise TypeError("goal register must be a string")
        # Validate without changing the user's int/Expression representation.
        as_expression(goal.expected)
    return result


def solve_sequence_effect(effect: SequenceEffect, goals: FinalGoals | None = None, *,
                          checkpoints: Mapping[int, Mapping[str, Value]] | None = None,
                          constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                          timeout_ms: int | None = 5000, diagnose: bool = False,
                          minimize_conflict: bool = False,
                          report_execution_errors: bool = False) -> SequenceSolveResult:
    """Check all sequence conditions together in one solver context.

    Checkpoints constrain registers AFTER a zero-based step. The final return
    target is unconstrained unless the caller specifies a RIP goal/constraint.
    Initial snapshot, intermediates and constraints share the same symbol names
    and input memory array. No per-step models are chosen or concatenated.
    Final goals may be a mapping or a list of RegisterGoal; list duplicates are
    retained as simultaneous equalities. None/empty adds no final constraints.
    Optional diagnostics run only AFTER the original solve returns UNSAT, in
    separate contexts. minimize_conflict requires diagnose=True.

    With report_execution_errors, an execution fault or input-forced undefined
    execution is UNKNOWN rather than a normal goal UNSAT. This presentation
    policy does not change the normal-execution constraints or SAT witnesses.
    An UNSAT with definedness obligations may need one additional goal-free
    validity check to classify an input-forced execution failure.
    """
    if type(diagnose) is not bool or type(minimize_conflict) is not bool:
        raise TypeError("diagnose and minimize_conflict must be bools")
    if minimize_conflict and not diagnose:
        raise ValueError("minimize_conflict requires diagnose=True")
    if type(report_execution_errors) is not bool:
        raise TypeError("report_execution_errors must be a bool")
    saved_constraints = tuple(constraints)
    final_goals = _normalize_goals(goals)
    final_equalities = tuple(Equality(effect.final_state.get_register(goal.register), goal.expected)
                             for goal in final_goals)
    problem = SolverContext(effect.initial_state, memory_name=memory_name, timeout_ms=timeout_ms)
    if diagnose:
        from .diagnostics import (ConstraintCategory as Category, ConstraintRecord,
                                  ConstraintSource as Source, _diagnose_unsat)
        memory_count = len(effect.initial_state.memory_bindings())
        sources = [(Category.MEMORY, Source(memory_binding_index=index))
                   if index < memory_count else (Category.INITIAL, Source())
                   for index in range(len(problem.constraints))]
    for step in effect.steps:
        previous_count = len(problem.constraints)
        problem.add_execution_conditions(step.after)
        if diagnose:
            sources.extend((Category.AUXILIARY, Source(sequence_index=step.index))
                           for _ in range(len(problem.constraints) - previous_count))
        # Preserve observations from earlier steps even if a later step overwrites
        # their registers. This is still not an instruction-level access trace.
        for _, value in step.after.register_bindings():
            problem.translator.translate(value)
    for flow_index, equality in enumerate(effect.constraints):
        problem.add(equality)
        if diagnose:
            flow = effect.control_flow[flow_index]
            sources.append((Category.INITIAL if flow.source_index is None else Category.CONTROL_FLOW,
                            Source(sequence_index=flow.target_index, source_sequence_index=flow.source_index,
                                   register="rip")))
    for index, conditions in (checkpoints or {}).items():
        if type(index) is not int or not 0 <= index < len(effect.steps):
            raise ValueError("checkpoint index must identify a step in the sequence")
        for register, expected in conditions.items():
            problem.add_equal(effect.steps[index].after.get_register(register), expected)
            if diagnose:
                sources.append((Category.CHECKPOINT, Source(sequence_index=index, checkpoint_index=index,
                                                            register=register)))
    goal_start = len(problem.constraints)
    for goal_index, equality in enumerate(final_equalities):
        problem.add(equality)
        if diagnose:
            sources.append((Category.FINAL_GOAL, Source(sequence_index=len(effect.steps) - 1,
                                                        goal_index=goal_index, register=final_goals[goal_index].register)))
    for auxiliary_index, equality in enumerate(saved_constraints):
        problem.add(equality)
        if diagnose:
            sources.append((Category.AUXILIARY, Source(auxiliary_index=auxiliary_index)))
    accumulated = problem.constraints
    # Keep the existing insertion order and distinguish provenance by position,
    # not equality: identical constraints may also come from checkpoints/users.
    sequence_constraints = accumulated[:goal_start] + accumulated[goal_start + len(final_equalities):]
    solution = problem.solve()
    if report_execution_errors:
        goal_definedness = tuple(condition for equality in final_equalities
                                 for operand in (equality.left, equality.right)
                                 for condition in value_requirements(operand))
        if effect.final_state.exception is not None:
            solution = SolveResult(SolveStatus.UNKNOWN, reason_unknown=effect.final_state.exception)
        elif (solution.status is SolveStatus.UNSAT
              and (goal_definedness or any(step.after.execution_conditions for step in effect.steps))):
            # Separate input-forced execution failure from a contradictory final
            # goal or return target. Reuse the same solver; no path search or
            # conflict diagnosis is involved. Never infer failure from UNKNOWN.
            validity = SolverContext(effect.initial_state, memory_name=memory_name,
                                     timeout_ms=timeout_ms)
            for step in effect.steps:
                validity.add_execution_conditions(step.after)
            for condition in goal_definedness:
                validity.add_equal(condition, True)
            # These are the already frozen auxiliary equalities, not goal/flow
            # equalities; an unreachable return remains ordinary UNSAT.
            for equality in saved_constraints:
                validity.add(equality)
            checked = validity.solve()
            if checked.status is SolveStatus.UNSAT:
                solution = SolveResult(SolveStatus.UNKNOWN, reason_unknown=
                    "no defined normal execution under the initial conditions "
                    "(division exception or undefined value/flag)")
            elif checked.status is SolveStatus.UNKNOWN:
                solution = SolveResult(SolveStatus.UNKNOWN, reason_unknown=
                    "normal execution validity is UNKNOWN: " + (checked.reason_unknown or "solver limit"))
    diagnostics = None
    if diagnose and solution.status is SolveStatus.UNSAT:
        if len(sources) != len(accumulated):
            raise RuntimeError("diagnostic metadata does not cover all solver constraints")
        records = tuple(ConstraintRecord(index, category, accumulated[index], source)
                        for index, (category, source) in enumerate(sources))
        diagnostics = _diagnose_unsat(records, minimize_conflict=minimize_conflict,
                                     memory_name=memory_name, timeout_ms=timeout_ms)
    return SequenceSolveResult(effect, solution, accumulated,
                               final_goals, final_equalities, sequence_constraints, diagnostics)


def solve_sequence(gadgets: Iterable[Gadget], goals: FinalGoals | None = None, *,
                   initial_state: CPUState | None = None, check_entry: bool = False,
                   checkpoints: Mapping[int, Mapping[str, Value]] | None = None,
                   constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                   timeout_ms: int | None = 5000, diagnose: bool = False,
                   minimize_conflict: bool = False,
                   report_execution_errors: bool = False) -> SequenceSolveResult:
    """Analyze and validate a caller-supplied sequence; never search or reorder."""
    effect = analyze_sequence(gadgets, initial_state, check_entry=check_entry)
    return solve_sequence_effect(effect, goals, checkpoints=checkpoints, constraints=constraints,
                                 memory_name=memory_name, timeout_ms=timeout_ms,
                                 diagnose=diagnose, minimize_conflict=minimize_conflict,
                                 report_execution_errors=report_execution_errors)


def solve_program_path(path, goals: Mapping[str, Value] | None = None, *,
                       constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                       timeout_ms: int | None = 5000) -> SolveResult:
    """Solve a completed fixed-program path with its original constraints.

    Uses the same solver and result type as fixed gadgets. A stopped path is
    UNKNOWN, including step/path limits and unresolved/unsupported execution;
    satisfying its intermediate registers does not establish a completed goal.
    """
    from .program import ProgramPath, ProgramStatus

    if not isinstance(path, ProgramPath):
        raise TypeError("a fixed-program ProgramPath is required")
    if path.status is not ProgramStatus.COMPLETE:
        return SolveResult(SolveStatus.UNKNOWN, reason_unknown=path.reason or path.status.value)
    return solve_effect(path.state.copy(), {} if goals is None else goals,
                        initial_state=path.state,
                        constraints=(*path.constraints, *tuple(constraints)),
                        memory_name=memory_name, timeout_ms=timeout_ms)
