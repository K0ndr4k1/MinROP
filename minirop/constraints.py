"""Solver-independent equality constraints over miniROP values."""

from dataclasses import dataclass

from .symbolic import BoolValue, Value


@dataclass(frozen=True)
class Equality:
    left: Value | BoolValue
    right: Value | BoolValue


@dataclass(frozen=True)
class RegisterGoal:
    """A final register equality, not an assignment.

    A list preserves repeated goals for the same register. The expected value
    uses the existing int/Expression system, including initial-state values.
    Register/value validation is performed when binding the goal to a state.
    """

    register: str
    expected: Value
