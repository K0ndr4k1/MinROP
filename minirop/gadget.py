"""Immutable discovery records, independent of Capstone objects."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Operand:
    """Decoder-neutral operand metadata; width is measured in bits."""

    kind: Literal["register", "immediate", "memory", "unsupported"]
    width: int
    register: str | None = None
    immediate: int | None = None
    base: str | None = None
    index: str | None = None
    scale: int = 1
    displacement: int = 0
    segment: str | None = None


@dataclass(frozen=True)
class Instruction:
    address: int
    mnemonic: str
    op_str: str
    size: int
    bytes: bytes
    operands: tuple[Operand, ...] = ()
    prefixes: tuple[int, ...] = ()

    def __str__(self) -> str:
        return f"{self.mnemonic} {self.op_str}".rstrip()


@dataclass(frozen=True)
class Gadget:
    address: int
    bytes: bytes
    instructions: tuple[Instruction, ...]

    @property
    def end_address(self) -> int:
        """Exclusive end address."""
        return self.address + len(self.bytes)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)
