"""Byte-aligned enumeration of candidates ending at an unprefixed C3 ret."""

from collections.abc import Iterable, Iterator

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86 import X86_OP_REG, X86_OP_IMM, X86_OP_MEM

from .elf import CodeRegion
from .gadget import Gadget, Instruction, Operand


def _legacy_prefixes(data: bytes) -> tuple[int, ...]:
    # Capstone may omit ignored REP prefixes from its normalized prefix array.
    # Preserve the encoded prefix stream so lifting can reject it conservatively.
    legacy = {0xf0, 0xf2, 0xf3, 0x2e, 0x36, 0x3e, 0x26, 0x64, 0x65, 0x66, 0x67}
    prefixes = []
    for byte in data:
        if byte in legacy:
            prefixes.append(byte)
        elif not 0x40 <= byte <= 0x4f:  # REX in 64-bit mode
            break
    return tuple(prefixes)


class Disassembler:
    def __init__(self) -> None:
        self._engine = Cs(CS_ARCH_X86, CS_MODE_64)
        self._engine.skipdata = False
        self._engine.detail = True

    def disassemble(self, data: bytes, address: int = 0) -> tuple[Instruction, ...]:
        result = []
        for i in self._engine.disasm(data, address):
            operands = []
            for op in i.operands:
                if op.type == X86_OP_REG:
                    operand = Operand("register", op.size * 8, register=i.reg_name(op.reg))
                elif op.type == X86_OP_IMM:
                    operand = Operand("immediate", op.size * 8, immediate=op.imm)
                elif op.type == X86_OP_MEM:
                    operand = Operand("memory", op.size * 8,
                                      base=i.reg_name(op.mem.base) or None,
                                      index=i.reg_name(op.mem.index) or None,
                                      scale=op.mem.scale, displacement=op.mem.disp,
                                      segment=i.reg_name(op.mem.segment) or None)
                else:
                    operand = Operand("unsupported", op.size * 8)
                operands.append(operand)
            result.append(Instruction(i.address, i.mnemonic, i.op_str, i.size, bytes(i.bytes),
                                      tuple(operands), _legacy_prefixes(bytes(i.bytes))))
        return tuple(result)


def ret_offsets(data: bytes) -> Iterator[int]:
    start = 0
    while (offset := data.find(b"\xc3", start)) != -1:
        yield offset
        start = offset + 1


class GadgetScanner:
    def __init__(self, max_gadget_bytes: int = 20, max_instructions: int = 5) -> None:
        if max_gadget_bytes < 1 or max_instructions < 1:
            raise ValueError("gadget limits must be positive")
        self.max_gadget_bytes = max_gadget_bytes
        self.max_instructions = max_instructions
        self.disassembler = Disassembler()

    def scan(self, regions: Iterable[CodeRegion]) -> list[Gadget]:
        found: dict[tuple[int, bytes], Gadget] = {}
        for region in regions:
            for ret in ret_offsets(region.bytes):
                for start in range(max(0, ret + 1 - self.max_gadget_bytes), ret + 1):
                    data = region.bytes[start:ret + 1]
                    address = region.address + start
                    instructions = self.disassembler.disassemble(data, address)
                    if not instructions or len(instructions) > self.max_instructions:
                        continue
                    if sum(i.size for i in instructions) != len(data):
                        continue
                    last = instructions[-1]
                    if last.address != region.address + ret or last.bytes != b"\xc3" or last.mnemonic != "ret":
                        continue
                    # An earlier return ends the sequence before the chosen endpoint.
                    if any(i.mnemonic.startswith("ret") for i in instructions[:-1]):
                        continue
                    found[(address, data)] = Gadget(address, data, instructions)
        return sorted(found.values(), key=lambda g: (g.address, len(g.bytes), g.bytes))
