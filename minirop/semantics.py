"""Conservative lifting from miniROP instruction records to miniROP IR."""

from .flags import CONDITION_ALIASES, normalize_condition
from .gadget import Gadget, Instruction, Operand
from .ir import (Add, AddRegister, Address, And, Extend, Lea, Leave, Load, Mov, Neg,
                 Operation, Or, Pop, Push, Ret, Shl, Shr, Store, Sub, SubRegister, Xchg, Xor, Inc, Dec, Not, Cmp, Test, Setcc, Cmovcc, Jcc, Multiply, Adc, Sbb, Clc, Stc, Cmc, Sar, IntegerOp)
from .symbolic import WORD_MASK
from .state import GPR_NAMES, REGISTER_VIEWS


class UnsupportedInstruction(ValueError):
    def __init__(self, instruction: Instruction, reason: str):
        self.instruction = instruction
        self.reason = reason
        location = f"0x{instruction.address:x}" if type(instruction.address) is int else repr(instruction.address)
        super().__init__(f"{location}: {instruction}: {reason}")


def _register(operand: Operand) -> str | None:
    view = REGISTER_VIEWS.get(operand.register)
    if operand.kind == "register" and view is not None and view[0] in GPR_NAMES and operand.width == view[1]:
        return operand.register
    return None


def _address(operand: Operand, instruction: Instruction) -> Address | None:
    if (operand.kind != "memory" or operand.width not in (8, 16, 32, 64)
            or operand.segment is not None or type(operand.displacement) is not int
            or type(operand.scale) is not int or operand.scale not in (1, 2, 4, 8)):
        return None
    if operand.base == "rip":
        if operand.index is not None or operand.scale != 1 or not -(1 << 31) <= operand.displacement < (1 << 31):
            return None
        if (type(instruction.address) is not int or not 0 <= instruction.address <= WORD_MASK
                or type(instruction.size) is not int or not 1 <= instruction.size <= 15
                or instruction.size != len(instruction.bytes)):
            raise UnsupportedInstruction(instruction, "RIP-relative requires valid instruction address and size metadata")
        return Address("rip", displacement=operand.displacement,
                       next_rip=(instruction.address + instruction.size) & WORD_MASK)
    if ((operand.base is None or operand.base in GPR_NAMES)
            and (operand.index is None or operand.index in GPR_NAMES and operand.index != "rsp")
            and (operand.base is not None or operand.index is not None)):
        return Address(operand.base, operand.index, operand.scale, operand.displacement)
    return None


def _encoded_prefixes(data: bytes) -> tuple[bytes, int | None]:
    rex = None
    while data:
        if 0x40 <= data[0] <= 0x4f:
            rex = data[0]
        elif data[0] in (0x66, 0x67, 0xf0, 0xf2, 0xf3, 0x2e, 0x36, 0x3e, 0x26, 0x64, 0x65):
            rex = None  # A legacy prefix after REX makes that REX ineffective.
        else:
            break
        data = data[1:]
    return data, rex


def _encoded_immediate(instruction: Instruction, opcode: int, byte_width: int) -> int | None:
    data, _ = _encoded_prefixes(instruction.bytes)
    if not data or data[0] != opcode or len(data) < byte_width + 1:
        return None
    return int.from_bytes(data[-byte_width:], "little", signed=True)


def _comparison_immediate(instruction: Instruction, width: int) -> int | None:
    data, _ = _encoded_prefixes(instruction.bytes)
    if not data:
        return None
    opcode = data[0]
    if instruction.mnemonic == "cmp":
        if opcode in (0x80, 0x81, 0x83):
            if len(data) < 2 or (data[1] >> 3) & 7 != 7:
                return None
        elif opcode not in (0x3c, 0x3d):
            return None
        byte_width = 1 if opcode in (0x80, 0x83, 0x3c) else min(width, 32) // 8
    else:
        if opcode in (0xf6, 0xf7):
            if len(data) < 2 or (data[1] >> 3) & 7 != 0:
                return None
        elif opcode not in (0xa8, 0xa9):
            return None
        byte_width = 1 if opcode in (0xf6, 0xa8) else min(width, 32) // 8
    return _encoded_immediate(instruction, opcode, byte_width)


def _modrm_end(data: bytes, offset: int = 1) -> int | None:
    """End of ModRM/SIB/displacement, used to validate the encoded immediate size."""
    if len(data) <= offset:
        return None
    mod, rm = data[offset] >> 6, data[offset] & 7
    end = offset + 1
    if mod != 3 and rm == 4:
        if len(data) <= end:
            return None
        base = data[end] & 7
        end += 1
        if mod == 0 and base == 5:
            end += 4
    elif mod == 0 and rm == 5:
        end += 4
    end += 1 if mod == 1 else 4 if mod == 2 else 0
    return end if end <= len(data) else None


def _encoded_width(instruction, data, rex, *, byte=False):
    if byte:
        return 8
    if rex is not None and rex & 8:
        return 64
    prefixes = instruction.bytes[:len(instruction.bytes) - len(data)]
    return 16 if 0x66 in prefixes else 32


def _lift_integer_arithmetic(instruction: Instruction) -> Operation | None:
    operands, mnemonic = instruction.operands, instruction.mnemonic
    data, rex = _encoded_prefixes(instruction.bytes)
    if not data:
        return None
    if not operands and mnemonic in ("clc", "stc", "cmc"):
        opcode, operation = {"clc": (0xf8, Clc), "stc": (0xf9, Stc), "cmc": (0xf5, Cmc)}[mnemonic]
        return operation() if data == bytes([opcode]) else None
    if mnemonic in ("mul", "imul") and len(operands) == 1:
        source = operands[0]
        value = _register(source) or _address(source, instruction)
        width = _encoded_width(instruction, data, rex, byte=data[0] == 0xf6)
        if (value is not None and source.width == width and data[0] in (0xf6, 0xf7)
                and _modrm_end(data) == len(data) and (data[1] >> 3) & 7 == (5 if mnemonic == "imul" else 4)
                and (data[1] >> 6 == 3) == isinstance(value, str)):
            return Multiply(value, width, signed=mnemonic == "imul")
    if mnemonic == "imul" and len(operands) in (2, 3):
        destination, source = operands[:2]
        target, value = _register(destination), _register(source) or _address(source, instruction)
        width = _encoded_width(instruction, data, rex)
        if target is None or value is None or destination.width != width or source.width != width:
            return None
        if len(operands) == 2:
            if (data[:2] == b"\x0f\xaf" and _modrm_end(data, 2) == len(data)
                    and (data[2] >> 6 == 3) == isinstance(value, str)):
                return Multiply(value, width, signed=True, dst=target)
        elif data[0] in (0x69, 0x6b):
            immediate = operands[2]
            byte_width = 1 if data[0] == 0x6b else min(width, 32) // 8
            end = _modrm_end(data)
            if (end is not None and len(data) == end + byte_width
                    and immediate.kind == "immediate" and immediate.width == width
                    and type(immediate.immediate) is int
                    and (data[1] >> 6 == 3) == isinstance(value, str)):
                encoded = int.from_bytes(data[end:], "little", signed=True)
                mask = (1 << width) - 1
                if encoded & mask == immediate.immediate & mask:
                    return Multiply(value, width, signed=True, dst=target, immediate=encoded)
    if mnemonic in ("add", "or", "adc", "sbb", "and", "sub", "xor") and len(operands) == 2:
        destination, source = operands
        left = _register(destination) or _address(destination, instruction)
        right = _register(source) or _address(source, instruction)
        width = destination.width
        if left is None or width not in (8, 16, 32, 64) or source.width != width:
            return None
        opcode, base = data[0], {"add":0, "or":8, "adc":16, "sbb":24, "and":32, "sub":40, "xor":48}[mnemonic]
        if source.kind == "immediate" and type(source.immediate) is int:
            if opcode in (0x80, 0x81, 0x83):
                end = _modrm_end(data)
                group = base // 8
                if end is None or (data[1] >> 3) & 7 != group:
                    return None
                if (data[1] >> 6 == 3) != isinstance(left, str):
                    return None
                byte_width = 1 if opcode in (0x80, 0x83) else min(width, 32) // 8
            elif opcode in (base + 4, base + 5):
                end = 1
                if left != {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]:
                    return None
                byte_width = 1 if opcode == base + 4 else min(width, 32) // 8
            else:
                return None
            encoded_width = _encoded_width(instruction, data, rex, byte=opcode in (0x80, base + 4))
            if width != encoded_width or len(data) != end + byte_width or opcode == 0x83 and width == 8:
                return None
            encoded = int.from_bytes(data[end:], "little", signed=True)
            if encoded & ((1 << width) - 1) != source.immediate & ((1 << width) - 1):
                return None
            right = encoded
        else:
            if opcode not in range(base, base + 4) or _modrm_end(data) != len(data):
                return None
            if width != _encoded_width(instruction, data, rex, byte=opcode in (base, base + 2)):
                return None
            if opcode in (base, base + 1) and not isinstance(right, str):
                return None
            if opcode in (base + 2, base + 3) and not isinstance(left, str):
                return None
            rm_value = left if opcode in (base, base + 1) else right
            if (data[1] >> 6 == 3) != isinstance(rm_value, str):
                return None
        if right is not None and not (isinstance(left, Address) and isinstance(right, Address)):
            if mnemonic in ("adc", "sbb"):
                return (Adc if mnemonic == "adc" else Sbb)(left, right, width)
            return IntegerOp(mnemonic, left, right, width)
    if mnemonic == "sar" and len(operands) == 2:
        destination, count = operands
        target = _register(destination)
        width = _encoded_width(instruction, data, rex, byte=data[0] in (0xc0, 0xd0))
        if (target is not None and destination.width == width and count.kind == "immediate"
                and type(count.immediate) is int and len(data) >= 2 and data[1] >> 6 == 3
                and (data[1] >> 3) & 7 == 7):
            if data[0] in (0xd0, 0xd1) and len(data) == 2 and count.immediate == 1:
                return Sar(target, 1)
            if (data[0] in (0xc0, 0xc1) and len(data) == 3 and count.width == 8
                    and count.immediate == data[2]):
                return Sar(target, count.immediate & (63 if width == 64 else 31))
    return None



def _lift_remaining_integer(instruction):
    operands, name = instruction.operands, instruction.mnemonic
    data, rex = _encoded_prefixes(instruction.bytes)
    if not data:
        return None
    width = _encoded_width(instruction, data, rex)
    implicit = {"cbw": (0x98, 16), "cwde": (0x98, 32), "cdqe": (0x98, 64),
                "cwd": (0x99, 16), "cdq": (0x99, 32), "cqo": (0x99, 64)}
    if name in implicit and not operands and data == bytes([implicit[name][0]]) and width == implicit[name][1]:
        return IntegerOp(name, width=width)
    if name in ("div", "idiv") and len(operands) == 1:
        source = _register(operands[0]) or _address(operands[0], instruction)
        width = _encoded_width(instruction, data, rex, byte=data[0] == 0xf6)
        if (source is not None and operands[0].width == width and data[0] in (0xf6, 0xf7)
                and _modrm_end(data) == len(data) and (data[1] >> 3) & 7 == (6 if name == "div" else 7)
                and (data[1] >> 6 == 3) == isinstance(source, str)):
            return IntegerOp(name, src=source, width=width)
    if name in ("shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr") and len(operands) == 2:
        destination, count = operands
        target = _register(destination) or _address(destination, instruction)
        group = {"rol":0, "ror":1, "rcl":2, "rcr":3, "shl":4, "sal":4, "shr":5, "sar":7}[name]
        width = _encoded_width(instruction, data, rex, byte=data[0] in (0xc0, 0xd0, 0xd2))
        end = _modrm_end(data)
        if (target is None or destination.width != width or end is None or (data[1] >> 3) & 7 != group
                or (data[1] >> 6 == 3) != isinstance(target, str)):
            return None
        source = None
        if data[0] in (0xc0, 0xc1) and len(data) == end + 1 and count.kind == "immediate" and count.width == 8 and count.immediate == data[-1]:
            source = count.immediate
        elif data[0] in (0xd0, 0xd1) and len(data) == end and count.kind == "immediate" and count.immediate == 1:
            source = 1
        elif data[0] in (0xd2, 0xd3) and len(data) == end and _register(count) == "cl":
            source = "cl"
        if source is not None:
            if name in ("shl", "sal", "shr", "sar") and isinstance(target, str) and type(source) is int:
                return {"shl":Shl, "sal":Shl, "shr":Shr, "sar":Sar}[name](target, source & (63 if width == 64 else 31))
            return IntegerOp("shl" if name == "sal" else name, target, source, width)
    if name == "bswap" and len(operands) == 1:
        target = _register(operands[0])
        if target is not None and width in (32, 64) and operands[0].width == width and len(data) == 2 and data[0] == 0x0f and 0xc8 <= data[1] <= 0xcf:
            return IntegerOp(name, target, width=width)
    if len(operands) != 2:
        return None
    destination, source = operands
    left = _register(destination) or _address(destination, instruction)
    right = _register(source) or _address(source, instruction)
    if name in ("bsf", "bsr", "popcnt", "xadd", "cmpxchg", "bt", "bts", "btr", "btc"):
        byte = name in ("xadd", "cmpxchg") and data[:2] in (b"\x0f\xc0", b"\x0f\xb0")
        width = _encoded_width(instruction, data, rex, byte=byte)
        if left is None or destination.width != width or len(data) < 3:
            return None
        if name in ("bsf", "bsr", "popcnt"):
            opcode = {"bsf":0xbc, "bsr":0xbd, "popcnt":0xb8}[name]
            if (not isinstance(left, str) or right is None or width == 8 or source.width != width
                    or data[:2] != bytes([0x0f, opcode]) or _modrm_end(data, 2) != len(data)
                    or (data[2] >> 6 == 3) != isinstance(right, str)
                    or name == "popcnt" and 0xf3 not in instruction.prefixes):
                return None
        elif name in ("xadd", "cmpxchg"):
            opcode = (0xc0 if name == "xadd" else 0xb0) + int(width != 8)
            if (not isinstance(right, str) or source.width != width or data[:2] != bytes([0x0f, opcode])
                    or _modrm_end(data, 2) != len(data) or (data[2] >> 6 == 3) != isinstance(left, str)):
                return None
        else:
            if width == 8:
                return None
            if source.kind == "immediate":
                end = _modrm_end(data, 2)
                group = {"bt":4, "bts":5, "btr":6, "btc":7}[name]
                if (data[:2] != b"\x0f\xba" or end is None or len(data) != end + 1
                        or (data[2] >> 3) & 7 != group or source.width != 8 or source.immediate != data[-1]):
                    return None
                right = source.immediate
            else:
                opcode = {"bt":0xa3, "bts":0xab, "btr":0xb3, "btc":0xbb}[name]
                if (not isinstance(right, str) or source.width != width or data[:2] != bytes([0x0f, opcode])
                        or _modrm_end(data, 2) != len(data)):
                    return None
            if (data[2] >> 6 == 3) != isinstance(left, str):
                return None
        return IntegerOp(name, left, right, width)
    return None

_JCC_CONDITIONS = ("o", "no", "b", "ae", "e", "ne", "be", "a",
                   "s", "ns", "p", "np", "l", "ge", "le", "g")


def _lift_jcc(instruction: Instruction) -> Jcc:
    if (type(instruction.address) is not int or not 0 <= instruction.address <= WORD_MASK
            or type(instruction.size) is not int or not 1 <= instruction.size <= 15
            or not isinstance(instruction.bytes, bytes) or instruction.size != len(instruction.bytes)):
        raise UnsupportedInstruction(instruction, "Jcc requires valid instruction address and size metadata")
    operands = instruction.operands
    if (len(operands) != 1 or operands[0].kind != "immediate"
            or operands[0].width != 64 or type(operands[0].immediate) is not int):
        raise UnsupportedInstruction(instruction, "Jcc requires a direct relative target")
    data, _ = _encoded_prefixes(instruction.bytes)
    encoded_prefixes = instruction.bytes[:len(instruction.bytes) - len(data)]
    if any(not 0x40 <= byte <= 0x4f for byte in encoded_prefixes):
        raise UnsupportedInstruction(instruction, "unsupported Jcc prefix encoding")
    if len(data) == 2 and 0x70 <= data[0] <= 0x7f:
        code, displacement = data[0] & 15, data[1:]
    elif len(data) == 6 and data[0] == 0x0f and 0x80 <= data[1] <= 0x8f:
        code, displacement = data[1] & 15, data[2:]
    else:
        raise UnsupportedInstruction(instruction, "Jcc requires rel8 or rel32 encoding")
    condition = normalize_condition(instruction.mnemonic[1:])
    if condition != _JCC_CONDITIONS[code]:
        raise UnsupportedInstruction(instruction, "Jcc condition does not match encoding")
    fallthrough = (instruction.address + instruction.size) & WORD_MASK
    target = (fallthrough + int.from_bytes(displacement, "little", signed=True)) & WORD_MASK
    if target != (operands[0].immediate & WORD_MASK):
        raise UnsupportedInstruction(instruction, "Jcc target does not match relative displacement")
    return Jcc(condition, target, fallthrough)


def lift_instruction(instruction: Instruction) -> Operation:
    operands = instruction.operands
    # Only operand-size 66 is admitted, for actual 16-bit forms. Segment,
    # address-size, REP/LOCK and redundant unsupported forms remain rejected.
    implicit16 = instruction.mnemonic in ("cbw", "cwd")
    popcnt_prefix = (instruction.mnemonic == "popcnt" and instruction.prefixes in ((0xf3,), (0x66, 0xf3), (0xf3, 0x66)))
    if instruction.prefixes and not popcnt_prefix and (instruction.prefixes != (0x66,) or not implicit16 and (not operands or operands[0].width != 16)):
        raise UnsupportedInstruction(instruction, "unsupported legacy prefixes")
    _, rex = _encoded_prefixes(instruction.bytes)
    if rex is not None and any(op.register in ("ah", "bh", "ch", "dh") for op in operands):
        raise UnsupportedInstruction(instruction, "high-byte registers cannot be encoded with REX")
    if rex is None:
        names = [name for op in operands for name in (op.register, op.base, op.index) if name is not None]
        if any(name in ("spl", "bpl", "sil", "dil") or
               name in REGISTER_VIEWS and REGISTER_VIEWS[name][0] in GPR_NAMES[8:] for name in names):
            raise UnsupportedInstruction(instruction, "new byte/extended registers require REX")
    remaining = {"div", "idiv", "shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr",
                 "cbw", "cwde", "cdqe", "cwd", "cdq", "cqo", "bswap", "bsf", "bsr",
                 "bt", "bts", "btr", "btc", "xadd", "cmpxchg", "popcnt"}
    if instruction.mnemonic in remaining:
        operation = _lift_remaining_integer(instruction)
        if operation is not None:
            return operation
        raise UnsupportedInstruction(instruction, "unsupported integer encoding, operand form or width")
    if instruction.mnemonic in ("add", "sub", "and", "or", "xor") and any(op.kind == "memory" for op in operands):
        operation = _lift_integer_arithmetic(instruction)
        if operation is not None:
            return operation
        raise UnsupportedInstruction(instruction, "unsupported memory arithmetic form")
    if instruction.mnemonic in ("inc", "dec", "neg", "not") and len(operands) == 1 and operands[0].kind == "memory":
        data, rex = _encoded_prefixes(instruction.bytes)
        name = instruction.mnemonic
        target = _address(operands[0], instruction)
        byte = data[0] in (0xfe, 0xf6)
        width = _encoded_width(instruction, data, rex, byte=byte)
        group = {"inc":0, "dec":1, "not":2, "neg":3}[name]
        opcode = (0xfe if byte else 0xff) if name in ("inc", "dec") else (0xf6 if byte else 0xf7)
        if target is not None and operands[0].width == width and data[0] == opcode and _modrm_end(data) == len(data) and (data[1] >> 3) & 7 == group and data[1] >> 6 != 3:
            return IntegerOp(name, target, width=width)
        raise UnsupportedInstruction(instruction, "unsupported unary memory encoding")
    if instruction.mnemonic in ("mul", "imul", "adc", "sbb", "clc", "stc", "cmc", "sar"):
        operation = _lift_integer_arithmetic(instruction)
        if operation is not None:
            return operation
        raise UnsupportedInstruction(instruction, "unsupported integer arithmetic encoding, operand form or width")
    if instruction.mnemonic.startswith("j") and instruction.mnemonic[1:] in CONDITION_ALIASES:
        return _lift_jcc(instruction)
    if instruction.mnemonic == "ret" and not operands and instruction.bytes == b"\xc3":
        return Ret()
    if instruction.mnemonic == "pop" and len(operands) == 1:
        register = _register(operands[0])
        if register is not None and operands[0].width == 64:
            return Pop(register)
    if instruction.mnemonic == "leave" and not operands and instruction.bytes == b"\xc9":
        return Leave()
    if instruction.mnemonic == "push" and len(operands) == 1:
        source = operands[0]
        register = _register(source)
        if register is not None and source.width == 64:
            return Push(register)
        if source.kind == "immediate" and source.width == 64 and type(source.immediate) is int:
            for opcode, byte_width in ((0x6a, 1), (0x68, 4)):
                value = _encoded_immediate(instruction, opcode, byte_width)
                if value is not None and (value & WORD_MASK) == (source.immediate & WORD_MASK):
                    return Push(value)
    if (instruction.mnemonic.startswith("set") and len(operands) == 1
            and instruction.mnemonic[3:] in CONDITION_ALIASES and operands[0].width == 8):
        destination = _register(operands[0]) or _address(operands[0], instruction)
        if destination is not None:
            return Setcc(normalize_condition(instruction.mnemonic[3:]), destination)
    if len(operands) == 1:
        register = _register(operands[0])
        if register is not None:
            if instruction.mnemonic == "inc":
                return Inc(register)
            if instruction.mnemonic == "dec":
                return Dec(register)
            if instruction.mnemonic == "neg":
                return Neg(register)
            if instruction.mnemonic == "not":
                return Not(register)
    if len(operands) == 2:
        dst, source = operands
        register, source_register = _register(dst), _register(source)
        mnemonic = instruction.mnemonic
        if mnemonic.startswith("cmov") and mnemonic[4:] in CONDITION_ALIASES:
            if (register is not None and source_register is not None
                    and dst.width == source.width and dst.width in (16, 32, 64)):
                return Cmovcc(normalize_condition(mnemonic[4:]), register, source_register)
        if mnemonic in ("cmp", "test") and dst.width in (8, 16, 32, 64) and source.width == dst.width:
            left = register if register is not None else _address(dst, instruction)
            right = source_register if source_register is not None else _address(source, instruction)
            if source.kind == "immediate" and type(source.immediate) is int:
                encoded = _comparison_immediate(instruction, dst.width)
                mask = (1 << dst.width) - 1
                if encoded is not None and (encoded & mask) == (source.immediate & mask):
                    right = encoded
            # No memory/memory form; TEST's memory operand must be on the left.
            if (left is not None and right is not None
                    and not (isinstance(left, Address) and isinstance(right, Address))
                    and not (mnemonic == "test" and isinstance(right, Address))):
                return (Cmp if mnemonic == "cmp" else Test)(left, right, dst.width)
        if mnemonic in ("movzx", "movsx", "movsxd") and register is not None:
            widths = (source.width, dst.width)
            if mnemonic == "movsxd":
                legal = widths == (32, 64) and rex is not None and bool(rex & 8)
            else:
                legal = widths in ((8, 16), (8, 32), (8, 64), (16, 32), (16, 64))
            if legal:
                value = source_register if source_register is not None else _address(source, instruction)
                if value is not None:
                    return Extend(register, value, source.width, mnemonic != "movzx")
        if mnemonic == "mov":
            address = _address(dst, instruction)
            if address is not None:
                if source_register is not None and source.width == dst.width:
                    return Store(address, source_register, dst.width)
                if source.kind == "immediate" and source.width == dst.width and type(source.immediate) is int:
                    encoded_width = min(dst.width, 32) // 8
                    value = _encoded_immediate(instruction, 0xc6 if dst.width == 8 else 0xc7, encoded_width)
                    mask = (1 << dst.width) - 1
                    if value is not None and (value & mask) == (source.immediate & mask):
                        return Store(address, value, dst.width)
        if register is not None:
            immediate = (source.immediate if source.kind == "immediate" and source.width == dst.width
                         and type(source.immediate) is int else None)
            matched_source = source_register if source.width == dst.width else None
            value = matched_source if matched_source is not None else immediate
            if mnemonic in ("mov", "movabs") and value is not None:
                return Mov(register, value)
            if mnemonic in ("add", "sub"):
                if matched_source is not None:
                    return (AddRegister if mnemonic == "add" else SubRegister)(register, matched_source)
                if immediate is not None:
                    return (Add if mnemonic == "add" else Sub)(register, immediate)
            if mnemonic in ("xor", "and", "or") and value is not None:
                return {"xor": Xor, "and": And, "or": Or}[mnemonic](register, value)
            if mnemonic == "xchg" and matched_source is not None:
                return Xchg(register, matched_source)
            if (mnemonic in ("shl", "sal", "shr") and source.kind == "immediate"
                    and source.width == 8 and type(source.immediate) is int):
                return (Shr if mnemonic == "shr" else Shl)(register, source.immediate & (63 if dst.width == 64 else 31))
            address = _address(source, instruction)
            if address is not None:
                if mnemonic == "lea" and dst.width in (16, 32, 64) and source.width == dst.width:
                    return Lea(register, address)
                if mnemonic == "mov" and source.width == dst.width:
                    return Load(register, address, source.width)
    raise UnsupportedInstruction(instruction, "unsupported opcode, operand form or width")


def lift_gadget(gadget: Gadget) -> tuple[Operation, ...]:
    if not gadget.instructions:
        raise ValueError("empty gadget")
    cursor = gadget.address
    for instruction in gadget.instructions:
        if instruction.address != cursor or instruction.size != len(instruction.bytes):
            raise ValueError("inconsistent gadget instruction layout")
        cursor += instruction.size
    if b"".join(i.bytes for i in gadget.instructions) != gadget.bytes:
        raise ValueError("gadget bytes do not match instruction bytes")
    operations = tuple(lift_instruction(i) for i in gadget.instructions)
    for instruction, operation in zip(gadget.instructions, operations):
        if isinstance(operation, Jcc):
            raise UnsupportedInstruction(instruction, "Jcc requires fixed-program execution")
    if not isinstance(operations[-1], Ret) or any(isinstance(op, Ret) for op in operations[:-1]):
        raise ValueError("gadget must end with exactly one return")
    return operations
