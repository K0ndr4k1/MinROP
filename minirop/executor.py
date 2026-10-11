"""Apply IR to a copy of state; no Capstone dependency or machine execution."""

from collections.abc import Iterable

from .gadget import Gadget
from .ir import (Add, AddRegister, Address, And, Extend, Lea, Leave, Load, Mov, Neg,
                 Operation, Or, Pop, Push, Ret, Shl, Shr, Store, Sub, SubRegister, Xchg, Xor, Inc, Dec, Not, Cmp, Test, Setcc, Cmovcc, Multiply, Adc, Sbb, Clc, Stc, Cmc, Sar, IntegerOp)
from .flags import (compute_add_flags, compute_sub_flags, compute_shift_flags,
                    compute_logic_flags, compute_carry_arithmetic, compute_multiply_flags,
                    UNDEFINED, UndefinedFlag, ConditionalFlag, flag_parts, conditional_flag, merge_count_flags, compute_result_flags, evaluate_condition, normalize_condition)
from .semantics import lift_gadget
from .state import CPUState, GPR_NAMES, InvalidRegister, Value, register_view
from .symbolic import (Expression, NonConcreteValue, add_values, as_expression,
                       sub_values, xor_values, and_values, or_values, shl_values, shr_values,
                       check_width, sign_extend, zero_extend, WORD_MASK, select_value, extract_value, multiply_values, sar_values, bool_not, bool_binary, compare_values, integer_value, GuardedValue, Constant, BoolConstant)


def _concrete(value: object) -> int:
    if type(value) is not int:
        raise NonConcreteValue("arithmetic and memory addresses require concrete integers")
    return value


def _address_value(value: object) -> int | Expression:
    if type(value) is int:
        return value
    return as_expression(value)


def _validate_address(address: Address) -> list[str]:
    if not isinstance(address, Address):
        raise TypeError("IR address required")
    if type(address.scale) is not int or address.scale not in (1, 2, 4, 8):
        raise ValueError("address scale must be 1, 2, 4 or 8")
    _concrete(address.displacement)
    if address.base == "rip":
        if (type(address.next_rip) is not int or not 0 <= address.next_rip <= WORD_MASK
                or address.index is not None or address.scale != 1
                or not -(1 << 31) <= address.displacement < (1 << 31)):
            raise ValueError("RIP-relative requires next_rip metadata and a signed disp32 without index")
        return []
    if address.next_rip is not None:
        raise ValueError("next_rip metadata requires RIP-relative addressing")
    if address.base is None and address.index is None:
        raise ValueError("base or index required")
    names = [name for name in (address.base, address.index) if name is not None]
    if any(name not in GPR_NAMES for name in names):
        raise InvalidRegister("address registers must be full 64-bit GPRs")
    if address.index == "rsp":
        raise InvalidRegister("RSP cannot be a SIB index")
    return names


def _validate_operation(op: Operation) -> None:
    registers = []
    if isinstance(op, IntegerOp):
        legal = {"div", "idiv", "shl", "shr", "sar", "rol", "ror", "rcl", "rcr",
                 "cbw", "cwde", "cdqe", "cwd", "cdq", "cqo", "bswap", "bsf", "bsr",
                 "bt", "bts", "btr", "btc", "xadd", "cmpxchg", "popcnt",
                 "add", "sub", "and", "or", "xor", "inc", "dec", "neg", "not"}
        check_width(op.width)
        if op.mnemonic not in legal:
            raise ValueError("unsupported integer IR")
        implicit = {"cbw":16, "cwde":32, "cdqe":64, "cwd":16, "cdq":32, "cqo":64}
        if op.mnemonic in implicit:
            if op.dst is not None or op.src is not None or op.width != implicit[op.mnemonic]:
                raise ValueError("invalid implicit integer form")
        elif op.mnemonic in ("div", "idiv"):
            if op.dst is not None or not isinstance(op.src, (str, Address)):
                raise ValueError("division requires one register/memory source")
        else:
            if not isinstance(op.dst, (str, Address)):
                raise ValueError("integer destination required")
            if op.mnemonic in ("bswap", "bsf", "bsr", "popcnt") and not isinstance(op.dst, str):
                raise ValueError("register destination required")
            if op.mnemonic == "bswap" and op.width not in (32, 64):
                raise ValueError("BSWAP requires 32/64 bits")
            if op.mnemonic in ("bsf", "bsr", "popcnt", "bt", "bts", "btr", "btc") and op.width == 8:
                raise ValueError("unsupported byte width")
            if op.mnemonic in ("xadd", "cmpxchg") and not isinstance(op.src, str):
                raise ValueError("register source required")
            if op.mnemonic in ("bsf", "bsr", "popcnt") and not isinstance(op.src, (str, Address)):
                raise ValueError("bit scan/count requires register or memory source")
            if op.mnemonic in ("bt", "bts", "btr", "btc") and not (isinstance(op.src, str) or type(op.src) is int and 0 <= op.src <= 255):
                raise ValueError("bit index requires a register or imm8")
            if op.mnemonic in ("shl", "shr", "sar", "rol", "ror", "rcl", "rcr") and not (type(op.src) is int or op.src == "cl"):
                raise ValueError("immediate or CL count required")
            if op.mnemonic in ("shl", "shr", "sar", "rol", "ror", "rcl", "rcr") and type(op.src) is int and not 0 <= op.src <= 255:
                raise ValueError("encoded shift/rotate immediate must fit imm8")
            if op.mnemonic in ("add", "sub", "and", "or", "xor") and type(op.src) is int:
                bits = min(op.width, 32)
                maximum = (1 << (bits - 1)) - 1 if op.width == 64 else (1 << bits) - 1
                if not -(1 << (bits - 1)) <= op.src <= maximum:
                    raise ValueError("arithmetic immediate exceeds a legal imm8/imm16/imm32 encoding")
            unary = op.mnemonic in ("bswap", "inc", "dec", "neg", "not")
            if unary != (op.src is None):
                raise ValueError("wrong integer operand count")
        if isinstance(op.dst, Address) and isinstance(op.src, Address):
            raise ValueError("memory/memory instruction unsupported")
        for operand in (op.dst, op.src):
            if isinstance(operand, str):
                registers.append(operand)
                if operand != "cl" or op.mnemonic not in ("shl", "shr", "sar", "rol", "ror", "rcl", "rcr"):
                    if register_view(operand)[1] != op.width:
                        raise ValueError("integer operand widths differ")
            elif isinstance(operand, Address):
                registers.extend(_validate_address(operand))
            elif operand is not None and type(operand) is not int:
                raise TypeError("invalid integer operand")
    elif isinstance(op, Pop):
        registers = [op.register]
    elif isinstance(op, (Mov, Xor, AddRegister, SubRegister, And, Or, Xchg, Shl, Shr, Sar)):
        registers = [op.dst]
        if isinstance(op.src, str):
            registers.append(op.src)
        elif type(op.src) is not int:
            raise TypeError("invalid IR source operand")
        if isinstance(op, (AddRegister, SubRegister, Xchg)) and not isinstance(op.src, str):
            raise TypeError("register source required")
        if isinstance(op, (Shl, Shr, Sar)) and type(op.src) is not int:
            raise TypeError("immediate shift count required")
    elif isinstance(op, (Neg, Inc, Dec, Not)):
        registers = [op.dst]
    elif isinstance(op, (Cmp, Test, Adc, Sbb)):
        check_width(op.width)
        if not isinstance(op.left, (str, Address)):
            raise TypeError("comparison lhs must be a register or memory")
        if isinstance(op.left, Address) and isinstance(op.right, Address):
            raise ValueError("memory/memory comparison unsupported")
        if isinstance(op, Test) and isinstance(op.right, Address):
            raise ValueError("TEST memory operand must be on the left")
        for operand in (op.left, op.right):
            if isinstance(operand, str):
                registers.append(operand)
                if register_view(operand)[1] != op.width:
                    raise ValueError("comparison operand widths differ")
            elif isinstance(operand, Address):
                registers.extend(_validate_address(operand))
            elif type(operand) is not int:
                raise TypeError("invalid comparison operand")
    elif isinstance(op, Multiply):
        check_width(op.width)
        if type(op.signed) is not bool:
            raise TypeError("multiply signed selector must be Boolean")
        if isinstance(op.src, str):
            registers.append(op.src)
            if register_view(op.src)[1] != op.width:
                raise ValueError("multiply source widths differ")
        elif isinstance(op.src, Address):
            registers.extend(_validate_address(op.src))
        else:
            raise TypeError("multiply requires register or memory source")
        if op.dst is not None:
            registers.append(op.dst)
            if not op.signed or op.width == 8 or register_view(op.dst)[1] != op.width:
                raise ValueError("explicit IMUL requires matching 16/32/64-bit destination")
        if op.immediate is not None:
            if type(op.immediate) is not int or op.dst is None:
                raise TypeError("IMUL immediate requires an explicit destination and integer value")
            immediate_bits = min(op.width, 32)
            if not -(1 << (immediate_bits - 1)) <= op.immediate < (1 << (immediate_bits - 1)):
                raise ValueError("IMUL immediate must fit its signed imm16/imm32 encoding")
    elif isinstance(op, Setcc):
        normalize_condition(op.condition)
        if isinstance(op.dst, str):
            registers = [op.dst]
            if register_view(op.dst)[1] != 8:
                raise ValueError("SETcc requires an 8-bit destination")
        elif isinstance(op.dst, Address):
            registers = _validate_address(op.dst)
        else:
            raise TypeError("invalid SETcc destination")
    elif isinstance(op, Cmovcc):
        normalize_condition(op.condition)
        registers = [op.dst, op.src]
        width = register_view(op.dst)[1]
        if width not in (16, 32, 64) or register_view(op.src)[1] != width:
            raise ValueError("CMOVcc requires matching 16/32/64-bit registers")
    elif isinstance(op, Extend):
        registers = [op.dst]
        if isinstance(op.src, str):
            registers.append(op.src)
            if register_view(op.src)[1] != op.source_width:
                raise ValueError("extension source width differs from register width")
        elif isinstance(op.src, Address):
            registers.extend(_validate_address(op.src))
        else:
            raise TypeError("extension requires a register or memory source")
        check_width(op.source_width)
        if type(op.signed) is not bool:
            raise TypeError("extension signed flag must be bool")
        widths = (op.source_width, register_view(op.dst)[1])
        if (widths not in ((8, 16), (8, 32), (8, 64), (16, 32), (16, 64))
                and not (op.signed and widths == (32, 64))):
            raise ValueError("unsupported extension width combination")
    elif isinstance(op, (Lea, Load, Store)):
        registers = _validate_address(op.address)
        if isinstance(op, Store):
            if isinstance(op.src, str):
                registers.append(op.src)
            elif type(op.src) is not int:
                raise TypeError("invalid store source")
        else:
            registers.append(op.dst)
        if isinstance(op, (Load, Store)):
            check_width(op.width)
            name = op.dst if isinstance(op, Load) else op.src
            if isinstance(name, str) and register_view(name)[1] != op.width:
                raise ValueError("memory and register widths differ")
    elif isinstance(op, Push):
        registers = ["rsp"]
        if isinstance(op.src, str):
            registers.append(op.src)
        elif type(op.src) is not int:
            raise TypeError("invalid push source")
    elif isinstance(op, (Add, Sub)):
        registers = [op.dst]
        _concrete(op.immediate)
    elif not isinstance(op, (Ret, Leave, Clc, Stc, Cmc)):
        raise TypeError(f"unsupported IR operation: {type(op).__name__}")
    for register in registers:
        parent, width, _ = register_view(register)
        if parent not in GPR_NAMES:
            raise InvalidRegister(f"IR requires a general register: {register}")
        if isinstance(op, (Pop, Push)) and width != 64:
            raise InvalidRegister("stack operations require 64-bit registers")
    if isinstance(op, (Mov, Xor, AddRegister, SubRegister, And, Or, Xchg)) and isinstance(op.src, str):
        if register_view(op.dst)[1] != register_view(op.src)[1]:
            raise ValueError("register widths differ")
    if isinstance(op, Lea) and register_view(op.dst)[1] == 8:
        raise InvalidRegister("LEA requires a 16/32/64-bit destination")


def _effective_address(address: Address, state: CPUState) -> int | Expression:
    if address.base == "rip":
        return _address_value(add_values(address.next_rip, address.displacement))
    value = state.get_register(address.base) if address.base is not None else 0
    if address.index is not None:
        index = state.get_register(address.index)
        for _ in range(address.scale.bit_length() - 1):
            index = add_values(index, index)
        value = add_values(value, index)
    return _address_value(add_values(value, address.displacement))


def _read_operand(operand: str | int | Address, state: CPUState, width: int):
    if isinstance(operand, str):
        return state.get_register(operand)
    if isinstance(operand, Address):
        return state.read_memory(_effective_address(operand, state), width)
    return operand


def _update_flags(state: CPUState, flags: dict) -> None:
    for name, value in flags.items():
        state.set_flag(name, value)



def _set_operand(destination, value, state, width, address=None):
    if isinstance(destination, str):
        state.set_register(destination, value)
    else:
        state.store_memory(_effective_address(destination, state) if address is None else address, value, width)


def _bit(value, bit):
    return bool_not(compare_values("eq", and_values(value, 1 << bit), 0))


def _count_update(name, original, result, count, width, old, carry=False):
    if count == 0:
        return {}
    if name in ("shl", "shr", "sar"):
        return compute_shift_flags(original, result, count, width, left=name == "shl", arithmetic=name == "sar")
    effective = count % (width + 1 if name in ("rcl", "rcr") else width)
    if name in ("rcl", "rcr"):
        cf = compare_values("eq", integer_value(name + "_cf", (original, count, select_value(carry, 1, 0)), width), 1)
        # A full carry-ring turn preserves CF, even though OF is undefined.
        cf = old.get("CF", UNDEFINED) if effective == 0 else cf
    else:
        cf = _bit(result, 0 if name == "rol" else width - 1)
    if count == 1:
        of = bool_binary("xor", _bit(result, width - 1), cf) if name in ("rol", "rcl") else bool_binary("xor", _bit(result, width - 1), _bit(result, width - 2))
    else:
        of = UNDEFINED
    return {"CF": cf, "OF": of}


def _execute_integer(op, state):
    name, width = op.mnemonic, op.width
    mask = (1 << width) - 1
    accumulator = {8:"al", 16:"ax", 32:"eax", 64:"rax"}[width]
    if name in ("cbw", "cwde", "cdqe"):
        source = {16:"al", 32:"ax", 64:"eax"}[width]
        state.set_register({16:"ax", 32:"eax", 64:"rax"}[width], sign_extend(state.get_register(source), width // 2, width))
        return
    if name in ("cwd", "cdq", "cqo"):
        high = select_value(_bit(state.get_register(accumulator), width - 1), mask, 0)
        state.set_register({16:"dx", 32:"edx", 64:"rdx"}[width], high)
        return
    if name in ("div", "idiv"):
        divisor = _read_operand(op.src, state, width)
        low = state.get_register(accumulator)
        high = state.get_register({8:"ah", 16:"dx", 32:"edx", 64:"rdx"}[width])
        values = (low, high, divisor)
        valid = compare_values("eq", integer_value("divok", values, width, signed=name == "idiv"), 1)
        state.require(valid)
        if valid is False or isinstance(valid, BoolConstant) and not valid.value:
            state.exception = "#DE: division by zero or quotient overflow"
            return
        quotient = integer_value("divq", values, width, signed=name == "idiv")
        remainder = integer_value("divr", values, width, signed=name == "idiv")
        if width == 8:
            state.set_register("ax", or_values(quotient, shl_values(remainder, 8)))
        else:
            state.set_register(accumulator, quotient)
            state.set_register({16:"dx", 32:"edx", 64:"rdx"}[width], remainder)
        _update_flags(state, {flag:UNDEFINED for flag in ("CF", "OF", "ZF", "SF", "PF", "AF")})
        return
    address = _effective_address(op.dst, state) if isinstance(op.dst, Address) else None
    if name in ("bt", "bts", "btr", "btc"):
        index = _read_operand(op.src, state, width)
        if address is not None and isinstance(op.src, str):
            signed_index = sign_extend(index, width, 64) if width < 64 else index
            word_index = integer_value("sar", (signed_index, width.bit_length() - 1), 64)
            address = add_values(address, shl_values(word_index, (width // 8).bit_length() - 1))
        original = state.read_memory(address, width) if address is not None else state.get_register(op.dst)
        bit_index = and_values(index, width - 1)
        bit_mask = integer_value("shl", (1, bit_index), width)
        state.set_flag("CF", bool_not(compare_values("eq", and_values(original, bit_mask), 0)))
        _update_flags(state, {flag:UNDEFINED for flag in ("OF", "SF", "AF", "PF")})
        if name != "bt":
            value = or_values(original, bit_mask) if name == "bts" else and_values(original, xor_values(bit_mask, mask)) if name == "btr" else xor_values(original, bit_mask)
            _set_operand(op.dst, value, state, width, address)
        return
    original = (state.read_memory(address, width) if address is not None else state.get_register(op.dst)) if name not in ("bsf", "bsr", "popcnt") else None
    source = _read_operand(op.src, state, width) if op.src is not None else None
    if name == "bswap":
        value = 0
        for byte in range(width // 8):
            value = or_values(value, shl_values(and_values(shr_values(original, 8 * byte), 255), width - 8 - 8 * byte))
        state.set_register(op.dst, value)
        return
    if name in ("bsf", "bsr", "popcnt"):
        source = extract_value(source, width)
        zero = compare_values("eq", source, 0)
        value = 0
        if name == "popcnt":
            for bit in range(width):
                value = add_values(value, and_values(shr_values(source, bit), 1))
            flags = {flag:False for flag in ("CF", "OF", "SF", "AF", "PF")}
        else:
            for bit in (reversed(range(width)) if name == "bsf" else range(width)):
                value = select_value(_bit(source, bit), bit, value)
            if zero is not False:
                value = GuardedValue(as_expression(value), bool_not(zero))
            flags = {flag:UNDEFINED for flag in ("CF", "OF", "SF", "AF", "PF")}
        state.set_register(op.dst, value)
        _update_flags(state, dict(flags, ZF=zero))
        return
    if name in ("shl", "shr", "sar", "rol", "ror", "rcl", "rcr"):
        count = and_values(source, 63 if width == 64 else 31)
        if isinstance(count, Constant):
            count = count.value
        old = dict(state.flag_bindings())
        carry, carry_defined = flag_parts(old.get("CF", UNDEFINED))
        through_carry = name in ("rcl", "rcr")
        values = (original, count, select_value(carry, 1, 0)) if through_carry else (original, count)
        value = integer_value(name, values, width)
        if through_carry:
            # No CF read is required for an effective zero count.
            nonzero = False
            for index in range(64 if width == 64 else 32):
                if index % (width + 1):
                    nonzero = bool_binary("or", nonzero, compare_values("eq", count, index))
            required = bool_binary("or", bool_not(nonzero), carry_defined)
            if required is False:
                state.get_flag("CF")  # Explicit undefined/uninitialized read error.
            state.require(required)
        if type(count) is int:
            flags = _count_update(name, original, value, count, width, old, carry)
        else:
            updates = [_count_update(name, original, integer_value(name, (original, index, select_value(carry, 1, 0)) if through_carry else (original, index), width), index, width, old, carry)
                       for index in range(64 if width == 64 else 32)]
            flags = merge_count_flags(count, updates, old)
        _set_operand(op.dst, value, state, width, address)
        _update_flags(state, flags)
        return
    if name == "xadd":
        value = add_values(original, source)
        flags = compute_add_flags(original, source, width)
        state.set_register(op.src, original)
        _set_operand(op.dst, value, state, width, address)
        _update_flags(state, flags)
        return
    if name == "cmpxchg":
        old_acc = state.get_register(accumulator)
        equal = compare_values("eq", old_acc, original)
        flags = compute_sub_flags(old_acc, original, width)
        parent = "rax"
        old_parent = state.get_register(parent)
        state.set_register(accumulator, original)
        failed = state.get_register(parent)
        state.set_register(parent, select_value(equal, old_parent, failed))
        _set_operand(op.dst, select_value(equal, source, original), state, width, address)
        _update_flags(state, flags)
        return
    if name in ("add", "sub", "inc", "dec", "neg"):
        lhs, rhs = (0, original) if name == "neg" else (original, 1 if name in ("inc", "dec") else source)
        subtract = name in ("sub", "dec", "neg")
        value = (sub_values if subtract else add_values)(lhs, rhs)
        flags = (compute_sub_flags if subtract else compute_add_flags)(lhs, rhs, width)
        if name in ("inc", "dec"):
            flags.pop("CF")
    else:
        value = xor_values(original, mask) if name == "not" else {"and":and_values, "or":or_values, "xor":xor_values}[name](original, source)
        flags = {} if name == "not" else compute_logic_flags(value, width)
    _set_operand(op.dst, value, state, width, address)
    _update_flags(state, flags)


def execute(operations: Iterable[Operation], state: CPUState[Value]) -> CPUState[Value]:
    """RIP changes only on Ret; other ops have no instruction-fetch semantics."""
    operations = tuple(operations)
    if any(isinstance(op, Ret) for op in operations[:-1]):
        raise ValueError("operations after return cannot be executed")
    for op in operations:
        _validate_operation(op)
    result = state.copy()
    for op in operations:
        if result.exception is not None:
            break
        if isinstance(op, IntegerOp):
            _execute_integer(op, result)
        elif isinstance(op, (Pop, Ret)):
            rsp = _address_value(result.get_register("rsp"))
            value = result.read_memory(rsp)
            # POP RSP writes the popped value AFTER the stack increment.
            result.set_register("rsp", add_values(rsp, 8))
            result.set_register(op.register if isinstance(op, Pop) else "rip", value)
        elif isinstance(op, Mov):
            value = result.get_register(op.src) if isinstance(op.src, str) else op.src
            result.set_register(op.dst, value)
        elif isinstance(op, Extend):
            source = (result.get_register(op.src) if isinstance(op.src, str)
                      else result.read_memory(_effective_address(op.src, result), op.source_width))
            extend = sign_extend if op.signed else zero_extend
            result.set_register(op.dst, extend(source, op.source_width, register_view(op.dst)[1]))
        elif isinstance(op, (Add, Sub, Inc, Dec)):
            width = register_view(op.dst)[1]
            left = result.get_register(op.dst)
            right = 1 if isinstance(op, (Inc, Dec)) else op.immediate
            subtract = isinstance(op, (Sub, Dec))
            value = (sub_values if subtract else add_values)(left, right)
            flags = (compute_sub_flags if subtract else compute_add_flags)(left, right, width)
            if isinstance(op, (Inc, Dec)):
                flags.pop("CF")  # Preserve even an unknown/undefined previous CF.
            result.set_register(op.dst, value)
            _update_flags(result, flags)
        elif isinstance(op, (AddRegister, SubRegister, Xor, And, Or, Shl, Shr, Sar)):
            width = register_view(op.dst)[1]
            if isinstance(op, Xor) and op.dst == op.src:
                value = 0
            else:
                left = result.get_register(op.dst)
                source = result.get_register(op.src) if isinstance(op.src, str) else op.src
                operation = {AddRegister: add_values, SubRegister: sub_values,
                             Xor: xor_values, And: and_values, Or: or_values,
                             Shl: shl_values, Shr: shr_values, Sar: sar_values}[type(op)]
                if isinstance(op, (Shl, Shr, Sar)):
                    source &= 63 if width == 64 else 31
                value = operation(left, source, width) if isinstance(op, Sar) else operation(left, source)
            result.set_register(op.dst, value)
            if isinstance(op, (AddRegister, SubRegister)):
                compute = compute_add_flags if isinstance(op, AddRegister) else compute_sub_flags
                _update_flags(result, compute(left, source, width))
            elif isinstance(op, (Shl, Shr, Sar)):
                _update_flags(result, compute_shift_flags(left, value, source, width,
                                                         left=isinstance(op, Shl), arithmetic=isinstance(op, Sar)))
            else:
                _update_flags(result, compute_logic_flags(value, width))
        elif isinstance(op, Neg):
            old = result.get_register(op.dst)
            result.set_register(op.dst, sub_values(0, old))
            _update_flags(result, compute_sub_flags(0, old, register_view(op.dst)[1]))
        elif isinstance(op, Not):
            result.set_register(op.dst, xor_values(result.get_register(op.dst), WORD_MASK))
        elif isinstance(op, (Clc, Stc)):
            result.set_flag("CF", isinstance(op, Stc))
        elif isinstance(op, Cmc):
            raw = dict(result.flag_bindings()).get("CF")
            value = (conditional_flag(bool_not(raw.value), raw.defined) if isinstance(raw, ConditionalFlag) else
                     UNDEFINED if isinstance(raw, UndefinedFlag) else bool_not(result.get_flag("CF")))
            result.set_flag("CF", value)
        elif isinstance(op, Multiply):
            source = _read_operand(op.src, result, op.width)
            accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[op.width]
            left = (op.immediate if op.immediate is not None else
                    result.get_register(op.dst if op.dst is not None else accumulator))
            low, high = multiply_values(left, source, op.width, signed=op.signed)
            flags = compute_multiply_flags(low, high, op.width, signed=op.signed)
            if op.dst is not None:
                result.set_register(op.dst, low)
            elif op.width == 8:
                result.set_register("ax", or_values(low, shl_values(high, 8)))
            else:
                result.set_register(accumulator, low)
                result.set_register({16: "dx", 32: "edx", 64: "rdx"}[op.width], high)
            _update_flags(result, flags)
        elif isinstance(op, (Adc, Sbb)):
            carry = result.get_flag("CF")
            left = _read_operand(op.left, result, op.width)
            right = _read_operand(op.right, result, op.width)
            value, flags = compute_carry_arithmetic(left, right, carry, op.width, subtract=isinstance(op, Sbb))
            if isinstance(op.left, str):
                result.set_register(op.left, value)
            else:
                result.store_memory(_effective_address(op.left, result), value, op.width)
            _update_flags(result, flags)
        elif isinstance(op, (Cmp, Test)):
            left = _read_operand(op.left, result, op.width)
            right = _read_operand(op.right, result, op.width)
            flags = (compute_sub_flags(left, right, op.width) if isinstance(op, Cmp)
                     else compute_logic_flags(and_values(left, right), op.width))
            _update_flags(result, flags)
        elif isinstance(op, Setcc):
            condition = evaluate_condition(op.condition, result)
            value = select_value(condition, 1, 0)
            if isinstance(op.dst, str):
                result.set_register(op.dst, value)
            else:
                result.store_memory(_effective_address(op.dst, result), value, 8)
        elif isinstance(op, Cmovcc):
            condition = evaluate_condition(op.condition, result)
            source = result.get_register(op.src)
            parent, width, _ = register_view(op.dst)
            old = result.get_register(parent)
            # x86-64 clears the upper half of a 32-bit CMOV even when false.
            untaken = extract_value(old, 32) if width == 32 else old
            result.set_register(op.dst, source)
            taken = result.get_register(parent)
            result.set_register(parent, select_value(condition, taken, untaken))
        elif isinstance(op, Xchg):
            old_dst, old_src = result.get_register(op.dst), result.get_register(op.src)
            result.set_register(op.dst, old_src)
            result.set_register(op.src, old_dst)
        elif isinstance(op, (Lea, Load)):
            address = _effective_address(op.address, result)
            value = result.read_memory(address, op.width) if isinstance(op, Load) else address
            result.set_register(op.dst, value)
        elif isinstance(op, Store):
            address = _effective_address(op.address, result)
            value = result.get_register(op.src) if isinstance(op.src, str) else op.src
            result.store_memory(address, value, op.width)
        elif isinstance(op, Push):
            rsp = _address_value(result.get_register("rsp"))
            # PUSH RSP saves the old RSP, not the decremented one.
            value = result.get_register(op.src) if isinstance(op.src, str) else op.src
            new_rsp = sub_values(rsp, 8)
            result.store_memory(new_rsp, value)
            result.set_register("rsp", new_rsp)
        elif isinstance(op, Leave):
            rbp = _address_value(result.get_register("rbp"))
            value = result.read_memory(rbp)
            result.set_register("rsp", add_values(rbp, 8))
            result.set_register("rbp", value)
        else:
            raise TypeError(f"unsupported IR operation: {type(op).__name__}")
    return result


def execute_gadget(gadget: Gadget, state: CPUState[Value]) -> CPUState[Value]:
    return execute(lift_gadget(gadget), state)
