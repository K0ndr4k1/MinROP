"""Core arithmetic flags and condition codes; independent of state and Z3."""
from dataclasses import dataclass

from .symbolic import (BoolValue, add_values, sub_values, and_values, extract_value,
                       compare_values, bool_not, bool_binary, check_width, select_value, xor_values)

FLAG_NAMES = ("CF", "ZF", "SF", "OF", "PF")


@dataclass(frozen=True)
class UndefinedFlag:
    """Architecture-undefined or explicitly unmodelled; never a usable Boolean."""


UNDEFINED = UndefinedFlag()


class UninitializedFlag(ValueError):
    pass


class UndefinedFlagError(ValueError):
    pass


def flag_name(name: str) -> str:
    if not isinstance(name, str) or name.upper() not in FLAG_NAMES + ("AF",):
        raise ValueError(f"unsupported flag: {name}")
    return name.upper()


def read_flag(bindings, name: str) -> BoolValue:
    name = flag_name(name)
    if name not in bindings:
        raise UninitializedFlag(f"uninitialized flag: {name}")
    value = bindings[name]
    if isinstance(value, UndefinedFlag):
        raise UndefinedFlagError(f"undefined or unmodelled flag: {name}")
    if isinstance(value, ConditionalFlag):
        from .symbolic import GuardedBool
        return GuardedBool(value.value, value.defined)
    return value


def _sign(value, width):
    return bool_not(compare_values("eq", and_values(value, 1 << (width - 1)), 0))


def compute_result_flags(result, width):
    result = extract_value(result, width)
    parity = False
    for bit in range(8):
        set_bit = bool_not(compare_values("eq", and_values(result, 1 << bit), 0))
        parity = bool_binary("xor", parity, set_bit)
    return {"ZF": compare_values("eq", result, 0), "SF": _sign(result, width),
            "PF": bool_not(parity), "AF": UNDEFINED}


def _arithmetic_flags(lhs, rhs, width, subtract):
    lhs, rhs = extract_value(lhs, width), extract_value(rhs, width)
    result = extract_value((sub_values if subtract else add_values)(lhs, rhs), width)
    flags = compute_result_flags(result, width)
    left_sign, right_sign, result_sign = _sign(lhs, width), _sign(rhs, width), _sign(result, width)
    signs_differ = bool_binary("xor", left_sign, right_sign)
    flags["AF"] = bool_not(compare_values("eq", and_values(xor_values(xor_values(lhs, rhs), result), 16), 0))
    flags["CF"] = compare_values("ult", lhs, rhs) if subtract else compare_values("ult", result, lhs)
    flags["OF"] = bool_binary("and", signs_differ if subtract else bool_not(signs_differ),
                              bool_binary("xor", left_sign, result_sign))
    return flags


def compute_add_flags(lhs, rhs, width):
    return _arithmetic_flags(lhs, rhs, width, False)


def compute_sub_flags(lhs, rhs, width):
    return _arithmetic_flags(lhs, rhs, width, True)


def compute_logic_flags(result, width):
    return dict(compute_result_flags(result, width), CF=False, OF=False)


def compute_carry_arithmetic(lhs, rhs, carry, width, *, subtract):
    """ADC/SBB value and flags, retaining carry/borrow from both additions/subtractions."""
    check_width(width)
    lhs, rhs = extract_value(lhs, width), extract_value(rhs, width)
    carry_value = select_value(carry, 1, 0)
    operation = sub_values if subtract else add_values
    first = extract_value(operation(lhs, rhs), width)
    result = extract_value(operation(first, carry_value), width)
    flags = compute_result_flags(result, width)
    if subtract:
        first_carry = compare_values("ult", lhs, rhs)
        second_carry = compare_values("ult", first, carry_value)
    else:
        first_carry = compare_values("ult", first, lhs)
        second_carry = compare_values("ult", result, first)
    flags["CF"] = bool_binary("or", first_carry, second_carry)
    left_sign, right_sign = _sign(lhs, width), _sign(rhs, width)
    signs_differ = bool_binary("xor", left_sign, right_sign)
    flags["OF"] = bool_binary("and", signs_differ if subtract else bool_not(signs_differ),
                              bool_binary("xor", left_sign, _sign(result, width)))
    flags["AF"] = bool_not(compare_values("eq", and_values(xor_values(xor_values(lhs, rhs), result), 16), 0))
    return result, flags


def compute_multiply_flags(low, high, width, *, signed):
    expected_high = select_value(_sign(low, width), (1 << width) - 1, 0) if signed else 0
    overflow = bool_not(compare_values("eq", high, expected_high))
    return {"CF": overflow, "OF": overflow, "ZF": UNDEFINED,
            "SF": UNDEFINED, "PF": UNDEFINED, "AF": UNDEFINED}


def compute_shift_flags(original, result, count, width, *, left, arithmetic=False):
    """Immediate SHL/SHR/SAR flags; an empty update preserves count-zero flags."""
    check_width(width)
    if type(count) is not int or type(left) is not bool or type(arithmetic) is not bool:
        raise TypeError("integer shift count and Boolean direction required")
    if arithmetic and left:
        raise ValueError("arithmetic shift must be rightward")
    count &= 63 if width == 64 else 31
    if count == 0:
        return {}
    original = extract_value(original, width)
    flags = compute_result_flags(result, width)
    if count >= width and not arithmetic:
        flags["CF"] = UNDEFINED
    else:
        bit = width - count if left else min(count, width) - 1
        flags["CF"] = bool_not(compare_values("eq", and_values(original, 1 << bit), 0))
    if count == 1:
        if arithmetic:
            flags["OF"] = False
        else:
            flags["OF"] = bool_binary("xor", flags["SF"], flags["CF"]) if left else _sign(original, width)
    else:
        flags["OF"] = UNDEFINED
    return flags


CONDITION_ALIASES = {
    "o": "o", "no": "no", "b": "b", "c": "b", "nae": "b",
    "ae": "ae", "nc": "ae", "nb": "ae", "e": "e", "z": "e",
    "ne": "ne", "nz": "ne", "be": "be", "na": "be", "a": "a", "nbe": "a",
    "s": "s", "ns": "ns", "p": "p", "pe": "p", "np": "np", "po": "np",
    "l": "l", "nge": "l", "ge": "ge", "nl": "ge", "le": "le", "ng": "le",
    "g": "g", "nle": "g",
}


def normalize_condition(condition: str) -> str:
    try:
        return CONDITION_ALIASES[condition.lower()]
    except (KeyError, AttributeError) as error:
        raise ValueError(f"unsupported condition: {condition}") from error


def evaluate_condition(condition: str, flags) -> BoolValue:
    condition = normalize_condition(condition)
    # Read only the flags used by the instruction. Undefined unrelated flags
    # must not prevent evaluating a valid condition (e.g. SETZ only needs ZF).
    read = flags.get_flag if hasattr(flags, "get_flag") else lambda name: read_flag(flags, name)
    if condition in ("o", "no", "b", "ae", "e", "ne", "s", "ns", "p", "np"):
        name, negate = {"o": ("OF", False), "no": ("OF", True), "b": ("CF", False),
                        "ae": ("CF", True), "e": ("ZF", False), "ne": ("ZF", True),
                        "s": ("SF", False), "ns": ("SF", True), "p": ("PF", False),
                        "np": ("PF", True)}[condition]
        value = read(name)
        return bool_not(value) if negate else value
    if condition in ("be", "a"):
        value = bool_binary("or", read("CF"), read("ZF"))
        return bool_not(value) if condition == "a" else value
    different = bool_binary("xor", read("SF"), read("OF"))
    if condition in ("l", "ge"):
        return different if condition == "l" else bool_not(different)
    value = bool_binary("or", read("ZF"), different)
    return value if condition == "le" else bool_not(value)


@dataclass(frozen=True)
class ConditionalFlag:
    """Defined only for some symbolic counts; consuming it requires that guard."""
    value: BoolValue
    defined: BoolValue


def flag_parts(flag):
    if isinstance(flag, UndefinedFlag):
        return False, False
    if isinstance(flag, ConditionalFlag):
        return flag.value, flag.defined
    return flag, True


def conditional_flag(value, defined):
    from .symbolic import BoolConstant
    if isinstance(defined, BoolConstant):
        defined = defined.value
    return value if defined is True else UNDEFINED if defined is False else ConditionalFlag(value, defined)


def merge_count_flags(count, updates, old):
    """Finite count domain, using existing Boolean/ITE expressions."""
    merged = {}
    for name in {key for update in updates for key in update}:
        value, defined = False, False
        for index, update in enumerate(updates):
            predicate = compare_values("eq", count, index)
            current, valid = flag_parts(update.get(name, old.get(name, UNDEFINED)))
            value = bool_binary("or", value, bool_binary("and", predicate, current))
            defined = bool_binary("or", defined, bool_binary("and", predicate, valid))
        merged[name] = conditional_flag(value, defined)
    return merged
