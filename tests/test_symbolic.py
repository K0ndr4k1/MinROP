from dataclasses import FrozenInstanceError

import pytest

from minirop.symbolic import (
    Add, Constant, MemoryRead, NonConcreteValue, Sub, Symbol, WORD_MASK, Xor,
    add_values, as_expression, evaluate, format_value, sub_values, xor_values,
)


def test_expression_structure_hash_and_immutability():
    x = Symbol("x")
    expr = Add(x, 8)
    assert expr == Add(Symbol("x"), Constant(8))
    assert expr.right == Constant(8)
    assert expr.bits == expr.left.bits == expr.right.bits == 64
    assert len({expr, Add(Symbol("x"), Constant(8)), Sub(x, Constant(8))}) == 2
    with pytest.raises(FrozenInstanceError):
        expr.left = Symbol("y")
    with pytest.raises(FrozenInstanceError):
        x.bits = 32


@pytest.mark.parametrize("value, expected", [(-1, WORD_MASK), (1 << 64, 0), (8, 8)])
def test_constant_wraparound(value, expected):
    assert Constant(value).value == expected
    assert as_expression(value) == Constant(expected)


@pytest.mark.parametrize("factory", [
    lambda: Symbol("x", bits=32), lambda: Constant(1, bits=8),
    lambda: Add(Symbol("x"), Constant(1), bits=128),
    lambda: MemoryRead(Symbol("x"), bits=32), lambda: Symbol(""),
])
def test_reject_bad_width_or_name(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("value", [None, "x", (), object(), 1.5])
def test_reject_unsupported_values(value):
    with pytest.raises(NonConcreteValue):
        add_values(Symbol("x"), value)
    with pytest.raises(NonConcreteValue):
        as_expression(value)


@pytest.mark.parametrize("operation, left, right, expected", [
    (add_values, 10, 8, 18), (add_values, WORD_MASK, 1, 0),
    (sub_values, 0, 1, WORD_MASK), (sub_values, 5, -1, 6),
    (xor_values, WORD_MASK, 0xff, WORD_MASK ^ 0xff),
])
def test_concrete_path_returns_int(operation, left, right, expected):
    result = operation(left, right)
    assert type(result) is int and result == expected
    assert operation(Constant(left), Constant(right)) == Constant(expected)


def test_symbolic_arithmetic_and_simplification():
    x, y = Symbol("x"), Symbol("y")
    assert add_values(x, 8) == Add(x, Constant(8))
    assert add_values(x, y) == Add(x, y)
    assert sub_values(x, y) == Sub(x, y)
    assert xor_values(x, y) == Xor(x, y)
    assert add_values(x, 0) == sub_values(x, 0) == xor_values(x, 0) == x
    assert add_values(0, x) == xor_values(0, x) == x
    assert xor_values(x, x) == Constant(0)
    assert add_values(add_values(x, 8), 8) == Add(x, Constant(16))
    assert add_values(sub_values(x, 8), 8) == x
    assert add_values(add_values(x, WORD_MASK), 1) == x


@pytest.mark.parametrize("expr, text", [
    (Symbol("x"), "x"), (Constant(16), "0x10"),
    (Add(Symbol("x"), Constant(8)), "(x + 0x8)"),
    (Sub(Symbol("x"), Constant(8)), "(x - 0x8)"),
    (Xor(Symbol("x"), Symbol("y")), "(x ^ y)"),
    (MemoryRead(Add(Symbol("rbp_0"), Constant(8))), "mem64[(rbp_0 + 0x8)]"),
    (Sub(Symbol("x"), Sub(Symbol("y"), Constant(1))), "(x - (y - 0x1))"),
])
def test_formatting(expr, text):
    assert str(expr) == format_value(expr) == text


@pytest.mark.parametrize("expr, bindings, expected", [
    (Add(Symbol("x"), Constant(8)), {"x": 10}, 18),
    (Add(Symbol("x"), Constant(1)), {"x": WORD_MASK}, 0),
    (Sub(Symbol("x"), Constant(1)), {"x": 0}, WORD_MASK),
    (Xor(Symbol("x"), Symbol("y")), {"x": 0xff, "y": 0xf}, 0xf0),
    (Symbol("x"), {"x": -1}, WORD_MASK),
    (Add(Sub(Symbol("x"), Constant(1)), Constant(1)), {"x": 0}, 0),
])
def test_evaluate(expr, bindings, expected):
    assert evaluate(expr, bindings) == expected


def test_evaluate_memory_read_and_nested_address():
    expr = MemoryRead(Add(MemoryRead(Symbol("rsp_0")), Constant(8)))
    assert evaluate(expr, {"rsp_0": 0x1000}, memory={0x1000: 0x2000, 0x2008: -1}) == WORD_MASK


def test_evaluate_rejects_missing_or_noninteger_inputs():
    with pytest.raises(KeyError):
        evaluate(Symbol("x"), {})
    with pytest.raises(TypeError):
        evaluate(Symbol("x"), {"x": Symbol("y")})
    with pytest.raises(KeyError):
        evaluate(MemoryRead(Symbol("x")), {"x": 0x1000})
    with pytest.raises(KeyError):
        evaluate(MemoryRead(Symbol("x")), {"x": 0x1000}, memory={})
    with pytest.raises(KeyError):
        evaluate(MemoryRead(Constant(WORD_MASK)), {}, memory={})
