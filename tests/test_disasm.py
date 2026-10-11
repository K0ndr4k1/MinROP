import pytest

from minirop import CodeRegion, ELFImage, GadgetScanner
from minirop.disasm import Disassembler, ret_offsets


def region(data, address=0x1000):
    return CodeRegion("test", address, 0, data)


def test_disassembly():
    instructions = Disassembler().disassemble(b"\x5f\xc3", 0x1000)
    assert [(i.address, i.mnemonic, i.op_str, i.size, i.bytes) for i in instructions] == [
        (0x1000, "pop", "rdi", 1, b"\x5f"), (0x1001, "ret", "", 1, b"\xc3")]


def test_ret_detection():
    assert list(ret_offsets(b"\xc3\x90\xc2\x00\x00\xc3")) == [0, 5]
    assert list(ret_offsets(b"")) == []


def test_backward_scan_and_gadget():
    gadgets = GadgetScanner().scan([region(b"\x5e\x41\x5f\xc3")])
    assert [g.address for g in gadgets] == [0x1000, 0x1001, 0x1002, 0x1003]
    first = gadgets[0]
    assert first.bytes == b"\x5e\x41\x5f\xc3"
    assert first.end_address == 0x1004 and first.instruction_count == 3
    assert [str(i) for i in first.instructions] == ["pop rsi", "pop r15", "ret"]
    assert str(gadgets[2].instructions[0]) == "pop rdi"


def test_duplicates_preserve_distinct_addresses():
    a = region(b"\x5f\xc3")
    gadgets = GadgetScanner().scan([a, a, region(a.bytes, 0x2000)])
    assert [g.address for g in gadgets] == [0x1000, 0x1001, 0x2000, 0x2001]


def test_limits_and_boundaries():
    assert len(GadgetScanner(max_gadget_bytes=1).scan([region(b"\x5f\xc3")])) == 1
    assert len(GadgetScanner(max_instructions=1).scan([region(b"\x5f\xc3")])) == 1
    assert GadgetScanner().scan([region(b"\x5f"), region(b"\xc3", 0x1001)])[0].bytes == b"\xc3"
    assert GadgetScanner().scan([region(b"\x90")]) == []
    for kwargs in ({"max_gadget_bytes": 0}, {"max_instructions": -1}):
        with pytest.raises(ValueError):
            GadgetScanner(**kwargs)


@pytest.mark.parametrize("data", [b"\x0f\xc3", b"\xb8\x00\x00\x00\xc3", b"\xc3\xc3", b"\xc2\x00\x00\xc3"])
def test_invalid_or_premature_endpoint(data):
    gadgets = GadgetScanner().scan([region(data)])
    assert not any(g.address == 0x1000 and len(g.bytes) == len(data) for g in gadgets)
    assert any(g.bytes == b"\xc3" for g in gadgets)


def test_scans_beyond_text_but_not_readonly(elf_path):
    image = ELFImage.load(elf_path)
    gadgets = GadgetScanner().scan(image.executable_segments)
    assert any(g.address == 0x401007 and str(g.instructions[0]) == "leave" for g in gadgets)
    assert all(g.address < 0x402000 for g in gadgets)
