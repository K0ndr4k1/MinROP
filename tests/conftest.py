"""Small deterministic ELF fixture, requiring no external compiler."""

import struct

import pytest


def make_elf(*, machine=62, elf_class=2, elf_type=2, sections=True, flags=5):
    code = b"\x5f\xc3\x90\x5e\x41\x5f\xc3\xc9\xc3"
    data = bytearray(0x400)
    ident = b"\x7fELF" + bytes([elf_class, 1, 1, 0]) + bytes(8)
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, elf_type, machine,
                     1, 0x401000, 64, 0x300 if sections else 0, 0,
                     64, 56, 2, 64, 3 if sections else 0, 2 if sections else 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, flags, 0x100, 0x401000,
                     0x401000, len(code), len(code) + 16, 0x100)
    struct.pack_into("<IIQQQQQQ", data, 120, 1, 4, 0x180, 0x402000,
                     0x402000, 2, 2, 0x80)
    data[0x100:0x100 + len(code)] = code
    data[0x180:0x182] = b"\x5a\xc3"  # Non-executable region.
    if sections:
        names = b"\x00.text\x00.shstrtab\x00"
        data[0x200:0x200 + len(names)] = names
        # .text deliberately covers only part of the executable segment.
        struct.pack_into("<IIQQQQIIQQ", data, 0x340, 1, 1, 6, 0x401000,
                         0x100, 2, 0, 0, 1, 0)
        struct.pack_into("<IIQQQQIIQQ", data, 0x380, 7, 3, 0, 0,
                         0x200, len(names), 0, 0, 1, 0)
    return bytes(data)


@pytest.fixture
def elf_path(tmp_path):
    path = tmp_path / "sample.elf"
    path.write_bytes(make_elf())
    return path
