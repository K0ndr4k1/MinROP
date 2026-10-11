import struct

import pytest

from minirop import ELFImage, ELFLoadError
from conftest import make_elf


def test_load_segments_text_and_mapping(elf_path):
    image = ELFImage.load(elf_path)
    assert not image.is_pie
    assert len(image.executable_segments) == 1
    region = image.executable_segments[0]
    assert region.address == 0x401000
    assert len(region.bytes) == 9  # Memory-only tail is excluded.
    assert region.end_address == 0x401009
    assert image.text.bytes == b"\x5f\xc3"
    assert image.text.address == 0x401000
    assert region.address_to_offset(0x401001) == 0x101
    assert region.offset_to_address(0x101) == 0x401001
    with pytest.raises(ValueError):
        region.address_to_offset(region.end_address)
    with pytest.raises(ValueError):
        region.offset_to_address(0xff)


def test_sectionless_pie(tmp_path):
    path = tmp_path / "pie"
    path.write_bytes(make_elf(sections=False, elf_type=3))
    image = ELFImage.load(path)
    assert image.is_pie and image.text is None
    assert len(image.executable_segments) == 1


@pytest.mark.parametrize("data", [b"not ELF", b"\x7fELF", make_elf(machine=183),
                                  make_elf(elf_class=1), make_elf(elf_type=1)])
def test_bad_or_unsupported_input(tmp_path, data):
    path = tmp_path / "bad"
    path.write_bytes(data)
    with pytest.raises(ELFLoadError):
        ELFImage.load(path)


def test_truncated_segment(tmp_path):
    data = bytearray(make_elf())
    struct.pack_into("<Q", data, 64 + 8, 0x3ff)
    path = tmp_path / "truncated"
    path.write_bytes(data)
    with pytest.raises(ELFLoadError, match="truncated"):
        ELFImage.load(path)


def test_no_executable_segments(tmp_path):
    path = tmp_path / "readonly"
    path.write_bytes(make_elf(flags=4))
    assert ELFImage.load(path).executable_segments == ()


def test_missing_file(tmp_path):
    with pytest.raises(ELFLoadError):
        ELFImage.load(tmp_path / "missing")


def test_multiple_executable_segments(tmp_path):
    data = bytearray(make_elf())
    struct.pack_into("<I", data, 120 + 4, 5)
    path = tmp_path / "multiple"
    path.write_bytes(data)
    image = ELFImage.load(path)
    assert [r.address for r in image.executable_segments] == [0x401000, 0x402000]
    assert image.executable_segments[1].bytes == b"\x5a\xc3"


def test_file_size_exceeds_memory_size(tmp_path):
    data = bytearray(make_elf())
    struct.pack_into("<Q", data, 64 + 40, 1)
    path = tmp_path / "bad-size"
    path.write_bytes(data)
    with pytest.raises(ELFLoadError, match="exceeds"):
        ELFImage.load(path)
