"""Load file-backed executable regions; never synthesize BSS bytes."""

from dataclasses import dataclass
from pathlib import Path

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile


class ELFLoadError(ValueError):
    """Invalid, unsupported, or truncated ELF input."""


@dataclass(frozen=True)
class CodeRegion:
    name: str
    address: int
    file_offset: int
    bytes: bytes

    @property
    def end_address(self) -> int:
        return self.address + len(self.bytes)

    def address_to_offset(self, address: int) -> int:
        if not self.address <= address < self.end_address:
            raise ValueError("address outside file-backed region")
        return self.file_offset + address - self.address

    def offset_to_address(self, offset: int) -> int:
        if not self.file_offset <= offset < self.file_offset + len(self.bytes):
            raise ValueError("offset outside file-backed region")
        return self.address + offset - self.file_offset


@dataclass(frozen=True)
class ELFImage:
    path: Path
    executable_segments: tuple[CodeRegion, ...]
    text: CodeRegion | None
    is_pie: bool

    @classmethod
    def load(cls, path: str | Path) -> "ELFImage":
        path = Path(path)
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, 2)
                stream.seek(0)
                elf = ELFFile(stream)
                if elf.elfclass != 64 or not elf.little_endian:
                    raise ELFLoadError("expected little-endian ELF64")
                if elf["e_machine"] != "EM_X86_64":
                    raise ELFLoadError("expected x86-64 ELF (EM_X86_64)")
                if elf["e_type"] not in ("ET_EXEC", "ET_DYN"):
                    raise ELFLoadError("expected ET_EXEC or ET_DYN ELF")

                def read_region(name: str, address: int, offset: int, length: int) -> CodeRegion:
                    if offset < 0 or length < 0 or offset + length > size:
                        raise ELFLoadError(f"truncated region: {name}")
                    stream.seek(offset)
                    data = stream.read(length)
                    if len(data) != length:
                        raise ELFLoadError(f"truncated region: {name}")
                    return CodeRegion(name, address, offset, data)

                regions = []
                for index, segment in enumerate(elf.iter_segments()):
                    if segment["p_type"] == "PT_LOAD" and segment["p_flags"] & 1:
                        if segment["p_filesz"] > segment["p_memsz"]:
                            raise ELFLoadError("PT_LOAD file size exceeds memory size")
                        regions.append(read_region(
                            f"PT_LOAD[{index}]", segment["p_vaddr"],
                            segment["p_offset"], segment["p_filesz"],
                        ))
                section = elf.get_section_by_name(".text")
                text = None
                if section is not None and section["sh_type"] != "SHT_NOBITS":
                    text = read_region(".text", section["sh_addr"], section["sh_offset"], section["sh_size"])
                return cls(path, tuple(regions), text, elf["e_type"] == "ET_DYN")
        except (ELFError, OSError, ValueError) as exc:
            if isinstance(exc, ELFLoadError):
                raise
            raise ELFLoadError(f"cannot load {path}: {exc}") from exc
