import subprocess
import sys

import pytest

from minirop.cli import main


def test_cli(elf_path, capsys):
    assert main(["gadgets", str(elf_path)]) == 0
    output = capsys.readouterr()
    assert "0x401000: pop rdi ; ret\n" in output.out
    assert "0x401007: leave ; ret\n" in output.out
    assert output.err == ""


def test_cli_error(tmp_path, capsys):
    assert main(["gadgets", str(tmp_path / "missing")]) == 1
    assert "cannot load" in capsys.readouterr().err


def test_cli_bad_limit(elf_path):
    with pytest.raises(SystemExit) as error:
        main(["gadgets", str(elf_path), "--max-instructions", "0"])
    assert error.value.code == 2


def test_module_entrypoint(elf_path):
    result = subprocess.run(
        [sys.executable, "-m", "minirop", "gadgets", str(elf_path), "--max-instructions", "1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == ["0x401001: ret", "0x401006: ret", "0x401008: ret"]
