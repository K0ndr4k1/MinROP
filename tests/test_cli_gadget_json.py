"""ELF -> scanner JSON -> existing search, through actual command-line files."""

import json
import os
from pathlib import Path
import subprocess
import sys

from minirop.disasm import GadgetScanner
from minirop.elf import ELFImage


ROOT = Path(__file__).resolve().parents[1]


def cli(*arguments):
    return subprocess.run([sys.executable, "-B", "-m", "minirop", *map(str, arguments)],
                          cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1"),
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def exported(elf_path, tmp_path):
    path = tmp_path / "gadgets.json"
    result = cli("gadgets", elf_path, "-o", path, "--supported-only")
    assert result.returncode == 0, result.stderr
    return path


def conditions():
    return {"input_version": 1, "initial": {"registers": {"rdi": 0}},
            "goals": {"rdi": 7}, "max_depth": 1, "pruning": False}


def test_json_export_preserves_all_scanned_bytes_addresses_and_operands(elf_path, tmp_path):
    source = tmp_path / "sample_样例.elf"
    source.write_bytes(elf_path.read_bytes())
    destination = tmp_path / "all.json"
    result = cli("gadgets", source, "--output", destination)
    assert result.returncode == 0 and not result.stdout, result.stderr
    document = read(destination)
    assert document["input_version"] == 1 and document["architecture"] == "x86_64"
    assert document["source"] == source.as_posix()
    expected = GadgetScanner().scan(ELFImage.load(source).executable_segments)
    assert len(document["gadgets"]) == len(expected)
    assert len({entry["id"] for entry in document["gadgets"]}) == len(expected)
    for entry, gadget in zip(document["gadgets"], expected):
        assert int(entry["address"], 16) == gadget.address
        assert bytes.fromhex(entry["bytes"]) == gadget.bytes
        assert entry["instructions"] == list(map(str, gadget.instructions))


def test_supported_only_is_explicit_and_preserves_scan_order_and_ids(elf_path, tmp_path):
    data = bytearray(elf_path.read_bytes())
    data[0x100:0x103] = bytes.fromhex("0f 0b c3")  # UD2; RET is deliberately unsupported.
    elf_path.write_bytes(data)
    unfiltered = tmp_path / "all.json"
    assert cli("gadgets", elf_path, "-o", unfiltered).returncode == 0
    original = read(unfiltered)
    assert any("ud2" in entry["instructions"] for entry in original["gadgets"])
    config = tmp_path / "search.json"
    write(config, conditions())
    refused = cli("search", unfiltered, "--config", config)
    assert refused.returncode == 2 and "unsupported" in refused.stderr
    assert "STATUS:" not in refused.stdout
    filtered = tmp_path / "supported.json"
    result = cli("gadgets", elf_path, "-o", filtered, "--supported-only")
    assert result.returncode == 0 and "excluded" in result.stderr
    accepted = read(filtered)["gadgets"]
    assert accepted and len(accepted) < len(original["gadgets"])
    accepted_ids = {entry["id"] for entry in accepted}
    assert accepted == [entry for entry in original["gadgets"] if entry["id"] in accepted_ids]
    assert all("ud2" not in entry["instructions"] for entry in accepted)


def test_search_reads_gadget_json_and_matches_existing_candidate_input(elf_path, tmp_path):
    path = exported(elf_path, tmp_path)
    document = read(path)
    document.update(conditions())
    write(path, document)
    legacy = {**conditions(), "candidates": [
        {key: entry[key] for key in ("id", "address", "bytes")} for entry in document["gadgets"]]}
    legacy_path = tmp_path / "candidates.json"
    write(legacy_path, legacy)
    direct = cli("search", path)
    previous = cli("search", legacy_path)
    assert direct.returncode == previous.returncode == 0, (direct.stderr, previous.stderr)
    assert direct.stdout.splitlines()[:6] == previous.stdout.splitlines()[:6]
    assert "STATUS: FOUND" in direct.stdout and '"rdi": "0x7"' in direct.stdout


def test_search_config_uses_catalog_without_reopening_elf(elf_path, tmp_path):
    path = exported(elf_path, tmp_path)
    config = tmp_path / "config.json"
    write(config, conditions())
    elf_path.unlink()  # Search consumes the catalog; source is provenance only.
    result = cli("search", path, "--config", config)
    assert result.returncode == 0, result.stderr
    assert "Goals satisfied: True" in result.stdout
    assert read(path).keys() == {"input_version", "architecture", "source", "gadgets"}
    changed = conditions()
    changed["initial"]["stack"] = [0] * 8
    write(config, changed)
    result = cli("search", path, "--config", config)
    # No candidate in this fixture can supply RDI=7 with all stack words zero.
    assert result.returncode == 1 and "STATUS: NOT_FOUND" in result.stdout


def test_original_search_example_is_unchanged():
    result = cli("search", "examples/search.json")
    assert result.returncode == 0, result.stderr
    assert 'Selected candidate IDs: ["add_two"]' in result.stdout
    assert "Tested candidates: 2" in result.stdout


def test_invalid_catalog_and_configuration_fail_explicitly(elf_path, tmp_path):
    path = exported(elf_path, tmp_path)
    original = read(path)
    config = tmp_path / "config.json"
    write(config, conditions())
    result = cli("search", path)
    assert result.returncode == 2 and "--config" in result.stderr
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"gadgets": [', encoding="utf-8")
    result = cli("search", malformed, "--config", config)
    assert result.returncode == 2 and str(malformed) in result.stderr
    changed = read(path)
    changed["gadgets"][0]["instructions"] = ["pop rax", "ret"]
    write(path, changed)
    result = cli("search", path, "--config", config)
    assert result.returncode == 2 and "instructions" in result.stderr
    assert not result.stdout
    for field, value in (("candidates", []), ("goals", {"rdi": 9}), ("input_version", 2)):
        write(path, {**original, field: value})
        result = cli("search", path, "--config", config)
        assert result.returncode == 2 and field in result.stderr
        assert "STATUS:" not in result.stdout


def test_non_elf_and_output_overwrite_are_rejected(elf_path, tmp_path):
    bad = tmp_path / "not_elf"
    bad.write_text("not an ELF file", encoding="utf-8")
    output = tmp_path / "existing.json"
    output.write_text("keep this", encoding="utf-8")
    result = cli("gadgets", bad, "-o", output)
    assert result.returncode == 1 and "cannot load" in result.stderr and str(bad) in result.stderr
    assert output.read_text(encoding="utf-8") == "keep this"
    before = elf_path.read_bytes()
    result = cli("gadgets", elf_path, "-o", elf_path)
    assert result.returncode == 2 and "overwrite" in result.stderr
    assert elf_path.read_bytes() == before


def test_generated_elf_export_and_search_demo(elf_path, tmp_path):
    regenerated = tmp_path / "sample_gadgets.json"
    result = cli("gadgets", elf_path, "-o", regenerated, "--supported-only")
    assert result.returncode == 0, result.stderr
    document = read(regenerated)
    assert document["source"] == elf_path.as_posix()
    repeated = tmp_path / "repeated.json"
    result = cli("gadgets", elf_path, "-o", repeated, "--supported-only")
    assert result.returncode == 0, result.stderr
    assert read(repeated) == document
    target = next(entry for entry in document["gadgets"] if entry["instructions"] == ["pop rdi", "ret"])
    config = tmp_path / "config.json"
    write(config, {"input_version": 1,
                   "initial": {"registers": {"rdi": 0, "rsp": "0x8000"}, "stack": [7, "0x9000"]},
                   "goals": {"rdi": 7, "rip": "0x9000"}, "max_depth": 1, "pruning": False})
    result = cli("search", regenerated, "--config", config)
    assert result.returncode == 0, result.stderr
    assert "STATUS: FOUND" in result.stdout and "Solver status: SAT" in result.stdout
    assert f'Selected candidate IDs: ["{target["id"]}"]' in result.stdout
    assert '"rdi": "0x7"' in result.stdout and '"rip": "0x9000"' in result.stdout
