"""Actual file input and command-line acceptance, not hand-built IR fixtures."""

import json
from pathlib import Path
import subprocess
import sys

from minirop.cli import main
from minirop.solver import SolverContext, SolveResult, SolveStatus


ROOT = Path(__file__).resolve().parents[1]


def invoke(tmp_path, mode, document):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return subprocess.run([sys.executable, "-B", "-m", "minirop", mode, str(path)],
                          cwd=ROOT, capture_output=True, text=True, timeout=30)


def fixed(raw="48 83 c0 01 c3", **initial):
    return {"input_version": 1,
            "candidates": [{"id": "piece", "address": "0x4000", "bytes": raw}],
            "sequence": ["piece"], "initial": {"registers": initial}, "goals": {"rax": 1}}


def search_input():
    return json.loads((ROOT / "examples" / "search.json").read_text(encoding="utf-8"))


def test_file_sat_carries_memory_and_state_between_snippets(tmp_path):
    document = fixed("48 8b 03 48 83 c0 01 48 89 03 c3", rbx="0x2000", rsp="0x8000")
    document["candidates"].append({"id": "read_back", "address": "0x5000", "bytes": "8b 0b c3"})
    document["sequence"].append("read_back")
    document["initial"].update(memory=[{"address": "0x2000", "value": 6}], stack=["0x5000", "0x9000"])
    document["goals"] = {"rax": 7, "ecx": 7, "rsp": "0x8010", "rip": "0x9000"}
    result = invoke(tmp_path, "solve", document)
    assert result.returncode == 0, result.stderr
    assert 'STATUS: FOUND\nSolver status: SAT\nGoals satisfied: True' in result.stdout
    assert 'Selected candidate IDs: ["piece", "read_back"]' in result.stdout
    assert 'Sequence depth: 2' in result.stdout
    assert '"ecx": "0x7"' in result.stdout
    assert not result.stderr


def test_file_unsat_goal_and_return_flow(tmp_path):
    document = fixed(rax=0)
    document["goals"]["rax"] = 2
    result = invoke(tmp_path, "solve", document)
    assert result.returncode == 1 and "Solver status: UNSAT" in result.stdout
    assert "Goals satisfied: False" in result.stdout and "only within" in result.stdout
    # Final values would match, but an explicit return cannot reach the next snippet.
    document["candidates"].append({"id": "next", "address": "0x5000", "bytes": "48 83 c0 01 c3"})
    document["sequence"].append("next")
    document["initial"]["stack"] = ["0x6000"]
    result = invoke(tmp_path, "solve", document)
    assert result.returncode == 1 and "STATUS: NOT_FOUND" in result.stdout


def test_search_file_found_uses_input_ids_and_existing_order(tmp_path):
    result = invoke(tmp_path, "search", search_input())
    assert result.returncode == 0, result.stderr
    assert 'Selected candidate IDs: ["add_two"]' in result.stdout
    assert "Tested candidates: 2" in result.stdout
    assert "Solver status: SAT" in result.stdout


def test_search_file_not_found_is_bounded(tmp_path):
    document = search_input()
    document.update(max_depth=1)
    document["goals"] = {"rax": 3}
    result = invoke(tmp_path, "search", document)
    assert result.returncode == 1, result.stderr
    assert "STATUS: NOT_FOUND" in result.stdout and "Solver status: UNSAT" in result.stdout
    assert "Tested candidates: 2" in result.stdout and "depth bound" in result.stdout


def test_invalid_files_are_rejected_without_solver_status(tmp_path):
    documents = []
    document = fixed()
    document["candidates"][0]["bytes"] = "48 83"  # truncated encoding
    documents.append((document, "incomplete"))
    document = fixed()
    document["candidates"][0]["extra_operand"] = "rbx"
    documents.append((document, "unknown fields"))
    document = fixed()
    document["sequence"] = ["missing"]
    documents.append((document, "unknown candidate ID"))
    document = search_input()
    document["max_depth"] = True
    documents.append((document, "max_depth"))
    document = fixed(rax="0x10000000000000000")
    documents.append((document, "64-bit range"))
    for document, expected in documents:
        result = invoke(tmp_path, "search" if "max_depth" in document else "solve", document)
        assert result.returncode == 2 and expected in result.stderr
        assert "Solver status:" not in result.stdout
    path = tmp_path / "duplicate.json"
    path.write_text('{"input_version": 1, "input_version": 1}', encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", "-m", "minirop", "solve", str(path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "duplicate JSON field" in result.stderr


def test_unsupported_instruction_and_undefined_flag_are_explicit(tmp_path):
    for raw, expected in [("0f 0b c3", "unsupported"), ("48 f7 e3 0f 94 c1 c3", "undefined")]:
        result = invoke(tmp_path, "solve", fixed(raw))
        assert result.returncode == 2, result.stdout
        assert expected in result.stderr.lower() and "STATUS:" not in result.stdout


def test_file_symbolic_condition_reaches_solver(tmp_path):
    document = fixed("48 39 d8 48 0f 44 ca c3", rbx=10, rcx=7, rdx=42)
    document["goals"] = {"rax": 10, "rcx": 42}
    result = invoke(tmp_path, "solve", document)
    assert result.returncode == 0, result.stderr
    assert '"rcx": "0x2a"' in result.stdout and '"rax_0": 10' in result.stdout
    document["goals"]["rax"] = 11
    result = invoke(tmp_path, "solve", document)
    assert result.returncode == 1 and "Solver status: UNSAT" in result.stdout


def test_file_division_faults_remain_inconclusive(tmp_path):
    for registers in ({"rax": 10, "rdx": 0, "rbx": 0},  # division by zero
                      {"rax": 0, "rdx": 1, "rbx": 1}):  # quotient overflow
        document = fixed("48 f7 f3 c3", **registers)
        for mode in ("solve", "search"):
            if mode == "search":
                document.pop("sequence")
                document["max_depth"] = 1
                document["pruning"] = True
            result = invoke(tmp_path, mode, document)
            assert result.returncode == 3, result.stderr
            assert "STATUS: INCONCLUSIVE" in result.stdout and "Solver status: UNKNOWN" in result.stdout
            assert "Goals satisfied: Unknown" in result.stdout and "Solver status: UNSAT" not in result.stdout
    document = fixed("48 f7 f3 c3", rax=10, rdx=0, rbx=2)
    document["goals"] = {"rax": 5, "rdx": 0}
    assert invoke(tmp_path, "solve", document).returncode == 0
    document["goals"]["rax"] = 6
    assert "Solver status: UNSAT" in invoke(tmp_path, "solve", document).stdout
    # The fault can arise only after another candidate changes the divisor.
    document = fixed("48 c7 c3 00 00 00 00 c3", rax=10, rdx=0, rbx=2)
    document["candidates"].append({"id": "divide", "address": "0x5000", "bytes": "48 f7 f3 c3"})
    document["sequence"].append("divide")
    document["initial"]["stack"] = ["0x5000"]
    assert invoke(tmp_path, "solve", document).returncode == 3
    document.pop("sequence")
    document["initial"].pop("stack")
    document.update(max_depth=2, pruning=True)
    result = invoke(tmp_path, "search", document)
    assert result.returncode == 3 and "STATUS: INCONCLUSIVE" in result.stdout
    # A bad candidate must not prevent finding a later normal SAT witness.
    document = fixed("48 f7 f3 c3", rax=10, rdx=0, rbx=0)
    document.pop("sequence")
    document["candidates"].append({"id": "normal", "address": "0x5000", "bytes": "c3"})
    document.update(max_depth=1)
    document["goals"] = {"rax": 10}
    result = invoke(tmp_path, "search", document)
    assert result.returncode == 0 and 'Selected candidate IDs: ["normal"]' in result.stdout


def test_file_unknown_preserved_in_both_commands(tmp_path, monkeypatch, capsys):
    # Force the existing solver's UNKNOWN result deterministically; input still
    # comes through an actual file and the real parser/executor/CLI.
    monkeypatch.setattr(SolverContext, "solve", lambda self: SolveResult(
        SolveStatus.UNKNOWN, reason_unknown="controlled UNKNOWN"))
    for mode, document in (("solve", fixed()), ("search", search_input())):
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert main([mode, str(path)]) == 3
        output = capsys.readouterr()
        assert "STATUS: INCONCLUSIVE" in output.out and "Solver status: UNKNOWN" in output.out
        assert "Goals satisfied: Unknown" in output.out and not output.err


def test_readme_examples_are_real_runnable_files():
    for mode in ("solve", "search"):
        result = subprocess.run([sys.executable, "-B", "-m", "minirop", mode, f"examples/{mode}.json"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "STATUS: FOUND" in result.stdout and "Goals satisfied: True" in result.stdout
