"""Final CLI presentation acceptance using saved witnesses and real input files."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from minirop import cli as command
from minirop.search import search_sequences
from minirop.solver import SolverContext, SolveResult, SolveStatus


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    return subprocess.run([sys.executable, "-B", "-m", "minirop", *map(str, arguments)],
                          cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1"),
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def write_input(tmp_path, document=None):
    if document is None:
        document = {"input_version": 1,
                    "candidates": [{"id": "pop_rdi", "address": "0x4000", "bytes": "5f c3"}],
                    "initial": {"registers": {"rdi": 0, "rsp": "0x8000"},
                                "stack": [7, "0x9000"],
                                "memory": [{"address": "0x2000", "value": 42}]},
                    "goals": {"rdi": 7, "rip": "0x9000"}, "max_depth": 1}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_default_and_explicit_raw_output_are_compatible():
    default = run("search", "examples/search.json")
    explicit = run("search", "examples/search.json", "--view", "raw")
    assert default.returncode == explicit.returncode == 0
    assert default.stdout == explicit.stdout
    assert "Input witness (one model):" in default.stdout
    assert 'Selected candidate IDs: ["add_two"]' in default.stdout
    assert "Payload view:" not in default.stdout and not default.stderr and not explicit.stderr


def test_payload_view_shows_actual_initial_register_stack_and_memory_values(tmp_path):
    path = write_input(tmp_path)
    result = run("search", path, "--view", "payload")
    assert result.returncode == 0, result.stderr
    assert "Input witness (one model):" in result.stdout
    view = result.stdout.split("Payload view:\n", 1)[1]
    assert "Payload reconstruction unavailable." in view
    assert "[0] 0x4000 (pop_rdi):\n    pop rdi\n    ret" in view
    assert "RDI = 0x0" in view and "RSP = 0x8000" in view
    assert "Offsets are relative to initial RSP = 0x8000." in view
    assert "stack[0x00] = 0x0000000000000007" in view
    assert "stack[0x08] = 0x0000000000009000" in view
    assert "stack[0x00] = 0x0000000000004000" not in view
    assert "Initial memory values (M0):\nmemory[0x2000] = 0x000000000000002a" in view


def test_payload_view_keeps_the_solved_sequence_order(tmp_path):
    document = {"input_version": 1,
                "candidates": [{"id": "first", "address": "0x4000", "bytes": "5f c3"},
                               {"id": "second", "address": "0x5000", "bytes": "5e c3"}],
                "initial": {"registers": {"rdi": 0, "rsi": 0, "rsp": "0x8000"},
                            "stack": [7, "0x5000", 8, "0x9000"]},
                "goals": {"rdi": 7, "rsi": 8, "rip": "0x9000"}, "max_depth": 2}
    result = run("search", write_input(tmp_path, document), "--view", "payload")
    assert result.returncode == 0, result.stderr
    assert 'Selected candidate IDs: ["first", "second"]' in result.stdout
    assert result.stdout.index("[0] 0x4000 (first):") < result.stdout.index("[1] 0x5000 (second):")
    assert "stack[0x08] = 0x0000000000005000" in result.stdout
    assert "stack[0x10] = 0x0000000000000008" in result.stdout


def test_missing_initial_register_is_not_completed_as_zero(tmp_path):
    document = {"input_version": 1,
                "candidates": [{"id": "overwrite", "address": "0x4000", "bytes": "b8 0a 00 00 00 c3"}],
                "goals": {"rax": 10}, "max_depth": 1}
    result = run("search", write_input(tmp_path, document), "--view", "payload")
    assert result.returncode == 0, result.stderr
    view = result.stdout.split("Payload view:\n", 1)[1]
    registers = view.split("Initial register values:\n", 1)[1].split("Stack values:", 1)[0]
    assert "RAX =" not in registers
    assert "Unresolved initial registers: RAX" in registers
    assert 'Final goal registers: {"rax": "0xa"}' in result.stdout


def test_view_only_reads_saved_result_and_does_not_solve_or_mutate(tmp_path, monkeypatch, capsys):
    path = write_input(tmp_path)
    _, pool, goals, options = command.load_input(path, "search")
    saved = search_sequences(pool, goals, **options)
    effect = saved.solve_result.effect
    constraints = saved.solve_result.constraints
    assignments = dict(saved.solve_result.solution.assignments)
    memory = dict(saved.solve_result.solution.memory_values)
    calls = []

    def existing_result(*args, **kwargs):
        calls.append(1)
        # The real search returns members of the current CLI candidate pool.
        return replace(saved, sequence=(args[0][0],))

    def forbidden_check(self):
        raise AssertionError("presentation must not invoke a solver check")

    monkeypatch.setattr(command, "search_sequences", existing_result)
    monkeypatch.setattr(SolverContext, "solve", forbidden_check)
    assert command.main(["search", str(path)]) == 0
    raw = capsys.readouterr()
    assert command.main(["search", str(path), "--view", "payload"]) == 0
    formatted = capsys.readouterr()
    assert formatted.out.split("\nPayload view:", 1)[0] == raw.out
    assert not raw.err and not formatted.err and len(calls) == 2
    assert saved.solve_result.effect == effect and saved.solve_result.constraints == constraints
    assert dict(saved.solve_result.solution.assignments) == assignments
    assert dict(saved.solve_result.solution.memory_values) == memory


def test_non_sat_view_cannot_display_a_valid_sequence_or_witness(tmp_path, monkeypatch, capsys):
    document = {"input_version": 1,
                "candidates": [{"id": "fixed", "address": "0x4000", "bytes": "b8 01 00 00 00 c3"}],
                "goals": {"rax": 2}, "max_depth": 1}
    path = write_input(tmp_path, document)
    assert command.main(["search", str(path), "--view", "payload"]) == 1
    unsat = capsys.readouterr()
    assert "STATUS: NOT_FOUND" in unsat.out and "Solver status: UNSAT" in unsat.out
    monkeypatch.setattr(SolverContext, "solve", lambda self: SolveResult(
        SolveStatus.UNKNOWN, reason_unknown="controlled resource limit"))
    assert command.main(["search", str(path), "--view", "payload"]) == 3
    unknown = capsys.readouterr()
    assert "STATUS: INCONCLUSIVE" in unknown.out and "Solver status: UNKNOWN" in unknown.out
    for output in (unsat, unknown):
        view = output.out.split("Payload view:\n", 1)[1]
        assert "Selected sequence: unavailable (no SAT result)" in view
        assert "Initial register values: unavailable" in view
        assert "stack[" not in view and "Input witness" not in output.out
        assert not output.err


def test_version_command_needs_no_input_file():
    result = run("--version")
    assert result.returncode == 0 and result.stdout == "minirop 1.0\n" and not result.stderr


def test_view_and_input_errors_remain_explicit(tmp_path):
    invalid_mode = run("search", "examples/search.json", "--view", "invalid")
    assert invalid_mode.returncode == 2 and "invalid choice" in invalid_mode.stderr
    malformed = tmp_path / "broken.json"
    malformed.write_text('{"input_version": ', encoding="utf-8")
    invalid_file = run("search", malformed, "--view", "payload")
    assert invalid_file.returncode == 2 and str(malformed) in invalid_file.stderr
    for result in (invalid_mode, invalid_file):
        assert "STATUS:" not in result.stdout and "Payload view:" not in result.stdout


def test_delivery_examples_and_generated_elf_readable_search(elf_path, tmp_path):
    for name in ("solve.json", "search.json"):
        assert (ROOT / "examples" / name).is_file()
    exported = tmp_path / "sample_gadgets.json"
    scan = run("gadgets", elf_path, "-o", exported, "--supported-only")
    assert scan.returncode == 0, scan.stderr
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"input_version": 1,
                                 "initial": {"registers": {"rdi": 0, "rsp": "0x8000"},
                                             "stack": [7, "0x9000"]},
                                 "goals": {"rdi": 7, "rip": "0x9000"},
                                 "max_depth": 1, "pruning": False}), encoding="utf-8")
    found = run("search", exported, "--config", config, "--view", "payload")
    assert found.returncode == 0, found.stderr
    assert "STATUS: FOUND" in found.stdout and "Payload view:" in found.stdout
    assert "[0] 0x401000 (gadget_0x401000_0):\n    pop rdi\n    ret" in found.stdout
    assert "stack[0x00] = 0x0000000000000007" in found.stdout
    fixed = run("solve", "examples/solve.json")
    assert fixed.returncode == 0 and "Solver status: SAT" in fixed.stdout
    search = run("search", "examples/search.json")
    assert search.returncode == 0 and 'Selected candidate IDs: ["add_two"]' in search.stdout
