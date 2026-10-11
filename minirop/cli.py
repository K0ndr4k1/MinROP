"""File-driven solve/search and the original gadget presentation command."""

import argparse
import json
from pathlib import Path
import re
import sys

from .constraints import Equality
from .disasm import Disassembler, GadgetScanner
from .elf import ELFImage, ELFLoadError
from .gadget import Gadget
from .search import search_sequences
from .semantics import lift_gadget
from .solver import SolveStatus, solve_sequence
from .state import make_symbolic_initial_state, register_view
from .symbolic import Expression, WORD_MASK, evaluate, relative_offset


VERSION = "1.0"


def _object(value, location, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(allowed)
    if missing:
        raise ValueError(f"{location}: missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{location}: unknown fields: {', '.join(sorted(extra))}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_json_number(value):
    raise ValueError(f"invalid JSON number: {value}")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_object,
                          parse_constant=_invalid_json_number)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{error.lineno}:{error.colno}: {error.msg}") from error


def _export_gadgets(source, output, scanned, *, supported_only=False):
    destination = Path(output)
    if destination.resolve() == Path(source).resolve() or (destination.exists() and destination.samefile(source)):
        raise ValueError("output must not overwrite the input ELF")
    entries = []
    excluded = 0
    for index, gadget in enumerate(scanned):
        if supported_only:
            try:
                lift_gadget(gadget)
            except ValueError:
                excluded += 1
                continue
        entries.append({"id": f"gadget_0x{gadget.address:x}_{index}",
                        "address": f"0x{gadget.address:x}",
                        "instructions": [str(instruction) for instruction in gadget.instructions],
                        "bytes": gadget.bytes.hex()})
    document = {"input_version": 1, "architecture": "x86_64", "source": Path(source).as_posix(), "gadgets": entries}
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    message = f"minirop: wrote {len(entries)} gadgets to {output}"
    if supported_only:
        message += f"; excluded {excluded} unsupported gadgets (--supported-only)"
    print(message, file=sys.stderr)


def _integer(value, location, width=64, *, signed=True):
    if isinstance(value, str) and re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|[0-9]+)", value):
        value = int(value, 16 if "x" in value.lower() else 10)
    if type(value) is not int:
        raise ValueError(f"{location}: expected an integer or decimal/hex integer string")
    minimum = -(1 << (width - 1)) if signed else 0
    if not minimum <= value < (1 << width):
        raise ValueError(f"{location}: value is outside the {width}-bit range")
    return value & ((1 << width) - 1)


def _register_values(value, location):
    _object(value, location, value.keys() if isinstance(value, dict) else ())
    result = {}
    for register, raw in value.items():
        try:
            _, width, _ = register_view(register)
        except ValueError as error:
            raise ValueError(f"{location}.{register}: {error}") from error
        normalized = register.lower()
        if normalized in result:
            raise ValueError(f"{location}: duplicate register {normalized}")
        result[normalized] = _integer(raw, f"{location}.{register}", width)
    return result


def load_input(path, mode, config_path=None):
    """Strict minimal JSON adapter; decoded operands are never reconstructed by hand.

    The old analysis manifest remains an identity/report format. This runnable
    input describes actual supplied bytes, with no scanning or address discovery.
    A scanner JSON catalog uses the same decoding path; an optional search
    config supplies goals and bounds without copying gadget records.
    Initial register equalities stay in the same context as M0 and final goals.
    """
    document = _read_json(path)
    catalog = isinstance(document, dict) and "gadgets" in document
    if catalog and mode != "search":
        raise ValueError("gadget JSON catalogs are accepted by search only")
    if catalog and (type(document.get("input_version")) is not int or document["input_version"] != 1):
        raise ValueError("input.input_version: only version 1 is supported")
    if config_path is not None:
        if not catalog or mode != "search":
            raise ValueError("--config requires a gadget JSON catalog as search input")
        config = _object(_read_json(config_path), "config",
                         {"input_version", "architecture", "initial", "goals", "max_depth", "pruning", "check_entry", "timeout_ms"},
                         {"input_version", "goals", "max_depth"})
        for key, value in config.items():
            if key in document:
                if (key in {"input_version", "architecture"} and type(value) is type(document[key])
                        and value == document[key]):
                    continue
                raise ValueError(f"config.{key}: conflicts with a field already present in the gadget JSON")
            document[key] = value
    if catalog and ("goals" not in document or "max_depth" not in document):
        raise ValueError("gadget JSON requires goals and max_depth; supply --config SEARCH.json "
                         "or include these fields in the gadget JSON")
    collection = "gadgets" if catalog else "candidates"
    common = {"input_version", "architecture", "candidates", "initial", "goals", "check_entry", "timeout_ms"}
    if catalog:
        common = (common - {"candidates"}) | {"gadgets", "source"}
    _object(document, "input", common | ({"sequence"} if mode == "solve" else {"max_depth", "pruning"}),
            {"input_version", collection, "goals", "sequence" if mode == "solve" else "max_depth"})
    if type(document["input_version"]) is not int or document["input_version"] != 1:
        raise ValueError("input.input_version: only version 1 is supported")
    if document.get("architecture", "x86_64") != "x86_64":
        raise ValueError("input.architecture: only x86_64 is supported")
    if catalog and (not isinstance(document.get("source"), str) or not document["source"]):
        raise ValueError("input.source: expected the original ELF path as a nonempty string")
    entries = document[collection]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"input.{collection}: expected a nonempty finite array")
    candidates = {}
    code = {}
    decoder = Disassembler()
    for index, entry in enumerate(entries):
        location = f"input.{collection}[{index}]"
        fields = {"id", "address", "bytes"} | ({"instructions"} if catalog else set())
        _object(entry, location, fields, fields)
        identifier = entry["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", identifier):
            raise ValueError(f"{location}.id: use 1..128 letters, digits, '.', '_' or '-'")
        if identifier in candidates:
            raise ValueError(f"{location}.id: duplicate candidate ID {identifier}")
        address = _integer(entry["address"], f"{location}.address", signed=False)
        raw = entry["bytes"]
        if not isinstance(raw, str):
            raise ValueError(f"{location}.bytes: expected hexadecimal instruction bytes")
        try:
            data = bytes.fromhex(raw)
        except ValueError as error:
            raise ValueError(f"{location}.bytes: invalid hexadecimal instruction bytes") from error
        if not data or address + len(data) > WORD_MASK + 1:
            raise ValueError(f"{location}: nonempty code must fit the 64-bit address space")
        instructions = decoder.disassemble(data, address)
        if not instructions or sum(i.size for i in instructions) != len(data):
            raise ValueError(f"{location}.bytes: invalid or incomplete x86-64 instruction encoding")
        if catalog and entry["instructions"] != [str(instruction) for instruction in instructions]:
            raise ValueError(f"{location}.instructions: must match the ordered instructions decoded from bytes")
        gadget = Gadget(address, data, instructions)
        try:
            lift_gadget(gadget)
        except ValueError as error:
            raise ValueError(f"{location} ({identifier}): {error}") from error
        for offset, byte in enumerate(data):
            at = address + offset
            if at in code and code[at] != byte:
                raise ValueError(f"{location}: conflicting code bytes at 0x{at:x}")
            code[at] = byte
        candidates[identifier] = gadget

    initial = _object(document.get("initial", {}), "input.initial", {"registers", "flags", "memory", "stack"})
    stack = initial.get("stack", [])
    if not isinstance(stack, list):
        raise ValueError("input.initial.stack: expected an array of 64-bit words")
    state = make_symbolic_initial_state(max(8, len(stack)))
    constraints = tuple(Equality(state.get_register(name), value) for name, value in
                        _register_values(initial.get("registers", {}), "input.initial.registers").items())
    flags = _object(initial.get("flags", {}), "input.initial.flags", {"CF", "ZF", "SF", "OF", "PF", "AF"})
    for name, value in flags.items():
        if type(value) is not bool:
            raise ValueError(f"input.initial.flags.{name}: expected true or false")
        state.set_flag(name, value)
    memory = initial.get("memory", [])
    if not isinstance(memory, list):
        raise ValueError("input.initial.memory: expected an array of 64-bit word bindings")
    for index, binding in enumerate(memory):
        location = f"input.initial.memory[{index}]"
        _object(binding, location, {"address", "value"}, {"address", "value"})
        address = _integer(binding["address"], f"{location}.address", signed=False)
        value = _integer(binding["value"], f"{location}.value")
        if any(at == address for at, _ in state.memory_bindings()):
            raise ValueError(f"{location}: duplicate memory address 0x{address:x}")
        try:
            state.write_memory(address, value)
        except ValueError as error:
            raise ValueError(f"{location}: {error}") from error
    for index, value in enumerate(stack):
        state.set_stack_slot(index, _integer(value, f"input.initial.stack[{index}]"))
    goals = _register_values(document["goals"], "input.goals")
    if not goals:
        raise ValueError("input.goals: at least one final register goal is required")
    options = {"initial_state": state, "constraints": constraints,
               "report_execution_errors": True}
    check_entry = document.get("check_entry", False)
    if type(check_entry) is not bool:
        raise ValueError("input.check_entry: expected true or false")
    options["check_entry"] = check_entry
    timeout = document.get("timeout_ms", 5000)
    if type(timeout) is not int or not 1 <= timeout <= (1 << 31) - 1:
        raise ValueError("input.timeout_ms: expected a positive integer below 2^31")
    options["timeout_ms"] = timeout
    if mode == "solve":
        sequence = document["sequence"]
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("input.sequence: expected a nonempty array of candidate IDs")
        for index, identifier in enumerate(sequence):
            if not isinstance(identifier, str) or identifier not in candidates:
                raise ValueError(f"input.sequence[{index}]: unknown candidate ID {identifier!r}")
        selected = tuple(candidates[identifier] for identifier in sequence)
    else:
        depth = document["max_depth"]
        if type(depth) is not int or depth < 1:
            raise ValueError("input.max_depth: expected a positive integer")
        pruning = document.get("pruning", False)
        if type(pruning) is not bool:
            raise ValueError("input.pruning: expected true or false")
        options.update(max_depth=depth, pruning=pruning)
        selected = tuple(candidates.values())
    return candidates, selected, goals, options


def _payload_view(sequence, result, identifiers):
    """Format the saved SAT witness; never execute, solve, or complete a model."""
    lines = ["", "Payload view:", "Payload reconstruction unavailable."]
    if result is None or result.status is not SolveStatus.SAT:
        lines.extend(["Selected sequence: unavailable (no SAT result)",
                      "Initial register values: unavailable", "Stack values: unavailable",
                      "Initial memory values (M0): unavailable"])
        return "\n".join(lines)
    solution = result.solution
    initial = result.effect.initial_state

    def observed(value):
        try:
            number = evaluate(value, solution.assignments, memory=solution.memory_values)
        except (KeyError, ValueError, TypeError):
            return None
        return number if type(number) is int else None

    lines.extend(["Values describe one solved initial-state witness.", "", "Selected sequence:"])
    for index, gadget in enumerate(sequence):
        lines.append(f"[{index}] 0x{gadget.address:x} ({identifiers[id(gadget)]}):")
        lines.extend(f"    {instruction}" for instruction in gadget.instructions)

    lines.extend(["", "Initial register values:"])
    registers = []
    unresolved = []
    for name, value in initial.register_bindings():
        number = observed(value)
        if number is None:
            unresolved.append(name.upper())
        else:
            registers.append(f"{name.upper()} = 0x{number:x}")
    lines.extend(registers or ["unavailable (no solved initial register values)"])
    if unresolved:
        lines.append("Unresolved initial registers: " + ", ".join(unresolved))

    base = initial.get_register("rsp")
    rsp = observed(base)
    lines.extend(["", "Stack values:",
                  "Offsets are relative to initial RSP" + (f" = 0x{rsp:x}." if rsp is not None else " (unavailable).")])
    stack = []
    unresolved_slots = []
    stack_addresses = set()
    memory = dict(solution.memory_values)
    for address, value in initial.memory_bindings():
        pointer, word = observed(address), observed(value)
        offset = relative_offset(address, base) if isinstance(address, Expression) and isinstance(base, Expression) else None
        if offset is not None:
            label = f"-0x{-offset:02x}" if offset < 0 else f"0x{offset:02x}"
            if word is not None:
                stack.append(f"stack[{label}] = 0x{word:016x}")
                if pointer is not None:
                    stack_addresses.add(pointer)
            else:
                unresolved_slots.append(label)
        elif pointer is not None and word is not None:
            memory[pointer] = word
    lines.extend(stack or ["unavailable (no solved initial stack slots)"])
    if unresolved_slots:
        lines.append("Unresolved stack offsets: " + ", ".join(unresolved_slots))
    lines.extend(["", "Initial memory values (M0):"])
    words = [f"memory[0x{address:x}] = 0x{word:016x}" for address, word in sorted(memory.items())
             if address not in stack_addresses]
    lines.extend(words or ["No additional solved memory words."])
    return "\n".join(lines)


def _run_analysis(path, mode, config_path=None, view="raw"):
    candidates, selected, goals, options = load_input(path, mode, config_path)
    if mode == "solve":
        result = solve_sequence(selected, goals, **options)
        status = {SolveStatus.SAT: "FOUND", SolveStatus.UNSAT: "NOT_FOUND",
                  SolveStatus.UNKNOWN: "INCONCLUSIVE"}[result.status]
        solver_status = result.status.value.upper()
        sequence = selected if result.status is SolveStatus.SAT else ()
        depth, tested = len(selected), 1
    else:
        search = search_sequences(selected, goals, **options)
        status = search.status.value.upper()
        solver_status = {"FOUND": "SAT", "NOT_FOUND": "UNSAT", "INCONCLUSIVE": "UNKNOWN"}[status]
        result = search.solve_result
        sequence, depth, tested = search.sequence, search.depth, search.tested_candidates
    satisfied = {"FOUND": "True", "NOT_FOUND": "False", "INCONCLUSIVE": "Unknown"}[status]
    identifiers = {id(gadget): name for name, gadget in candidates.items()}
    if status == "FOUND":
        solution = result.solution
        final = {name: f"0x{evaluate(result.effect.final_state.get_register(name), solution.assignments, memory=solution.memory_values):x}"
                 for name in goals}
    print(f"STATUS: {status}")
    print(f"Solver status: {solver_status}")
    print(f"Goals satisfied: {satisfied}")
    print("Selected candidate IDs: " + json.dumps([identifiers[id(g)] for g in sequence]))
    print(f"Sequence depth: {depth if depth is not None else '-'}")
    print(f"Tested candidates: {tested}")
    if status == "FOUND":
        print("Final goal registers: " + json.dumps(final, sort_keys=True))
        print("Input witness (one model): " + json.dumps(dict(solution.assignments), sort_keys=True))
        if solution.memory_values:
            print("Initial memory witness: " + json.dumps(
                {f"0x{at:x}": f"0x{word:x}" for at, word in solution.memory_values.items()}, sort_keys=True))
        if view == "payload":
            print(_payload_view(sequence, result, identifiers))
        return 0
    if status == "NOT_FOUND":
        print("No solution only within the supplied sequence / finite candidate pool and depth bound.")
        if view == "payload":
            print(_payload_view(sequence, result, identifiers))
        return 1
    reason = result.solution.reason_unknown if result is not None else None
    print("Reason: " + (reason or "one or more candidates returned UNKNOWN or could not complete defined normal execution"))
    print("No claim of unsatisfiability is made.")
    if view == "payload":
        print(_payload_view(sequence, result, identifiers))
    return 3


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minirop")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    gadgets = subcommands.add_parser("gadgets", help="scan ELF64 x86-64 executable segments")
    gadgets.add_argument("elf")
    gadgets.add_argument("--max-gadget-bytes", type=positive_int, default=20)
    gadgets.add_argument("--max-instructions", type=positive_int, default=5)
    gadgets.add_argument("-o", "--output", help="write a UTF-8 gadget JSON catalog instead of text")
    gadgets.add_argument("--supported-only", action="store_true",
                         help="export only gadgets accepted by current semantics (requires --output)")
    for name in ("solve", "search"):
        command = subcommands.add_parser(name, help=f"{name} explicitly supplied x86-64 snippets from JSON")
        command.add_argument("input", help="UTF-8 JSON input file")
        if name == "search":
            command.add_argument("--config", help="JSON goals and search bounds for a gadget catalog")
            command.add_argument("--view", choices=("raw", "payload"), default="raw",
                                 help="append a readable initial-state witness view (default: raw)")
    args = parser.parse_args(argv)
    if args.command != "gadgets":
        try:
            return _run_analysis(args.input, args.command, getattr(args, "config", None),
                                 getattr(args, "view", "raw"))
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            print(f"minirop: {exc}", file=sys.stderr)
            return 2
    if args.supported_only and args.output is None:
        parser.error("--supported-only requires --output")
    try:
        image = ELFImage.load(args.elf)
        scanner = GadgetScanner(args.max_gadget_bytes, args.max_instructions)
        scanned = scanner.scan(image.executable_segments)
        if args.output is not None:
            _export_gadgets(args.elf, args.output, scanned, supported_only=args.supported_only)
        else:
            for gadget in scanned:
                print(f"0x{gadget.address:x}: " + " ; ".join(map(str, gadget.instructions)))
    except ELFLoadError as exc:
        print(f"minirop: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"minirop: {exc}", file=sys.stderr)
        return 2
    return 0
