"""Canonical input identity, independent of execution and constraint solving.

Version 1 covers the current x86-64 input model. It describes structure, not
algebraic equivalence, execution evidence, or producer authenticity.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType

from .constraints import Equality, RegisterGoal
from .flags import UNDEFINED
from .gadget import Gadget, Instruction, Operand
from .search import _finite_cost
from .state import CPUState, REGISTER_NAMES, make_symbolic_initial_state
from .symbolic import BoolSymbol, Add, Constant, InitialMemory, MemoryRead, Sub, Symbol, WORD_MASK, Xor


def _require(condition, message):
    if not condition:
        raise ValueError(f"invalid analysis manifest: {message}")


def _native(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        _require(isfinite(value), "nonfinite number")
        return value
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "string keys required")
        return {key: _native(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_native(item) for item in value]
    raise ValueError("invalid analysis manifest: unsupported runtime object")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_json_bytes(value) -> bytes:
    """Canonical JSON for this schema, never a dump of arbitrary objects."""
    return json.dumps(_native(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fields(value, names):
    _require(type(value) is dict and set(value) == set(names.split()), "descriptor fields")


def _int(value, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), "integer range")


def _list(value):
    _require(type(value) is list, "list required")


def _register(name):
    _require(type(name) is str, "register name")
    return CPUState._register(name)


def _expression(value, *, concrete=False):
    if type(value) is int:
        return dict(kind="integer" if concrete else "constant", value=value & WORD_MASK)
    if type(value) is Constant:
        return dict(kind="constant", value=value.value)
    if type(value) is Symbol:
        return dict(kind="symbol", name=value.name)
    if type(value) is MemoryRead:
        _require(isinstance(value.memory, InitialMemory) and value.width == 64,
                 "memory versions or access widths unsupported by manifest version 1")
        return dict(kind="memory_read", address=_expression(value.address))
    if type(value) in (Add, Sub, Xor):
        return dict(kind={Add: "add", Sub: "sub", Xor: "xor"}[type(value)],
                    left=_expression(value.left), right=_expression(value.right))
    raise ValueError("invalid analysis manifest: unsupported expression")


def _check_expression(node, *, state=False, root=True):
    _require(type(node) is dict and type(node.get("kind")) is str, "expression")
    kind = node["kind"]
    if kind == "unrepresented" and state and root:
        _fields(node, "kind")
    elif kind in ("constant", "integer"):
        _require(kind != "integer" or (state and root), "integer node context")
        _fields(node, "kind value")
        _int(node["value"], maximum=WORD_MASK)
    elif kind == "symbol":
        _fields(node, "kind name")
        _require(type(node["name"]) is str and bool(node["name"]), "symbol name")
    elif kind == "memory_read":
        _fields(node, "kind address")
        _check_expression(node["address"], root=False)
    elif kind in ("add", "sub", "xor"):
        _fields(node, "kind left right")
        _check_expression(node["left"], root=False)
        _check_expression(node["right"], root=False)
    else:
        raise ValueError("invalid analysis manifest: expression kind")


def _goals(goals):
    if goals is None:
        return []
    mapping = isinstance(goals, Mapping)
    entries = [RegisterGoal(k, v) for k, v in goals.items()] if mapping else list(goals)
    result = []
    for goal in entries:
        _require(type(goal) is RegisterGoal, "RegisterGoal required")
        result.append(dict(register=_register(goal.register), expected=_expression(goal.expected)))
    if mapping:
        # Case aliases are simultaneous goals; do not collapse them into a dict.
        result.sort(key=lambda item: (item["register"], canonical_json_bytes(item["expected"])))
    return [dict(index=index, **item) for index, item in enumerate(result)]


def _check_goals(goals):
    _list(goals)
    for index, goal in enumerate(goals):
        _fields(goal, "index register expected")
        _int(goal["index"])
        _require(goal["index"] == index and goal["register"] in REGISTER_NAMES, "goal identity")
        _check_expression(goal["expected"])


def _candidate(gadget, index):
    _require(type(gadget) is Gadget, "unsupported candidate identity")
    instructions = []
    for ins in gadget.instructions:
        _require(type(ins) is Instruction, "unsupported instruction")
        operands = []
        for op in ins.operands:
            _require(type(op) is Operand, "unsupported operand")
            # Version 1 cannot retain effective-address metadata. Fail closed
            # rather than fingerprint two different state transformations alike.
            _require((op.base, op.index, op.scale, op.displacement, op.segment)
                     == (None, None, 1, 0, None),
                     "effective-address metadata is unsupported by manifest version 1")
            operands.append(dict(kind=op.kind, width=op.width, register=op.register, immediate=op.immediate))
        instructions.append(dict(address=ins.address, mnemonic=ins.mnemonic, op_str=ins.op_str,
                                 size=ins.size, bytes=ins.bytes.hex(), operands=operands,
                                 prefixes=list(ins.prefixes)))
    return dict(index=index, address=gadget.address, bytes=gadget.bytes.hex(), instructions=instructions)


def _hex(value):
    _require(type(value) is str and len(value) % 2 == 0 and all(c in "0123456789abcdef" for c in value), "canonical bytes")


def _check_candidates(candidates):
    _list(candidates)
    for index, candidate in enumerate(candidates):
        _fields(candidate, "index address bytes instructions")
        _int(candidate["index"])
        _require(candidate["index"] == index, "candidate index")
        _int(candidate["address"], maximum=WORD_MASK)
        _hex(candidate["bytes"])
        _list(candidate["instructions"])
        for ins in candidate["instructions"]:
            _fields(ins, "address mnemonic op_str size bytes operands prefixes")
            _int(ins["address"], maximum=WORD_MASK)
            _int(ins["size"], 1)
            _hex(ins["bytes"])
            _require(type(ins["mnemonic"]) is str and type(ins["op_str"]) is str, "instruction text")
            _list(ins["prefixes"])
            for prefix in ins["prefixes"]:
                _int(prefix, maximum=255)
            _list(ins["operands"])
            for op in ins["operands"]:
                _fields(op, "kind width register immediate")
                _require(op["kind"] in ("register", "immediate", "memory", "unsupported"), "operand kind")
                _int(op["width"])
                _require(op["register"] is None or type(op["register"]) is str, "operand register")
                _require(op["immediate"] is None or type(op["immediate"]) is int, "operand immediate")


def _state(state):
    if state is None:
        state = make_symbolic_initial_state()
    if type(state) is not CPUState:
        return None
    _require(isinstance(state.memory_state, InitialMemory), "memory versions unsupported by manifest version 1")
    _require(all(value == (UNDEFINED if name == "AF" else BoolSymbol(f"{name.lower()}_0"))
                 for name, value in state.flag_bindings()),
             "explicit flag states unsupported by manifest version 1")
    registers, memory, base, slots = state.input_bindings()

    def value(item):
        try:
            return _expression(item, concrete=True)
        except ValueError:
            return dict(kind="unrepresented")

    return dict(registers=[dict(register=k, value=value(v)) for k, v in sorted(registers)],
                memory=[dict(address=k, value=value(v)) for k, v in sorted(memory)],
                stack_base=None if base is None else value(base),
                stack_slots=[dict(offset=k, value=value(v)) for k, v in sorted(slots)])


def _check_state(state):
    if state is None:
        return ["unrepresented_initial_state"]
    _fields(state, "registers memory stack_base stack_slots")
    incomplete = False
    base = state["stack_base"]
    if base is not None:
        _check_expression(base, state=True)
        _require(base["kind"] in ("symbol", "unrepresented"), "stack base")
        incomplete |= base["kind"] == "unrepresented"
    for section, key in (("registers", "register"), ("memory", "address"), ("stack_slots", "offset")):
        _list(state[section])
        keys = []
        for item in state[section]:
            _fields(item, f"{key} value")
            k = item[key]
            if key == "register":
                _require(type(k) is str and k in REGISTER_NAMES, "state register")
            elif key == "address":
                _int(k, maximum=WORD_MASK - 7)
            else:
                _int(k, -(1 << 63), (1 << 63) - 8)
                _require(k % 8 == 0 and base is not None, "stack offset")
            keys.append(k)
            _check_expression(item["value"], state=True)
            incomplete |= item["value"]["kind"] == "unrepresented"
        _require(keys == sorted(set(keys)), "state binding order or duplication")
        if key == "address":
            _require(all(b - a >= 8 for a, b in zip(keys, keys[1:])), "overlapping memory")
    return ["unrepresented_initial_state"] if incomplete else []


def _cost(value):
    try:
        value = _finite_cost(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid analysis manifest: cost") from error
    return 0.0 if type(value) is float and value == 0 else value


def _validate_payload(data):
    _fields(data, "manifest_version analysis_kind architecture candidates goals initial_state semantic_options analysis_options complete incomplete_reasons")
    _require(type(data["manifest_version"]) is int and data["manifest_version"] == 1, "unsupported manifest_version")
    _require(data["analysis_kind"] in ("fixed_sequence", "search"), "analysis kind")
    _require(data["architecture"] == "x86_64", "architecture")
    _check_candidates(data["candidates"])
    _check_goals(data["goals"])
    reasons = _check_state(data["initial_state"])
    semantic, analysis = data["semantic_options"], data["analysis_options"]
    _fields(semantic, "max_depth candidate_costs check_entry constraints checkpoints memory_name")
    _fields(analysis, "cost_mode pruning record_trace diagnose minimize_conflict minimize_conflicts max_diagnosed_candidates timeout_ms")
    for key in ("pruning", "record_trace", "diagnose", "minimize_conflict", "minimize_conflicts"):
        _require(type(analysis[key]) is bool, key)
    _require(type(semantic["check_entry"]) is bool, "check_entry")
    _require(type(semantic["memory_name"]) is str and bool(semantic["memory_name"]), "memory_name")
    _require(not analysis["minimize_conflict"] or analysis["diagnose"], "minimize_conflict requires diagnose")
    for key in ("max_diagnosed_candidates", "timeout_ms"):
        if analysis[key] is not None:
            _int(analysis[key])
    _list(semantic["constraints"])
    for constraint in semantic["constraints"]:
        _fields(constraint, "left right")
        _check_expression(constraint["left"])
        _check_expression(constraint["right"])
    _list(semantic["checkpoints"])
    steps = []
    for checkpoint in semantic["checkpoints"]:
        _fields(checkpoint, "step goals")
        _int(checkpoint["step"], maximum=len(data["candidates"]) - 1)
        steps.append(checkpoint["step"])
        _check_goals(checkpoint["goals"])
    _require(steps == sorted(set(steps)), "checkpoint order")
    if data["analysis_kind"] == "search":
        _int(semantic["max_depth"])
        _require(not steps, "search checkpoints unsupported")
        _require(analysis["cost_mode"] in ("default", "candidate_costs"), "cost mode")
        costs = semantic["candidate_costs"]
        _list(costs)
        _require(len(costs) == len(data["candidates"]), "cost vector length")
        semantic["candidate_costs"] = [_cost(v) for v in costs]
        if analysis["cost_mode"] == "default":
            _require(all(type(v) is int and v == 1 for v in costs), "default costs")
    else:
        _require(bool(data["candidates"]), "empty sequence")
        _require(semantic["max_depth"] is None and semantic["candidate_costs"] is None and
                 analysis["cost_mode"] is None, "fixed sequence search options")
        _require(not analysis["pruning"] and not analysis["record_trace"] and
                 not analysis["minimize_conflicts"] and analysis["max_diagnosed_candidates"] is None,
                 "fixed sequence reporting options")
    _require(type(data["complete"]) is bool and data["complete"] is (not reasons), "completeness")
    _require(data["incomplete_reasons"] == reasons, "incomplete reasons")


def _fingerprints(payload):
    semantic = {k: v for k, v in payload.items() if k != "analysis_options"}
    return (sha256(canonical_json_bytes(semantic)).hexdigest(),
            sha256(canonical_json_bytes(payload)).hexdigest())


@dataclass(frozen=True)
class AnalysisManifest:
    manifest_version: int
    analysis_kind: str
    architecture: str
    candidates: tuple
    goals: tuple
    initial_state: Mapping | None
    semantic_options: Mapping
    analysis_options: Mapping
    complete: bool
    incomplete_reasons: tuple[str, ...]
    semantic_fingerprint: str | None = None
    analysis_fingerprint: str | None = None

    def __post_init__(self):
        payload = self.to_dict()
        stored = payload.pop("semantic_fingerprint"), payload.pop("analysis_fingerprint")
        _validate_payload(payload)
        fingerprints = _fingerprints(payload)
        if stored != (None, None):
            _require(stored == fingerprints, "fingerprint mismatch")
        for name, value in payload.items():
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "semantic_fingerprint", fingerprints[0])
        object.__setattr__(self, "analysis_fingerprint", fingerprints[1])

    @property
    def fingerprint(self):
        """Convenience alias for semantic_fingerprint."""
        return self.semantic_fingerprint

    def to_dict(self):
        return {name: _native(getattr(self, name)) for name in self.__dataclass_fields__}

    def canonical_bytes(self, *, analysis=False):
        payload = self.to_dict()
        del payload["semantic_fingerprint"], payload["analysis_fingerprint"]
        if not analysis:
            del payload["analysis_options"]
        return canonical_json_bytes(payload)

    def validate(self):
        data = self.to_dict()
        stored = data.pop("semantic_fingerprint"), data.pop("analysis_fingerprint")
        _validate_payload(data)
        _require(stored == _fingerprints(data), "fingerprint mismatch")
        return True

    @classmethod
    def from_dict(cls, data):
        data = _native(data)
        _require(type(data) is dict and set(data) == set(cls.__dataclass_fields__), "manifest fields")
        for key in ("semantic_fingerprint", "analysis_fingerprint"):
            _require(type(data[key]) is str and len(data[key]) == 64, "fingerprint format")
        return cls(**data)


def create_analysis_manifest(*, sequence=None, candidates=None, goals=None, initial_state=None,
                             max_depth=None, check_entry=False, constraints=(), checkpoints=None,
                             memory_name="memory_0", timeout_ms=5000, pruning=False,
                             cost_fn=None, candidate_costs=None, record_trace=False, diagnose=False,
                             minimize_conflict=False, minimize_conflicts=False,
                             max_diagnosed_candidates=None) -> AnalysisManifest:
    """Describe supplied inputs without executing them or invoking a solver.

    cost_fn, when supplied, is evaluated once per candidate entry. It must be
    pure and deterministic, with the same values used by search. Prefer an
    explicit candidate_costs vector when costs were already captured. Unknown
    state values are marked incomplete; unsupported expressions elsewhere fail.
    """
    _require((sequence is None) != (candidates is None), "supply sequence OR candidates")
    fixed = sequence is not None
    pool = sequence if fixed else candidates
    _require(isinstance(pool, Sequence) and not isinstance(pool, (str, bytes)), "ordered sequence required")
    pool = tuple(pool)
    _require(cost_fn is None or candidate_costs is None, "cost_fn OR candidate_costs")
    if fixed:
        _require(cost_fn is None and candidate_costs is None, "fixed sequence costs")
        costs, mode = None, None
    else:
        mode = "default" if cost_fn is None and candidate_costs is None else "candidate_costs"
        costs = ([_cost(cost_fn(g)) for g in pool] if cost_fn is not None else
                 [_cost(v) for v in candidate_costs] if candidate_costs is not None else [1] * len(pool))
    equalities = []
    for equality in constraints:
        _require(type(equality) is Equality, "Equality required")
        equalities.append(dict(left=_expression(equality.left), right=_expression(equality.right)))
    checkpoint_data = []
    if checkpoints is not None:
        _require(isinstance(checkpoints, Mapping), "checkpoint mapping")
        for step, conditions in checkpoints.items():
            _int(step)
            _require(isinstance(conditions, Mapping), "checkpoint conditions")
            checkpoint_data.append(dict(step=step, goals=_goals(conditions)))
        checkpoint_data.sort(key=lambda item: item["step"])
    state = _state(initial_state)
    reasons = _check_state(state)
    payload = dict(manifest_version=1, analysis_kind="fixed_sequence" if fixed else "search",
                   architecture="x86_64", candidates=[_candidate(g, i) for i, g in enumerate(pool)],
                   goals=_goals(goals), initial_state=state,
                   semantic_options=dict(max_depth=max_depth, candidate_costs=costs,
                       check_entry=check_entry, constraints=equalities, checkpoints=checkpoint_data, memory_name=memory_name),
                   analysis_options=dict(cost_mode=mode, pruning=pruning, record_trace=record_trace,
                       diagnose=diagnose, minimize_conflict=minimize_conflict, minimize_conflicts=minimize_conflicts,
                       max_diagnosed_candidates=max_diagnosed_candidates, timeout_ms=timeout_ms),
                   complete=not reasons, incomplete_reasons=reasons)
    return AnalysisManifest(**payload)
