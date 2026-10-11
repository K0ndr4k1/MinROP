"""Input identity, conservative coverage and snapshot comparability regressions."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from minirop.constraints import Equality, RegisterGoal
from minirop.manifest import AnalysisManifest, canonical_json_bytes, create_analysis_manifest
from minirop.search import search_sequences
from minirop.search_diagnostics import diagnose_search_result
from minirop.snapshot import (AnalysisSnapshot, ComparabilityStatus, compare_analysis_snapshots,
                              create_analysis_snapshot, load_analysis_snapshot, save_analysis_snapshot)
from minirop.solver import SolverContext, solve_sequence
from minirop.state import CPUState, make_symbolic_initial_state
from minirop.symbolic import Add, Constant, MemoryRead, Sub, Symbol, Xor
from test_sequence import gadget


@pytest.fixture
def inputs():
    return dict(candidates=[gadget("48 c7 c0 01 00 00 00 c3", 0x4000), gadget("c3", 0x4100)],
                goals={"rax": 1}, initial_state=make_symbolic_initial_state(0), max_depth=2)


def build(inputs, **overrides):
    return create_analysis_manifest(**(inputs | overrides))


def test_deterministic_bytes_hash_and_roundtrip(inputs):
    one, two = build(inputs), build(inputs)
    assert one == two and one.to_dict() == two.to_dict()
    assert one.manifest_version == 1 and one.complete and not one.incomplete_reasons
    assert one.fingerprint == sha256(one.canonical_bytes()).hexdigest()
    assert one.analysis_fingerprint == sha256(one.canonical_bytes(analysis=True)).hexdigest()
    assert one.canonical_bytes() == two.canonical_bytes()
    assert len(one.fingerprint) == 64
    assert AnalysisManifest.from_dict(json.loads(json.dumps(one.to_dict()))) == one
    assert one.validate()


@pytest.mark.parametrize("change", ["order", "duplicate", "address", "bytes", "operand", "prefix", "instruction"])
def test_candidate_identity(inputs, change):
    pool = list(inputs["candidates"])
    if change == "order":
        pool.reverse()
    elif change == "duplicate":
        pool.insert(0, pool[0])
    elif change == "address":
        pool[0] = replace(pool[0], address=0x5000)
    elif change == "bytes":
        pool[0] = replace(pool[0], bytes=b"\xc3")
    else:
        ins = pool[0].instructions[0]
        if change == "operand":
            ins = replace(ins, operands=(ins.operands[0], replace(ins.operands[1], immediate=2)))
        elif change == "prefix":
            ins = replace(ins, prefixes=(0x66,))
        else:
            ins = replace(ins, mnemonic="add")
        pool[0] = replace(pool[0], instructions=(ins,) + pool[0].instructions[1:])
    changed = build(inputs, candidates=pool)
    assert changed.fingerprint != build(inputs).fingerprint
    assert [item["index"] for item in changed.candidates] == list(range(len(pool)))


def test_duplicate_candidate_descriptor_preserved(inputs):
    a, b = inputs["candidates"]
    manifest = build(inputs, candidates=[a, a, b])
    assert len(manifest.candidates) == 3
    first, second = [dict(item) for item in manifest.candidates[:2]]
    del first["index"], second["index"]
    assert first == second


@pytest.mark.parametrize("goals", [
    {"rax": 2}, [RegisterGoal("rax", 1), RegisterGoal("rax", 1)],
    [RegisterGoal("rax", 1), RegisterGoal("rax", 2)],
])
def test_goal_value_duplicates_conflicts(inputs, goals):
    manifest = build(inputs, goals=goals)
    assert manifest.fingerprint != build(inputs).fingerprint
    assert len(manifest.goals) == len(goals)


def test_dict_order_case_and_sorted_list_equivalence(inputs):
    a = build(inputs, goals={"RBX": 2, "RAX": 1})
    b = build(inputs, goals={"rax": 1, "rbx": Constant(2)})
    c = build(inputs, goals=[RegisterGoal("RAX", 1), RegisterGoal("RBX", 2)])
    assert a == b == c
    assert build(inputs, goals=[RegisterGoal("rbx", 2), RegisterGoal("rax", 1)]) != a
    aliases = build(inputs, goals={"RAX": 2, "rax": 1})
    assert len(aliases.goals) == 2
    assert aliases == build(inputs, goals={"rax": 1, "RAX": 2})


@pytest.mark.parametrize("expression", [Constant(-1), Symbol("符号"), Add(Symbol("x"), 1),
    Sub(Symbol("x"), 2), Xor(Symbol("x"), 3), MemoryRead(Add(Symbol("rsp_0"), 8))])
def test_supported_expression_roundtrip(inputs, expression):
    manifest = build(inputs, goals={"rax": expression})
    assert AnalysisManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.fingerprint != build(inputs).fingerprint


def test_unknown_goal_expression_rejected(inputs):
    with pytest.raises(ValueError, match="expression"):
        build(inputs, goals={"rax": object()})


@pytest.mark.parametrize("change", ["register", "memory", "slot", "base", "concrete_kind"])
def test_initial_state_identity(inputs, change):
    state = inputs["initial_state"].copy()
    if change == "register":
        state.set_register("RAX", 123)
    elif change == "memory":
        state.write_memory(0x2000, 9)
    elif change == "slot":
        state.set_stack_slot(-1, 4)
    elif change == "base":
        # The stack base must survive even if there are no stack slots.
        state = CPUState(stack_base=Symbol("different_base"))
        for register, value in inputs["initial_state"].register_bindings():
            state.set_register(register, value)
    else:
        inputs["initial_state"].set_register("rsp", 0x2000)
        state = inputs["initial_state"].copy()
        state.set_register("rsp", Constant(0x2000))
    assert build(inputs, initial_state=state).fingerprint != build(inputs).fingerprint


def test_state_insertion_order_and_default(inputs):
    a, b = CPUState(stack_base=Symbol("s")), CPUState(stack_base=Symbol("s"))
    for name in ("rax", "rbx"):
        a.set_register(name, 1)
    for name in ("rbx", "rax"):
        b.set_register(name, 1)
    for address in (0x1000, 0x2000):
        a.write_memory(address, 2)
    for address in (0x2000, 0x1000):
        b.write_memory(address, 2)
    a.set_stack_slot(1, 3)
    a.set_stack_slot(0, 4)
    b.set_stack_slot(0, 4)
    b.set_stack_slot(1, 3)
    assert build(inputs, initial_state=a) == build(inputs, initial_state=b)
    assert build(inputs, initial_state=None) == build(inputs, initial_state=make_symbolic_initial_state())


@pytest.mark.parametrize("option,value", [
    ("max_depth", 3), ("check_entry", True), ("memory_name", "other_memory"),
    ("constraints", [Equality(Symbol("x"), 1)]), ("candidate_costs", [1, 2]),
])
def test_semantic_options(inputs, option, value):
    before, after = build(inputs), build(inputs, **{option: value})
    assert before.fingerprint != after.fingerprint
    assert before.analysis_fingerprint != after.analysis_fingerprint


@pytest.mark.parametrize("option,value", [
    ("pruning", True), ("record_trace", True), ("diagnose", True),
    ("minimize_conflicts", True), ("max_diagnosed_candidates", 3), ("timeout_ms", 20),
])
def test_analysis_options(inputs, option, value):
    before, after = build(inputs), build(inputs, **{option: value})
    assert before.fingerprint == after.fingerprint
    assert before.analysis_fingerprint != after.analysis_fingerprint


def test_fixed_sequence_options(inputs):
    options = dict(sequence=inputs["candidates"], initial_state=inputs["initial_state"])
    base = create_analysis_manifest(**options)
    diagnostic = create_analysis_manifest(**options, diagnose=True, minimize_conflict=True)
    checkpoint = create_analysis_manifest(**options, checkpoints={0: {"rax": 1}})
    assert base.analysis_kind == "fixed_sequence"
    assert base.fingerprint == diagnostic.fingerprint != checkpoint.fingerprint
    assert base.analysis_fingerprint != diagnostic.analysis_fingerprint


def test_cost_vector_callback_and_negative_zero(inputs):
    calls = []

    def cost(candidate):
        calls.append(candidate)
        return 2 if candidate.address == 0x4000 else 3.5

    callback = build(inputs, cost_fn=cost)
    assert calls == inputs["candidates"]
    assert callback == build(inputs, candidate_costs=[2, 3.5])
    assert build(inputs, candidate_costs=[-0.0, 1]) == build(inputs, candidate_costs=[0.0, 1])
    assert build(inputs, candidate_costs=[1, 1]).fingerprint == build(inputs).fingerprint
    # Integer and floating point accumulation are distinct in Phase 8.
    assert build(inputs, candidate_costs=[1.0, 1]).fingerprint != build(inputs).fingerprint


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_invalid_cost(inputs, value):
    with pytest.raises(ValueError):
        build(inputs, candidate_costs=[value, 1])


@pytest.mark.parametrize("options", [
    {"max_depth": True}, {"max_depth": -1}, {"max_depth": None}, {"pruning": 1},
    {"candidate_costs": [1]}, {"cost_fn": lambda _: 1, "candidate_costs": [1, 1]},
    {"sequence": []}, {"checkpoints": {0: {"rax": 1}}}, {"minimize_conflict": True},
])
def test_invalid_configuration(inputs, options):
    with pytest.raises(ValueError):
        build(inputs, **options)


def test_missing_input_and_unordered_candidates(inputs):
    with pytest.raises(ValueError):
        create_analysis_manifest()
    with pytest.raises(ValueError):
        build(inputs, candidates=set(inputs["candidates"]))
    with pytest.raises(ValueError):
        create_analysis_manifest(sequence=[])


def test_incomplete_state_roundtrip(inputs):
    state = inputs["initial_state"].copy()
    state.set_register("rax", object())
    manifest = build(inputs, initial_state=state)
    assert not manifest.complete
    assert manifest.incomplete_reasons == ("unrepresented_initial_state",)
    assert AnalysisManifest.from_dict(manifest.to_dict()) == manifest
    assert not build(inputs, initial_state=object()).complete


@pytest.mark.parametrize("change", ["version", "missing", "index", "goal_index", "depth", "hash", "complete", "reason", "state", "cost", "extra"])
def test_tampered_manifest(inputs, change):
    data = build(inputs).to_dict()
    if change == "version":
        data["manifest_version"] = 2
    elif change == "missing":
        del data["goals"]
    elif change == "index":
        data["candidates"][0]["index"] = True
    elif change == "goal_index":
        data["goals"][0]["index"] = 1
    elif change == "depth":
        data["semantic_options"]["max_depth"] = 5
    elif change == "hash":
        data["semantic_fingerprint"] = "0" * 64
    elif change == "complete":
        data["complete"] = False
    elif change == "reason":
        data["incomplete_reasons"] = ["free form error"]
    elif change == "state":
        data["initial_state"]["stack_base"] = {"kind": "integer", "value": 5}
    elif change == "cost":
        data["semantic_options"]["candidate_costs"][0] = True
    else:
        data["ignored"] = 123
    with pytest.raises(ValueError):
        AnalysisManifest.from_dict(data)


def test_canonical_json_rejects_objects_and_nan():
    assert canonical_json_bytes({"z": "中", "a": 1}) == '{"a":1,"z":"中"}'.encode()
    for value in (object(), {1: 2}, float("nan")):
        with pytest.raises(ValueError):
            canonical_json_bytes(value)


def test_detached_immutable_manifest(inputs):
    manifest = build(inputs)
    saved = manifest.to_dict()
    inputs["candidates"].reverse()
    inputs["goals"]["rax"] = 99
    inputs["initial_state"].set_register("rax", 100)
    assert manifest.to_dict() == saved
    saved["goals"][0]["expected"]["value"] = 3
    assert manifest.goals[0]["expected"]["value"] == 1
    with pytest.raises(TypeError):
        manifest.semantic_options["max_depth"] = 4


def snap(inputs, **options):
    manifest = build(inputs, **options)
    result_options = {k: v for k, v in options.items() if k in ("pruning", "record_trace")}
    result = search_sequences(**inputs, **result_options)
    return create_analysis_snapshot(search_result=result, manifest=manifest), result


def test_v2_file_roundtrip_and_v1_compatibility(inputs, tmp_path):
    snapshot, result = snap(inputs, record_trace=True)
    assert snapshot.schema_version == 2
    path = tmp_path / "analysis.json"
    save_analysis_snapshot(snapshot, path)
    assert load_analysis_snapshot(path) == snapshot
    old = create_analysis_snapshot(search_result=result)
    old_data = old.to_dict()
    # v1 ignores future fields, and never claims input identity from them.
    old_data.update(manifest=snapshot.manifest.to_dict(), semantic_fingerprint="bad")
    loaded = AnalysisSnapshot.from_dict(old_data)
    assert loaded == old and loaded.manifest is None and loaded.semantic_fingerprint is None
    diff = compare_analysis_snapshots(old, snapshot)
    assert diff.comparability is ComparabilityStatus.UNKNOWN and diff.input_changed is None
    assert not diff.semantic_changed
    assert compare_analysis_snapshots(old, old).comparability is ComparabilityStatus.UNKNOWN


@pytest.mark.parametrize("field", ["manifest", "semantic_fingerprint", "analysis_fingerprint"])
def test_snapshot_tamper(inputs, field):
    snapshot, _ = snap(inputs)
    data = snapshot.to_dict()
    if field == "manifest":
        data[field]["goals"][0]["expected"]["value"] = 99
    else:
        data[field] = "0" * 64
    with pytest.raises(ValueError):
        AnalysisSnapshot.from_dict(data)


def test_identical_inputs_metadata_and_deterministic_diff(inputs):
    snapshot, result = snap(inputs)
    other = create_analysis_snapshot(search_result=result, manifest=snapshot.manifest, metadata={"label": "new"})
    diff = compare_analysis_snapshots(snapshot, other)
    assert diff.comparability is ComparabilityStatus.IDENTICAL_INPUTS and diff.input_changed is False
    assert diff.analysis_configuration_changed is False
    assert snapshot.semantic_fingerprint == other.semantic_fingerprint
    assert snapshot.analysis_fingerprint == other.analysis_fingerprint
    assert diff.to_dict() == compare_analysis_snapshots(snapshot, other).to_dict()
    assert diff.validate(snapshot, other)
    assert not diff.semantic_changed and not diff.operational_changed


def test_different_inputs_same_results(inputs):
    snapshot, result = snap(inputs)
    other = create_analysis_snapshot(search_result=result, manifest=build(inputs, goals={"rax": 2}))
    diff = compare_analysis_snapshots(snapshot, other)
    assert diff.comparability is ComparabilityStatus.DIFFERENT_INPUTS
    assert diff.input_changed and diff.changed and not diff.semantic_changed
    assert diff.comparability_warnings and diff.validate(snapshot, other)


def test_same_inputs_changed_result(inputs):
    found, _ = snap(inputs)
    # Deliberately attach to a changed result: attachment is not a proof/replay.
    changed_result = search_sequences(**(inputs | {"goals": [RegisterGoal("rax", 1), RegisterGoal("rax", 42)]}))
    missing = create_analysis_snapshot(search_result=changed_result, manifest=found.manifest)
    diff = compare_analysis_snapshots(missing, found)
    assert diff.comparability is ComparabilityStatus.IDENTICAL_INPUTS
    assert diff.input_changed is False and diff.semantic_changed


def test_partial_and_different_incomplete(inputs):
    result = search_sequences(**inputs)
    first = create_analysis_snapshot(search_result=result, manifest=build(inputs, initial_state=object()))
    second = create_analysis_snapshot(search_result=result, manifest=build(inputs, initial_state=object()))
    diff = compare_analysis_snapshots(first, second)
    assert diff.comparability is ComparabilityStatus.PARTIAL and diff.input_changed is None
    assert diff.comparability_warnings
    third = create_analysis_snapshot(search_result=result, manifest=build(inputs, initial_state=object(), goals={"rax": 2}))
    assert compare_analysis_snapshots(first, third).comparability is ComparabilityStatus.DIFFERENT_INPUTS


def test_pruning_configuration_comparison(inputs):
    before, _ = snap(inputs, record_trace=True)
    after, _ = snap(inputs, record_trace=True, pruning=True)
    diff = compare_analysis_snapshots(before, after)
    assert diff.comparability is ComparabilityStatus.IDENTICAL_INPUTS
    assert diff.analysis_configuration_changed is True and diff.input_changed is False
    assert not diff.semantic_changed and diff.validate(before, after)


@pytest.mark.parametrize("change", ["count", "depth", "trace", "pruning", "cost", "kind"])
def test_attach_mismatch(inputs, change):
    snapshot, result = snap(inputs, record_trace=True)
    options = dict(record_trace=True)
    if change == "count":
        options["candidates"] = inputs["candidates"][:1]
    elif change == "depth":
        options["max_depth"] = 0
    elif change == "trace":
        options["record_trace"] = False
    elif change == "pruning":
        options["pruning"] = True
    elif change == "cost":
        options["candidate_costs"] = [1, 2]
    if change == "kind":
        manifest = create_analysis_manifest(sequence=inputs["candidates"])
    else:
        manifest = build(inputs, **options)
    with pytest.raises(ValueError):
        create_analysis_snapshot(search_result=result, manifest=manifest)


def test_no_trace_depth_mismatch(inputs):
    result = search_sequences(**inputs)
    with pytest.raises(ValueError, match="depth"):
        create_analysis_snapshot(search_result=result, manifest=build(inputs, max_depth=0))


def test_fixed_snapshot_attach(inputs):
    sequence = inputs["candidates"][:1]
    result = solve_sequence(sequence, inputs["goals"], initial_state=inputs["initial_state"])
    manifest = create_analysis_manifest(sequence=sequence, goals=inputs["goals"], initial_state=inputs["initial_state"])
    snapshot = create_analysis_snapshot(solve_result=result, manifest=manifest)
    assert AnalysisSnapshot.from_dict(snapshot.to_dict()) == snapshot
    with pytest.raises(ValueError, match="length"):
        create_analysis_snapshot(solve_result=result, manifest=create_analysis_manifest(sequence=sequence * 2))


def test_report_attachment_zero_solves_and_inputs_unchanged(inputs, monkeypatch, tmp_path):
    options = inputs | {"goals": [RegisterGoal("rax", 1), RegisterGoal("rax", 2)], "record_trace": True}
    result = search_sequences(**options)
    report = diagnose_search_result(result, candidates=options["candidates"], goals=options["goals"],
                                    initial_state=options["initial_state"], max_diagnosed_candidates=1)
    trace_data, report_data = result.trace.to_dict(), report.to_dict()
    state_data = options["initial_state"].input_bindings()
    pool, goals = list(options["candidates"]), list(options["goals"])

    def forbidden(*args, **kwargs):
        pytest.fail("identity/report operations must not construct a SolverContext")

    monkeypatch.setattr(SolverContext, "__init__", forbidden)
    import z3
    monkeypatch.setattr(z3, "Solver", forbidden)
    manifest = create_analysis_manifest(**options, max_diagnosed_candidates=1)
    manifest.validate()
    snapshot = create_analysis_snapshot(search_result=result, search_diagnostics=report, manifest=manifest)
    standalone = create_analysis_snapshot(search_diagnostics=report, manifest=manifest)
    assert standalone.validate()
    path = tmp_path / "report.json"
    save_analysis_snapshot(snapshot, path)
    loaded = load_analysis_snapshot(path)
    assert compare_analysis_snapshots(snapshot, loaded).validate(snapshot, loaded)
    assert result.trace.to_dict() == trace_data and report.to_dict() == report_data
    assert options["initial_state"].input_bindings() == state_data
    assert pool == options["candidates"] and goals == options["goals"]
    with pytest.raises(ValueError, match="max_diagnosed_candidates"):
        create_analysis_snapshot(search_diagnostics=report, manifest=create_analysis_manifest(**options))


@pytest.mark.parametrize("section", ["register", "memory", "stack"])
def test_unknown_state_values_do_not_claim_completeness(inputs, section):
    state = inputs["initial_state"].copy()
    if section == "register":
        state.set_register("rax", object())
    elif section == "memory":
        state.write_memory(0x1000, object())
    else:
        state.set_stack_slot(0, object())
    manifest = build(inputs, initial_state=state)
    assert not manifest.complete and manifest.validate()
    assert AnalysisManifest.from_dict(manifest.to_dict()) == manifest


@pytest.mark.parametrize("change", ["cost", "goal", "state", "complete"])
def test_rehashed_structurally_invalid_data_still_rejected(inputs, change):
    data = build(inputs).to_dict()
    if change == "cost":
        data["semantic_options"]["candidate_costs"] = [True, 1]
    elif change == "goal":
        data["goals"][0]["expected"] = {"kind": "arbitrary", "value": 1}
    elif change == "state":
        data["initial_state"]["registers"].reverse()
    else:
        data["complete"] = False
        data["incomplete_reasons"] = ["unrepresented_initial_state"]
    del data["semantic_fingerprint"], data["analysis_fingerprint"]
    semantic = {key: value for key, value in data.items() if key != "analysis_options"}
    sf = sha256(canonical_json_bytes(semantic)).hexdigest()
    af = sha256(canonical_json_bytes(data)).hexdigest()
    with pytest.raises(ValueError):
        AnalysisManifest.from_dict(data | {"semantic_fingerprint": sf, "analysis_fingerprint": af})


def test_found_cost_mismatch(inputs):
    result = search_sequences(**inputs, cost_fn=lambda _: 2, record_trace=True)
    with pytest.raises(ValueError, match="cost mismatch"):
        create_analysis_snapshot(search_result=result, manifest=build(inputs, candidate_costs=[3, 3], record_trace=True))


def test_v2_without_manifest_and_inconsistent_hashes(inputs):
    result = search_sequences(**inputs)
    data = create_analysis_snapshot(search_result=result).to_dict()
    data.update(schema_version=2, manifest=None, semantic_fingerprint=None, analysis_fingerprint=None)
    snapshot = AnalysisSnapshot.from_dict(data)
    assert compare_analysis_snapshots(snapshot, snapshot).comparability is ComparabilityStatus.UNKNOWN
    data["semantic_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="require manifest"):
        AnalysisSnapshot.from_dict(data)


def test_fixed_checkpoint_canonical_order(inputs):
    options = dict(sequence=inputs["candidates"], goals=inputs["goals"])
    a = create_analysis_manifest(**options, checkpoints={1: {"rbx": 2, "rax": 1}, 0: {"rax": 0}})
    b = create_analysis_manifest(**options, checkpoints={0: {"RAX": 0}, 1: {"rax": 1, "rbx": 2}})
    assert a == b
    with pytest.raises(ValueError):
        create_analysis_manifest(**options, checkpoints={2: {"rax": 0}})
