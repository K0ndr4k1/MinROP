import json
from dataclasses import replace

import pytest

from minirop.constraints import RegisterGoal
from minirop.search import search_sequences
from minirop.search_diagnostics import diagnose_search_result
from minirop.snapshot import (AnalysisKind, AnalysisSnapshot, ChangeCategory, FieldChange,
                              SnapshotDiff, compare_analysis_snapshots, create_analysis_snapshot,
                              load_analysis_snapshot, save_analysis_snapshot)
from minirop.solver import SolveResult, SolveStatus, SolverContext, solve_sequence
from minirop.state import make_symbolic_initial_state
from test_sequence import gadget


@pytest.fixture
def setup():
    a = gadget("48 c7 c0 01 00 00 00 c3", 0x4000)
    b = gadget("48 c7 c3 02 00 00 00 c3", 0x4100)
    initial = make_symbolic_initial_state(0)
    initial.set_register("rax", 0)
    initial.set_register("rbx", 0)
    goals = {"rax": 1, "rbx": 2}
    return [a, b], goals, initial


def found_snapshot(setup, trace=True):
    pool, goals, initial = setup
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial, record_trace=trace)
    return create_analysis_snapshot(search_result=result), result


def report_snapshot(setup, limit=None, combined=False):
    pool, _, initial = setup
    goals = [RegisterGoal("rax", 3), RegisterGoal("rax", 4)]
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial, record_trace=True)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial,
                                    max_diagnosed_candidates=limit)
    return create_analysis_snapshot(search_result=result if combined else None, search_diagnostics=report), result, report


@pytest.mark.parametrize("trace", [False, True])
def test_found_snapshot_identity_and_no_guessing(setup, trace):
    snapshot, result = found_snapshot(setup, trace)
    assert snapshot.schema_version == 1 and snapshot.analysis_kind is AnalysisKind.SEARCH
    assert snapshot.outcome["status"] == "found" and snapshot.outcome["depth"] == result.depth == 2
    assert snapshot.outcome["cost"] == result.cost
    assert snapshot.outcome["found_sequence_indices"] == ((0, 1) if trace else None)
    assert snapshot.provenance["max_depth"] == (2 if trace else None)
    assert snapshot.provenance["candidate_count"] == (2 if trace else None)
    assert snapshot.provenance["pruning"] is (False if trace else None)
    assert snapshot.search_summary["solver_calls"] == result.solver_calls
    assert snapshot.trace_summary is None if not trace else snapshot.trace_summary["event_count"] == len(result.trace.events)
    assert snapshot.validate()


def test_not_found_and_inconclusive_snapshots(setup, monkeypatch):
    import minirop.search as search_module
    pool, goals, initial = setup
    failed = search_sequences(pool, goals, max_depth=1, initial_state=initial, record_trace=True)
    snapshot = create_analysis_snapshot(search_result=failed)
    assert snapshot.outcome["status"] == "not_found"
    assert all(snapshot.outcome[key] is None for key in ("depth", "cost", "found_sequence_indices"))
    baseline = solve_sequence([pool[0]], initial_state=initial)
    monkeypatch.setattr(search_module, "solve_sequence", lambda *a, **kw: replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)))
    unknown = search_sequences(pool, max_depth=1, record_trace=True)
    snapshot = create_analysis_snapshot(search_result=unknown)
    assert snapshot.outcome["status"] == "inconclusive"
    assert snapshot.search_summary["unknown_candidates"] == 2
    assert snapshot.validate()


@pytest.mark.parametrize("status", [SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN])
def test_fixed_sequence_snapshots(status):
    initial = make_symbolic_initial_state(0)
    sequence = [gadget("c3", 0x4000)]
    goals = [RegisterGoal("rax", 1)]
    if status is SolveStatus.UNSAT:
        goals.append(RegisterGoal("rax", 2))
    result = solve_sequence(sequence, goals, initial_state=initial, diagnose=True, minimize_conflict=True)
    if status is SolveStatus.UNKNOWN:
        result = replace(result, solution=SolveResult(status))
    snapshot = create_analysis_snapshot(solve_result=result)
    assert snapshot.analysis_kind is AnalysisKind.FIXED_SEQUENCE
    assert snapshot.outcome["status"] == status.value
    assert snapshot.outcome["sequence_length"] == 1
    assert snapshot.outcome["constraint_count"] == len(goals)
    assert snapshot.outcome["goals_satisfied"] is result.goals_satisfied
    if status is SolveStatus.UNSAT:
        assert snapshot.diagnostic_summary["conflict_ids"] == (0, 1)
        assert snapshot.diagnostic_summary["conflict_categories"] == ("final_goal",)
        assert snapshot.diagnostic_summary["irreducible"] is True
        assert snapshot.outcome["diagnostic_solver_calls"] == 3
    else:
        assert snapshot.diagnostic_summary is None
    assert AnalysisSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))) == snapshot


@pytest.mark.parametrize("combined", [False, True])
def test_report_aggregate_snapshot(setup, combined):
    snapshot, _, report = report_snapshot(setup, limit=3, combined=combined)
    assert snapshot.analysis_kind is (AnalysisKind.SEARCH if combined else AnalysisKind.SEARCH_DIAGNOSTICS)
    summary = snapshot.diagnostic_summary
    assert summary["eligible_candidates"] == report.eligible_candidates == 6
    assert summary["candidates_diagnosed"] == 3 and summary["truncated"] is True
    assert (summary["coverage_numerator"], summary["coverage_denominator"]) == (3, 6)
    assert summary["required_category_counts"] == {key.value: count for key, count in report.required_category_counts.items()}
    assert summary["total_analysis_solver_calls"] == report.total_solver_calls
    raw = json.dumps(snapshot.to_dict())
    assert "candidate_summaries" not in raw and "records" not in raw
    assert "sequence_indices" not in json.dumps(snapshot.to_dict()["diagnostic_summary"])
    assert snapshot.validate()


def test_determinism_roundtrip_and_detached_metadata(setup):
    _, result = found_snapshot(setup)
    metadata = {"label": "基线", "nullable": None, "build": 1}
    s1 = create_analysis_snapshot(search_result=result, metadata=metadata)
    s2 = create_analysis_snapshot(search_result=result, metadata=metadata)
    assert s1 == s2 and s1.to_dict() == s2.to_dict()
    assert AnalysisSnapshot.from_dict(json.loads(json.dumps(s1.to_dict(), allow_nan=False))) == s1
    metadata["label"] = "changed"
    assert s1.metadata["label"] == "基线"
    data = s1.to_dict()
    data["outcome"]["found_sequence_indices"].append(99)
    assert s1.outcome["found_sequence_indices"] == (0, 1)
    with pytest.raises(TypeError):
        s1.outcome["depth"] = 99


@pytest.mark.parametrize("damage", [
    "missing_version", "version", "bool_version", "missing_field", "negative_count", "bool_count", "status",
    "nan", "inf", "negative_cost", "index", "bool_index", "depth", "inconclusive", "not_found_cost",
    "calls", "unknown", "metadata_nan", "metadata_object", "unknown_float", "trace_identity",
])
def test_invalid_search_snapshot_rejected(setup, damage):
    snapshot, _ = found_snapshot(setup)
    data = snapshot.to_dict()
    if damage == "missing_version":
        del data["schema_version"]
    elif damage == "version":
        data["schema_version"] = 999
    elif damage == "bool_version":
        data["schema_version"] = True
    elif damage == "missing_field":
        del data["outcome"]["depth"]
    elif damage == "negative_count":
        data["search_summary"]["tested_candidates"] = -1
    elif damage == "bool_count":
        data["search_summary"]["tested_candidates"] = True
    elif damage == "status":
        data["outcome"]["status"] = "maybe"
    elif damage in ("nan", "inf", "negative_cost"):
        data["outcome"]["cost"] = {"nan": float("nan"), "inf": float("inf"), "negative_cost": -1}[damage]
    elif damage in ("index", "bool_index"):
        data["outcome"]["found_sequence_indices"] = [0, -1 if damage == "index" else True]
    elif damage == "depth":
        data["outcome"]["depth"] = 3
    elif damage == "inconclusive":
        data["outcome"]["status"] = "inconclusive"
    elif damage == "not_found_cost":
        data["outcome"]["status"] = "not_found"
    elif damage == "calls":
        data["search_summary"]["solver_calls"] = 99
    elif damage == "unknown":
        data["search_summary"]["unknown_candidates"] = 99
    elif damage == "metadata_nan":
        data["metadata"]["bad"] = float("nan")
    elif damage == "metadata_object":
        data["metadata"]["bad"] = object()
    elif damage == "unknown_float":
        data["future"] = {"bad": float("inf")}
    elif damage == "trace_identity":
        data["outcome"]["found_sequence_indices"] = None
    with pytest.raises(ValueError):
        AnalysisSnapshot.from_dict(data)


def test_loader_tolerates_unknown_fields_but_drops_them(setup):
    snapshot, _ = found_snapshot(setup)
    data = snapshot.to_dict()
    data["future"] = "nonbreaking"
    data["outcome"]["future"] = [1, 2]
    assert AnalysisSnapshot.from_dict(data) == snapshot


@pytest.mark.parametrize("field,value", [("candidates_diagnosed", 99), ("truncated", False),
                                        ("coverage_denominator", 99), ("total_analysis_solver_calls", 0)])
def test_invalid_diagnostic_relationships(setup, field, value):
    snapshot, _, _ = report_snapshot(setup, limit=2)
    data = snapshot.to_dict()
    data["diagnostic_summary"][field] = value
    with pytest.raises(ValueError):
        AnalysisSnapshot.from_dict(data)


def test_empty_snapshot_sources_and_incompatible_sources(setup):
    with pytest.raises(ValueError):
        create_analysis_snapshot()
    pool, _, initial = setup
    result = solve_sequence(pool[:1], initial_state=initial)
    _, search = found_snapshot(setup)
    with pytest.raises(ValueError):
        create_analysis_snapshot(solve_result=result, search_result=search)
    _, failed, report = report_snapshot(setup)
    with pytest.raises(ValueError):
        create_analysis_snapshot(search_result=search, search_diagnostics=report)


def test_all_snapshot_operations_do_not_solve_or_mutate_inputs(setup, monkeypatch, tmp_path):
    snapshot, result, report = report_snapshot(setup, combined=True)
    before = result.trace.to_dict(), report.to_dict(), result.solver_calls
    def unexpected(*args, **kwargs):
        pytest.fail("snapshot operation must not solve")
    monkeypatch.setattr(SolverContext, "solve", unexpected)
    monkeypatch.chdir(tmp_path)
    created = create_analysis_snapshot(search_result=result, search_diagnostics=report)
    assert list(tmp_path.iterdir()) == []
    assert created == snapshot and created.validate()
    restored = AnalysisSnapshot.from_dict(created.to_dict())
    diff = compare_analysis_snapshots(created, restored)
    assert diff.validate(created, restored) and not diff.changed
    save_analysis_snapshot(snapshot, tmp_path / "历史.json")
    assert load_analysis_snapshot(tmp_path / "历史.json") == snapshot
    first = (tmp_path / "历史.json").read_bytes()
    save_analysis_snapshot(snapshot, tmp_path / "历史.json")
    assert (tmp_path / "历史.json").read_bytes() == first
    assert before == (result.trace.to_dict(), report.to_dict(), result.solver_calls)


def test_real_same_found_with_fewer_calls_is_only_operational():
    a = gadget("48 83 c0 01 c3", 0x4000)
    b = gadget("48 c7 c0 00 00 00 00 c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rip", a.address)
    for index in range(5):
        initial.set_stack_slot(index, a.address)
    kwargs = dict(goals={"rax": 5}, initial_state=initial, max_depth=5, check_entry=True, record_trace=True)
    plain = search_sequences([a, b], **kwargs)
    pruned = search_sequences([a, b], pruning=True, **kwargs)
    before, after = [create_analysis_snapshot(search_result=item) for item in (plain, pruned)]
    assert plain.sequence == pruned.sequence == (a,) * 5
    assert plain.solver_calls == 31 and pruned.solver_calls == 17
    diff = compare_analysis_snapshots(before, after)
    assert diff.changed and not diff.semantic_changed and diff.operational_changed
    assert any(change.field == "search_summary.solver_calls" and (change.before, change.after) == (31, 17) for change in diff.changes)
    assert diff.validate(before, after)


@pytest.mark.parametrize("field", ["found_sequence_indices", "depth", "cost", "shortest_depth_proven", "lowest_cost_at_depth_proven"])
def test_outcome_changes_are_semantic(setup, field):
    before, _ = found_snapshot(setup)
    data = before.to_dict()
    if field == "found_sequence_indices":
        data["outcome"][field] = [1, 0]
    elif field == "depth":
        data["outcome"][field] = 1
        data["outcome"]["found_sequence_indices"] = [0]
    elif field == "cost":
        data["outcome"][field] = 3
    elif field == "shortest_depth_proven":
        data["outcome"][field] = False
        data["search_summary"].update(unknown_candidates=1, unknown_solver_calls=1, shallowest_unknown_depth=1)
    else:
        data["outcome"][field] = False
    after = AnalysisSnapshot.from_dict(data)
    diff = compare_analysis_snapshots(before, after)
    assert diff.semantic_changed and any(change.field == f"outcome.{field}" for change in diff.outcome_changes)
    assert diff.validate(before, after)


def test_status_change_is_semantic(setup):
    pool, goals, initial = setup
    found, _ = found_snapshot(setup)
    failed = create_analysis_snapshot(search_result=search_sequences(pool, goals, max_depth=1, initial_state=initial))
    assert compare_analysis_snapshots(failed, found).semantic_changed


@pytest.mark.parametrize("field", ["required_category_counts", "conflict_patterns"])
def test_diagnostic_changes_are_semantic(setup, field):
    before, _, _ = report_snapshot(setup)
    data = before.to_dict()
    summary = data["diagnostic_summary"]
    if field == "required_category_counts":
        summary[field] = {}
    else:
        summary[field] = [{"categories": [], "count": summary["candidates_diagnosed"], "determined": False}]
    after = AnalysisSnapshot.from_dict(data)
    diff = compare_analysis_snapshots(before, after)
    assert diff.semantic_changed and diff.diagnostic_changes
    assert diff.validate(before, after)


def test_coverage_and_truncation_are_not_semantic(setup):
    before, _, _ = report_snapshot(setup, limit=6)
    data = before.to_dict()
    summary = data["diagnostic_summary"]
    summary.update(eligible_candidates=7, coverage_denominator=7, truncated=True)
    after = AnalysisSnapshot.from_dict(data)
    diff = compare_analysis_snapshots(before, after)
    assert not diff.semantic_changed and diff.operational_changed
    assert any(change.field == "diagnostic_summary.truncated" for change in diff.diagnostic_changes)


def test_metadata_change_and_null_presence(setup):
    before, result = found_snapshot(setup)
    after = create_analysis_snapshot(search_result=result, metadata={"label": "experiment", "extra": None})
    diff = compare_analysis_snapshots(before, after)
    assert diff.changed and not diff.semantic_changed and not diff.operational_changed
    assert {change.field for change in diff.metadata_changes} == {"metadata.label", "metadata.extra"}
    assert diff.validate(before, after)


def test_candidate_universe_difference_warns_without_blocking(setup):
    before, _ = found_snapshot(setup)
    data = before.to_dict()
    data["provenance"]["candidate_count"] = 3
    after = AnalysisSnapshot.from_dict(data)
    diff = compare_analysis_snapshots(before, after)
    assert diff.comparability_warnings and not diff.semantic_changed


def test_different_kinds_and_schemas_rejected(setup):
    search, _ = found_snapshot(setup)
    report, _, _ = report_snapshot(setup)
    with pytest.raises(ValueError, match="same analysis kind"):
        compare_analysis_snapshots(search, report)
    with pytest.raises(ValueError, match="schema_version"):
        compare_analysis_snapshots(search, replace(search, schema_version=999))


def test_diff_determinism_json_and_validation(setup):
    before, result = found_snapshot(setup)
    after = create_analysis_snapshot(search_result=result, metadata={"label": "after"})
    first = compare_analysis_snapshots(before, after)
    second = compare_analysis_snapshots(before, after)
    assert first == second and first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()
    assert first.validate(before, after)
    damaged = replace(first, changes=(replace(first.changes[0], category=ChangeCategory.SEARCH),))
    with pytest.raises(ValueError, match="group"):
        damaged.validate()
    noop = SnapshotDiff(AnalysisKind.SEARCH, (FieldChange("outcome.cost", 2, 2, ChangeCategory.OUTCOME, True),))
    with pytest.raises(ValueError, match="unchanged"):
        noop.validate()


def test_snapshot_does_not_iterate_detailed_trace_or_candidate_reports(setup):
    _, result, report = report_snapshot(setup, combined=True)
    class NoIteration(tuple):
        def __iter__(self):
            pytest.fail("snapshot must not traverse detailed history")
    result = replace(result, trace=replace(result.trace, events=NoIteration(result.trace.events)))
    report = replace(report, candidate_summaries=NoIteration(report.candidate_summaries))
    snapshot = create_analysis_snapshot(search_result=result, search_diagnostics=report)
    assert snapshot.validate()


def test_unknown_report_replay_coverage_roundtrip(setup, monkeypatch):
    import minirop.search_diagnostics as report_module
    _, result, _ = report_snapshot(setup)
    pool, _, initial = setup
    baseline = solve_sequence(pool[:1], initial_state=initial)
    monkeypatch.setattr(report_module, "solve_sequence", lambda *a, **kw: replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)))
    report = diagnose_search_result(result, candidates=pool, goals=None, initial_state=initial)
    snapshot = create_analysis_snapshot(search_result=result, search_diagnostics=report)
    assert snapshot.diagnostic_summary["truncated"] is False
    assert snapshot.diagnostic_summary["replay_consistent"] is False
    assert snapshot.diagnostic_summary["coverage_numerator"] == 0
    assert AnalysisSnapshot.from_dict(snapshot.to_dict()) == snapshot
