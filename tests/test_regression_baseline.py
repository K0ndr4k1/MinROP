"""Policy tests use structurally valid stored records to simulate regressions.

Changed conclusions are intentional fixtures, not claims about current solver
correctness. Phase 14 must classify records without replaying their analysis.
"""

from dataclasses import replace
import json

import pytest

from minirop.constraints import RegisterGoal
from minirop.manifest import create_analysis_manifest
from minirop.regression import (BaselineEntry, CasePresence, CasePresencePolicy, ImprovementReason as I,
    RegressionBaseline, RegressionPolicy, RegressionReason as R, RegressionStatus as S, RegressionWarning as W,
    create_regression_baseline, evaluate_regression, load_regression_baseline, save_regression_baseline)
from minirop.search import search_sequences
from minirop.search_diagnostics import diagnose_search_result
from minirop.snapshot import AnalysisSnapshot, ComparabilityStatus, create_analysis_snapshot
from minirop.solver import solve_sequence
from minirop.state import make_symbolic_initial_state
from test_sequence import gadget


@pytest.fixture(scope="module")
def templates():
    pool = [gadget("c3", 0x4000), gadget("c3", 0x4100)]
    initial = make_symbolic_initial_state(0)
    options = dict(candidates=pool, initial_state=initial, max_depth=4)
    result = search_sequences(**options)
    plain = create_analysis_snapshot(search_result=result, manifest=create_analysis_manifest(**options))
    traced = create_analysis_snapshot(search_result=search_sequences(**options, record_trace=True),
                                     manifest=create_analysis_manifest(**options, record_trace=True))
    fixed_options = dict(sequence=pool[:1], initial_state=initial)
    fixed = create_analysis_snapshot(solve_result=solve_sequence(pool[:1], initial_state=initial),
                                     manifest=create_analysis_manifest(**fixed_options))
    conflict_goals = [RegisterGoal("rax", 1), RegisterGoal("rax", 2)]
    diag_options = options | dict(goals=conflict_goals, record_trace=True, max_depth=1)
    failed = search_sequences(**diag_options)
    diagnostics = diagnose_search_result(failed, candidates=pool, goals=conflict_goals, initial_state=initial)
    diagnostic = create_analysis_snapshot(search_diagnostics=diagnostics,
                                          manifest=create_analysis_manifest(**diag_options))
    fixed_diagnostic = create_analysis_snapshot(
        solve_result=solve_sequence(pool[:1], conflict_goals, initial_state=initial, diagnose=True),
        manifest=create_analysis_manifest(**fixed_options, goals=conflict_goals, diagnose=True))
    return dict(plain=plain, traced=traced, fixed=fixed, diagnostic=diagnostic, fixed_diagnostic=fixed_diagnostic)


def search_record(templates, status="found", *, depth=2, cost=2, calls=10,
                  unknown=0, shortest=True, cost_proof=True, trace=False, indices=None):
    data = templates["traced" if trace else "plain"].to_dict()
    if status == "inconclusive":
        unknown = max(unknown, 1)
    data["search_summary"].update(generated_candidates=calls, tested_candidates=calls, solver_calls=calls,
        unknown_candidates=unknown, unknown_solver_calls=unknown,
        shallowest_unknown_depth=(depth if shortest else 1) if unknown else None)
    if status == "found":
        data["outcome"].update(depth=depth, cost=cost, shortest_depth_proven=shortest,
                              lowest_cost_at_depth_proven=cost_proof,
                              found_sequence_indices=(indices if indices is not None else [0] * depth) if trace else None)
    else:
        data["outcome"] = {key: None for key in data["outcome"]}
    data["outcome"]["status"] = status
    if trace:
        data["trace_summary"]["final_status"] = status
    return AnalysisSnapshot.from_dict(data)


def fixed_record(templates, status):
    data = templates["fixed"].to_dict()
    data["outcome"].update(status=status, goals_satisfied=None if status == "unknown" else status == "sat")
    return AnalysisSnapshot.from_dict(data)


def evaluate_pair(before, after, **policy):
    baseline = create_regression_baseline({"case": before})
    report = evaluate_regression(baseline, {"case": after}, policy=RegressionPolicy(**policy))
    assert report.validate(baseline, {"case": after})
    return report.case_results[0]


@pytest.mark.parametrize("old,new,status,reason,improvement", [
    ("found", "not_found", S.FAIL, R.SEARCH_LOST_FOUND_RESULT, None),
    ("found", "inconclusive", S.FAIL, R.SEARCH_BECAME_INCONCLUSIVE, None),
    ("not_found", "found", S.PASS, None, I.FOUND_RESULT_GAINED),
    ("inconclusive", "found", S.PASS, None, I.FOUND_RESULT_GAINED),
    ("not_found", "inconclusive", S.FAIL, R.SEARCH_BECAME_INCONCLUSIVE, None),
    ("inconclusive", "not_found", S.PASS, None, I.SEARCH_BECAME_CONCLUSIVE),
])
def test_search_status_directions(templates, old, new, status, reason, improvement):
    case = evaluate_pair(search_record(templates, old), search_record(templates, new))
    assert case.status is status and case.diff.semantic_changed
    if reason:
        assert reason in case.reasons
    if improvement:
        assert improvement in case.improvements
    if old == "found" and new == "inconclusive":
        assert R.SEARCH_LOST_FOUND_RESULT in case.reasons


@pytest.mark.parametrize("old,new,status,reason", [
    ("sat", "unsat", S.FAIL, R.SAT_UNSAT_FLIP), ("unsat", "sat", S.FAIL, R.SAT_UNSAT_FLIP),
    ("sat", "unknown", S.FAIL, R.STATUS_REGRESSION), ("unsat", "unknown", S.FAIL, R.STATUS_REGRESSION),
    ("unknown", "sat", S.PASS, None), ("unknown", "unsat", S.PASS, None),
])
def test_fixed_status_directions(templates, old, new, status, reason):
    case = evaluate_pair(fixed_record(templates, old), fixed_record(templates, new))
    assert case.status is status
    if reason:
        assert reason in case.reasons
    else:
        assert I.SOLVE_BECAME_CONCLUSIVE in case.improvements


@pytest.mark.parametrize("old,new,status,reason,improvement", [
    ({"depth": 2}, {"depth": 3}, S.FAIL, R.DEPTH_INCREASE, None),
    ({"depth": 3}, {"depth": 2}, S.PASS, None, I.DEPTH_DECREASE),
    ({"cost": 2}, {"cost": 3}, S.FAIL, R.COST_INCREASE, None),
    ({"cost": 2, "cost_proof": False}, {"cost": 3, "cost_proof": False}, S.REVIEW, R.COST_INCREASE, None),
    ({"cost": 3}, {"cost": 2}, S.PASS, None, I.COST_DECREASE),
    ({}, {"unknown": 1, "shortest": False}, S.FAIL, R.SHORTEST_PROOF_LOST, None),
    ({"unknown": 1, "shortest": False}, {}, S.PASS, None, I.SHORTEST_PROOF_GAINED),
    ({}, {"cost_proof": False}, S.FAIL, R.COST_PROOF_LOST, None),
    ({"cost_proof": False}, {}, S.PASS, None, I.COST_PROOF_GAINED),
])
def test_result_quality(templates, old, new, status, reason, improvement):
    case = evaluate_pair(search_record(templates, **old), search_record(templates, **new))
    assert case.status is status
    if reason:
        assert reason in case.reasons
    if improvement:
        assert improvement in case.improvements


def test_cost_only_compared_at_same_found_depth(templates):
    case = evaluate_pair(search_record(templates, depth=3, cost=2), search_record(templates, depth=2, cost=3))
    assert case.status is S.PASS and R.COST_INCREASE not in case.reasons


@pytest.mark.parametrize("strict", [False, True])
def test_sequence_change_policy(templates, strict):
    before = search_record(templates, trace=True, indices=[0, 1])
    after = search_record(templates, trace=True, indices=[1, 0])
    case = evaluate_pair(before, after, fail_on_found_sequence_change=strict)
    assert case.status is (S.FAIL if strict else S.PASS)
    assert W.FOUND_SEQUENCE_CHANGED in case.warnings


def test_trace_identity_unavailable_not_a_sequence_change(templates):
    case = evaluate_pair(search_record(templates, trace=True), search_record(templates), fail_on_found_sequence_change=True)
    assert case.status is S.PASS and W.FOUND_SEQUENCE_UNAVAILABLE in case.warnings


@pytest.mark.parametrize("strict", [False, True])
def test_new_unknown_policy(templates, strict):
    case = evaluate_pair(search_record(templates), search_record(templates, unknown=1), fail_on_new_unknown=strict)
    assert case.status is (S.FAIL if strict else S.PASS)
    assert W.NEW_UNKNOWN in case.warnings


def test_unknown_reduction(templates):
    case = evaluate_pair(search_record(templates, unknown=1), search_record(templates))
    assert case.status is S.PASS and I.UNKNOWN_REDUCED in case.improvements


@pytest.mark.parametrize("option,old,new", [
    ("fail_on_status_regression", {}, {"status": "not_found"}),
    ("fail_on_depth_increase", {}, {"depth": 3}),
    ("fail_on_cost_increase", {}, {"cost": 3}),
    ("fail_on_optimality_loss", {}, {"cost_proof": False}),
])
def test_disabling_failure_rule(templates, option, old, new):
    case = evaluate_pair(search_record(templates, **old), search_record(templates, **new), **{option: False})
    assert case.status is S.PASS and W.SEMANTIC_CHANGE_ALLOWED in case.warnings


def reattach(snapshot, manifest):
    data = snapshot.to_dict()
    data.update(schema_version=2, manifest=manifest.to_dict(), semantic_fingerprint=manifest.semantic_fingerprint,
                analysis_fingerprint=manifest.analysis_fingerprint)
    return AnalysisSnapshot.from_dict(data)


@pytest.mark.parametrize("kind", ["different", "partial", "unknown"])
def test_comparability_gate(templates, kind):
    before, after = search_record(templates), search_record(templates, "not_found", calls=30)
    if kind == "different":
        manifest = replace(after.manifest, goals=(), semantic_options={**after.manifest.semantic_options, "check_entry": True},
                           semantic_fingerprint=None, analysis_fingerprint=None)
        after = reattach(after, manifest)
    elif kind == "partial":
        manifest = replace(before.manifest, initial_state=None, complete=False,
                           incomplete_reasons=("unrepresented_initial_state",), semantic_fingerprint=None, analysis_fingerprint=None)
        before, after = reattach(before, manifest), reattach(after, manifest)
    else:
        data = after.to_dict()
        data["schema_version"] = 1
        after = AnalysisSnapshot.from_dict(data)
    case = evaluate_pair(before, after, max_solver_call_increase=0)
    assert case.status is S.REVIEW
    assert R.SEARCH_LOST_FOUND_RESULT not in case.reasons
    assert R.SOLVER_CALL_BUDGET_EXCEEDED not in case.reasons
    assert evaluate_pair(before, after, require_identical_inputs=False).status is S.FAIL


def test_config_change_pass_but_does_not_excuse_degradation(templates):
    before = search_record(templates)
    manifest = replace(before.manifest, analysis_options={**before.manifest.analysis_options, "pruning": True},
                       semantic_fingerprint=None, analysis_fingerprint=None)
    after = reattach(before, manifest)
    case = evaluate_pair(before, after)
    assert case.status is S.PASS and W.ANALYSIS_CONFIGURATION_CHANGED in case.warnings
    case = evaluate_pair(before, reattach(search_record(templates, "not_found"), manifest))
    assert case.status is S.FAIL


@pytest.mark.parametrize("delta,limit,status", [(6, None, S.PASS), (6, 5, S.FAIL), (5, 5, S.PASS), (0, 0, S.PASS), (-2, 0, S.PASS)])
def test_solver_call_threshold(templates, delta, limit, status):
    case = evaluate_pair(search_record(templates), search_record(templates, calls=10 + delta), max_solver_call_increase=limit)
    assert case.status is status and case.solver_calls_delta == delta
    if delta < 0:
        assert I.SOLVER_CALLS_DECREASED in case.improvements
    elif delta > 0:
        assert W.SOLVER_CALLS_INCREASED in case.warnings


@pytest.mark.parametrize("field,strict_option,reason", [
    ("required_category_counts", "fail_on_required_category_change", R.REQUIRED_CATEGORY_CHANGED),
    ("conflict_patterns", "fail_on_conflict_pattern_change", R.CONFLICT_PATTERN_CHANGED),
])
def test_diagnostic_review_or_fail(templates, field, strict_option, reason):
    before = templates["diagnostic"]
    data = before.to_dict()
    data["diagnostic_summary"][field] = {} if field == "required_category_counts" else [
        {"categories": [], "count": data["diagnostic_summary"]["candidates_diagnosed"], "determined": False}]
    after = AnalysisSnapshot.from_dict(data)
    case = evaluate_pair(before, after)
    assert case.status is S.REVIEW and reason in case.reasons
    assert evaluate_pair(before, after, **{strict_option: True}).status is S.FAIL


def test_fixed_diagnostic_required_categories(templates):
    before = templates["fixed_diagnostic"]
    data = before.to_dict()
    data["diagnostic_summary"]["required_categories"] = []
    after = AnalysisSnapshot.from_dict(data)
    assert evaluate_pair(before, after).status is S.REVIEW
    assert evaluate_pair(before, after, fail_on_required_category_change=True).status is S.FAIL


def test_gaining_conclusion_with_diagnostics_is_still_improvement(templates):
    after = templates["fixed_diagnostic"]
    data = after.to_dict()
    data["outcome"].update(status="unknown", goals_satisfied=None, diagnostics_present=False, diagnostic_solver_calls=0)
    data["diagnostic_summary"] = None
    data["provenance"]["diagnostics_included"] = False
    before = AnalysisSnapshot.from_dict(data)
    case = evaluate_pair(before, after)
    assert case.status is S.PASS and I.SOLVE_BECAME_CONCLUSIVE in case.improvements
    assert W.DIAGNOSTICS_CHANGED in case.warnings


def test_gaining_found_does_not_fail_for_removed_patterns(templates):
    before = templates["diagnostic"]
    data = before.to_dict()
    data["outcome"].update(status="found", conclusive=True)
    summary = data["diagnostic_summary"]
    summary.update(candidates_considered=0, candidates_diagnosed=0, coverage_numerator=0,
                   replay_solver_calls=0, diagnostic_solver_calls=0, total_analysis_solver_calls=0,
                   category_counts={}, required_category_counts={}, unknown_category_counts={},
                   conflict_patterns=[], irreducible_category_patterns=[], truncated=False)
    after = AnalysisSnapshot.from_dict(data)
    case = evaluate_pair(before, after, fail_on_conflict_pattern_change=True)
    assert case.status is S.PASS and I.FOUND_RESULT_GAINED in case.improvements


def test_diagnostic_detail_change_review(templates):
    before = templates["fixed_diagnostic"]
    data = before.to_dict()
    data["diagnostic_summary"]["conflict_categories"] = []
    case = evaluate_pair(before, AnalysisSnapshot.from_dict(data))
    assert case.status is S.REVIEW and R.DIAGNOSTIC_DETAIL_CHANGED in case.reasons


def test_diagnostic_solver_calls_operational_default(templates):
    before = templates["diagnostic"]
    data = before.to_dict()
    data["diagnostic_summary"]["diagnostic_solver_calls"] += 3
    data["diagnostic_summary"]["total_analysis_solver_calls"] += 3
    after = AnalysisSnapshot.from_dict(data)
    case = evaluate_pair(before, after)
    assert case.status is S.PASS and case.solver_calls_delta == 3
    assert evaluate_pair(before, after, max_solver_call_increase=2).status is S.FAIL


def test_multiple_reasons_not_fail_fast(templates):
    case = evaluate_pair(search_record(templates), search_record(templates, depth=3, unknown=1, shortest=False, calls=20),
                         max_solver_call_increase=5)
    assert case.status is S.FAIL
    assert {R.DEPTH_INCREASE, R.SHORTEST_PROOF_LOST, R.SOLVER_CALL_BUDGET_EXCEEDED} <= set(case.reasons)
    assert case.reasons == tuple(reason for reason in R if reason in case.reasons)


def test_baseline_order_roundtrip_tags_and_metadata(templates, tmp_path):
    snapshot = templates["plain"]
    a = BaselineEntry("A", snapshot, "description", ["search", "search", "tag"])
    b = BaselineEntry("B", snapshot)
    one, two = create_regression_baseline([b, a]), create_regression_baseline([a, b])
    assert one == two and one.to_dict() == two.to_dict()
    assert one.baseline_schema_version == 1 and a.tags == ("search", "tag")
    assert RegressionBaseline.from_dict(json.loads(json.dumps(one.to_dict()))) == one
    path = tmp_path / "基线.json"
    save_regression_baseline(one, path)
    first = path.read_bytes()
    save_regression_baseline(two, path)
    assert path.read_bytes() == first and load_regression_baseline(path) == one
    report = evaluate_regression(one, {"B": snapshot, "A": snapshot})
    assert report.status is S.PASS and report.summary.passed_cases == 2
    assert report.validate(one, {"A": snapshot, "B": snapshot})


@pytest.mark.parametrize("case_id", ["", " ", " leading", "trailing ", "\n", 1, None])
def test_invalid_case_id(templates, case_id):
    with pytest.raises(ValueError):
        create_regression_baseline({case_id: templates["plain"]})
    with pytest.raises(ValueError):
        evaluate_regression(create_regression_baseline({}), {case_id: templates["plain"]})


def test_case_identity_exact_and_duplicate_rejected(templates):
    a = BaselineEntry("case", templates["plain"])
    with pytest.raises(ValueError, match="duplicate"):
        create_regression_baseline([a, a])
    baseline = create_regression_baseline([a])
    report = evaluate_regression(baseline, {"Case": a.snapshot})
    assert report.missing_cases == ("case",) and report.new_cases == ("Case",)
    assert report.summary.matched_cases == 0


@pytest.mark.parametrize("mode", list(CasePresencePolicy))
@pytest.mark.parametrize("missing", [False, True])
def test_presence_policy(templates, mode, missing):
    entries = {"case": templates["plain"]}
    baseline = create_regression_baseline(entries if missing else {})
    report = evaluate_regression(baseline, {} if missing else entries,
                                 policy=RegressionPolicy(**{("missing_case" if missing else "new_case"): mode}))
    assert report.status.value == ("pass" if mode is CasePresencePolicy.ALLOW else mode.value)
    assert report.case_results[0].diff is None and report.validate()


def test_empty_baseline_and_defaults(templates):
    empty = create_regression_baseline({})
    assert evaluate_regression(empty, {}).status is S.PASS
    entries = {"case": templates["plain"]}
    assert evaluate_regression(empty, entries).status is S.REVIEW
    assert evaluate_regression(create_regression_baseline(entries), {}).status is S.FAIL


def test_overall_status_and_all_cases_evaluated(templates):
    snapshot = search_record(templates)
    baseline = create_regression_baseline({"pass": snapshot, "fail": snapshot})
    current = {"pass": snapshot, "fail": search_record(templates, "not_found"), "new": snapshot}
    report = evaluate_regression(baseline, current)
    assert report.status is S.FAIL and [case.case_id for case in report.case_results] == ["fail", "new", "pass"]
    assert (report.summary.passed_cases, report.summary.failed_cases, report.summary.review_cases) == (1, 1, 1)
    assert report.summary.identical_input_cases == 2 and report.summary.unknown_comparability_cases == 0
    assert report.summary.reason_counts[R.SEARCH_LOST_FOUND_RESULT] == 1
    current["fail"] = snapshot
    assert evaluate_regression(baseline, current).status is S.REVIEW
    del current["new"]
    assert evaluate_regression(baseline, current).status is S.PASS


def test_kind_mismatch_records_failure_and_continues(templates):
    baseline = create_regression_baseline({"wrong": templates["plain"], "right": templates["plain"]})
    report = evaluate_regression(baseline, {"wrong": templates["fixed"], "right": templates["plain"]})
    assert report.status is S.FAIL and report.summary.matched_cases == 2
    case = report.case_results[1]
    assert case.reasons == (R.ANALYSIS_KIND_CHANGED,) and case.diff is None
    assert report.validate()


def test_metadata_only_report_and_policy_serialization(templates):
    before = templates["plain"]
    after = replace(before, metadata={"label": "current"}, producer_version="next")
    case = evaluate_pair(before, after)
    assert case.status is S.PASS and case.diff.changed and not case.diff.semantic_changed
    policy = RegressionPolicy(new_case=CasePresencePolicy.ALLOW, max_solver_call_increase=5)
    assert RegressionPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy


@pytest.mark.parametrize("kwargs", [
    {"policy_version": True}, {"policy_version": 2}, {"require_identical_inputs": 1},
    {"fail_on_new_unknown": "yes"}, {"max_solver_call_increase": True},
    {"max_solver_call_increase": -1}, {"max_solver_call_increase": 1.5}, {"missing_case": "fail"},
])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        RegressionPolicy(**kwargs)


@pytest.mark.parametrize("change", ["version", "missing", "duplicate", "snapshot", "fingerprint"])
def test_invalid_baseline_load(templates, change):
    data = create_regression_baseline({"case": templates["plain"]}).to_dict()
    if change == "version":
        data["baseline_schema_version"] = 2
    elif change == "missing":
        del data["entries"]
    elif change == "duplicate":
        data["entries"].append(data["entries"][0])
    elif change == "snapshot":
        data["entries"][0]["snapshot"]["schema_version"] = 999
    else:
        data["entries"][0]["snapshot"]["manifest"]["semantic_fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        RegressionBaseline.from_dict(data)


def test_policy_from_dict_rejects_missing_unknown_and_invalid_enum():
    data = RegressionPolicy().to_dict()
    for corrupted in ({key: value for key, value in data.items() if key != "policy_version"},
                      data | {"custom_rule": "something"}, data | {"new_case": "ignore"}):
        with pytest.raises(ValueError):
            RegressionPolicy.from_dict(corrupted)


def test_invalid_current_rejected_before_evaluation(templates):
    baseline = create_regression_baseline({"case": templates["plain"]})
    with pytest.raises(ValueError):
        evaluate_regression(baseline, {"case": replace(templates["plain"], schema_version=99)})
    with pytest.raises(ValueError):
        evaluate_regression(baseline, {"case": "file.json"})


@pytest.mark.parametrize("change", ["summary", "status", "presence", "diff", "reason", "count_type", "coverage"])
def test_report_validation(templates, change):
    baseline = create_regression_baseline({"case": templates["plain"]})
    report = evaluate_regression(baseline, {"case": templates["plain"]})
    if change == "summary":
        report = replace(report, summary=replace(report.summary, failed_cases=1))
    elif change == "status":
        report = replace(report, status=S.FAIL)
    elif change == "count_type":
        report = replace(report, summary=replace(report.summary, total_cases=True))
    elif change == "coverage":
        report = replace(report, current_cases=())
    else:
        case = report.case_results[0]
        if change == "presence":
            case = replace(case, presence=CasePresence.NEW)
        elif change == "diff":
            case = replace(case, diff=None)
        else:
            case = replace(case, reasons=(R.DEPTH_INCREASE, R.DEPTH_INCREASE))
        report = replace(report, case_results=(case,))
    with pytest.raises(ValueError):
        report.validate()


def test_reuses_diff_no_analysis_calls_and_inputs_unchanged(templates, monkeypatch, tmp_path):
    import minirop.regression as module
    import minirop.search as search_module
    import minirop.search_diagnostics as diagnostic_module
    import minirop.solver as solver_module
    import z3

    def forbidden(*args, **kwargs):
        pytest.fail("regression operations must not run analysis")

    monkeypatch.setattr(z3, "Solver", forbidden)
    monkeypatch.setattr(solver_module.SolverContext, "__init__", forbidden)
    monkeypatch.setattr(solver_module, "solve_sequence", forbidden)
    monkeypatch.setattr(search_module, "search_sequences", forbidden)
    monkeypatch.setattr(diagnostic_module, "diagnose_search_result", forbidden)
    calls = []
    original = module.compare_analysis_snapshots

    def compare(before, after):
        result = original(before, after)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "compare_analysis_snapshots", compare)
    current = {"b": templates["diagnostic"], "a": templates["plain"]}
    saved_current = {key: value.to_dict() for key, value in current.items()}
    policy = RegressionPolicy()
    saved_policy = policy.to_dict()
    baseline = create_regression_baseline(current)
    saved_baseline = baseline.to_dict()
    path = tmp_path / "baseline.json"
    save_regression_baseline(baseline, path)
    assert load_regression_baseline(path) == baseline
    first = evaluate_regression(baseline, current, policy=policy)
    assert len(calls) == 2 and all(case.diff is calls[index] for index, case in enumerate(first.case_results))
    second = evaluate_regression(baseline, dict(reversed(list(current.items()))), policy=policy)
    assert first.to_dict() == second.to_dict()
    assert first.validate(baseline, current) and json.dumps(first.to_dict())
    assert saved_current == {key: value.to_dict() for key, value in current.items()}
    assert baseline.to_dict() == saved_baseline and policy.to_dict() == saved_policy
    exported = baseline.to_dict()
    exported["entries"][0]["case_id"] = "changed"
    assert baseline.entries[0].case_id == "a"
    with pytest.raises(TypeError):
        first.summary.reason_counts[R.MISSING_CASE] = 1
