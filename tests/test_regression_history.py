"""History describes stored reports, and never re-runs their evaluation."""

from dataclasses import replace
import json

import pytest

from minirop.regression import (RegressionReport, RegressionStatus as S, RegressionReason as R,
    RegressionWarning as W, ImprovementReason as I, create_regression_baseline, evaluate_regression)
from minirop.regression_history import (HistoryPresence, HistoryRun, RegressionHistory, StabilityStatus as ST,
    TrendStatus as T, analyze_regression_history, create_regression_history, save_regression_history, load_regression_history)
from minirop.snapshot import AnalysisSnapshot, SnapshotDiff
from test_regression_baseline import templates, search_record


@pytest.fixture
def reports(templates):
    base = search_record(templates)
    baseline = create_regression_baseline({"case": base})
    return {
        "P": evaluate_regression(baseline, {"case": base}),
        "F": evaluate_regression(baseline, {"case": search_record(templates, depth=3)}),
        "R": evaluate_regression(create_regression_baseline({}), {"case": base}),
        "A": evaluate_regression(create_regression_baseline({}), {}),
        "M": evaluate_regression(baseline, {}),
    }


def analyze(reports, sequence):
    return analyze_regression_history(create_regression_history([(f"run-{index}", reports[key])
                                      for index, key in enumerate(sequence)]))


def test_empty_history():
    history = create_regression_history([])
    result = analyze_regression_history(history)
    assert result.run_count == result.case_count == 0
    assert result.current_overall_status is None and result.overall_failure_streak == 0
    assert not result.case_histories and not result.operational_trends
    assert result.summary.runs == 0 and json.dumps(result.to_dict())
    assert RegressionHistory.from_dict(history.to_dict()) == history


@pytest.mark.parametrize("sequence,expected", [
    ("P", ST.INSUFFICIENT_HISTORY), ("F", ST.INSUFFICIENT_HISTORY), ("R", ST.INSUFFICIENT_HISTORY),
    ("PPP", ST.STABLE_PASS), ("PPF", ST.NEW_FAILURE), ("PFF", ST.PERSISTENT_FAILURE),
    ("FF", ST.PERSISTENT_FAILURE), ("FPP", ST.RECOVERED), ("PFP", ST.FLAKY),
    ("PFPP", ST.FLAKY), ("PFPFP", ST.FLAKY), ("RRR", ST.STABLE_REVIEW),
    ("FRF", ST.MIXED), ("RP", ST.MIXED), ("PRP", ST.MIXED),
    ("PAP", ST.MIXED), ("PAA", ST.INSUFFICIENT_HISTORY), ("PPA", ST.MIXED),
    ("AAPPP", ST.STABLE_PASS),
])
def test_stability_definitions(reports, sequence, expected):
    case = analyze(reports, sequence).case_histories[0]
    assert case.stability is expected
    assert len(case.observations) == len(sequence)


def test_explicit_run_order_label_and_metadata(reports):
    history = create_regression_history([
        HistoryRun("z", reports["P"], label="first", metadata={"timestamp": "2099-01-01"}),
        HistoryRun("A", reports["F"], metadata={"timestamp": "1900-01-01"}),
        ("a", reports["F"]),
    ])
    assert [run.run_id for run in history.runs] == ["z", "A", "a"]
    case = analyze_regression_history(history).case_histories[0]
    assert [item.run_id for item in case.observations] == ["z", "A", "a"]
    assert case.stability is ST.PERSISTENT_FAILURE
    other = create_regression_history([("z", reports["P"]), ("A", reports["F"]), ("a", reports["F"])])
    assert analyze_regression_history(history) == analyze_regression_history(other)


@pytest.mark.parametrize("identity", ["", " ", " leading", "trailing ", "\n", 1, None])
def test_invalid_run_id(reports, identity):
    with pytest.raises(ValueError):
        create_regression_history([(identity, reports["P"])])


def test_duplicate_run_id(reports):
    with pytest.raises(ValueError, match="duplicate"):
        create_regression_history([("same", reports["P"]), ("same", reports["F"])])


@pytest.mark.parametrize("runs", [{}, set(), "run", [("too", "many", "values")]])
def test_explicit_sequence_required(runs):
    with pytest.raises(ValueError):
        create_regression_history(runs)


def test_missing_new_absent_are_distinct(reports):
    case = analyze(reports, "ARMP").case_histories[0]
    assert [item.presence for item in case.observations] == [
        HistoryPresence.ABSENT, HistoryPresence.NEW, HistoryPresence.MISSING, HistoryPresence.MATCHED]
    absent, new, missing, matched = case.observations
    assert absent.status is None and absent.comparability is None and absent.diff is None
    assert absent.semantic_changed is None and absent.operational_changed is None
    assert new.status is S.REVIEW and missing.status is S.FAIL
    assert new.diff is None and missing.diff is None and matched.diff is not None
    assert case.first_seen_run == "run-1" and case.last_seen_run == "run-3"
    assert case.continuous_since_first_seen and not case.has_gaps


def test_gap_breaks_streak_and_does_not_bridge_transitions(reports):
    case = analyze(reports, "FFAFFA").case_histories[0]
    assert case.has_gaps and not case.continuous_since_first_seen
    assert case.current_status is S.FAIL and case.last_seen_run == "run-4"
    assert case.current_failure_streak == 0 and case.max_failure_streak == 2
    assert not case.current_failure_reasons and not case.transitions
    assert case.stability is ST.MIXED
    case = analyze(reports, "FAP").case_histories[0]
    assert case.last_recovery_run is None and not case.transitions


def test_failure_streak_first_last_and_current_reasons(reports):
    case = analyze(reports, "PFFF").case_histories[0]
    assert case.current_failure_streak == case.max_failure_streak == 3
    assert case.current_pass_streak == case.current_review_streak == 0
    assert case.first_failure_run == "run-1" and case.last_failure_run == "run-3"
    assert case.reason_counts[R.DEPTH_INCREASE] == 3
    assert case.recurring_reasons == case.current_failure_reasons == (R.DEPTH_INCREASE,)


def test_max_streak_and_latest_recovery(reports):
    case = analyze(reports, "FFPFFFP").case_histories[0]
    assert case.max_failure_streak == 3 and case.current_failure_streak == 0
    assert case.current_pass_streak == 1 and case.last_recovery_run == "run-6"
    assert case.last_failure_run == "run-5"
    assert len(case.transitions) == 3


def test_review_and_pass_streaks(reports):
    result = analyze(reports, "PRRR")
    assert result.case_histories[0].current_review_streak == 3
    assert result.persistent_reviews == ("case",) and result.summary.persistent_reviews == 1
    assert analyze(reports, "FPPP").case_histories[0].current_pass_streak == 3


def test_reason_intersection_improvements_and_warnings(templates):
    baseline = create_regression_baseline({"case": search_record(templates, calls=20)})
    current = [search_record(templates, depth=3, calls=10, cost_proof=False),
               search_record(templates, depth=3, calls=12),
               search_record(templates, depth=3, calls=15, cost_proof=False)]
    records = [evaluate_regression(baseline, {"case": snapshot}) for snapshot in current]
    case = analyze_regression_history(create_regression_history([(str(i), report) for i, report in enumerate(records)])).case_histories[0]
    assert case.reason_counts == {R.DEPTH_INCREASE: 3, R.COST_PROOF_LOST: 2}
    assert case.current_failure_reasons == (R.DEPTH_INCREASE,)
    assert case.improvement_counts[I.SOLVER_CALLS_DECREASED] == 3
    assert case.recurring_reasons == (R.DEPTH_INCREASE, R.COST_PROOF_LOST)
    assert [key for key in case.reason_counts] == [reason for reason in R if reason in case.reason_counts]


def test_warning_counts(templates):
    baseline = create_regression_baseline({"case": search_record(templates, calls=10)})
    report = evaluate_regression(baseline, {"case": search_record(templates, calls=15)})
    result = analyze_regression_history(create_regression_history([("1", report), ("2", report)]))
    assert result.case_histories[0].warning_counts[W.SOLVER_CALLS_INCREASED] == 2


def test_overall_timeline_summary_and_sorted_cases(reports, templates):
    analysis = analyze(reports, "PPRFF")
    assert [item.status for item in analysis.overall_timeline] == [S.PASS, S.PASS, S.REVIEW, S.FAIL, S.FAIL]
    assert analysis.current_overall_status is S.FAIL and analysis.overall_failure_streak == 2
    assert [(item.before, item.after) for item in analysis.overall_transitions] == [(S.PASS, S.REVIEW), (S.REVIEW, S.FAIL)]
    snapshot = search_record(templates)
    report = evaluate_regression(create_regression_baseline({"z": snapshot, "a": snapshot}), {"z": snapshot, "a": snapshot})
    history = create_regression_history([("first", report), ("last", reports["F"])])
    analysis = analyze_regression_history(history)
    assert [case.case_id for case in analysis.case_histories] == ["a", "case", "z"]
    assert analysis.status_counts == {S.PASS: 2, S.FAIL: 1, S.REVIEW: 0}
    assert analysis.summary.current_pass == 2 and analysis.case_count == 3


@pytest.mark.parametrize("values,trend", [
    ([30, 25, 20], T.IMPROVING), ([10, 12, 15], T.DEGRADING), ([10, 10, 10], T.STABLE),
    ([10, 8, 12], T.MIXED), ([10], T.INSUFFICIENT_DATA), ([30, 30, 20], T.IMPROVING),
])
def test_solver_metric_trends(templates, values, trend):
    baseline = create_regression_baseline({"case": search_record(templates, calls=50)})
    reports = [evaluate_regression(baseline, {"case": search_record(templates, calls=value)}) for value in values]
    analysis = analyze_regression_history(create_regression_history([(str(i), report) for i, report in enumerate(reports)]))
    case = analysis.case_histories[0]
    series = {series.metric: series for series in case.metric_series}
    for name in ("solver_calls", "generated_candidates", "tested_candidates"):
        assert series[name].trend is trend
        assert [item.value for item in series[name].observations] == values
        assert series[name].first == values[0] and series[name].latest == values[-1]
        assert series[name].minimum == min(values) and series[name].maximum == max(values)
        assert series[name].net_change == values[-1] - values[0]
    assert [item.value for item in series["solver_calls_delta"].observations] == [value - 50 for value in values]
    assert all(report.status is S.PASS for report in reports)


def test_unchanged_absolute_values_are_unknown_not_zero(templates):
    snapshot = search_record(templates, calls=10)
    baseline = create_regression_baseline({"case": snapshot})
    same = evaluate_regression(baseline, {"case": snapshot})
    series = {item.metric: item for item in analyze_regression_history(
        create_regression_history([("same", same), ("still-same", same)])).operational_trends}
    assert series["solver_calls_delta"].trend is T.STABLE
    assert [item.value for item in series["solver_calls_delta"].observations] == [0, 0]
    assert series["solver_calls"].numeric_count == 0 and series["solver_calls"].latest is None
    assert series["solver_calls"].trend is T.INSUFFICIENT_DATA
    assert series["generated_candidates_delta"].numeric_count == 0


def test_metric_gaps_are_explicit_and_not_filled(templates, reports):
    baseline = create_regression_baseline({"case": search_record(templates, calls=50)})
    one = evaluate_regression(baseline, {"case": search_record(templates, calls=30)})
    two = evaluate_regression(baseline, {"case": search_record(templates, calls=20)})
    result = analyze_regression_history(create_regression_history([("1", one), ("2", reports["A"]), ("3", two)]))
    series = next(item for item in result.operational_trends if item.metric == "solver_calls")
    assert [item.value for item in series.observations] == [30, None, 20]
    assert series.numeric_count == 2 and series.has_gaps and series.trend is T.IMPROVING
    assert result.case_histories[0].stability is ST.MIXED


def test_pruning_has_no_value_direction(templates):
    baseline = create_regression_baseline({"case": search_record(templates)})
    records = []
    for count in (1, 2, 3):
        data = search_record(templates).to_dict()
        data["search_summary"]["pruned_candidates"] = count
        records.append(evaluate_regression(baseline, {"case": AnalysisSnapshot.from_dict(data)}))
    result = analyze_regression_history(create_regression_history([(str(i), report) for i, report in enumerate(records)]))
    for series in result.operational_trends:
        if series.metric.startswith("pruned_candidates"):
            assert series.direction is None and series.trend is T.MIXED
            assert [item.value for item in series.observations] == [1, 2, 3]


def test_diagnostic_metric_series(templates):
    before = templates["diagnostic"]
    baseline = create_regression_baseline({"case": before})
    records = []
    for increment in (3, 2, 1):
        data = before.to_dict()
        data["diagnostic_summary"]["diagnostic_solver_calls"] += increment
        data["diagnostic_summary"]["total_analysis_solver_calls"] += increment
        records.append(evaluate_regression(baseline, {"case": AnalysisSnapshot.from_dict(data)}))
    result = analyze_regression_history(create_regression_history([(str(i), report) for i, report in enumerate(records)]))
    series = next(item for item in result.operational_trends if item.metric == "diagnostic_solver_calls_delta")
    assert [item.value for item in series.observations] == [3, 2, 1] and series.trend is T.IMPROVING
    total = next(item for item in result.operational_trends if item.metric == "solver_calls")
    assert total.numeric_count == 3 and total.trend is T.IMPROVING


def test_stability_summary_classifications(reports):
    for sequence, field in (("PPF", "newly_failed_cases"), ("PFF", "persistent_failures"),
                            ("FPP", "recovered_cases"), ("PFP", "flaky_cases")):
        analysis = analyze(reports, sequence)
        assert getattr(analysis, field) == ("case",)
    assert analyze(reports, "PPF").summary.new_failures == 1
    assert analyze(reports, "PFF").summary.persistent_failures == 1
    assert analyze(reports, "FPP").summary.recovered == 1
    assert analyze(reports, "PFP").summary.flaky == 1


def test_fixed_diagnostic_metrics(templates):
    before = templates["fixed_diagnostic"]
    data = before.to_dict()
    data["outcome"]["diagnostic_solver_calls"] += 2
    report = evaluate_regression(create_regression_baseline({"case": before}),
                                 {"case": AnalysisSnapshot.from_dict(data)})
    result = analyze_regression_history(create_regression_history([("run", report)]))
    series = {item.metric: item for item in result.operational_trends}
    assert series["diagnostic_solver_calls_delta"].latest == 2
    assert series["diagnostic_solver_calls"].latest == data["outcome"]["diagnostic_solver_calls"]


@pytest.mark.parametrize("metadata", [{"nested": {}}, {"nan": float("nan")}, {1: "value"}])
def test_run_metadata_validation(reports, metadata):
    with pytest.raises(ValueError):
        HistoryRun("run", reports["P"], metadata=metadata)


def test_report_from_dict_derived_ids_and_count_types(reports):
    for key in ("matched_cases", "baseline_cases"):
        data = reports["P"].to_dict()
        data[key] = []
        with pytest.raises(ValueError):
            RegressionReport.from_dict(data)
    data = reports["P"].to_dict()
    data["summary"]["passed_cases"] = True
    with pytest.raises(ValueError):
        RegressionReport.from_dict(data)


@pytest.mark.parametrize("key", ["P", "F", "R", "M", "A"])
def test_report_and_diff_roundtrip(reports, key):
    report = reports[key]
    assert RegressionReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report
    for case in report.case_results:
        if case.diff is not None:
            assert SnapshotDiff.from_dict(json.loads(json.dumps(case.diff.to_dict()))) == case.diff


def test_history_json_roundtrip_and_analysis_determinism(reports, tmp_path):
    history = create_regression_history([HistoryRun("z", reports["P"], "基线", {"date": "explicit", "number": 1}),
                                         ("a", reports["F"]), ("b", reports["R"])])
    assert RegressionHistory.from_dict(json.loads(json.dumps(history.to_dict()))) == history
    path = tmp_path / "历史.json"
    save_regression_history(history, path)
    assert load_regression_history(path) == history
    before = path.read_bytes()
    save_regression_history(history, path)
    assert path.read_bytes() == before
    first, second = analyze_regression_history(history), analyze_regression_history(history)
    assert first == second and first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict())


@pytest.mark.parametrize("change", ["version", "bool_version", "missing", "duplicate", "run_id", "report_summary", "report_status", "case_presence", "case_reasons", "case_comparability"])
def test_tampered_history(reports, change):
    data = create_regression_history([("run", reports["F"])]).to_dict()
    report = data["runs"][0]["report"]
    if change == "version":
        data["history_schema_version"] = 2
    elif change == "bool_version":
        data["history_schema_version"] = True
    elif change == "missing":
        del data["runs"]
    elif change == "duplicate":
        data["runs"].append(data["runs"][0])
    elif change == "run_id":
        data["runs"][0]["run_id"] = " "
    elif change == "report_summary":
        report["summary"]["failed_cases"] += 1
    elif change == "report_status":
        report["status"] = "pass"
    elif change == "case_presence":
        report["case_results"][0]["presence"] = "new"
    elif change == "case_reasons":
        report["case_results"][0]["reasons"] *= 2
    else:
        report["case_results"][0]["comparability"] = "unknown"
    with pytest.raises(ValueError):
        RegressionHistory.from_dict(data)


@pytest.mark.parametrize("change", ["changed", "semantic", "input", "category", "presence", "unknown_field", "bool_type"])
def test_tampered_diff(reports, change):
    data = reports["F"].case_results[0].diff.to_dict()
    if change == "changed":
        data["changed"] = False
    elif change == "semantic":
        data["semantic_changed"] = False
    elif change == "input":
        data["input_changed"] = True
    elif change == "category":
        data["changes"][0]["category"] = "unknown"
    elif change == "presence":
        data["changes"][0]["after_present"] = False
    elif change == "unknown_field":
        data["extras"] = 123
    else:
        data["changed"] = 1
    with pytest.raises(ValueError):
        SnapshotDiff.from_dict(data)


def test_no_execution_comparison_or_reevaluation(reports, monkeypatch, tmp_path):
    import minirop.regression as regression
    import minirop.snapshot as snapshot
    import minirop.search as search
    import minirop.search_diagnostics as diagnostics
    import minirop.solver as solver
    import z3

    def forbidden(*args, **kwargs):
        pytest.fail("history must only read recorded reports")

    monkeypatch.setattr(z3, "Solver", forbidden)
    monkeypatch.setattr(solver.SolverContext, "__init__", forbidden)
    monkeypatch.setattr(solver, "solve_sequence", forbidden)
    monkeypatch.setattr(search, "search_sequences", forbidden)
    monkeypatch.setattr(diagnostics, "diagnose_search_result", forbidden)
    monkeypatch.setattr(regression, "evaluate_regression", forbidden)
    monkeypatch.setattr(regression, "compare_analysis_snapshots", forbidden)
    monkeypatch.setattr(snapshot, "compare_analysis_snapshots", forbidden)
    before = {key: report.to_dict() for key, report in reports.items()}
    history = create_regression_history([("r1", reports["P"]), ("r2", reports["F"]), ("r3", reports["F"])])
    assert history.validate()
    analysis = analyze_regression_history(history)
    assert analysis.persistent_failures == ("case",)
    path = tmp_path / "history.json"
    save_regression_history(history, path)
    restored = load_regression_history(path)
    assert restored == history and analyze_regression_history(restored).to_dict() == analysis.to_dict()
    assert {key: report.to_dict() for key, report in reports.items()} == before
    exported = history.to_dict()
    exported["runs"][0]["run_id"] = "changed"
    assert history.runs[0].run_id == "r1"
    with pytest.raises(TypeError):
        analysis.case_histories[0].reason_counts[R.DEPTH_INCREASE] = 50
    with pytest.raises(TypeError):
        history.runs[0].metadata["new"] = "changed"
