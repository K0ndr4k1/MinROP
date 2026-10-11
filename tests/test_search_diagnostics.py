import json
from dataclasses import FrozenInstanceError, replace

import pytest

import minirop.diagnostics as diagnostic_module
import minirop.search as search_module
import minirop.search_diagnostics as report_module
from minirop.constraints import Equality, RegisterGoal
from minirop.diagnostics import ConstraintCategory as Category
from minirop.search import SearchStatus, search_sequences
from minirop.search_diagnostics import diagnose_search_result
from minirop.search_trace import TraceEventKind
from minirop.solver import SolveResult, SolveStatus, SolverContext, solve_sequence
from minirop.state import make_symbolic_initial_state
from test_sequence import gadget


@pytest.fixture
def setup():
    pool = [gadget("c3", address) for address in (0x4000, 0x4100, 0x4200)]
    goals = [RegisterGoal("rax", 1), RegisterGoal("rax", 2)]
    initial = make_symbolic_initial_state(0)
    result = search_sequences(pool, goals, max_depth=1, initial_state=initial, record_trace=True)
    return pool, goals, initial, result


def report_for(setup, **kwargs):
    pool, goals, initial, result = setup
    return diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial, **kwargs)


def test_full_report_counts_and_real_solver_calls(setup, monkeypatch):
    calls = []
    original = SolverContext.solve
    def counted(self):
        calls.append(self)
        return original(self)
    monkeypatch.setattr(SolverContext, "solve", counted)
    report = report_for(setup)
    assert report.search_status is SearchStatus.NOT_FOUND and report.conclusive is True
    assert report.eligible_candidates == report.candidates_diagnosed == report.candidates_considered == 3
    assert report.coverage == (3, 3) and report.truncated is False
    assert report.category_counts == report.required_category_counts == {Category.FINAL_GOAL: 3}
    assert report.unknown_category_counts == {}
    assert report.diagnostic_solver_calls == 3 and report.replay_solver_calls == 3
    assert len(calls) == report.total_solver_calls == 6
    assert len({id(item) for item in calls}) == 6
    assert report.conflict_patterns[0].categories == (Category.FINAL_GOAL,)
    assert report.conflict_patterns[0].count == 3 and report.conflict_patterns[0].determined is True
    assert all(item.irreducible is None for item in report.candidate_summaries)
    assert report.irreducible_category_patterns == ()
    assert report.validate(setup[-1])


@pytest.mark.parametrize("limit,diagnosed,truncated", [(None, 3, False), (0, 0, True), (1, 1, True), (3, 3, False), (9, 3, False)])
def test_budgets(setup, limit, diagnosed, truncated):
    report = report_for(setup, max_diagnosed_candidates=limit)
    assert report.candidates_diagnosed == diagnosed and report.truncated is truncated
    assert report.coverage == (diagnosed, 3)
    assert report.diagnostic_solver_calls == diagnosed
    assert report.validate(setup[-1])


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2"])
def test_invalid_budgets(setup, limit):
    with pytest.raises(ValueError, match="max_diagnosed_candidates"):
        report_for(setup, max_diagnosed_candidates=limit)


def test_no_trace_rejected_without_solving(setup, monkeypatch):
    pool, goals, initial, result = setup
    def unexpected(*args, **kwargs):
        pytest.fail("must not run a replacement search/solve")
    monkeypatch.setattr(report_module, "solve_sequence", unexpected)
    with pytest.raises(ValueError, match="recorded SearchTrace"):
        diagnose_search_result(replace(result, trace=None), candidates=pool, goals=goals, initial_state=initial)


def test_trace_order_including_cost_is_preserved(setup):
    pool, goals, initial, _ = setup
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial,
                              cost_fn={pool[0]: 3, pool[1]: 1, pool[2]: 2}.__getitem__, record_trace=True)
    kwargs = dict(candidates=pool, goals=goals, initial_state=initial, max_diagnosed_candidates=5)
    r1 = diagnose_search_result(result, **kwargs)
    r2 = diagnose_search_result(result, **kwargs)
    assert [item.sequence_indices for item in r1.candidate_summaries] == [(1,), (2,), (0,), (1, 1), (1, 2)]
    assert r1.to_dict() == r2.to_dict()
    assert r1.validate(result)


def test_repeated_indices_reconstructed_from_original_pool(setup, monkeypatch):
    pool, goals, initial, _ = setup
    pool = pool[:2]
    result = search_sequences(pool, goals, max_depth=3, initial_state=initial, record_trace=True)
    seen = []
    original = report_module.solve_sequence
    def traced(sequence, **kwargs):
        seen.append(sequence)
        return original(sequence, **kwargs)
    monkeypatch.setattr(report_module, "solve_sequence", traced)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)
    item_index = next(i for i, item in enumerate(report.candidate_summaries) if item.sequence_indices == (0, 0, 1))
    assert seen[item_index] == (pool[0], pool[0], pool[1])
    assert report.candidates_diagnosed == 14 and report.validate(result)


@pytest.mark.parametrize("damage", ["too_few", "too_many", "index"])
def test_invalid_pool_or_indices_rejected(setup, damage):
    pool, goals, initial, result = setup
    if damage == "too_few":
        pool = pool[:1]
    elif damage == "too_many":
        pool = pool + pool[:1]
    else:
        events = list(result.trace.events)
        events[0] = replace(events[0], sequence_indices=(99,))
        result = replace(result, trace=replace(result.trace, events=tuple(events)))
    with pytest.raises(ValueError):
        diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)


def test_categories_count_candidates_not_constraints_and_nonrequired_pattern():
    pool = [gadget("c3", address) for address in (0x4000, 0x4100)]
    initial = make_symbolic_initial_state(1)
    initial.set_register("rax", 0)
    initial.set_stack_slot(0, 0xDEAD)
    goals = [RegisterGoal("rax", 1), RegisterGoal("rax", 1)]
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial, record_trace=True)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)
    assert report.category_counts == {Category.FINAL_GOAL: 6, Category.CONTROL_FLOW: 4, Category.MEMORY: 6}
    assert report.required_category_counts == {Category.FINAL_GOAL: 2}
    assert report.conflict_patterns[0].categories == ()
    assert report.conflict_patterns[0].count == 4 and report.conflict_patterns[0].determined is False
    assert report.conflict_patterns[1].categories == (Category.FINAL_GOAL,)
    assert report.conflict_patterns[1].count == 2
    assert report.validate(result)


def test_pattern_aggregation_and_stable_tie_order(setup, monkeypatch):
    pool, goals, initial, _ = setup
    both = solve_sequence([gadget("58 48 83 ec 08 c3", 0x5000), gadget("c3", 0x5100)],
                          {"rax": 1}, initial_state=initial, diagnose=True)
    final = solve_sequence([pool[0]], goals, initial_state=initial, diagnose=True)
    assert both.diagnostics.categories_required == (Category.CONTROL_FLOW, Category.FINAL_GOAL)
    responses = iter([final, final, both])
    monkeypatch.setattr(report_module, "solve_sequence", lambda *args, **kwargs: next(responses))
    report = report_for(setup)
    assert [(p.categories, p.count) for p in report.conflict_patterns] == [
        ((Category.FINAL_GOAL,), 2), ((Category.CONTROL_FLOW, Category.FINAL_GOAL), 1)]
    responses = iter([final, both, final])
    limited = report_for(setup, max_diagnosed_candidates=2)
    assert [p.categories for p in limited.conflict_patterns] == [(Category.CONTROL_FLOW, Category.FINAL_GOAL), (Category.FINAL_GOAL,)]


def test_unknown_category_removal_has_separate_counts(setup, monkeypatch):
    monkeypatch.setattr(diagnostic_module, "_solve_subset", lambda *args, **kwargs: SolveStatus.UNKNOWN)
    report = report_for(setup)
    assert report.unknown_category_counts == {Category.FINAL_GOAL: 3}
    assert report.required_category_counts == {}
    assert report.conflict_patterns[0].determined is False
    assert report.validate(setup[-1])


def test_control_flow_removal_unknown_is_not_counted_required(setup, monkeypatch):
    pool, goals, initial, _ = setup
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial, record_trace=True)
    original = diagnostic_module._solve_subset
    def simulated(records, **kwargs):
        if len(records) == 2 and all(record.category is Category.FINAL_GOAL for record in records):
            return SolveStatus.UNKNOWN
        return original(records, **kwargs)
    monkeypatch.setattr(diagnostic_module, "_solve_subset", simulated)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)
    assert report.unknown_category_counts == {Category.CONTROL_FLOW: 9}
    assert report.required_category_counts == {Category.FINAL_GOAL: 12}
    assert report.category_counts == {Category.FINAL_GOAL: 12, Category.CONTROL_FLOW: 9}
    assert report.validate(result)


def test_minimization_matches_direct_phase_ten(setup):
    pool, goals, initial, _ = setup
    report = report_for(setup, minimize_conflicts=True)
    for item in report.candidate_summaries:
        direct = solve_sequence([pool[i] for i in item.sequence_indices], goals, initial_state=initial,
                                diagnose=True, minimize_conflict=True)
        assert item.diagnostics == direct.diagnostics
        assert item.irreducible is True and item.conflict_size == 2
    assert report.diagnostic_solver_calls == 9
    assert report.irreducible_category_patterns[0].categories == (Category.FINAL_GOAL,)
    assert report.irreducible_category_patterns[0].count == 3
    assert report.validate(setup[-1])


@pytest.mark.parametrize("all_unknown", [False, True])
def test_unknown_candidates_not_replayed_and_search_remains_inconclusive(setup, monkeypatch, all_unknown):
    pool, goals, initial, _ = setup
    baseline = solve_sequence([pool[0]], goals, initial_state=initial)
    def simulated(sequence, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if all_unknown or sequence == (pool[0],) else baseline
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences(pool, goals, max_depth=1, initial_state=initial, record_trace=True)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)
    assert report.search_status is SearchStatus.INCONCLUSIVE and report.conclusive is False
    assert report.unknown_candidates == (3 if all_unknown else 1)
    assert report.eligible_candidates == report.candidates_diagnosed == (0 if all_unknown else 2)
    assert all(item.sequence_indices != (0,) for item in report.candidate_summaries)
    if all_unknown:
        assert report.conflict_patterns == () and report.diagnostic_solver_calls == 0
    assert report.validate(result)


def test_found_does_not_diagnose_earlier_failures(setup, monkeypatch):
    _, _, initial, _ = setup
    pool = [gadget("48 c7 c0 01 00 00 00 c3", 0x4000), gadget("48 c7 c0 02 00 00 00 c3", 0x4100)]
    result = search_sequences(pool, {"rax": 2}, max_depth=2, initial_state=initial, record_trace=True)
    def unexpected(*args, **kwargs):
        pytest.fail("FOUND must not diagnose failures")
    monkeypatch.setattr(report_module, "solve_sequence", unexpected)
    report = diagnose_search_result(result, candidates=pool, goals={"rax": 2}, initial_state=initial)
    assert report.eligible_candidates == 1 and report.candidates_diagnosed == 0
    assert report.conclusive is True and report.truncated is False
    assert report.total_solver_calls == 0 and report.coverage == (0, 1)
    assert report.validate(result)


def test_pruning_proofs_not_counted_as_additional_candidates(setup):
    pool, goals, initial, _ = setup
    extra = [Equality(1, 2)]
    result = search_sequences(pool, goals, max_depth=3, initial_state=initial, constraints=extra,
                              pruning=True, record_trace=True)
    assert result.prefix_checks == result.pruned_candidates == 3
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial, constraints=extra)
    assert report.eligible_candidates == report.candidates_diagnosed == 3
    assert report.pruned_prefixes == 3
    solved_ids = {e.event_id for e in result.trace.events if e.kind is TraceEventKind.SOLVED}
    assert all(item.solved_event_id in solved_ids for item in report.candidate_summaries)
    assert report.validate(result)


@pytest.mark.parametrize("empty,depth", [(True, 3), (False, 0)])
def test_empty_search_reports(setup, empty, depth):
    pool, goals, initial, _ = setup
    if empty:
        pool = []
    result = search_sequences(pool, goals, max_depth=depth, record_trace=True)
    report = diagnose_search_result(result, candidates=pool, goals=goals, initial_state=initial)
    assert report.coverage == (0, 0) and report.total_solver_calls == 0
    assert report.truncated is False and report.validate(result)


def test_inputs_statistics_trace_and_trial_states_unchanged(setup, monkeypatch, tmp_path):
    pool, goals, initial, result = setup
    saved = result.trace.to_dict()
    state_before = (initial.register_bindings(), initial.memory_bindings())
    before = (pool.copy(), goals.copy(), result.solver_calls, result.tested_candidates, result.prefix_checks)
    seen = []
    original = report_module.solve_sequence
    def isolated(sequence, **kwargs):
        state = kwargs["initial_state"]
        assert (state.register_bindings(), state.memory_bindings()) == state_before
        assert tuple(kwargs["goals"]) == tuple(goals)
        assert kwargs["constraints"] == (Equality(3, 3),)
        assert kwargs["memory_name"] == "report_memory" and kwargs["timeout_ms"] == 2000
        assert kwargs["check_entry"] is True
        replay = original(sequence, **kwargs)
        state.set_register("rax", 99)
        seen.append(state)
        return replay
    monkeypatch.setattr(report_module, "solve_sequence", isolated)
    monkeypatch.chdir(tmp_path)
    report = diagnose_search_result(result, candidates=pool, goals=iter(goals), initial_state=initial,
                                    constraints=iter([Equality(3, 3)]), check_entry=True,
                                    memory_name="report_memory", timeout_ms=2000)
    assert before == (pool, goals, result.solver_calls, result.tested_candidates, result.prefix_checks)
    assert result.trace.to_dict() == saved and (initial.register_bindings(), initial.memory_bindings()) == state_before
    assert len({id(state) for state in seen}) == 3
    assert list(tmp_path.iterdir()) == []
    assert report.validate(result)


def test_json_and_immutability(setup):
    report = report_for(setup)
    data = report.to_dict()
    text = json.dumps(data, allow_nan=False)
    assert json.loads(text) == data
    assert "object at" not in text and "0x" not in text
    data["candidate_summaries"][0]["sequence_indices"].append(99)
    assert report.candidate_summaries[0].sequence_indices == (0,)
    with pytest.raises(TypeError):
        report.category_counts[Category.FINAL_GOAL] = 99
    with pytest.raises(FrozenInstanceError):
        report.truncated = True


@pytest.mark.parametrize("damage", ["diagnosed", "eligible", "truncated", "counts", "calls", "index", "pattern"])
def test_validation_detects_corruption_without_solving(setup, monkeypatch, damage):
    report = report_for(setup)
    def unexpected(*args, **kwargs):
        pytest.fail("validation must not solve")
    monkeypatch.setattr(SolverContext, "solve", unexpected)
    assert report.validate(setup[-1])
    if damage == "diagnosed":
        report = replace(report, candidates_diagnosed=4)
    elif damage == "eligible":
        report = replace(report, eligible_candidates=1)
    elif damage == "truncated":
        report = replace(report, truncated=True)
    elif damage == "counts":
        report = replace(report, category_counts={Category.FINAL_GOAL: 99})
    elif damage == "calls":
        report = replace(report, diagnostic_solver_calls=99)
    elif damage == "index":
        items = (replace(report.candidate_summaries[0], sequence_indices=(99,)),) + report.candidate_summaries[1:]
        report = replace(report, candidate_summaries=items)
    else:
        report = replace(report, conflict_patterns=())
    with pytest.raises(ValueError):
        report.validate()


@pytest.mark.parametrize("status", [SolveStatus.SAT, SolveStatus.UNKNOWN])
def test_changed_replay_status_is_preserved_not_aggregated(setup, monkeypatch, status):
    pool, goals, initial, _ = setup
    replay = solve_sequence([pool[0]], goals, initial_state=initial)
    monkeypatch.setattr(report_module, "solve_sequence", lambda *args, **kwargs: replace(replay, solution=SolveResult(status)))
    report = report_for(setup, max_diagnosed_candidates=2)
    assert report.candidates_considered == 2 and report.candidates_diagnosed == 0
    assert report.diagnostic_solver_calls == 0 and report.total_solver_calls == 2
    assert report.replay_consistent is False and report.truncated is True
    assert report.coverage == (0, 3) and report.conflict_patterns == ()
    assert report.validate(setup[-1])


def test_exceptions_propagate(setup, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("invalid replay input")
    monkeypatch.setattr(report_module, "solve_sequence", broken)
    with pytest.raises(ValueError, match="invalid replay input"):
        report_for(setup)
