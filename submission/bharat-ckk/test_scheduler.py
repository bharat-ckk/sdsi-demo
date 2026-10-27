"""
Comprehensive test suite for the Job Scheduler (PS5).
Covers: core correctness, edge cases, error handling, and performance.

Run with:  pytest test_scheduler.py -v
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from scheduler import (
    Job,
    ScheduleResult,
    _check_missing_deps,
    _compute_levels,
    _critical_path,
    _detect_cycles,
    _downstream_jobs,
    _forward_pass,
    _group_into_stages,
    _topological_sort,
    load_jobs,
    schedule,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def simple_dag() -> dict[str, Job]:
    """A:3  B:4   C:2(←A)  D:5(←A,B)  E:1(←C,D)  F:3(←D)  G:2(←E,F)"""
    return {
        "A": Job(id="A", duration=3, depends_on=[]),
        "B": Job(id="B", duration=4, depends_on=[]),
        "C": Job(id="C", duration=2, depends_on=["A"]),
        "D": Job(id="D", duration=5, depends_on=["A", "B"]),
        "E": Job(id="E", duration=1, depends_on=["C", "D"]),
        "F": Job(id="F", duration=3, depends_on=["D"]),
        "G": Job(id="G", duration=2, depends_on=["E", "F"]),
    }


@pytest.fixture
def simple_jobs_list():
    return [
        {"id": "A", "duration": 3, "dependsOn": []},
        {"id": "B", "duration": 4, "dependsOn": []},
        {"id": "C", "duration": 2, "dependsOn": ["A"]},
        {"id": "D", "duration": 5, "dependsOn": ["A", "B"]},
        {"id": "E", "duration": 1, "dependsOn": ["C", "D"]},
        {"id": "F", "duration": 3, "dependsOn": ["D"]},
        {"id": "G", "duration": 2, "dependsOn": ["E", "F"]},
    ]


# ─────────────────────────────────────────────
#  M1: Graph validation
# ─────────────────────────────────────────────


class TestCycleDetection:
    def test_no_cycle(self, simple_dag):
        has_cycle, path = _detect_cycles(simple_dag)
        assert not has_cycle
        assert path == []

    def test_simple_cycle(self):
        # A → B → A
        jobs = {
            "A": Job(id="A", duration=1, depends_on=["B"]),
            "B": Job(id="B", duration=1, depends_on=["A"]),
        }
        has_cycle, path = _detect_cycles(jobs)
        assert has_cycle
        # path should contain the cycle
        assert "A" in path
        assert "B" in path

    def test_three_node_cycle(self):
        # A → B → C → A
        jobs = {
            "A": Job(id="A", duration=1, depends_on=["C"]),
            "B": Job(id="B", duration=1, depends_on=["A"]),
            "C": Job(id="C", duration=1, depends_on=["B"]),
        }
        has_cycle, path = _detect_cycles(jobs)
        assert has_cycle
        assert len(path) >= 3

    def test_self_loop(self):
        jobs = {"A": Job(id="A", duration=1, depends_on=["A"])}
        has_cycle, path = _detect_cycles(jobs)
        assert has_cycle
        assert path == ["A", "A"]

    def test_cycle_in_subgraph(self):
        # A → B, B → C, C → D, D → B  (A is fine, B-C-D cycle)
        jobs = {
            "A": Job(id="A", duration=1, depends_on=[]),
            "B": Job(id="B", duration=1, depends_on=["A", "D"]),
            "C": Job(id="C", duration=1, depends_on=["B"]),
            "D": Job(id="D", duration=1, depends_on=["C"]),
        }
        has_cycle, path = _detect_cycles(jobs)
        assert has_cycle

    def test_disconnected_dag_no_cycle(self):
        jobs = {
            "A": Job(id="A", duration=1, depends_on=[]),
            "B": Job(id="B", duration=1, depends_on=[]),
            "C": Job(id="C", duration=1, depends_on=[]),
        }
        has_cycle, path = _detect_cycles(jobs)
        assert not has_cycle


class TestMissingDependencies:
    def test_no_missing(self, simple_dag):
        assert _check_missing_deps(simple_dag) == {}

    def test_depends_on_nonexistent(self):
        jobs = {
            "A": Job(id="A", duration=1, depends_on=["Z"]),
            "B": Job(id="B", duration=1, depends_on=[]),
        }
        missing = _check_missing_deps(jobs)
        assert missing == {"A": ["Z"]}

    def test_multiple_missing(self):
        jobs = {
            "A": Job(id="A", duration=1, depends_on=["X", "Y"]),
        }
        missing = _check_missing_deps(jobs)
        assert missing == {"A": ["X", "Y"]}

    def test_all_deps_missing(self):
        jobs = {
            "A": Job(id="A", duration=1, depends_on=["B", "C"]),
        }
        missing = _check_missing_deps(jobs)
        assert missing == {"A": ["B", "C"]}


# ─────────────────────────────────────────────
#  M2: Topological sort
# ─────────────────────────────────────────────


class TestTopologicalSort:
    def test_linear_chain(self):
        # A → B → C
        jobs = {
            "A": Job(id="A", duration=1, depends_on=[]),
            "B": Job(id="B", duration=1, depends_on=["A"]),
            "C": Job(id="C", duration=1, depends_on=["B"]),
        }
        order = _topological_sort(jobs)
        assert order == ["A", "B", "C"]

    def test_diamond(self):
        # A → B, A → C, B → D, C → D
        jobs = {
            "A": Job(id="A", duration=1, depends_on=[]),
            "B": Job(id="B", duration=1, depends_on=["A"]),
            "C": Job(id="C", duration=1, depends_on=["A"]),
            "D": Job(id="D", duration=1, depends_on=["B", "C"]),
        }
        order = _topological_sort(jobs)
        assert order.index("A") == 0
        assert order.index("D") == 3
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_respects_dependencies(self, simple_dag):
        order = _topological_sort(simple_dag)
        for jid, job in simple_dag.items():
            for dep in job.depends_on:
                assert order.index(dep) < order.index(jid), (
                    f"{dep} must come before {jid}"
                )

    def test_single_job(self):
        jobs = {"A": Job(id="A", duration=1, depends_on=[])}
        assert _topological_sort(jobs) == ["A"]

    def test_empty(self):
        assert _topological_sort({}) == []


# ─────────────────────────────────────────────
#  M3: Parallel stages
# ─────────────────────────────────────────────


class TestStages:
    def test_worker_limit_1_is_sequential(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _compute_levels(simple_dag, topo)
        stages = _group_into_stages(simple_dag, topo, worker_limit=1)
        # Each stage has at most 1 job
        for stage in stages:
            assert len(stage) == 1

    def test_worker_limit_infinite(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _compute_levels(simple_dag, topo)
        stages = _group_into_stages(simple_dag, topo, worker_limit=100)
        # All jobs at the same level can run in one stage
        for stage in stages:
            assert all(simple_dag[j].level == simple_dag[stage[0]].level for j in stage)

    def test_worker_limit_respected(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _compute_levels(simple_dag, topo)
        for limit in range(1, 4):
            stages = _group_into_stages(simple_dag, topo, worker_limit=limit)
            for stage in stages:
                assert len(stage) <= limit, f"Worker limit {limit} violated in stage {stage}"

    def test_all_jobs_scheduled(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _compute_levels(simple_dag, topo)
        stages = _group_into_stages(simple_dag, topo, worker_limit=2)
        scheduled = {j for stage in stages for j in stage}
        assert scheduled == set(simple_dag.keys())

    def test_dependency_order_in_stages(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _compute_levels(simple_dag, topo)
        stages = _group_into_stages(simple_dag, topo, worker_limit=2)
        # A job must not appear before its dependency in the stage list
        for jid, job in simple_dag.items():
            jid_stage = next(i for i, s in enumerate(stages) if jid in s)
            for dep in job.depends_on:
                dep_stage = next(i for i, s in enumerate(stages) if dep in s)
                assert dep_stage <= jid_stage, f"{dep} stage {dep_stage} must be ≤ {jid} stage {jid_stage}"


# ─────────────────────────────────────────────
#  M4: Critical path
# ─────────────────────────────────────────────


class TestCriticalPath:
    def test_minimum_completion_time(self, simple_dag):
        topo = _topological_sort(simple_dag)
        _forward_pass(simple_dag, topo)
        _backward_pass(simple_dag, topo)
        cp, t = _critical_path(simple_dag)
        # Manual: A(3) → D(5) → E(1) → G(2) = 11 OR B(4) → D(5) → ... = 11
        # A start 0, finish 3; D start max(3,4)=4, finish 9
        # F start 9, finish 12; E start max(5,9)=9, finish 10
        # G start max(10,12)=12, finish 14
        assert t == 14.0
        # Critical path: A→D→F→G  or  B→D→F→G
        assert all(j in cp for j in ["D", "G"])
        # Verify zero slack on critical path jobs
        for jid in cp:
            j = simple_dag[jid]
            assert abs(j.earliest_start - j.latest_start) < 1e-9

    def test_single_job(self):
        jobs = {"A": Job(id="A", duration=5, depends_on=[])}
        _forward_pass(jobs, ["A"])
        _backward_pass(jobs, ["A"])
        cp, t = _critical_path(jobs)
        assert cp == ["A"]
        assert t == 5.0

    def test_parallel_paths(self):
        # A → B(10) → D
        # A → C(5)  → D   — B is on critical path
        jobs = {
            "A": Job(id="A", duration=2, depends_on=[]),
            "B": Job(id="B", duration=10, depends_on=["A"]),
            "C": Job(id="C", duration=5, depends_on=["A"]),
            "D": Job(id="D", duration=1, depends_on=["B", "C"]),
        }
        topo = _topological_sort(jobs)
        _forward_pass(jobs, topo)
        _backward_pass(jobs, topo)
        cp, t = _critical_path(jobs)
        # A(2) → B(10) → D(1) = 13
        assert t == 13.0
        assert "B" in cp
        assert "C" not in cp  # C has slack


# ─────────────────────────────────────────────
#  M5: Failure simulation
# ─────────────────────────────────────────────


class TestFailureSimulation:
    def test_root_failure_skips_all_downstream(self, simple_dag):
        # Failing A should skip C, D, E, F, G
        skipped = _downstream_jobs(simple_dag, "A")
        assert set(skipped) == {"C", "D", "E", "F", "G"}

    def test_leaf_failure_skips_nothing(self, simple_dag):
        skipped = _downstream_jobs(simple_dag, "G")
        assert skipped == []

    def test_intermediate_failure(self, simple_dag):
        # Failing D skips E, F, G
        skipped = _downstream_jobs(simple_dag, "D")
        assert set(skipped) == {"E", "F", "G"}

    def test_nonexistent_job(self, simple_dag):
        skipped = _downstream_jobs(simple_dag, "Z")
        assert skipped == []

    def test_empty_graph(self):
        skipped = _downstream_jobs({}, "A")
        assert skipped == []


# ─────────────────────────────────────────────
#  Integration tests via schedule()
# ─────────────────────────────────────────────


class TestScheduleIntegration:
    def test_full_pipeline_valid(self, simple_jobs_list):
        result = schedule(simple_jobs_list, worker_limit=2)
        assert result.valid
        assert not result.has_cycle
        assert result.missing_deps == {}
        assert len(result.topological_order) == 7
        assert result.minimum_completion_time > 0
        assert len(result.stages) > 0

    def test_full_pipeline_with_failure(self, simple_jobs_list):
        result = schedule(simple_jobs_list, worker_limit=2, failed_job_id="D")
        assert result.valid
        assert set(result.skipped_jobs) == {"E", "F", "G"}

    def test_cycle_halts_m2_m4(self):
        jobs = [
            {"id": "A", "duration": 1, "dependsOn": ["B"]},
            {"id": "B", "duration": 1, "dependsOn": ["A"]},
        ]
        result = schedule(jobs)
        assert not result.valid
        assert result.has_cycle
        assert result.topological_order == []
        assert result.stages == []
        assert result.minimum_completion_time == 0.0

    def test_missing_dep_halts(self):
        jobs = [
            {"id": "A", "duration": 1, "dependsOn": ["Z"]},
        ]
        result = schedule(jobs)
        assert not result.valid
        assert result.missing_deps == {"A": ["Z"]}

    def test_disconnected_jobs(self):
        jobs = [
            {"id": "X", "duration": 5, "dependsOn": []},
            {"id": "Y", "duration": 3, "dependsOn": []},
        ]
        result = schedule(jobs, worker_limit=2)
        assert result.valid
        assert set(result.topological_order) == {"X", "Y"}
        assert result.minimum_completion_time == 5.0

    def test_linear_chain_schedule(self):
        jobs = [
            {"id": "1", "duration": 2, "dependsOn": []},
            {"id": "2", "duration": 3, "dependsOn": ["1"]},
            {"id": "3", "duration": 1, "dependsOn": ["2"]},
        ]
        result = schedule(jobs, worker_limit=2)
        assert result.topological_order == ["1", "2", "3"]
        assert result.minimum_completion_time == 6.0
        assert result.critical_path == ["1", "2", "3"]

    def test_fan_out_fan_in(self):
        # A → B, A → C, A → D, then B,C,D → E
        jobs = [
            {"id": "A", "duration": 2, "dependsOn": []},
            {"id": "B", "duration": 5, "dependsOn": ["A"]},
            {"id": "C", "duration": 3, "dependsOn": ["A"]},
            {"id": "D", "duration": 1, "dependsOn": ["A"]},
            {"id": "E", "duration": 2, "dependsOn": ["B", "C", "D"]},
        ]
        result = schedule(jobs, worker_limit=3)
        assert result.valid
        # Critical path: A(2) → B(5) → E(2) = 9
        assert result.minimum_completion_time == 9.0


# ─────────────────────────────────────────────
#  Edge cases & error handling
# ─────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_duration_jobs(self):
        jobs = [
            {"id": "A", "duration": 0, "dependsOn": []},
            {"id": "B", "duration": 0, "dependsOn": ["A"]},
        ]
        result = schedule(jobs)
        assert result.valid
        assert result.minimum_completion_time == 0.0

    def test_large_durations(self):
        jobs = [
            {"id": "A", "duration": 1e9, "dependsOn": []},
            {"id": "B", "duration": 1e9, "dependsOn": []},
        ]
        result = schedule(jobs)
        assert result.valid
        assert result.minimum_completion_time == 1e9

    def test_float_durations(self):
        jobs = [
            {"id": "A", "duration": 2.5, "dependsOn": []},
            {"id": "B", "duration": 3.7, "dependsOn": ["A"]},
        ]
        result = schedule(jobs)
        assert result.minimum_completion_time == pytest.approx(6.2)

    def test_single_job_no_deps(self):
        result = schedule([{"id": "X", "duration": 42, "dependsOn": []}])
        assert result.valid
        assert result.topological_order == ["X"]
        assert result.minimum_completion_time == 42.0
        assert result.critical_path == ["X"]

    def test_empty_depends_on_omitted(self):
        # dependsOn key may be absent
        result = schedule([{"id": "A", "duration": 1}])
        assert result.valid


class TestInputValidation:
    def test_empty_jobs_list(self):
        with pytest.raises(ValueError, match="At least one job"):
            schedule([])

    def test_non_list_input(self):
        with pytest.raises(ValueError, match="must be a list"):
            schedule({"jobs": []})  # type: ignore[arg-type]

    def test_missing_id(self):
        with pytest.raises(ValueError, match="'id' field"):
            schedule([{"duration": 1, "dependsOn": []}])

    def test_blank_id(self):
        with pytest.raises(ValueError, match="non-empty string"):
            schedule([{"id": "  ", "duration": 1, "dependsOn": []}])

    def test_duplicate_id(self):
        with pytest.raises(ValueError, match="Duplicate job id"):
            schedule([
                {"id": "A", "duration": 1, "dependsOn": []},
                {"id": "A", "duration": 1, "dependsOn": []},
            ])

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative number"):
            schedule([{"id": "A", "duration": -5, "dependsOn": []}])

    def test_depends_on_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            schedule([{"id": "A", "duration": 1, "dependsOn": "B"}])

    def test_depends_on_contains_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            schedule([{"id": "A", "duration": 1, "dependsOn": [1, 2]}])

    def test_worker_limit_less_than_1(self):
        with pytest.raises(ValueError, match="worker_limit"):
            schedule([{"id": "A", "duration": 1, "dependsOn": []}], worker_limit=0)

    def test_non_dict_entry(self):
        with pytest.raises(ValueError, match="must be a dict"):
            schedule(["not_a_dict"])  # type: ignore[listitem]


class TestLoadJobs:
    def test_loads_valid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"jobs": [{"id": "A", "duration": 1, "dependsOn": []}]}, f)
            path = f.name
        try:
            jobs = load_jobs(path)
            assert len(jobs) == 1
            assert jobs[0]["id"] == "A"
        finally:
            Path(path).unlink()

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_jobs("/nonexistent/path/file.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not valid json")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_jobs(path)
        finally:
            Path(path).unlink()

    def test_missing_jobs_key(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"tasks": []}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="'jobs' key"):
                load_jobs(path)
        finally:
            Path(path).unlink()

    def test_not_a_dict_root(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([1, 2, 3], f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="must be an object"):
                load_jobs(path)
        finally:
            Path(path).unlink()


# ─────────────────────────────────────────────
#  Performance awareness
# ─────────────────────────────────────────────


class TestPerformance:
    def test_large_dag(self):
        """Build a chain of 500 jobs — should complete well under 1 second."""
        n = 500
        jobs_list = []
        for i in range(n):
            deps = [str(i - 1)] if i > 0 else []
            jobs_list.append({"id": str(i), "duration": 1, "dependsOn": deps})
        result = schedule(jobs_list)
        assert result.valid
        assert len(result.topological_order) == n
        assert result.minimum_completion_time == n

    def test_wide_dag(self):
        """100 independent jobs — tests parallel stage grouping."""
        n = 100
        jobs_list = []
        for i in range(n):
            jobs_list.append({"id": f"J{i}", "duration": 1, "dependsOn": []})
        result = schedule(jobs_list, worker_limit=10)
        assert result.valid
        # All can in principle run in one level — but worker limit controls stages
        total_scheduled = sum(len(s) for s in result.stages)
        assert total_scheduled == n
        for stage in result.stages:
            assert len(stage) <= 10