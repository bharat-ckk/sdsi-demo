"""
Job Scheduler with Dependency Resolution (DAG)
==============================================
A mini build/ETL orchestrator that validates, schedules, and analyses
a directed acyclic graph (DAG) of jobs.

Capabilities (M1–M5):
  M1 — Validate the graph: cycle detection (with path), missing dependencies
  M2 — Produce a valid execution order (topological sort, Kahn's algorithm)
  M3 — Group jobs into parallel stages with a configurable worker limit N
  M4 — Compute the critical path and minimum total completion time
  M5 — Simulate a failure: list every downstream job that is skipped

Input format (JSON):
  {
    "jobs": [
      { "id": "A", "duration": 5, "dependsOn": [] },
      { "id": "B", "duration": 3, "dependsOn": ["A"] },
      ...
    ]
  }

Author: Bharat KS  |  SDSI Agentic AI Hackathon  |  Productivity track
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────


@dataclass
class Job:
    """A single job in the DAG."""

    id: str
    duration: float
    depends_on: list[str] = field(default_factory=list)

    # Computed during scheduling (M3/M4)
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    latest_start: float = float("inf")
    latest_finish: float = float("inf")
    level: int = 0  # depth in the DAG (for M3 stage grouping)


@dataclass
class ScheduleResult:
    """Aggregated output of the scheduler."""

    # M1
    valid: bool
    has_cycle: bool
    cycle_path: list[str]
    missing_deps: dict[str, list[str]]  # job_id → missing-dependency ids

    # M2
    topological_order: list[str]

    # M3
    stages: list[list[str]]  # each inner list = parallel stage
    worker_limit: int

    # M4
    critical_path: list[str]
    minimum_completion_time: float

    # M5
    failed_job_id: Optional[str]
    skipped_jobs: list[str]


# ──────────────────────────────────────────────
#  Validation helpers (M1)
# ──────────────────────────────────────────────


def _detect_cycles(
    jobs: dict[str, Job],
) -> tuple[bool, list[str]]:
    """Return (has_cycle, a cycle-path list if one was found).

    Uses DFS with three-colour marking:
      WHITE (0) — unvisited
      GRAY  (1) — in the current recursion stack
      BLACK (2) — fully processed
    """

    adj: dict[str, list[str]] = {jid: list(j.depends_on) for jid, j in jobs.items()}
    colour: dict[str, int] = {jid: 0 for jid in jobs}

    def dfs(node: str, stack: list[str]) -> bool:
        colour[node] = 1
        stack.append(node)
        for parent in adj.get(node, []):
            if colour.get(parent, -1) == 0:
                if dfs(parent, stack):
                    return True
            elif colour.get(parent) == 1:
                # Found a back-edge → cycle
                cycle_start = stack.index(parent)
                stack.append(parent)  # close the loop for display
                return True
        colour[node] = 2
        stack.pop()
        return False

    for jid in jobs:
        if colour[jid] == 0:
            stack: list[str] = []
            if dfs(jid, stack):
                return True, stack
    return False, []


def _check_missing_deps(jobs: dict[str, Job]) -> dict[str, list[str]]:
    """Return {job_id: [missing_dep, ...]} for every job that references ids not in the graph."""
    valid_ids = set(jobs)
    missing: dict[str, list[str]] = {}
    for jid, job in jobs.items():
        absent = [d for d in job.depends_on if d not in valid_ids]
        if absent:
            missing[jid] = absent
    return missing


# ──────────────────────────────────────────────
#  Topological sort (M2) — Kahn's algorithm
# ──────────────────────────────────────────────


def _topological_sort(
    jobs: dict[str, Job],
) -> list[str]:
    """Return a valid execution order respecting all dependencies."""
    in_degree: dict[str, int] = {jid: 0 for jid in jobs}
    children: dict[str, list[str]] = {jid: [] for jid in jobs}

    for jid, job in jobs.items():
        for dep in job.depends_on:
            if dep in jobs:
                in_degree[jid] += 1
                children[dep].append(jid)

    queue: deque[str] = deque(jid for jid, deg in in_degree.items() if deg == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order


# ──────────────────────────────────────────────
#  Computation helpers (M3, M4)
# ──────────────────────────────────────────────


def _forward_pass(jobs: dict[str, Job], topo: list[str]) -> None:
    """Compute earliest start/finish in topological order."""
    children: dict[str, list[str]] = {jid: [] for jid in jobs}
    for jid, job in jobs.items():
        for dep in job.depends_on:
            if dep in jobs:
                children[dep].append(jid)

    for jid in topo:
        job = jobs[jid]
        # Earliest start = max earliest finish of all dependencies
        if job.depends_on:
            job.earliest_start = max(
                (jobs[d].earliest_finish for d in job.depends_on if d in jobs),
                default=0,
            )
        else:
            job.earliest_start = 0
        job.earliest_finish = job.earliest_start + job.duration


def _backward_pass(jobs: dict[str, Job], topo: list[str]) -> None:
    """Compute latest start/finish in reverse topological order."""
    children: dict[str, list[str]] = {jid: [] for jid in jobs}
    for jid, job in jobs.items():
        for dep in job.depends_on:
            if dep in jobs:
                children[dep].append(jid)

    # Seed: all terminal nodes have LFT = project completion
    max_eft = max(j.earliest_finish for j in jobs.values())

    for jid in jobs:
        jobs[jid].latest_finish = max_eft  # initialise

    for jid in reversed(topo):
        job = jobs[jid]
        succs = children.get(jid, [])
        if succs:
            job.latest_finish = min(jobs[c].latest_start for c in succs)
            job.latest_start = job.latest_finish - job.duration
        else:
            job.latest_finish = max_eft
            job.latest_start = job.latest_finish - job.duration


def _compute_levels(jobs: dict[str, Job], topo: list[str]) -> None:
    """Assign each job a level = longest path from any source in edges.

    Level is used for stage-based grouping (M3).
    """
    for jid in topo:
        job = jobs[jid]
        if not job.depends_on:
            job.level = 0
        else:
            job.level = max(
                (jobs[d].level + 1 for d in job.depends_on if d in jobs),
                default=0,
            )


def _group_into_stages(
    jobs: dict[str, Job],
    topo: list[str],
    worker_limit: int,
) -> list[list[str]]:
    """Group jobs into parallel stages respecting dependencies and worker cap.

    Uses a simulation approach:
      At each discrete time instant, schedule ready jobs up to worker_limit.
      "Ready" = all dependencies have finished.
    """
    remaining_deps: dict[str, set[str]] = {}
    for jid, job in jobs.items():
        remaining_deps[jid] = set(job.depends_on)

    # Collect completion times keyed by (time, job_id) for deterministic ordering
    # We simulate by ordering jobs by earliest_start, then pick the next N ready.
    ready = [jid for jid in topo if not remaining_deps[jid]]
    # sort by earliest_start for deterministic simulation
    ready.sort(key=lambda j: (jobs[j].earliest_start, j))

    stages: list[list[str]] = []
    finished: set[str] = set()

    while ready:
        # Pick up to worker_limit jobs to start this stage
        stage = ready[:worker_limit]
        stages.append(stage)
        # Mark them as finished
        for jid in stage:
            finished.add(jid)
        ready = ready[worker_limit:]  # remainder

        # Determine next ready jobs (all deps among finished are satisfied)
        # Key insight: in a DAG with forward pass, the next ready jobs
        # are those whose *last* finishing dependency is in the finished set.
        potentially_ready: list[str] = []
        for jid, rdeps in remaining_deps.items():
            if jid in finished or not rdeps:
                continue
            if rdeps.issubset(finished):
                potentially_ready.append(jid)

        potentially_ready.sort(key=lambda j: (jobs[j].earliest_start, j))
        ready.extend(potentially_ready)
        ready.sort(key=lambda j: (jobs[j].earliest_start, j))

        # Remove the now-ready jobs from remaining_deps tracking
        for jid in potentially_ready:
            remaining_deps[jid] = set()

    return stages


def _critical_path(jobs: dict[str, Job]) -> tuple[list[str], float]:
    """Return (critical path job ids, minimum completion time).

    Critical path = jobs where earliest_start == latest_start (zero slack).
    """
    max_eft = max(j.earliest_finish for j in jobs.values())

    # Collect zero-slack jobs, sorted by earliest start
    zero_slack = sorted(
        [j for j in jobs.values() if abs(j.earliest_start - j.latest_start) < 1e-9],
        key=lambda j: j.earliest_start,
    )

    # Build the longest zero-slack chain
    if not zero_slack:
        return [], max_eft

    # Build adjacency within zero-slack jobs
    children: dict[str, list[str]] = {j.id: [] for j in zero_slack}
    for j in zero_slack:
        for dep in j.depends_on:
            if dep in jobs and abs(jobs[dep].earliest_start - jobs[dep].latest_start) < 1e-9:
                children.setdefault(dep, []).append(j.id)

    # Find longest path in the zero-slack subgraph (greedy: pick the one that
    # yields max finish time, which is correct for a CP chain)
    starts = [j.id for j in zero_slack if not j.depends_on or not any(
        abs(jobs[d].earliest_start - jobs[d].latest_start) < 1e-9
        for d in j.depends_on
    )]

    best_path: list[str] = []
    for start in starts:
        path = [start]
        current = start
        while current in children:
            nxt = children[current]
            if not nxt:
                break
            # Pick the child with the latest finish
            current = max(nxt, key=lambda c: jobs[c].earliest_finish)
            path.append(current)
        if len(path) > len(best_path):
            best_path = path

    return best_path, max_eft


def _downstream_jobs(jobs: dict[str, Job], failed_id: str) -> list[str]:
    """Return all jobs (transitively) downstream of the failed job.

    BFS over the reverse graph (job → dependants).
    """
    if failed_id not in jobs:
        return []

    # Build reverse adjacency (job → jobs that depend on it)
    dependants: dict[str, list[str]] = {jid: [] for jid in jobs}
    for jid, job in jobs.items():
        for dep in job.depends_on:
            if dep in jobs:
                dependants[dep].append(jid)

    visited: set[str] = {failed_id}
    queue: deque[str] = deque([failed_id])
    downstream: list[str] = []

    while queue:
        node = queue.popleft()
        for child in dependants.get(node, []):
            if child not in visited:
                visited.add(child)
                downstream.append(child)
                queue.append(child)

    # Sort by earliest_start for deterministic output
    downstream.sort(key=lambda j: jobs[j].earliest_start)
    return downstream


# ──────────────────────────────────────────────
#  Core entry point
# ──────────────────────────────────────────────


def schedule(
    jobs_list: list[dict[str, Any]],
    worker_limit: int = 2,
    failed_job_id: Optional[str] = None,
) -> ScheduleResult:
    """Run the full pipeline (M1–M5) and return aggregated results.

    Parameters
    ----------
    jobs_list : list[dict]
        Raw job definitions from JSON.
    worker_limit : int, default=2
        Maximum parallel workers for M3 stage grouping.
    failed_job_id : str or None
        If set, compute downstream skipped jobs for M5.

    Returns
    -------
    ScheduleResult
    """
    # ── Validation ──────────────────────────────
    if not isinstance(jobs_list, list):
        raise ValueError("Input must be a list of job objects")
    if worker_limit < 1:
        raise ValueError("worker_limit must be ≥ 1")

    # Build job map
    jobs: dict[str, Job] = {}
    seen_ids: set[str] = set()
    for entry in jobs_list:
        if not isinstance(entry, dict):
            raise ValueError(f"Each job entry must be a dict; got {type(entry).__name__}")
        jid = entry.get("id")
        if jid is None:
            raise ValueError("Each job entry must have an 'id' field")
        if not isinstance(jid, str) or not jid.strip():
            raise ValueError(f"Job 'id' must be a non-empty string; got {jid!r}")
        if jid in seen_ids:
            raise ValueError(f"Duplicate job id: {jid!r}")
        seen_ids.add(jid)

        duration = entry.get("duration", 0)
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(
                f"Job {jid!r}: 'duration' must be a non-negative number; got {duration!r}"
            )

        depends_on = entry.get("dependsOn", [])
        if not isinstance(depends_on, list):
            raise ValueError(
                f"Job {jid!r}: 'dependsOn' must be a list; got {type(depends_on).__name__}"
            )
        for i, dep in enumerate(depends_on):
            if not isinstance(dep, str):
                raise ValueError(
                    f"Job {jid!r}: dependsOn[{i}] must be a string; got {type(dep).__name__}"
                )

        jobs[jid] = Job(id=jid, duration=float(duration), depends_on=list(depends_on))

    if not jobs:
        raise ValueError("At least one job is required")

    # ── M1: Validate ────────────────────────────
    has_cycle, cycle_path = _detect_cycles(jobs)
    missing_deps = _check_missing_deps(jobs)
    valid = not has_cycle and not missing_deps

    # ── M2: Topological sort ───────────────────
    topo = _topological_sort(jobs) if valid else []

    # ── M3/M4: Forward/backward passes ──────────
    if valid:
        _forward_pass(jobs, topo)
        _backward_pass(jobs, topo)
        _compute_levels(jobs, topo)
        stages = _group_into_stages(jobs, topo, worker_limit)
        cp, min_time = _critical_path(jobs)
    else:
        stages = []
        cp = []
        min_time = 0.0

    # ── M5: Failure simulation ──────────────────
    skipped: list[str] = []
    if failed_job_id and failed_job_id in jobs:
        skipped = _downstream_jobs(jobs, failed_job_id)
    elif failed_job_id and failed_job_id not in jobs and failed_job_id.strip():
        # Referenced a non-existent job — report nothing for M5
        pass

    return ScheduleResult(
        valid=valid,
        has_cycle=has_cycle,
        cycle_path=cycle_path,
        missing_deps=missing_deps,
        topological_order=topo,
        stages=stages,
        worker_limit=worker_limit,
        critical_path=cp,
        minimum_completion_time=min_time,
        failed_job_id=failed_job_id if (failed_job_id and failed_job_id in jobs) else None,
        skipped_jobs=skipped,
    )


# ──────────────────────────────────────────────
#  Utility: load from file
# ──────────────────────────────────────────────


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    """Load job definitions from a JSON file.  Surface parse/IO errors clearly."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    if not p.is_file():
        raise IsADirectoryError(f"Expected a file, got a directory: {p.resolve()}")

    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"File is not valid UTF-8: {p.resolve()}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p.resolve()}: {exc}")

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object with a 'jobs' key")
    if "jobs" not in data:
        raise ValueError("JSON root must contain a 'jobs' key")
    return data["jobs"]