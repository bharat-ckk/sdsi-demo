# PROMPTS.md — AI Collaboration Log

## PS5: Job Scheduler with Dependency Resolution (DAG)

**Collaborator**: Claude Code (Claude Agent SDK, DeepSeek V4 Pro)
**Date**: 2026-10-27

---

## Iteration 1: Architecture & design

**Prompt**: "PS5: Job Scheduler with Dependency Resolution (DAG) … folder: submission/bharat/"

The AI proposed:
- Pure Python (stdlib-only for engine, pytest for testing)
- Separate modules: `scheduler.py` (engine), `cli.py` (runner), `test_scheduler.py`
- Algorithms: DFS 3-colour cycles, Kahn's for topo sort, simulation-based stages,
  forward/backward passes for critical path, BFS for downstream failure

**Decision**: Accepted the architecture. Single-pass implementation with all
five milestones in one file + CLI.

---

## Iteration 2: Core engine implementation

**Prompt**: (Continuation — the AI wrote `scheduler.py`)

Key design choices made during implementation:
- Dataclass `Job` holds computed fields (EST, EFT, LST, LFT, level)
- `ScheduleResult` dataclass aggregates all M1–M5 outputs
- Stage grouping uses a simulation approach rather than pure level-grouping
  to properly respect the worker limit
- Critical path extracted by building the longest zero-slack chain
- Input validation is eager — checks types, duplicates, ranges before computation

---

## Iteration 3: CLI runner

**Prompt**: (Continuation — the AI wrote `cli.py`)

- argparse-based CLI with `--workers` and `--fail` flags
- Pretty-printed output for each milestone
- Graceful handling: if graph is invalid, M2–M4 are skipped but M5 is still shown

---

## Iteration 4: Test suite

**Prompt**: (Continuation — the AI wrote `test_scheduler.py`)

Coverage plan:
- M1: no cycle, simple cycle, 3-node cycle, self-loop, cycle in subgraph,
  disconnected DAG, missing deps (none, single, multiple, all missing)
- M2: linear chain, diamond, respects all deps, single job, empty
- M3: worker=1 sequential, worker=∞ parallel, limit respected, all jobs
  scheduled, dependency order maintained
- M4: min completion time, single job, parallel paths (slack vs no-slack)
- M5: root failure, leaf failure, intermediate, nonexistent job, empty graph
- Integration: full pipeline valid, with failure, cycle halts M2–M4,
  missing dep halts, disconnected jobs, fan-out/fan-in
- Edge cases: zero duration, large durations, float durations, single job,
  omitted dependsOn
- Validation: empty input, non-list, missing id, blank id, duplicate id,
  negative duration, non-list dependsOn, non-string dep, worker_limit < 1
- File I/O: valid JSON, missing file, invalid JSON, missing jobs key,
  non-object root
- Performance: 500-node chain, 100 independent jobs

---

## Iteration 5: Documentation

**Prompt**: (Continuation — the AI wrote README.md and PROMPTS.md)

README covers: quick start, input format, M1–M5 descriptions, architecture
table, edge cases, security notes.

---

## What worked well

- The simulation-based stage grouping correctly handles cases where a
  level-based approach would over-pack a stage
- Eager input validation catches problems before any computation runs
- The 3-colour cycle detection reports the exact cycle path, not just "yes/no"

## What could be improved

- Stage grouping could use a priority-queue event-driven simulation
  for more accurate time-based scheduling
- Critical path extraction could be done with a single forward pass + DP
  rather than the two-pass CPM approach (though the two-pass is pedagogically clearer)
- Could add support for job weights/priorities within the same stage