# PS5: Job Scheduler with Dependency Resolution (DAG)

A mini build/ETL orchestrator that validates, schedules, and analyses a
directed acyclic graph (DAG) of jobs.

## Quick start

```bash
# Install deps (pytest only needed for tests)
pip install pytest

# Run on the sample input
python cli.py sample_jobs.json

# With custom worker limit
python cli.py sample_jobs.json --workers 4

# Simulate a failure
python cli.py sample_jobs.json --fail D

# Run tests
pytest test_scheduler.py -v
```

## Input format

```json
{
  "jobs": [
    { "id": "A", "duration": 3, "dependsOn": [] },
    { "id": "B", "duration": 4, "dependsOn": [] },
    { "id": "C", "duration": 2, "dependsOn": ["A"] },
    { "id": "D", "duration": 5, "dependsOn": ["A", "B"] },
    { "id": "E", "duration": 1, "dependsOn": ["C", "D"] },
    { "id": "F", "duration": 3, "dependsOn": ["D"] },
    { "id": "G", "duration": 2, "dependsOn": ["E", "F"] }
  ]
}
```

- `id` — unique string identifier
- `duration` — non-negative number (float allowed)
- `dependsOn` — list of job ids this job must wait for (may be empty/omitted)

## What it does (M1–M5)

### M1 — Graph validation
- **Cycle detection**: DFS with 3-colour marking. Reports the full cycle path if found.
- **Missing dependency detection**: flags every job that references an unknown id.

### M2 — Topological sort
- **Kahn's algorithm** (BFS, in-degree queue). Produces a valid linear execution order.

### M3 — Parallel stages with worker limit
- Groups jobs into stages where each stage runs at most N jobs concurrently.
- Uses a simulation approach: at each step, pick up to N ready jobs, advance
  time, then pick the next ready set whose dependencies are all satisfied.
- A stage is a set of jobs that can all start at the same logical time.

### M4 — Critical path
- Forward pass: compute earliest start / earliest finish (EST/EFT).
- Backward pass: compute latest start / latest finish (LST/LFT).
- Zero-slack jobs form the critical path. Delaying any of them delays the
  whole project.
- Minimum total completion time = max EFT across all jobs.

### M5 — Failure simulation
- Given a failed job id, BFS over the reverse dependency graph lists every
  transitively downstream job that must be skipped.

## Architecture

```
scheduler.py    Core engine — pure functions, no I/O
cli.py          CLI runner — argparse → load → schedule → print
test_scheduler.py  pytest suite — 40+ tests covering all milestones
sample_jobs.json   Example input
PROMPTS.md      AI collaboration log
```

### Key design decisions

| Concern | Approach |
|---|---|
| **Determinism** | All tie-breaking sorts by (earliest_start, id) |
| **Float safety** | Slack comparison uses `abs(…) < 1e-9` |
| **Validation** | Input validated eagerly before any computation |
| **Empty deps** | `dependsOn` key is optional (defaults to `[]`) |
| **Zero-duration** | Supported — does not break forward/backward passes |
| **Disconnected DAG** | Multiple roots/leaves handled naturally |

## Edge cases covered

- Self-loops (`"A" depends on "A"`)
- Multiple disconnected components
- Single-job graph
- Zero-duration jobs
- Float durations
- Duplicate job ids
- Blank/whitespace-only ids
- Negative durations
- Non-existent failed job id (M5)
- worker_limit = 1 (sequential)
- Very large DAGs (500+ nodes)
- Missing `dependsOn` field entirely

## Security & robustness

- JSON loaded with `json.loads` (no arbitrary code exec)
- Path traversal not possible (only reads the file at the given path)
- All user inputs validated with clear error messages
- No external runtime dependencies beyond stdlib + pytest