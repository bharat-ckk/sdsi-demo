"""
CLI runner for the Job Scheduler (PS5).
Usage:
  python cli.py sample_jobs.json [--workers N] [--fail JOB_ID]
"""

import argparse
import json
import sys
from pathlib import Path

from scheduler import load_jobs, schedule


def _escape_path(obj: list[str]) -> str:
    return " → ".join(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Job Scheduler with Dependency Resolution (DAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py sample_jobs.json
  python cli.py sample_jobs.json --workers 4
  python cli.py sample_jobs.json --fail B
""",
    )
    parser.add_argument("input", help="Path to JSON job definitions")
    parser.add_argument(
        "--workers", "-n", type=int, default=2,
        help="Maximum parallel workers for M3 stage grouping (default: 2)",
    )
    parser.add_argument(
        "--fail", type=str, default=None,
        help="Job ID to simulate failure for (M5)",
    )
    args = parser.parse_args()

    # Load
    try:
        jobs_list = load_jobs(args.input)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(f"Error loading input: {exc}", file=sys.stderr)
        sys.exit(1)

    # Run
    try:
        result = schedule(jobs_list, worker_limit=args.workers, failed_job_id=args.fail)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    jobs_map = {j["id"]: j for j in jobs_list}

    # ── Report ────────────────────────────────────
    print("=" * 60)
    print("  JOB SCHEDULER — DAG Analysis")
    print("=" * 60)

    print(f"\n📋 Jobs loaded: {len(jobs_list)}")

    # M1
    print("\n── M1: Graph Validation ──")
    if result.valid:
        print("  ✅ Graph is valid (no cycles, no missing dependencies).")
    else:
        if result.has_cycle:
            print(f"  ❌ Cycle detected: {_escape_path(result.cycle_path)}")
        if result.missing_deps:
            for jid, deps in result.missing_deps.items():
                print(f"  ⚠️  Job '{jid}' depends on unknown: {', '.join(deps)}")
        print("  ⛔ Skipping M2–M4 (invalid graph).")
        # Still show M5 if requested
        if result.failed_job_id:
            print("\n── M5: Failure Simulation ──")
            skipped = result.skipped_jobs
            print(f"  Failed job: {result.failed_job_id}")
            print(f"  Skipped downstream: {skipped if skipped else '(none)'}")
        return

    # M2
    print("\n── M2: Topological Order ──")
    print(f"  {' → '.join(result.topological_order)}")

    # M3
    print(f"\n── M3: Parallel Stages (worker limit = {result.worker_limit}) ──")
    for i, stage in enumerate(result.stages, 1):
        durs = [f"{jid}({jobs_map[jid]['duration']})" for jid in stage]
        print(f"  Stage {i}: [{', '.join(durs)}]")

    # M4
    print("\n── M4: Critical Path ──")
    print(f"  Critical path: {' → '.join(result.critical_path) if result.critical_path else '(none)'}")
    print(f"  Minimum completion time: {result.minimum_completion_time:.1f}")

    # M5
    if args.fail:
        print("\n── M5: Failure Simulation ──")
        print(f"  Failed job: {args.fail}")
        print(f"  Skipped downstream: {result.skipped_jobs if result.skipped_jobs else '(none)'}")


if __name__ == "__main__":
    main()