#!/usr/bin/env python3
"""Report GitHub Actions consumption for a repository.

Lists recent workflow runs, converts each job's wall time into billed
minutes using GitHub's runner multipliers (ubuntu 1x, windows 2x,
macOS 10x), and prints a per-run and per-workflow summary.

Usage:
    python scripts/actions_usage.py                      # this repo, last 50 runs
    python scripts/actions_usage.py --days 30            # only runs from the last 30 days
    python scripts/actions_usage.py --repo owner/name    # another repository
    GITHUB_TOKEN=ghp_... python scripts/actions_usage.py  # authenticated (higher API limits)

Public repositories can be queried anonymously, but GitHub rate-limits
unauthenticated requests to 60/hour — set GITHUB_TOKEN for repeated use.
Billed minutes are an estimate: GitHub actually meters per second and
applies the multiplier, this script rounds each job up to whole minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

RUNNER_MULTIPLIERS = {
    "ubuntu": 1,
    "windows": 2,
    "macos": 10,
}


def default_repo() -> str:
    """Parse owner/name from the git origin remote."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "guaguastandup/OmniTranscribe"
    for prefix in ("https://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url[len(prefix):].removesuffix(".git")
    return url


def api_get(url: str, token: str | None) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def job_billed_minutes(job: dict) -> tuple[int, int]:
    """Return (estimated billed minutes, multiplier) for one job."""
    started = parse_time(job["started_at"])
    completed = parse_time(job["completed_at"]) if job.get("completed_at") else datetime.now(timezone.utc)
    seconds = max(1, (completed - started).total_seconds())
    runner = (job.get("runner_name") or "").lower()
    multiplier = 1
    for name, factor in RUNNER_MULTIPLIERS.items():
        if name in runner:
            multiplier = factor
            break
    import math

    return math.ceil(seconds / 60) * multiplier, multiplier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=default_repo(), help="owner/name (default: this repo's origin)")
    parser.add_argument("--limit", type=int, default=50, help="how many recent runs to fetch (1-100)")
    parser.add_argument("--days", type=int, default=None, help="only count runs newer than N days")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=args.days) if args.days else None
    )

    print(f"Repository: {args.repo}")
    if not token:
        print("(tip: set GITHUB_TOKEN to avoid the 60 req/hour anonymous API limit)")
    print()

    data = api_get(
        f"https://api.github.com/repos/{args.repo}/actions/runs?per_page={args.limit}", token
    )
    runs = data.get("workflow_runs", [])

    if not runs:
        print("No workflow runs found.")
        return

    per_workflow: dict[str, tuple[int, int]] = {}
    header = f"{'RUN':<40} {'COMMIT':<8} {'STATUS':<10} {'EST. MIN':>8}"
    print(header)
    print("-" * len(header))

    for run in runs:
        started = parse_time(run["run_started_at"])
        if cutoff and started < cutoff:
            continue
        jobs = api_get(
            f"https://api.github.com/repos/{args.repo}/actions/runs/{run['id']}/jobs?per_page=100",
            token,
        ).get("jobs", [])

        billed = 0
        for job in jobs:
            minutes, _ = job_billed_minutes(job)
            billed += minutes

        name = (run["display_title"] or "")[:38]
        status = run.get("conclusion") or run["status"]
        print(f"{name:<40} {run['head_sha'][:7]:<8} {status:<10} {billed:>8}")

        wf_total = per_workflow.setdefault(run["name"], [0, 0])
        wf_total[0] += billed
        wf_total[1] += 1

    print()
    print("Estimated billed minutes by workflow:")
    for workflow, (total, count) in sorted(per_workflow.items(), key=lambda kv: -kv[1][0]):
        print(f"  {workflow}: {total} minutes across {count} run(s)")
    grand_total = sum(t[0] for t in per_workflow.values())
    print(f"  TOTAL: {grand_total} minutes")
    if any("Desktop Release" in name for name in per_workflow):
        print("\nNote: Desktop Release runs include macOS (10x) and Windows (2x) runners.")
    print("This only counts THIS repository. Account-wide usage: GitHub → Settings → Billing → Actions.")


if __name__ == "__main__":
    sys.exit(main())
