#!/usr/bin/env python3
"""
Daily training digest — compares current training_status.json with ~24h ago
and posts a summary comment on the Training Dashboard GitHub Issue.

Called by .github/workflows/daily-summary.yml on a daily schedule.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_JSON = REPO_ROOT / "training_status.json"
DASHBOARD_LABEL = "training-dashboard"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_current() -> dict[str, Any]:
    if not STATUS_JSON.exists():
        print(f"ERROR: {STATUS_JSON} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(STATUS_JSON.read_text())


def find_commit_near_hours_ago(hours: int) -> str | None:
    """Find the git commit closest to N hours ago that touched training_status.json."""
    since = f"{hours} hours ago"
    result = subprocess.run(
        [
            "git",
            "log",
            f"--since={since}",
            "--reverse",
            "--format=%H",
            "--",
            "training_status.json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]

    # Fallback: get the last commit before the cutoff
    result = subprocess.run(
        [
            "git",
            "log",
            f"--until={since}",
            "--format=%H",
            "-1",
            "--",
            "training_status.json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]

    return None


def load_from_commit(commit: str) -> dict[str, Any] | None:
    """Load training_status.json from a specific git commit."""
    result = subprocess.run(
        ["git", "show", f"{commit}:training_status.json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout)
    return None


def runs_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["run_id"]: r for r in data.get("runs", [])}


# ---------------------------------------------------------------------------
# Digest generation
# ---------------------------------------------------------------------------


def format_delta(current: float | int, previous: float | int) -> str:
    """Format a numeric delta with direction indicator."""
    diff = current - previous
    if isinstance(diff, float):
        if diff < 0:
            return f"{diff:.6f} (improved)"
        if diff > 0:
            return f"+{diff:.6f} (regressed)"
        return "no change"
    if diff > 0:
        return f"+{diff}"
    if diff < 0:
        return str(diff)
    return "no change"


def generate_digest(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## Daily Digest \u2014 {now[:10]}",
        "",
        f"> Comparing current state with ~24h ago | Generated at {now}",
        "",
    ]

    cur_runs = runs_by_id(current)

    if previous is None:
        lines.append("No previous snapshot found (first digest or no history).")
        lines.append("")
        _append_current_summary(lines, current)
        return "\n".join(lines)

    prev_runs = runs_by_id(previous)

    # --- Per-run deltas ---
    lines.extend([
        "### Run Progress",
        "",
        "| Version | Data | Status | Epoch (24h ago -> now) | Best Loss (24h ago -> now) | Note |",
        "|---------|------|--------|-----------------------|---------------------------|------|",
    ])

    all_run_ids = sorted(set(list(cur_runs.keys()) + list(prev_runs.keys())))
    for run_id in all_run_ids:
        cur = cur_runs.get(run_id)
        prev = prev_runs.get(run_id)

        if cur and not prev:
            version = cur.get("version", "?")
            data = cur.get("data", "")
            status = cur.get("status", "?")
            ep = cur.get("last_epoch", 0)
            total = cur.get("total_epochs", 0)
            best = cur.get("best_loss")
            best_str = f"{best:.6f}" if best is not None else "\u2014"
            lines.append(
                f"| {version} | {data} | {status} | NEW -> {ep}/{total} | NEW -> {best_str} | new run |"
            )
            continue

        if prev and not cur:
            version = prev.get("version", "?")
            data = prev.get("data", "")
            lines.append(
                f"| {version} | {data} | GONE | was {prev.get('last_epoch', '?')}/{prev.get('total_epochs', '?')} | \u2014 | run disappeared |"
            )
            continue

        assert cur is not None and prev is not None
        version = cur.get("version", "?")
        data = cur.get("data", "")
        status = cur.get("status", "?")
        prev_status = prev.get("status", "?")
        status_str = (
            f"{prev_status} -> {status}" if status != prev_status else status
        )

        prev_ep = prev.get("last_epoch", 0)
        cur_ep = cur.get("last_epoch", 0)
        total = cur.get("total_epochs", 0)
        ep_delta = cur_ep - prev_ep
        ep_str = f"{prev_ep} -> {cur_ep}/{total} (+{ep_delta})" if ep_delta else f"{cur_ep}/{total}"

        prev_best = prev.get("best_loss")
        cur_best = cur.get("best_loss")
        if prev_best is not None and cur_best is not None:
            loss_delta = format_delta(cur_best, prev_best)
            loss_str = f"{prev_best:.6f} -> {cur_best:.6f} ({loss_delta})"
        elif cur_best is not None:
            loss_str = f"{cur_best:.6f}"
        else:
            loss_str = "\u2014"

        note = cur.get("note", "")
        lines.append(
            f"| {version} | {data} | {status_str} | {ep_str} | {loss_str} | {note} |"
        )

    lines.append("")

    # --- Summary stats ---
    training_count = sum(
        1 for r in cur_runs.values() if r.get("status") == "training"
    )
    completed_count = sum(
        1 for r in cur_runs.values() if r.get("status") == "completed"
    )
    stopped_count = sum(
        1 for r in cur_runs.values() if r.get("status") == "stopped"
    )
    total_epoch_progress = sum(
        cur_runs[rid].get("last_epoch", 0) - prev_runs.get(rid, {}).get("last_epoch", 0)
        for rid in cur_runs
        if rid in prev_runs
    )

    lines.extend([
        "### Summary",
        "",
        f"- **Active training:** {training_count} runs",
        f"- **Completed:** {completed_count} runs",
        f"- **Stopped:** {stopped_count} runs",
        f"- **Total epoch progress (24h):** +{total_epoch_progress}",
        "",
    ])

    pending = current.get("pending", [])
    if pending:
        lines.extend(["### Pending Runs", ""])
        for p in pending:
            lines.append(
                f"- **{p['version']}** ({p['data']}): {p['status']} \u2014 {p.get('note', '')}"
            )
        lines.append("")

    return "\n".join(lines)


def _append_current_summary(lines: list[str], current: dict[str, Any]) -> None:
    lines.extend([
        "### Current State",
        "",
        "| Version | Data | Status | Progress | Best Loss |",
        "|---------|------|--------|----------|-----------|",
    ])
    for r in current.get("runs", []):
        version = r.get("version", "?")
        data = r.get("data", "")
        status = r.get("status", "?")
        ep = r.get("last_epoch", 0)
        total = r.get("total_epochs", 0)
        best = r.get("best_loss")
        best_str = f"{best:.6f}" if best is not None else "\u2014"
        lines.append(f"| {version} | {data} | {status} | {ep}/{total} | {best_str} |")
    lines.append("")


# ---------------------------------------------------------------------------
# GitHub Issue management
# ---------------------------------------------------------------------------


def find_dashboard_issue() -> int | None:
    """Find the dashboard issue number."""
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--label",
            DASHBOARD_LABEL,
            "--state",
            "open",
            "--json",
            "number",
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode == 0 and result.stdout.strip():
        issues = json.loads(result.stdout)
        if issues:
            return issues[0]["number"]
    return None


def post_digest_comment(issue_number: int, digest: str) -> None:
    """Post digest as a comment on the dashboard issue."""
    result = subprocess.run(
        [
            "gh",
            "issue",
            "comment",
            str(issue_number),
            "--body",
            digest,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"Failed to post comment: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Posted daily digest on issue #{issue_number}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    current = load_current()

    commit = find_commit_near_hours_ago(24)
    previous = load_from_commit(commit) if commit else None

    digest = generate_digest(current, previous)

    # Write to file (for workflow artifact / local testing)
    digest_path = REPO_ROOT / "daily_digest.md"
    digest_path.write_text(digest)
    print(digest)

    # Post to dashboard issue
    issue_num = find_dashboard_issue()
    if issue_num:
        post_digest_comment(issue_num, digest)
    else:
        print(
            "WARNING: No dashboard issue found. "
            "Run monitor_training.py first to create one.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
