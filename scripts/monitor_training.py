#!/usr/bin/env python3
"""
Training anomaly detection — compares current vs previous training_status.json
and creates/updates GitHub Issues via `gh` CLI.

Called by .github/workflows/training-monitor.yml after each status update push.

Anomaly categories:
  - stall:            status=training but no checkpoint update in >6h
  - loss_regression:  best_loss increased between snapshots
  - unexpected_stop:  status changed training->stopped without note change
  - completion:       run reached total_epochs
  - new_run:          run appeared that wasn't in previous snapshot
  - run_disappeared:  run was in previous snapshot but gone now
  - low_progress:     epoch progress rate = 0 between updates
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_JSON = REPO_ROOT / "training_status.json"
DASHBOARD_LABEL = "training-dashboard"
ALERT_LABEL = "training-alert"

# Thresholds
STALL_HOURS = 6.0  # hours without checkpoint update while status=training


@dataclass(frozen=True)
class Anomaly:
    severity: str  # critical, warning, info
    run_id: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_current() -> dict[str, Any]:
    if not STATUS_JSON.exists():
        print(f"ERROR: {STATUS_JSON} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(STATUS_JSON.read_text())


def load_previous() -> dict[str, Any] | None:
    """Load previous version of training_status.json from git history."""
    try:
        result = subprocess.run(
            ["git", "show", "HEAD~1:training_status.json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def runs_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["run_id"]: r for r in data.get("runs", [])}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def detect_anomalies(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    now = datetime.now()
    cur_runs = runs_by_id(current)
    prev_runs = runs_by_id(previous) if previous else {}

    for run_id, run in cur_runs.items():
        status = run.get("status", "unknown")
        version = run.get("version", run_id[:30])

        # --- Stall detection ---
        if status == "training":
            try:
                last_mod = datetime.strptime(
                    run["last_modified"], "%Y-%m-%d %H:%M"
                )
                hours_since = (now - last_mod).total_seconds() / 3600
                if hours_since > STALL_HOURS:
                    anomalies.append(
                        Anomaly(
                            severity="critical",
                            run_id=run_id,
                            category="stall",
                            message=(
                                f"**{version}** has not produced a new checkpoint "
                                f"in {hours_since:.1f}h (threshold: {STALL_HOURS}h)"
                            ),
                            details={"hours_since_update": round(hours_since, 1)},
                        )
                    )
            except (KeyError, ValueError):
                pass

        # --- Loss regression ---
        if run_id in prev_runs:
            prev_best = prev_runs[run_id].get("best_loss")
            cur_best = run.get("best_loss")
            if (
                prev_best is not None
                and cur_best is not None
                and cur_best > prev_best
            ):
                anomalies.append(
                    Anomaly(
                        severity="warning",
                        run_id=run_id,
                        category="loss_regression",
                        message=(
                            f"**{version}** best_loss regressed: "
                            f"{prev_best:.6f} -> {cur_best:.6f}"
                        ),
                        details={
                            "previous_best_loss": prev_best,
                            "current_best_loss": cur_best,
                        },
                    )
                )

        # --- Unexpected stop ---
        if run_id in prev_runs:
            prev_status = prev_runs[run_id].get("status", "unknown")
            if prev_status == "training" and status == "stopped":
                prev_note = prev_runs[run_id].get("note", "")
                cur_note = run.get("note", "")
                if prev_note == cur_note:
                    anomalies.append(
                        Anomaly(
                            severity="critical",
                            run_id=run_id,
                            category="unexpected_stop",
                            message=(
                                f"**{version}** stopped unexpectedly "
                                f"(was training, now stopped, no note change)"
                            ),
                        )
                    )

        # --- Completion ---
        if run_id in prev_runs:
            prev_status = prev_runs[run_id].get("status", "unknown")
            if prev_status != "completed" and status == "completed":
                total = run.get("total_epochs", "?")
                anomalies.append(
                    Anomaly(
                        severity="info",
                        run_id=run_id,
                        category="completion",
                        message=(
                            f"**{version}** completed training "
                            f"({total} epochs)"
                        ),
                    )
                )

        # --- Low progress rate ---
        if run_id in prev_runs and status == "training":
            prev_ep = prev_runs[run_id].get("last_epoch", 0)
            cur_ep = run.get("last_epoch", 0)
            total = run.get("total_epochs", 0)
            delta_ep = cur_ep - prev_ep
            if delta_ep == 0 and prev_runs[run_id].get("status") == "training":
                anomalies.append(
                    Anomaly(
                        severity="warning",
                        run_id=run_id,
                        category="low_progress",
                        message=(
                            f"**{version}** made 0 epoch progress since last update "
                            f"(stuck at ep{cur_ep}/{total})"
                        ),
                        details={"epoch": cur_ep, "total": total},
                    )
                )

    # --- New runs ---
    for run_id in cur_runs:
        if run_id not in prev_runs:
            run = cur_runs[run_id]
            version = run.get("version", run_id[:30])
            anomalies.append(
                Anomaly(
                    severity="info",
                    run_id=run_id,
                    category="new_run",
                    message=f"**{version}** new run detected: `{run_id}`",
                    details={
                        "data": run.get("data", ""),
                        "total_epochs": run.get("total_epochs", 0),
                    },
                )
            )

    # --- Disappeared runs ---
    for run_id in prev_runs:
        if run_id not in cur_runs:
            prev = prev_runs[run_id]
            version = prev.get("version", run_id[:30])
            anomalies.append(
                Anomaly(
                    severity="warning",
                    run_id=run_id,
                    category="run_disappeared",
                    message=(
                        f"**{version}** run `{run_id}` was in previous snapshot "
                        f"but no longer found"
                    ),
                )
            )

    return anomalies


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    anomalies: list[Anomaly], current: dict[str, Any]
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Training Monitor Report",
        "",
        f"> Generated at {now}",
        "",
    ]

    if not anomalies:
        lines.append("All runs nominal. No anomalies detected.")
        lines.append("")
        _append_summary_table(lines, current)
        return "\n".join(lines)

    # Group by severity
    critical = [a for a in anomalies if a.severity == "critical"]
    warnings = [a for a in anomalies if a.severity == "warning"]
    infos = [a for a in anomalies if a.severity == "info"]

    if critical:
        lines.extend(["## Critical", ""])
        for a in critical:
            lines.append(f"- [{a.category}] {a.message}")
        lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        for a in warnings:
            lines.append(f"- [{a.category}] {a.message}")
        lines.append("")

    if infos:
        lines.extend(["## Info", ""])
        for a in infos:
            lines.append(f"- [{a.category}] {a.message}")
        lines.append("")

    _append_summary_table(lines, current)
    return "\n".join(lines)


def _append_summary_table(lines: list[str], current: dict[str, Any]) -> None:
    lines.extend(
        [
            "## Current Status",
            "",
            "| Version | Data | Status | Progress | Best Loss | Last Modified |",
            "|---------|------|--------|----------|-----------|---------------|",
        ]
    )
    for r in current.get("runs", []):
        version = r.get("version", "?")
        data = r.get("data", "")
        status = r.get("status", "?")
        last_ep = r.get("last_epoch", 0)
        total = r.get("total_epochs", 0)
        progress = f"{last_ep}/{total}" if total else str(last_ep)
        best = r.get("best_loss")
        best_str = f"{best:.6f}" if best is not None else "\u2014"
        last_mod = r.get("last_modified", "")
        lines.append(
            f"| {version} | {data} | {status} | {progress} | {best_str} | {last_mod} |"
        )
    lines.append("")


# ---------------------------------------------------------------------------
# GitHub Issue management
# ---------------------------------------------------------------------------


def find_dashboard_issue() -> int | None:
    """Find existing dashboard issue by label."""
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


def create_dashboard_issue(body: str) -> int:
    """Create the dashboard issue."""
    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            "Training Dashboard",
            "--body",
            body,
            "--label",
            DASHBOARD_LABEL,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"Failed to create issue: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    url = result.stdout.strip()
    print(f"Created dashboard issue: {url}")
    return int(url.rstrip("/").split("/")[-1])


def update_dashboard_issue(issue_number: int, body: str) -> None:
    """Update dashboard issue body."""
    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--body",
            body,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    print(f"Updated dashboard issue #{issue_number}")


def ensure_label_exists(label: str, color: str, description: str) -> None:
    """Create label if it doesn't exist."""
    subprocess.run(
        [
            "gh",
            "label",
            "create",
            label,
            "--color",
            color,
            "--description",
            description,
            "--force",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def create_alert_issue(anomaly: Anomaly) -> None:
    """Create a separate issue for critical/warning anomalies."""
    title = f"[{anomaly.severity.upper()}] {anomaly.category}: {anomaly.run_id[:40]}"
    body = (
        f"## {anomaly.category}\n\n"
        f"{anomaly.message}\n\n"
        f"**Run ID:** `{anomaly.run_id}`\n"
        f"**Severity:** {anomaly.severity}\n"
    )
    if anomaly.details:
        body += f"\n**Details:**\n```json\n{json.dumps(anomaly.details, indent=2)}\n```\n"

    subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--label",
            ALERT_LABEL,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    print(f"Created alert issue: {title}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    current = load_current()
    previous = load_previous()

    anomalies = detect_anomalies(current, previous)
    report = generate_report(anomalies, current)

    # Write report to file (for workflow artifact)
    report_path = REPO_ROOT / "monitor_report.md"
    report_path.write_text(report)
    print(report)

    # Ensure labels exist
    ensure_label_exists(DASHBOARD_LABEL, "0075ca", "Training status dashboard")
    ensure_label_exists(ALERT_LABEL, "d93f0b", "Training anomaly alert")

    # Update or create dashboard issue
    issue_num = find_dashboard_issue()
    if issue_num:
        update_dashboard_issue(issue_num, report)
    else:
        issue_num = create_dashboard_issue(report)

    # Create alert issues for critical/warning anomalies
    for a in anomalies:
        if a.severity in ("critical", "warning"):
            create_alert_issue(a)


if __name__ == "__main__":
    main()
