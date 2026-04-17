#!/usr/bin/env python3
"""
Sync training_status.json to a Google Sheet.

Each run gets two row types:
  - A "current snapshot" row in the main sheet (updated in-place per run_id)
  - An append-only "history" row in a second sheet (one row per update)

Called by the update workflow after generating training_status.json.

Requires:
  - Service account JSON key at ~/.config/training-tracker/service-account.json
    (or path via GOOGLE_SERVICE_ACCOUNT_KEY env var)
  - Sheet shared with the service account email as editor
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gspread

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_JSON = REPO_ROOT / "training_status.json"

SHEET_ID = "1Pc6UF3Up_jLPttetXGJDAp-iu4yc4-tDdxkWqckBrHs"
KEY_PATH = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_KEY",
    str(Path.home() / ".config/training-tracker/service-account.json"),
)

# Column layout for the "Status" sheet
STATUS_HEADERS = [
    "run_id",
    "version",
    "data",
    "vfi",
    "status",
    "best_epoch",
    "best_loss",
    "last_epoch",
    "last_loss",
    "lr",
    "total_epochs",
    "progress",
    "num_checkpoints",
    "start_date",
    "last_modified",
    "note",
    "updated_at",
]

# Column layout for the "History" sheet (append-only log)
HISTORY_HEADERS = [
    "timestamp",
    "run_id",
    "version",
    "data",
    "status",
    "best_epoch",
    "best_loss",
    "last_epoch",
    "last_loss",
    "total_epochs",
    "num_checkpoints",
]

# Column layout for the "Datasets" sheet (static reference)
DATASET_HEADERS = [
    "dataset",
    "type",
    "structural_width",
    "aux_lines",
    "vfi",
    "samples",
    "design_goal",
]

DATASET_ROWS = [
    ["train", "real", "original", "all", "—", "BONES + StudioSeven", "Baseline real-motion data"],
    ["train2", "synthetic", "1.0–4.0px (80% in 1–2px)", "all @ 1px", "—", "2000 seq × 6f = 12k", "Foundational synthetic for V9 LoRA"],
    ["train3", "synthetic+LTX", "same as train2", "all @ 1px", "LTX-2.3", "12k × 3 variants = 36k", "Diffusion-noise robustness"],
    ["train4", "synthetic", "1.1–1.2px only", "all @ 1px", "—", "500 seq × 6f = 3k", "Tight structural line control"],
    ["train5", "synthetic+LTX", "1.1–1.2px only", "all @ 1px", "LTX-2.3", "3k × 3 variants = 9k", "Tight lines + diffusion"],
    ["train6", "synthetic", "1.1–1.2px", "none (structural only)", "—", "500 seq × 6f = 3k", "Eliminate aux-line noise"],
    ["train7", "synthetic+LTX", "1.1–1.2px", "none (structural only)", "LTX-2.3", "3k × 3 variants = 9k", "Structural-only + diffusion"],
    ["train8", "synthetic", "1.1–1.2px", "none + non-overlapping", "—", "500 seq × 6f = 3k", "Cleanest signal: no overlap, no shapes"],
    ["train9", "synthetic", "1.1–1.2px", "none + non-overlapping", "—", "600 seq × 6f = 3.6k", "BASELINE: train8 + 50% straight + dynamic calcSteps"],
    ["train10", "synthetic+LTX", "same as train9", "none + non-overlapping", "LTX-2.3", "3.6k × 3 variants = 10.8k", "train9 + diffusion augmentation"],
]


# Column layout for the "Dataset Status" sheet (dynamic, updated each sync)
DATASET_STATUS_HEADERS = [
    "dataset",
    "type",
    "design_goal",
    "status",
    "colorlines",
    "gt_images",
    "midframes",
    "anilines",
    "hints",
    "updated_at",
]


def load_status() -> dict[str, Any]:
    if not STATUS_JSON.exists():
        print(f"ERROR: {STATUS_JSON} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(STATUS_JSON.read_text())


def fmt(val: Any) -> str:
    """Format a value for the spreadsheet."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.8f}"
    return str(val)


def ensure_sheet(
    spreadsheet: gspread.Spreadsheet, title: str, headers: list[str]
) -> gspread.Worksheet:
    """Get or create a worksheet with the given headers."""
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=len(headers))
        ws.append_row(headers, value_input_option="RAW")
        ws.format("1", {"textFormat": {"bold": True}})
    return ws


def sync_status_sheet(
    ws: gspread.Worksheet, runs: list[dict[str, Any]]
) -> None:
    """Update the Status sheet — one row per run_id, updated in-place."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get existing data to find run_id positions
    existing = ws.get_all_values()
    run_id_to_row: dict[str, int] = {}
    for i, row in enumerate(existing):
        if i == 0:
            continue  # skip header
        if row:
            run_id_to_row[row[0]] = i + 1  # 1-indexed

    for run in runs:
        last_ep = run.get("last_epoch", 0)
        total = run.get("total_epochs", 0)
        progress = f"{last_ep}/{total}" if total else str(last_ep)

        row_data = [
            run.get("run_id", ""),
            run.get("version", ""),
            run.get("data", ""),
            run.get("vfi") or "",
            run.get("status", ""),
            fmt(run.get("best_epoch", 0)),
            fmt(run.get("best_loss")),
            fmt(run.get("last_epoch", 0)),
            fmt(run.get("last_loss")),
            run.get("lr", ""),
            fmt(run.get("total_epochs", 0)),
            progress,
            fmt(run.get("num_checkpoints", 0)),
            run.get("start_date", ""),
            run.get("last_modified", ""),
            run.get("note", ""),
            now,
        ]

        run_id = run.get("run_id", "")
        if run_id in run_id_to_row:
            row_num = run_id_to_row[run_id]
            ws.update(
                values=[row_data],
                range_name=f"A{row_num}:Q{row_num}",
                value_input_option="RAW",
            )
        else:
            ws.append_row(row_data, value_input_option="RAW")

    print(f"Status sheet: updated {len(runs)} runs")


def sync_history_sheet(
    ws: gspread.Worksheet, runs: list[dict[str, Any]]
) -> None:
    """Append a snapshot row per active run (training/stopped) to the History sheet."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_to_append: list[list[str]] = []
    for run in runs:
        # Only log runs that are still active or recently stopped
        if run.get("status") not in ("training", "stopped"):
            continue

        rows_to_append.append([
            now,
            run.get("run_id", ""),
            run.get("version", ""),
            run.get("data", ""),
            run.get("status", ""),
            fmt(run.get("best_epoch", 0)),
            fmt(run.get("best_loss")),
            fmt(run.get("last_epoch", 0)),
            fmt(run.get("last_loss")),
            fmt(run.get("total_epochs", 0)),
            fmt(run.get("num_checkpoints", 0)),
        ])

    if rows_to_append:
        ws.append_rows(rows_to_append, value_input_option="RAW")
        print(f"History sheet: appended {len(rows_to_append)} rows")
    else:
        print("History sheet: no active runs to log")


def sync_datasets_sheet(ws: gspread.Worksheet) -> None:
    """Write the static dataset legend (clear and rewrite each time)."""
    # Row 1 is the header (created by ensure_sheet), data starts at row 2
    existing = ws.get_all_values()
    if len(existing) > 1:
        # Already has data rows — check if content matches
        current_data = existing[1:]
        if current_data == DATASET_ROWS:
            print("Datasets sheet: unchanged")
            return
        # Clear old data rows (keep header)
        last_row = len(existing)
        ws.batch_clear([f"A2:G{last_row}"])

    ws.append_rows(DATASET_ROWS, value_input_option="RAW")
    print(f"Datasets sheet: wrote {len(DATASET_ROWS)} dataset entries")


def sync_dataset_status_sheet(
    ws: gspread.Worksheet, datasets: list[dict[str, Any]]
) -> None:
    """Update the Dataset Status sheet — one row per dataset, clear and rewrite."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clear old data rows (keep header)
    existing = ws.get_all_values()
    if len(existing) > 1:
        ws.batch_clear([f"A2:J{len(existing)}"])

    rows: list[list[str]] = []
    for ds in datasets:
        rows.append([
            ds.get("dataset", ""),
            ds.get("type", ""),
            ds.get("design_goal", ""),
            ds.get("status", ""),
            str(ds.get("colorlines", 0)),
            str(ds.get("gt_images", 0)),
            str(ds.get("midframes", 0)),
            str(ds.get("anilines", 0)),
            str(ds.get("hints", 0)),
            now,
        ])

    if rows:
        ws.append_rows(rows, value_input_option="RAW")
    print(f"Dataset Status sheet: wrote {len(rows)} datasets")


def main() -> None:
    if not Path(KEY_PATH).exists():
        print(f"ERROR: Service account key not found at {KEY_PATH}", file=sys.stderr)
        sys.exit(1)

    data = load_status()
    runs = data.get("runs", [])
    datasets = data.get("datasets", [])

    gc = gspread.service_account(filename=KEY_PATH)
    spreadsheet = gc.open_by_key(SHEET_ID)

    status_ws = ensure_sheet(spreadsheet, "Status", STATUS_HEADERS)
    history_ws = ensure_sheet(spreadsheet, "History", HISTORY_HEADERS)
    datasets_ws = ensure_sheet(spreadsheet, "Datasets", DATASET_HEADERS)
    dataset_status_ws = ensure_sheet(spreadsheet, "Dataset Status", DATASET_STATUS_HEADERS)

    sync_status_sheet(status_ws, runs)
    sync_history_sheet(history_ws, runs)
    sync_datasets_sheet(datasets_ws)
    sync_dataset_status_sheet(dataset_status_ws, datasets)

    print(f"Synced to Google Sheet: https://docs.google.com/spreadsheets/d/{SHEET_ID}")


if __name__ == "__main__":
    main()
