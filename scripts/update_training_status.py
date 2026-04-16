#!/usr/bin/env python3
"""
Scan training checkpoints and generate training_status.md + training_status.json.

Runs via GitHub Actions (self-hosted runner) every 3 hours, or manually:
  python scripts/update_training_status.py

Scans all v*/checkpoints/ directories under LOG_ROOT.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = Path("/storage/HDD20T/ryan/ColoredLineArt")
DATA_ROOT = Path("/storage/SSD3/ryan/programs/ColoredLineArt/data")
OUTPUT_MD = REPO_ROOT / "training_status.md"
OUTPUT_JSON = REPO_ROOT / "training_status.json"

# ---------------------------------------------------------------------------
# Known run metadata (fields not derivable from checkpoints alone).
# Add new runs here when starting training.
# ---------------------------------------------------------------------------
RUN_META: dict[str, dict[str, Any]] = {
    "2026-04-03_14-05-10_lora_r8_a16_decoder_balanced": {
        "version": "v9-lora",
        "data": "train2 (synthetic)",
        "vfi": None,
        "total_epochs": 2000,
        "lr": "5e-05",
        "note": "resumed from ep196 on 0405",
    },
    "2026-04-03_14-03-40_balanced_sched_cosine_loss_focal_lovasz_cldice": {
        "version": "v9-finetune",
        "data": "train2 (synthetic)",
        "vfi": None,
        "total_epochs": 1000,
        "lr": "1e-04",
        "note": "full finetune",
    },
    "2026-04-13_00-32-23_lora_r8_a16_decoder_balanced": {
        "version": "v9-lora",
        "data": "train3 (synthetic+LTX)",
        "vfi": "LTX-2.3",
        "total_epochs": 10000,
        "lr": "5e-05",
        "note": "stopped ep99 for GPU realloc",
    },
    "2026-04-14_09-24-43_lora_r8_a16_decoder_balanced": {
        "version": "v9-lora",
        "data": "train4 (1.1-1.2px SS=4)",
        "vfi": None,
        "total_epochs": 10000,
        "lr": "5e-05",
        "note": "structural 1.1-1.2px, aux 1px, defer_fixed_width",
    },
}

# ---------------------------------------------------------------------------
# Pending / planned runs (not yet started, no checkpoints).
# ---------------------------------------------------------------------------
PENDING_RUNS: list[dict[str, str]] = [
    {
        "version": "v9-lora-ltx",
        "data": "train5 (train4+LTX)",
        "vfi": "LTX-2.3",
        "status": "data generating",
        "note": "LTX interpolation on train4 GT; run_train5_pipeline.sh",
    },
    {
        "version": "v9-finetune-ltx",
        "data": "synthetic+LTX",
        "vfi": "LTX-2.3",
        "status": "pending",
        "note": "needs scheduling",
    },
]


def load_checkpoint_meta(path: str) -> dict[str, Any]:
    """Load non-tensor metadata from a .pth checkpoint."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}

    skip = {
        "model_state_dict",
        "lora_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    return {k: v for k, v in ckpt.items() if k not in skip}


def scan_run(run_dir: Path, variant: str) -> dict[str, Any] | None:
    """Scan a single run directory for checkpoint info."""
    run_id = run_dir.name
    pth_files = sorted(run_dir.glob("*.pth"))
    if not pth_files:
        return None

    # Best checkpoint
    best_lora = run_dir / "best_lora.pth"
    best_model = run_dir / "best_model.pth"
    best_path = (
        best_lora
        if best_lora.exists()
        else (best_model if best_model.exists() else None)
    )
    best_meta = load_checkpoint_meta(str(best_path)) if best_path else {}

    # Latest periodic checkpoint (exclude best_*)
    periodic = [f for f in pth_files if "best" not in f.name]
    last_meta = load_checkpoint_meta(str(periodic[-1])) if periodic else {}

    # Current epoch: max of best_meta.epoch, last_meta.epoch, filename parse
    best_ep = best_meta.get("epoch", 0)
    max_file_ep = 0
    for f in periodic:
        parts = f.stem.split("ep")
        if len(parts) == 2 and parts[1].isdigit():
            max_file_ep = max(max_file_ep, int(parts[1]))
    current_ep = max(best_ep, last_meta.get("epoch", 0), max_file_ep)

    last_modified_ts = max(f.stat().st_mtime for f in pth_files)

    info: dict[str, Any] = {
        "run_id": run_id,
        "variant": variant,
        "best_epoch": best_ep,
        "best_loss": best_meta.get("loss"),
        "last_epoch": current_ep,
        "last_loss": last_meta.get("loss") or best_meta.get("loss"),
        "lora_config": best_meta.get("lora_config") or last_meta.get("lora_config"),
        "base_checkpoint": best_meta.get("base_checkpoint"),
        "num_checkpoints": len(pth_files),
        "start_date": run_id[:10],
        "last_modified": datetime.fromtimestamp(last_modified_ts).strftime(
            "%Y-%m-%d %H:%M"
        ),
    }

    # Merge metadata: run_meta.json (auto-generated by training scripts) > RUN_META (legacy)
    meta_json = run_dir / "run_meta.json"
    if meta_json.exists():
        try:
            file_meta = json.loads(meta_json.read_text())
            # Map run_meta.json fields to info fields
            for key in ("version", "data", "vfi", "total_epochs", "lr", "note",
                        "base_checkpoint", "lora_config"):
                if key in file_meta and file_meta[key] is not None:
                    info[key] = file_meta[key]
        except Exception:
            pass

    # Legacy fallback: hardcoded RUN_META (overrides run_meta.json if present)
    if run_id in RUN_META:
        info.update(RUN_META[run_id])

    return info


def detect_status(info: dict[str, Any]) -> str:
    """Determine run status based on checkpoint state."""
    total = info.get("total_epochs", 0)
    current = info.get("last_epoch", 0)

    if total and current >= total:
        return "completed"

    # Checkpoint modified recently (within 3.5 hours — matches update cadence)
    try:
        last_mod = datetime.strptime(info["last_modified"], "%Y-%m-%d %H:%M")
        if (datetime.now() - last_mod).total_seconds() < 12600:
            return "training"
    except Exception:
        pass

    if current > 0:
        return "stopped"

    return "unknown"


def format_loss(val: float | None) -> str:
    if val is None:
        return "\u2014"
    return f"{val:.6f}"


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

# Expected subdirectories for a complete dataset
_DATASET_SUBDIRS = [
    "colorlines",
    "gt_images",
    "LTX-2",
    "LTX-2_AniLines_gray",
    "LTX-2_hints_cotracker",
]

# Dataset design metadata (static, keyed by directory name)
DATASET_DESIGN: dict[str, dict[str, str]] = {
    "train": {
        "type": "real",
        "structural_width": "original",
        "aux_lines": "all",
        "vfi": "\u2014",
        "expected_samples": "BONES + StudioSeven",
        "design_goal": "Baseline real-motion data",
    },
    "train2": {
        "type": "synthetic",
        "structural_width": "1.0\u20134.0px (80% in 1\u20132px)",
        "aux_lines": "all @ 1px",
        "vfi": "\u2014",
        "expected_samples": "2000 seq \u00d7 6f = 12k",
        "design_goal": "Foundational synthetic for V9 LoRA",
    },
    "train3": {
        "type": "synthetic+LTX",
        "structural_width": "same as train2",
        "aux_lines": "all @ 1px",
        "vfi": "LTX-2.3",
        "expected_samples": "12k \u00d7 3 variants = 36k",
        "design_goal": "Diffusion-noise robustness",
    },
    "train4": {
        "type": "synthetic",
        "structural_width": "1.1\u20131.2px only",
        "aux_lines": "all @ 1px",
        "vfi": "\u2014",
        "expected_samples": "500 seq \u00d7 6f = 3k",
        "design_goal": "Tight structural line control",
    },
    "train5": {
        "type": "synthetic+LTX",
        "structural_width": "1.1\u20131.2px only",
        "aux_lines": "all @ 1px",
        "vfi": "LTX-2.3",
        "expected_samples": "3k \u00d7 3 variants = 9k",
        "design_goal": "Tight lines + diffusion",
    },
    "train6": {
        "type": "synthetic",
        "structural_width": "1.1\u20131.2px",
        "aux_lines": "none (structural only)",
        "vfi": "\u2014",
        "expected_samples": "500 seq \u00d7 6f = 3k",
        "design_goal": "Eliminate aux-line noise",
    },
    "train7": {
        "type": "synthetic+LTX",
        "structural_width": "1.1\u20131.2px",
        "aux_lines": "none (structural only)",
        "vfi": "LTX-2.3",
        "expected_samples": "3k \u00d7 3 variants = 9k",
        "design_goal": "Structural-only + diffusion",
    },
    "train8": {
        "type": "synthetic",
        "structural_width": "1.1\u20131.2px",
        "aux_lines": "none + non-overlapping",
        "vfi": "\u2014",
        "expected_samples": "500 seq \u00d7 6f = 3k",
        "design_goal": "Cleanest signal: no overlap, no shapes",
    },
}


def _count_pngs(directory: Path) -> int:
    """Count PNG files recursively (fast glob)."""
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob("*.png"))


def scan_datasets() -> list[dict[str, Any]]:
    """Scan data/train* directories and report file counts + status."""
    results: list[dict[str, Any]] = []
    if not DATA_ROOT.exists():
        return results

    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("train"):
            continue

        name = entry.name
        counts: dict[str, int] = {}
        for sub in _DATASET_SUBDIRS:
            counts[sub] = _count_pngs(entry / sub)

        # Determine pipeline status
        has_colorlines = counts["colorlines"] > 0
        has_gt = counts["gt_images"] > 0
        has_midframes = counts["LTX-2"] > 0
        has_anilines = counts["LTX-2_AniLines_gray"] > 0
        has_hints = counts["LTX-2_hints_cotracker"] > 0

        if has_colorlines and has_gt and has_midframes and has_anilines and has_hints:
            status = "ready"
        elif has_midframes and not has_anilines:
            status = "need AniLines"
        elif has_anilines and not has_hints:
            status = "need hints"
        elif has_midframes and counts["LTX-2"] < _expected_ltx_count(name, counts):
            status = "LTX generating"
        elif not has_colorlines and not has_gt and not has_midframes:
            status = "empty"
        else:
            status = "incomplete"

        design = DATASET_DESIGN.get(name, {})
        results.append({
            "dataset": name,
            "type": design.get("type", "?"),
            "structural_width": design.get("structural_width", "?"),
            "aux_lines": design.get("aux_lines", "?"),
            "vfi": design.get("vfi", "\u2014"),
            "design_goal": design.get("design_goal", ""),
            "expected_samples": design.get("expected_samples", ""),
            "status": status,
            "colorlines": counts["colorlines"],
            "gt_images": counts["gt_images"],
            "midframes": counts["LTX-2"],
            "anilines": counts["LTX-2_AniLines_gray"],
            "hints": counts["LTX-2_hints_cotracker"],
        })

    return results


def _expected_ltx_count(name: str, counts: dict[str, int]) -> int:
    """Rough expected LTX-2 count for +LTX datasets (3x base gt_images)."""
    base_map = {"train3": "train2", "train5": "train4", "train7": "train6"}
    if name not in base_map:
        return counts.get("LTX-2", 0)
    # For +LTX datasets, expect 3× the base gt count
    base_gt = counts.get("gt_images", 0)
    if base_gt > 0:
        return base_gt * 3
    # Hardcoded fallback
    expected = {"train3": 36000, "train5": 9000, "train7": 9000}
    return expected.get(name, 0)


def discover_variants() -> list[tuple[str, Path]]:
    """Find all v*/checkpoints/ directories under LOG_ROOT."""
    variants: list[tuple[str, Path]] = []
    if not LOG_ROOT.exists():
        return variants
    for entry in sorted(LOG_ROOT.iterdir()):
        ckpt_root = entry / "checkpoints"
        if entry.is_dir() and ckpt_root.is_dir():
            variants.append((entry.name, ckpt_root))
    return variants


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------


def generate_markdown(runs: list[dict[str, Any]], datasets: list[dict[str, Any]] | None = None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Training Status",
        "",
        f"> Auto-generated by `scripts/update_training_status.py` at {now}",
        "",
        "## Active & Completed Runs",
        "",
        "| Date | Version | Data | VFI | Status | Best Loss (ep) | Last Loss (ep) | LR | Total Epochs | Progress | Checkpoints | Note |",
        "|------|---------|------|-----|--------|----------------|----------------|----|-------------|----------|-------------|------|",
    ]

    for r in runs:
        status = detect_status(r)
        status_label = {
            "completed": "done",
            "training": "TRAINING",
            "stopped": "stopped",
            "unknown": "?",
        }.get(status, status)

        best_ep = r.get("best_epoch", "?")
        best_loss = format_loss(r.get("best_loss"))
        last_ep = r.get("last_epoch", "?")
        last_loss = format_loss(r.get("last_loss"))
        total = r.get("total_epochs", "?")
        progress = f"{last_ep}/{total}" if total else str(last_ep)
        version = r.get("version", r["run_id"].split("_", 3)[-1][:20])
        data = r.get("data", "\u2014")
        vfi = r.get("vfi") or "\u2014"
        lr = r.get("lr", "\u2014")
        note = r.get("note", "")
        num_ckpt = r.get("num_checkpoints", 0)
        date = r.get("start_date", "")

        lines.append(
            f"| {date} | {version} | {data} | {vfi} | {status_label} "
            f"| {best_loss} (ep{best_ep}) | {last_loss} (ep{last_ep}) "
            f"| {lr} | {total} | {progress} | {num_ckpt} | {note} |"
        )

    # TensorBoard paths
    lines.extend(["", "## TensorBoard Logs", ""])
    for r in runs:
        run_id = r["run_id"]
        variant = r.get("variant", "?")
        version = r.get("version", variant)
        tb = f"{LOG_ROOT}/{variant}/logs/{run_id}"
        lines.append(f"- **{version}** `{r.get('start_date', '')}`: `{tb}`")

    # Dataset legend
    lines.extend([
        "",
        "## Dataset Legend",
        "",
        "| Dataset | Type | Structural Width | Aux Lines | VFI | Samples | Design Goal |",
        "|---------|------|-----------------|-----------|-----|---------|-------------|",
        "| train | real | original | all | \u2014 | BONES + StudioSeven | Baseline real-motion data |",
        "| train2 | synthetic | 1.0\u20134.0px (80% in 1\u20132px) | all @ 1px | \u2014 | 2000 seq \u00d7 6f = 12k | Foundational synthetic for V9 LoRA |",
        "| train3 | synthetic+LTX | same as train2 | all @ 1px | LTX-2.3 | 12k \u00d7 3 variants = 36k | Diffusion-noise robustness |",
        "| train4 | synthetic | **1.1\u20131.2px** only | all @ 1px | \u2014 | 500 seq \u00d7 6f = 3k | Tight structural line control |",
        "| train5 | synthetic+LTX | **1.1\u20131.2px** only | all @ 1px | LTX-2.3 | 3k \u00d7 3 variants = 9k | Tight lines + diffusion |",
        "| train6 | synthetic | 1.1\u20131.2px | **none** (structural only) | \u2014 | 500 seq \u00d7 6f = 3k | Eliminate aux-line noise |",
        "| train7 | synthetic+LTX | 1.1\u20131.2px | **none** (structural only) | LTX-2.3 | 3k \u00d7 3 variants = 9k | Structural-only + diffusion |",
        "| train8 | synthetic | 1.1\u20131.2px | **none** + non-overlapping | \u2014 | 500 seq \u00d7 6f = 3k | Cleanest signal: no overlap, no shapes |",
        "",
        "**Structural lines**: MainLine (black), ContourLine_A (orange), ContourLine_B (purple)",
        "**Auxiliary lines**: Highlight_I/II, Shadow_I/II, ColorBoundary_A/B",
        "",
        "Progression: train2 \u2192 train4 (tighter widths) \u2192 train6 (structural only) \u2192 train8 (non-overlapping). "
        "Odd-numbered variants (train3/5/7) add LTX-2.3 diffusion augmentation to their predecessor.",
    ])

    # Dataset status (live file counts)
    if datasets:
        lines.extend([
            "",
            "## Dataset Status",
            "",
            "| Dataset | Status | Colorlines | GT Images | Mid-frames | AniLines | Hints |",
            "|---------|--------|-----------|-----------|------------|----------|-------|",
        ])
        for ds in datasets:
            status_label = {
                "ready": "\u2705 ready",
                "need AniLines": "\u23f3 need AniLines",
                "need hints": "\u23f3 need hints",
                "LTX generating": "\u23f3 LTX generating",
                "incomplete": "\u26a0\ufe0f incomplete",
                "empty": "\u274c empty",
            }.get(ds["status"], ds["status"])
            lines.append(
                f"| {ds['dataset']} | {status_label} "
                f"| {ds['colorlines']:,} | {ds['gt_images']:,} "
                f"| {ds['midframes']:,} | {ds['anilines']:,} "
                f"| {ds['hints']:,} |"
            )

    # Pending runs
    lines.extend(
        [
            "",
            "## Pending",
            "",
            "| Version | Data | VFI | Status | Note |",
            "|---------|------|-----|--------|------|",
        ]
    )
    for p in PENDING_RUNS:
        lines.append(
            f"| {p['version']} | {p['data']} | {p['vfi']} | {p['status']} | {p['note']} |"
        )

    lines.extend(["", "---", f"*Last updated: {now}*", ""])
    return "\n".join(lines)


def generate_json(runs: list[dict[str, Any]], datasets: list[dict[str, Any]] | None = None) -> str:
    """Machine-readable status for monitor/digest scripts."""
    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "runs": [],
        "pending": PENDING_RUNS,
        "datasets": datasets or [],
    }
    for r in runs:
        run_data = {
            "run_id": r["run_id"],
            "variant": r.get("variant", ""),
            "version": r.get("version", ""),
            "data": r.get("data", ""),
            "vfi": r.get("vfi"),
            "status": detect_status(r),
            "best_epoch": r.get("best_epoch", 0),
            "best_loss": r.get("best_loss"),
            "last_epoch": r.get("last_epoch", 0),
            "last_loss": r.get("last_loss"),
            "lr": r.get("lr", ""),
            "total_epochs": r.get("total_epochs", 0),
            "num_checkpoints": r.get("num_checkpoints", 0),
            "start_date": r.get("start_date", ""),
            "last_modified": r.get("last_modified", ""),
            "note": r.get("note", ""),
            "lora_config": _serialize(r.get("lora_config")),
            "base_checkpoint": r.get("base_checkpoint"),
        }
        data["runs"].append(run_data)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _serialize(obj: Any) -> Any:
    """Make torch-loaded objects JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


def strip_volatile(s: str) -> str:
    """Strip timestamp lines for content comparison."""
    return "\n".join(
        line
        for line in s.splitlines()
        if not line.startswith("*Last updated:")
        and not line.startswith("> Auto-generated")
        and not line.strip().startswith('"generated_at"')
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Git add, commit, push if changed (for local cron fallback)",
    )
    args = parser.parse_args()

    # Scan all variants dynamically
    all_runs: list[dict[str, Any]] = []
    for variant_name, ckpt_root in discover_variants():
        for run_dir in sorted(ckpt_root.iterdir()):
            if not run_dir.is_dir():
                continue
            info = scan_run(run_dir, variant=variant_name)
            if info:
                all_runs.append(info)

    # Sort: training first, then by date desc
    status_order = {"training": 0, "stopped": 1, "completed": 2, "unknown": 3}
    all_runs.sort(
        key=lambda r: (
            status_order.get(detect_status(r), 9),
            r.get("start_date", ""),
        )
    )

    # Scan dataset directories
    all_datasets = scan_datasets()

    md = generate_markdown(all_runs, datasets=all_datasets)
    js = generate_json(all_runs, datasets=all_datasets)

    # Check if content changed (ignore timestamps)
    old_md = OUTPUT_MD.read_text() if OUTPUT_MD.exists() else ""
    old_json = OUTPUT_JSON.read_text() if OUTPUT_JSON.exists() else ""

    changed = strip_volatile(md) != strip_volatile(old_md) or strip_volatile(
        js
    ) != strip_volatile(old_json)

    if not changed:
        print("No changes detected, skipping.")
        return

    OUTPUT_MD.write_text(md)
    OUTPUT_JSON.write_text(js)
    print(f"Updated {OUTPUT_MD}")
    print(f"Updated {OUTPUT_JSON}")

    if args.auto_commit:
        os.chdir(REPO_ROOT)
        subprocess.run(
            ["git", "add", "training_status.md", "training_status.json"], check=True
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], capture_output=True
        )
        if result.returncode == 0:
            print("No git diff, skipping commit.")
            return
        subprocess.run(
            ["git", "commit", "-m", "chore: update training status"], check=True
        )
        subprocess.run(["git", "push"], check=True)
        print("Committed and pushed.")


if __name__ == "__main__":
    main()
