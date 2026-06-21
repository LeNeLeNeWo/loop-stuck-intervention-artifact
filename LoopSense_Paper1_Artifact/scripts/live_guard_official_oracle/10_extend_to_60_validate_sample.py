from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from official_common import ARMS, DATASET_NAME, DATASET_SPLIT, ensure_output, read_csv_dicts, repo_root, utc_now, write_csv, write_json


REQUIRED_FILES = [
    "OFFICIAL_ORACLE_PREREGISTRATION.md",
    "official_oracle_task_sample.csv",
    "official_oracle_run_plan.json",
    "agent_run_log.csv",
    "patch_manifest.csv",
    "oracle_eval_results.csv",
    "live_guard_official_results.csv",
    "live_guard_official_summary_by_arm.csv",
    "live_guard_official_pairwise_effects.csv",
    "live_guard_official_report.md",
]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def order(row: dict[str, str]) -> int:
    return int(row.get("selected_order") or 999999)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen 60-task extension for official-oracle live guard experiment.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)

    sample_path = out / "official_oracle_task_sample.csv"
    sample = sorted(read_csv_dicts(sample_path), key=order)
    runs = read_csv_dicts(out / "agent_run_log.csv")
    arms = set(ARMS)
    run_pairs = {(r.get("released_task_id", ""), r.get("arm", "")): r for r in runs}
    first30 = [r for r in sample if order(r) <= 30]
    next30 = [r for r in sample if 31 <= order(r) <= 60]
    completed_first30_ids = {r.get("released_task_id", "") for r in runs}
    sample_first30_ids = {r.get("released_task_id", "") for r in first30}
    first30_rows_ok = all((r["released_task_id"], arm) in run_pairs for r in first30 for arm in arms)
    first30_no_extra = all(
        (r.get("released_task_id", "") in sample_first30_ids and r.get("arm", "") in arms)
        for r in runs
        if r.get("status")
    )
    source_ok = all(r.get("task_source") == f"{DATASET_NAME}:{DATASET_SPLIT}" for r in sample)
    trace_ok = all(r.get("traceability_level") == "public_benchmark_fallback" for r in sample)
    selected60_ok = len(sample) >= 60 and all(truthy(r.get("selected_for_60")) for r in sample[:60])
    selected30_ok = all(truthy(r.get("selected_for_30")) for r in first30)
    next30_existing = [(r, arm) for r in next30 for arm in arms if (r["released_task_id"], arm) in run_pairs]
    required_missing = [name for name in REQUIRED_FILES if not (out / name).exists()]

    extension_rows = []
    for row in sample[:60]:
        selected_order = order(row)
        for arm in ARMS:
            existing = run_pairs.get((row["released_task_id"], arm))
            run_needed = selected_order > 30 and existing is None
            reason = "completed_prefix" if selected_order <= 30 else ("missing_extension_row" if run_needed else "already_present_extension_row")
            extension_rows.append(
                {
                    "released_task_id": row["released_task_id"],
                    "benchmark_instance_id": row["benchmark_instance_id"],
                    "selected_order": selected_order,
                    "arm": arm,
                    "run_needed": run_needed,
                    "existing_status": existing.get("status", "") if existing else "",
                    "reason": reason,
                }
            )
    write_csv(
        out / "official_oracle_extension_tasks_31_60.csv",
        [r for r in extension_rows if 31 <= int(r["selected_order"]) <= 60],
        ["released_task_id", "benchmark_instance_id", "selected_order", "arm", "run_needed", "existing_status", "reason"],
    )
    write_json(
        out / "official_oracle_run_plan_60.json",
        {
            "generated_at": utc_now(),
            "mode": "extend_to_60",
            "tasks": 60,
            "completed_prefix_tasks": 30,
            "scheduled_selected_order": "31:60",
            "scheduled_task_arm_rows": sum(1 for r in extension_rows if r["run_needed"]),
            "arms": list(ARMS),
            "model": "deepseek-v4-flash",
            "runner": "local mini-SWE-agent mini_version=2.3.0",
            "guard": "file_revisit_guard window=5,repeats=3",
            "max_steps": 50,
            "timeout_minutes": 45,
            "oracle": f"{DATASET_NAME}:{DATASET_SPLIT} official SWE-bench harness",
            "resume": True,
        },
    )

    checks = {
        "required_files_present": not required_missing,
        "sample_has_at_least_60_tasks": len(sample) >= 60,
        "selected_order_unique_1_60": [order(r) for r in sample[:60]] == list(range(1, 61)),
        "selected_for_30_prefix_true": selected30_ok,
        "selected_for_60_prefix_true": selected60_ok,
        "first30_completed_rows_match_prefix": first30_rows_ok and first30_no_extra and completed_first30_ids == sample_first30_ids,
        "completed_90_rows_for_first30_x_3_arms": len(runs) == 90 and first30_rows_ok,
        "tasks31_60_not_completed": len(next30_existing) == 0,
        "task_source_ok": source_ok,
        "traceability_level_ok": trace_ok,
        "arms_ok": all(r.get("arm") in arms for r in runs),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    lines = [
        "# Extend To 60 Sample Validation",
        "",
        f"Generated: {utc_now()}",
        f"Status: **{status}**",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- sample_rows: {len(sample)}",
            f"- first30_tasks: {len(first30)}",
            f"- next30_tasks: {len(next30)}",
            f"- existing_run_rows: {len(runs)}",
            f"- next30_existing_task_arm_rows: {len(next30_existing)}",
            f"- required_missing: {', '.join(required_missing) if required_missing else 'none'}",
            "",
            "## Decision",
            "",
            "Proceed with tasks 31-60 only." if status == "PASS" else "STOP: frozen-prefix validation failed; do not run extension.",
        ]
    )
    (out / "extend_to_60_sample_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validation status: {status}")
    print(out / "extend_to_60_sample_validation.md")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
