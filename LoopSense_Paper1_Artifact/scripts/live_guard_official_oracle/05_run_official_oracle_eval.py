from __future__ import annotations

import argparse
import json
import shlex
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from official_common import DATASET_NAME, DATASET_SPLIT, ensure_output, read_csv_dicts, repo_root, swebench_python, utc_now, write_csv, write_json, wsl_bash, wsl_path


FIELDS = [
    "released_task_id",
    "benchmark_instance_id",
    "arm",
    "patch_generated",
    "oracle_status",
    "resolved",
    "tests_pass",
    "fail_to_pass_count",
    "pass_to_pass_count",
    "oracle_error_type",
    "oracle_runtime",
    "notes",
]


def parse_task_range(value: str) -> tuple[int, int]:
    if ":" not in value:
        n = int(value)
        return n, n
    start, end = value.split(":", 1)
    return int(start), int(end)


def allowed_task_ids(out: Path, task_range: str | None) -> set[str] | None:
    if not task_range:
        return None
    start, end = parse_task_range(task_range)
    rows = read_csv_dicts(out / "official_oracle_task_sample.csv")
    return {
        row["released_task_id"]
        for row in rows
        if start <= int(row.get("selected_order") or 999999) <= end
    }


def latest_patch_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        latest[(row["released_task_id"], row["arm"])] = row
    return latest


def parse_reports(report_dir: Path, arm: str, run_id: str) -> dict[str, str]:
    resolved: set[str] = set()
    unresolved: set[str] = set()
    errors: set[str] = set()
    for path in list(report_dir.rglob("*.json")) + list(Path.cwd().glob(f"*.{run_id}.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(data, dict) and "resolved_ids" in data:
            resolved.update(map(str, data.get("resolved_ids") or []))
            unresolved.update(map(str, data.get("unresolved_ids") or []))
            errors.update(map(str, data.get("error_ids") or []))
        for iid, payload in data.items() if isinstance(data, dict) else []:
            if isinstance(payload, dict) and "resolved" in payload:
                (resolved if payload.get("resolved") else unresolved).add(str(iid))
    out = {}
    for iid in resolved:
        out[iid] = "resolved"
    for iid in unresolved:
        out.setdefault(iid, "unresolved")
    for iid in errors:
        out.setdefault(iid, "error")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official SWE-bench oracle evaluation for generated patches.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arms", nargs="+", default=["NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--eval-no-patch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-range", default=None, help="Inclusive selected_order range, e.g. 31:60.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    patch_manifest = out / "patch_manifest.csv"
    allowed = allowed_task_ids(out, args.task_range)
    rows = list(latest_patch_rows(read_csv_dicts(patch_manifest)).values())
    if allowed is not None:
        rows = [row for row in rows if row.get("released_task_id") in allowed]
    existing = {(r["released_task_id"], r["arm"]) for r in read_csv_dicts(out / "oracle_eval_results.csv")} if args.resume else set()
    all_results: list[dict[str, Any]] = read_csv_dicts(out / "oracle_eval_results.csv") if args.resume and (out / "oracle_eval_results.csv").exists() else []
    by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("arm") in args.arms:
            by_arm[row["arm"]].append(row)

    py = swebench_python(root)
    for arm, arm_rows in by_arm.items():
        eval_rows = []
        skipped_rows = []
        for row in arm_rows:
            if (row["released_task_id"], row["arm"]) in existing:
                continue
            patch_generated = str(row.get("patch_generated", "")).lower() in {"true", "1", "yes"}
            if patch_generated or args.eval_no_patch:
                eval_rows.append(row)
            else:
                skipped_rows.append(row)
        for row in skipped_rows:
            all_results.append(
                {
                    "released_task_id": row["released_task_id"],
                    "benchmark_instance_id": row["benchmark_instance_id"],
                    "arm": arm,
                    "patch_generated": False,
                    "oracle_status": "no_patch_not_evaluated",
                    "resolved": False,
                    "tests_pass": False,
                    "fail_to_pass_count": "",
                    "pass_to_pass_count": "",
                    "oracle_error_type": "no_patch",
                    "oracle_runtime": 0,
                    "notes": "No patch generated; official no-op evaluation not requested.",
                }
            )
        if not eval_rows:
            continue
        pred_path = out / "oracle_predictions" / f"{arm}_predictions.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in eval_rows:
                patch_path = Path(row.get("patch_path_local", ""))
                patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
                f.write(
                    json.dumps(
                        {
                            "instance_id": row["benchmark_instance_id"],
                            "model_name_or_path": f"{row.get('model_name_or_path') or 'deepseek'}__{arm}",
                            "model_patch": patch,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        report_dir = out / "oracle_eval_reports" / arm
        report_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"official_live_{arm.lower()}_{int(time.time())}"
        ids = " ".join(shlex.quote(row["benchmark_instance_id"]) for row in eval_rows)
        cmd = (
            f"{shlex.quote(wsl_path(py))} -m swebench.harness.run_evaluation "
            f"--dataset_name {shlex.quote(DATASET_NAME)} --split {shlex.quote(DATASET_SPLIT)} "
            f"--instance_ids {ids} "
            f"--predictions_path {shlex.quote(wsl_path(pred_path))} "
            f"--max_workers {args.max_workers} --timeout {args.timeout} "
            f"--cache_level instance --clean False --run_id {shlex.quote(run_id)} "
            f"--report_dir {shlex.quote(wsl_path(report_dir))}"
        )
        start = time.perf_counter()
        res = wsl_bash(cmd, timeout=args.timeout * max(1, len(eval_rows)) + 1800, cwd=root)
        runtime = time.perf_counter() - start
        (report_dir / "oracle_stdout_stderr.log").write_text(res["stdout"], encoding="utf-8")
        parsed = parse_reports(report_dir, arm, run_id)
        for row in eval_rows:
            status = parsed.get(row["benchmark_instance_id"])
            patch_generated = str(row.get("patch_generated", "")).lower() in {"true", "1", "yes"}
            if status == "resolved":
                oracle_status, resolved, err = "evaluated", True, ""
            elif status == "unresolved":
                oracle_status, resolved, err = "evaluated", False, ""
            elif status == "error":
                oracle_status, resolved, err = "oracle_error", False, "harness_error"
            elif not patch_generated and args.eval_no_patch and res["returncode"] == 0:
                oracle_status, resolved, err = "official_empty_patch", False, "empty_patch"
            else:
                oracle_status, resolved, err = "oracle_report_parse_failed", False, "report_parse_failed"
            all_results.append(
                {
                    "released_task_id": row["released_task_id"],
                    "benchmark_instance_id": row["benchmark_instance_id"],
                    "arm": arm,
                    "patch_generated": patch_generated,
                    "oracle_status": oracle_status,
                    "resolved": resolved,
                    "tests_pass": resolved,
                    "fail_to_pass_count": "",
                    "pass_to_pass_count": "",
                    "oracle_error_type": err,
                    "oracle_runtime": f"{runtime:.3f}",
                    "notes": f"run_id={run_id}; returncode={res['returncode']}",
                }
            )
    write_csv(out / "oracle_eval_results.csv", all_results, FIELDS)
    write_json(out / "oracle_eval_manifest.json", {"generated_at": utc_now(), "rows": len(all_results), "arms": args.arms})
    print(f"Oracle result rows: {len(all_results)}")
    print(f"Oracle results: {out / 'oracle_eval_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
