from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_common import (
    DATASET_NAME,
    DATASET_SPLIT,
    ensure_output,
    repo_root,
    run_cmd,
    swebench_python,
    utc_now,
    write_json,
    wsl_bash,
    wsl_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install or validate SWE-bench official oracle harness.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-noop-smoke", action="store_true")
    parser.add_argument("--smoke-instance", default="pydata__xarray-3095")
    parser.add_argument("--smoke-timeout", type=int, default=1800)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    logs = out / "oracle_logs"
    logs.mkdir(parents=True, exist_ok=True)

    py = swebench_python(root)
    docker = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=120, cwd=root)
    help_cmd = shlex_cmd = shlex_cmd = f"{wsl_path(py)} -m swebench.harness.run_evaluation --help | head -80"
    py_available = py.parent.exists()
    help_res = wsl_bash(help_cmd, timeout=120, cwd=root) if py_available else {"returncode": 127, "stdout": "swebench venv python missing", "timeout": False}
    import_cmd = (
        f"{wsl_path(py)} -c "
        + "'import swebench, sys; import swebench.harness.run_evaluation; "
        + 'print(sys.version.split()[0]); print(getattr(swebench, "__version__", "no_version"))' + "'"
    )
    import_res = wsl_bash(import_cmd, timeout=120, cwd=root) if py_available else {"returncode": 127, "stdout": "missing", "timeout": False}

    smoke_status = "not_run"
    smoke_report = ""
    if args.run_noop_smoke and py_available and docker["returncode"] == 0:
        pred_path = out / "oracle_noop_smoke_predictions.jsonl"
        pred_path.write_text(
            json.dumps(
                {
                    "instance_id": args.smoke_instance,
                    "model_name_or_path": "noop_smoke",
                    "model_patch": "",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report_dir = out / "oracle_noop_smoke_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"{wsl_path(py)} -m swebench.harness.run_evaluation "
            f"--dataset_name {DATASET_NAME} --split {DATASET_SPLIT} "
            f"--instance_ids {args.smoke_instance} "
            f"--predictions_path {shlex_quote(wsl_path(pred_path))} "
            f"--max_workers 1 --timeout {args.smoke_timeout} "
            f"--cache_level instance --clean False --run_id official_oracle_noop_smoke "
            f"--report_dir {shlex_quote(wsl_path(report_dir))}"
        )
        smoke = wsl_bash(cmd, timeout=args.smoke_timeout + 900, cwd=root)
        (logs / "noop_smoke_stdout.log").write_text(smoke["stdout"], encoding="utf-8")
        smoke_status = "pass" if smoke["returncode"] == 0 else "fail"
        smoke_report = smoke["stdout"][-4000:]

    if help_res["returncode"] == 0 and docker["returncode"] == 0:
        decision = "OFFICIAL_ORACLE_READY" if (not args.run_noop_smoke or smoke_status == "pass") else "ORACLE_READY_WITH_LIMITATIONS"
    else:
        decision = "ORACLE_NOT_AVAILABLE"

    payload = {
        "generated_at": utc_now(),
        "decision": decision,
        "harness": {
            "source": "existing local swebench package in prior backend-repair venv",
            "python": str(py),
            "help_returncode": help_res["returncode"],
            "import_returncode": import_res["returncode"],
            "import_output": import_res["stdout"][:1000],
        },
        "docker": {"returncode": docker["returncode"], "output": docker["stdout"][:200]},
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "noop_smoke": {"status": smoke_status, "instance": args.smoke_instance},
    }
    write_json(out / "oracle_install_report.json", payload)
    lines = [
        "# SWE-bench Official Oracle Validation",
        "",
        f"Decision: **{decision}**",
        "",
        "## Harness",
        "",
        "- Source: existing local SWE-bench package from prior backend-repair virtual environment.",
        f"- Python: `{py}`",
        f"- Import check: {'pass' if import_res['returncode'] == 0 else 'fail'}",
        f"- CLI help check: {'pass' if help_res['returncode'] == 0 else 'fail'}",
        f"- Docker: {'pass' if docker['returncode'] == 0 else 'fail'}",
        f"- Dataset: `{DATASET_NAME}:{DATASET_SPLIT}`",
        "",
        "## No-op Smoke",
        "",
        f"- Requested: {'yes' if args.run_noop_smoke else 'no'}",
        f"- Status: {smoke_status}",
        f"- Instance: `{args.smoke_instance}`",
    ]
    if smoke_report:
        lines.extend(["", "## Smoke Output Excerpt", "", "```text", smoke_report, "```"])
    if decision == "ORACLE_NOT_AVAILABLE":
        lines.extend(["", "## Blocker", "", "The SWE-bench oracle harness or Docker runtime is not available; no task-success evidence can be claimed."])
    (out / "oracle_install_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Oracle decision: {decision}")
    print(f"Report: {out / 'oracle_install_report.md'}")
    return 0 if decision != "ORACLE_NOT_AVAILABLE" else 2


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
