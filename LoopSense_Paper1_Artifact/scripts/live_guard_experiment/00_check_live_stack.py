from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from pathlib import Path

from live_guard_common import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    MINI_SWE_PACKAGE,
    csv_header,
    deepseek_env,
    ensure_output,
    ensure_preregistration,
    file_size,
    masked_url,
    post_deepseek_test,
    redacted_env_for_litellm,
    repo_root,
    run_cmd,
    system_snapshot,
    write_json,
)


PRIMARY_INPUTS = {
    "normalized_steps": Path("outputs/paper1_audit/normalized_steps.csv"),
    "normalized_spans": Path("outputs/paper1_audit/normalized_spans.csv"),
    "stop_counterfactual_triggers": Path("outputs/paper1_intervention_validation/stop_counterfactual_triggers.csv"),
    "provenance_summary": Path("outputs/paper1_provenance_audit/provenance_summary_by_pool.csv"),
}


def module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def network_check(host: str = "github.com", port: int = 443, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def classify(checks: dict) -> str:
    api = checks["deepseek_api"]["ok"]
    docker = checks["docker"]["ok"]
    mini = checks["runner_candidates"]["mini_swe_agent"]["available"]
    datasets = checks["python_modules"].get("datasets", False)
    public_oracle = checks["python_modules"].get("swebench", False)
    network = checks["network"]["ok"]
    if api and docker and mini and datasets and public_oracle:
        return "FULL_LIVE_TASK_EXPERIMENT_FEASIBLE"
    if api and docker and mini and datasets and network:
        return "LIVE_POLICY_EXPERIMENT_FEASIBLE_WITH_PUBLIC_BENCHMARK"
    if api:
        return "REPLAY_LITE_ONLY"
    return "NOT_FEASIBLE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live guard-policy experiment stack.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    out.mkdir(parents=True, exist_ok=True)
    prereg = ensure_preregistration(root)

    env_values = deepseek_env()
    api = post_deepseek_test()
    docker = run_cmd(["docker", "info", "--format", "{{json .ServerVersion}}"], timeout=120, cwd=root)
    git = run_cmd(["git", "--version"], timeout=30, cwd=root)
    mini_pkg = root / MINI_SWE_PACKAGE
    mini_help = run_cmd(
        ["python", "-m", "minisweagent.run.benchmarks.swebench", "--help"],
        timeout=120,
        cwd=root,
        env=redacted_env_for_litellm(),
    )
    net_ok, net_detail = network_check()

    inputs = {}
    for name, rel in PRIMARY_INPUTS.items():
        p = root / rel
        inputs[name] = {
            "path": str(rel),
            "exists": p.exists(),
            "size_bytes": file_size(p),
            "columns_prefix": csv_header(p)[:24],
        }

    modules = {name: module_present(name) for name in ["datasets", "swebench", "docker", "litellm", "pandas", "matplotlib"]}
    runner_candidates = {
        "mini_swe_agent": {
            "available": mini_pkg.exists() and mini_help["returncode"] == 0,
            "path": str(MINI_SWE_PACKAGE),
            "help_returncode": mini_help["returncode"],
            "help_timeout": mini_help["timeout"],
        },
        "swe_agent": {"available": (root / "apps/swe-agent").exists() or (root / "scripts/swe_agent").exists()},
        "openhands": {"available": (root / "apps/openhands").exists() or (root / "OpenHands").exists()},
    }

    checks = {
        "system": system_snapshot(root),
        "deepseek_env": {
            ENV_BASE_URL: "present" if env_values[ENV_BASE_URL] else "missing",
            ENV_API_KEY: "present; value not printed" if env_values[ENV_API_KEY] else "missing",
            ENV_MODEL: env_values[ENV_MODEL] or "missing",
            "endpoint": masked_url(env_values[ENV_BASE_URL]),
        },
        "deepseek_api": api,
        "docker": {"ok": docker["returncode"] == 0, "returncode": docker["returncode"], "output": docker["stdout"][:500]},
        "git": {"ok": git["returncode"] == 0, "output": git["stdout"][:200]},
        "network": {"ok": net_ok, "detail": net_detail},
        "python_modules": modules,
        "primary_inputs": inputs,
        "runner_candidates": runner_candidates,
        "oracle_candidates": {
            "swebench_python_package": modules.get("swebench", False),
            "docker_swebench_images": docker["returncode"] == 0,
            "prior_official_eval_blocked": (root / "experiments/paper1_loopbench/stage_p1_8_downstream_intervention_live_sweagent/p1_8_4_fresh_high_confidence_early_stop_pilot/official_eval/official_eval_summary.md").exists(),
        },
    }
    decision = classify(checks)
    checks["decision"] = decision
    write_json(out / "live_stack_check.json", checks)

    lines = [
        "# Live Guard Stack Check",
        "",
        f"Decision: **{decision}**",
        "",
        f"Preregistration: `{prereg.relative_to(root)}`",
        "",
        "## DeepSeek API",
        "",
        f"- `{ENV_BASE_URL}`: {checks['deepseek_env'][ENV_BASE_URL]}",
        f"- `{ENV_API_KEY}`: {checks['deepseek_env'][ENV_API_KEY]}",
        f"- `{ENV_MODEL}`: {checks['deepseek_env'][ENV_MODEL]}",
        f"- Endpoint: `{checks['deepseek_env']['endpoint']}`",
        f"- Test call: {'pass' if api.get('ok') else 'fail'}",
        f"- Latency seconds: {api.get('latency_seconds', 'NA')}",
        f"- Token usage keys: {', '.join(sorted((api.get('usage') or {}).keys())) if api.get('usage') else 'not returned'}",
        "",
        "## System",
        "",
        f"- Python: {checks['system']['python']}",
        f"- Platform: {checks['system']['platform']}",
        f"- Disk free GB: {checks['system']['disk_free_gb']}",
        f"- Git: {'pass' if checks['git']['ok'] else 'fail'}",
        f"- Docker daemon: {'pass' if checks['docker']['ok'] else 'fail'}",
        f"- Network: {'pass' if net_ok else 'fail'} ({net_detail})",
        "",
        "## Python Modules",
        "",
    ]
    for name, ok in modules.items():
        lines.append(f"- `{name}`: {'present' if ok else 'missing'}")
    lines.extend(["", "## Existing Data Inputs", "", "| Input | Exists | Size bytes | Columns prefix |", "|---|---:|---:|---|"])
    for name, row in inputs.items():
        lines.append(f"| {name} | {'yes' if row['exists'] else 'no'} | {row['size_bytes'] or 'NA'} | {', '.join(row['columns_prefix'])} |")
    lines.extend(["", "## Runner Candidates", ""])
    for name, row in runner_candidates.items():
        lines.append(f"- `{name}`: {'available' if row.get('available') else 'not available'}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full task-success claims require both a runnable agent/tool stack and an available task oracle.",
            "- If the SWE-bench oracle package remains unavailable, live runs may still support tool-level repetition/cost evidence, but not resolved/tests-pass claims.",
            "- No raw API key, authorization header, or local path is written to this report.",
        ]
    )
    (out / "live_stack_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Live stack decision: {decision}")
    print(f"Report: {out / 'live_stack_check.md'}")
    return 0 if decision != "NOT_FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
