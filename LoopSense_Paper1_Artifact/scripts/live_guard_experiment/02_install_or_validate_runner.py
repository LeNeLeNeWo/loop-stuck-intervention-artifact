from __future__ import annotations

import argparse
import json
from pathlib import Path

from live_guard_common import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    MINI_SWE_PACKAGE,
    deepseek_env,
    ensure_output,
    masked_url,
    redacted_env_for_litellm,
    repo_root,
    run_cmd,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate live guard runner and write DeepSeek mini-swe-agent config.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    runner_config = out / "runner_config"
    runner_config.mkdir(parents=True, exist_ok=True)

    vals = deepseek_env()
    model = vals[ENV_MODEL]
    config_path = runner_config / "deepseek_litellm_swebench.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                f'  model_name: "openai/{model}"',
                '  model_class: "litellm"',
                "  model_kwargs:",
                "    drop_params: true",
                "    temperature: 0.0",
                "    timeout: 60",
                '  cost_tracking: "ignore_errors"',
                "agent:",
                "  step_limit: 100",
                "  cost_limit: 0.0",
                "  wall_time_limit_seconds: 1800",
                "environment:",
                "  timeout: 60",
                "  container_timeout: 2h",
                "  pull_timeout: 600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mini_pkg = root / MINI_SWE_PACKAGE
    help_check = run_cmd(
        ["python", "-m", "minisweagent.run.benchmarks.swebench", "--help"],
        timeout=120,
        cwd=root,
        env=redacted_env_for_litellm(),
    )
    litellm_check = run_cmd(
        [
            "python",
            "-c",
            (
                "import os; from litellm import completion; "
                "os.environ['OPENAI_API_KEY']=os.environ.get('DEEPSEEK_API_KEY',''); "
                "os.environ['OPENAI_BASE_URL']=os.environ.get('DEEPSEEK_BASE_URL',''); "
                "os.environ['OPENAI_API_BASE']=os.environ.get('DEEPSEEK_BASE_URL',''); "
                "r=completion(model='openai/'+os.environ['DEEPSEEK_MODEL'],"
                "messages=[{'role':'user','content':'Reply with OK.'}],temperature=0,max_tokens=50,drop_params=True,timeout=30); "
                "print((r.choices[0].message.content or '').strip()[:40]); "
                "print(bool(r.choices[0].message.content or r.choices[0].message.tool_calls))"
            ),
        ],
        timeout=60,
        cwd=root,
        env=redacted_env_for_litellm(),
    )
    docker = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=120, cwd=root)
    dataset_check = run_cmd(
        ["python", "-c", "from datasets import load_dataset; print('datasets-ok')"],
        timeout=60,
        cwd=root,
        env=redacted_env_for_litellm(),
    )

    status = {
        "runner": "mini-swe-agent",
        "mini_swe_agent_package_path": str(MINI_SWE_PACKAGE),
        "mini_swe_agent_available": mini_pkg.exists() and help_check["returncode"] == 0,
        "deepseek_model": model,
        "deepseek_base_url": masked_url(vals[ENV_BASE_URL]),
        ENV_API_KEY: "present; value not printed" if vals[ENV_API_KEY] else "missing",
        "config_path": str(config_path.relative_to(root)),
        "help_check": {"returncode": help_check["returncode"], "timeout": help_check["timeout"]},
        "litellm_check": {
            "returncode": litellm_check["returncode"],
            "timeout": litellm_check["timeout"],
            "output": litellm_check["stdout"][:500],
        },
        "docker_check": {"returncode": docker["returncode"], "output": docker["stdout"][:200]},
        "datasets_check": {"returncode": dataset_check["returncode"], "output": dataset_check["stdout"][:200]},
    }
    ready = (
        status["mini_swe_agent_available"]
        and litellm_check["returncode"] == 0
        and docker["returncode"] == 0
        and dataset_check["returncode"] == 0
        and bool(vals[ENV_API_KEY])
        and bool(vals[ENV_BASE_URL])
        and bool(vals[ENV_MODEL])
    )
    status["runner_ready"] = ready
    write_json(out / "runner_install_report.json", status)

    lines = [
        "# Live Guard Runner Validation",
        "",
        f"Status: **{'READY' if ready else 'NOT_READY'}**",
        "",
        "## Runner",
        "",
        "- Selected runner: mini-swe-agent existing local package.",
        f"- Package path: `{MINI_SWE_PACKAGE}`",
        f"- CLI help check: {'pass' if help_check['returncode'] == 0 else 'fail'}",
        "",
        "## Model Configuration",
        "",
        f"- Model: `{model}`",
        f"- LiteLLM model string: `openai/{model}`",
        f"- Endpoint: `{masked_url(vals[ENV_BASE_URL])}`",
        f"- API key: {'present; value not printed' if vals[ENV_API_KEY] else 'missing'}",
        f"- Config: `{config_path.relative_to(root)}`",
        f"- LiteLLM smoke call: {'pass' if litellm_check['returncode'] == 0 else 'fail'}",
        "",
        "## Infrastructure",
        "",
        f"- Docker daemon: {'pass' if docker['returncode'] == 0 else 'fail'}",
        f"- datasets package: {'pass' if dataset_check['returncode'] == 0 else 'fail'}",
        "",
        "## Notes",
        "",
        "- No external runner repository was cloned because an existing mini-swe-agent package is available.",
        "- This validation does not run SWE-bench tasks.",
        "- Official task-success oracle availability is checked separately by the analysis/stack scripts.",
    ]
    (out / "runner_install_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Runner validation: {'READY' if ready else 'NOT_READY'}")
    print(f"Report: {out / 'runner_install_report.md'}")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
