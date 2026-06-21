from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_NAME = "paper1_live_guard_experiment"
OUTPUT_SUBDIR = Path("outputs") / EXPERIMENT_NAME
SCRIPT_SUBDIR = Path("scripts") / EXPERIMENT_NAME
MINI_SWE_PACKAGE = Path(
    "experiments/paper1_loopbench/stage_p1_6_validity_expansion/"
    "closed_loop_mini_validation/backend_acquisition_and_task_reconciliation/python_packages"
)

ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"

ARMS = ("NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD")
WARN_REPLAN_MESSAGE = (
    "You may be repeating. Before taking the next action, summarize the evidence gathered so far, "
    "state the current hypothesis, explain what has changed since the previous attempt, and choose "
    "the next debugging action only if it is justified by new evidence. If no useful new evidence "
    "exists, revise the plan instead of repeating the same action."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / OUTPUT_SUBDIR


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any, n: int = 16) -> str:
    raw = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_output(root: Path | None = None) -> Path:
    out = out_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def masked_url(url: str) -> str:
    if not url:
        return "missing"
    parsed = urllib.parse.urlparse(url)
    suffix = "/chat/completions" if (parsed.path or "").endswith("/chat/completions") else ""
    return f"{parsed.scheme}://{parsed.netloc}/<masked-path>{suffix}"


def deepseek_env() -> dict[str, str]:
    return {
        ENV_BASE_URL: os.environ.get(ENV_BASE_URL, "").strip(),
        ENV_API_KEY: os.environ.get(ENV_API_KEY, "").strip(),
        ENV_MODEL: os.environ.get(ENV_MODEL, "").strip(),
    }


def redacted_env_for_litellm() -> dict[str, str]:
    env = os.environ.copy()
    vals = deepseek_env()
    env["OPENAI_API_KEY"] = vals[ENV_API_KEY]
    env["OPENAI_BASE_URL"] = vals[ENV_BASE_URL]
    env["OPENAI_API_BASE"] = vals[ENV_BASE_URL]
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PIP_PROGRESS_BAR", "off")
    pkg = str((repo_root() / MINI_SWE_PACKAGE).resolve())
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def silence_mini_swe_startup() -> None:
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


def apply_deepseek_openai_env() -> None:
    vals = deepseek_env()
    if vals[ENV_API_KEY]:
        os.environ["OPENAI_API_KEY"] = vals[ENV_API_KEY]
    if vals[ENV_BASE_URL]:
        os.environ["OPENAI_BASE_URL"] = vals[ENV_BASE_URL]
        os.environ["OPENAI_API_BASE"] = vals[ENV_BASE_URL]


def redactor() -> Any:
    values = [
        os.environ.get(ENV_API_KEY),
        os.environ.get(ENV_BASE_URL),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("OPENAI_BASE_URL"),
        os.environ.get("OPENAI_API_BASE"),
    ]
    values = [v for v in values if v]

    def _redact(text: Any) -> str:
        out = "" if text is None else str(text)
        for value in values:
            out = out.replace(value, "<REDACTED_SECRET_OR_ENDPOINT>")
        out = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<REDACTED_TOKEN>", out)
        out = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <REDACTED_TOKEN>", out, flags=re.I)
        out = re.sub(r"[A-Za-z]:\\[^\s,;:]+", "<LOCAL_PATH>", out)
        out = re.sub(r"\\\\wsl\.localhost\\[^\s,;:]+", "<LOCAL_PATH>", out)
        out = re.sub(r"<LOCAL_PATH>,;:]+", "<LOCAL_PATH>", out)
        return out

    return _redact


def run_cmd(
    cmd: list[str],
    *,
    timeout: int = 120,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or repo_root(),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": redactor()(proc.stdout or ""),
            "timeout": False,
            "seconds": time.perf_counter() - start,
        }
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return {
            "returncode": -1,
            "stdout": redactor()(raw),
            "timeout": True,
            "seconds": time.perf_counter() - start,
        }


def post_deepseek_test(timeout: int = 30) -> dict[str, Any]:
    vals = deepseek_env()
    missing = [name for name, value in vals.items() if not value]
    if missing:
        return {"ok": False, "missing": missing, "error": "missing_environment"}
    payload = {
        "model": vals[ENV_MODEL],
        "messages": [
            {"role": "system", "content": "You are a JSON-only assistant."},
            {"role": "user", "content": "{\"ping\": true}"},
        ],
        "temperature": 0,
        "max_tokens": 80,
    }
    body = json.dumps(payload).encode("utf-8")
    url = completions_url(vals[ENV_BASE_URL])
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + vals[ENV_API_KEY],
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - start
        data = json.loads(raw)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return {
            "ok": resp.status == 200,
            "http_status": resp.status,
            "latency_seconds": elapsed,
            "endpoint": masked_url(url),
            "model": vals[ENV_MODEL],
            "content_nonempty": bool(content.strip()),
            "usage": usage,
        }
    except Exception as exc:
        return {
            "ok": False,
            "endpoint": masked_url(url),
            "model": vals[ENV_MODEL],
            "error": redactor()(f"{type(exc).__name__}: {exc}"),
        }


def docker_image_name(instance_id: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            return next(csv.reader(f), [])
    except Exception:
        return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def preregistration_text() -> str:
    return f"""# Live Guard Policy Experiment Preregistration

Experiment: `{EXPERIMENT_NAME}`

Generated: {utc_now()}

This preregistration is written before any full live guard-policy run. Dry-run and stack-validation commands may be used only to verify feasibility, logging, and safety.

## Research Questions

1. Does a warning/replan guard reduce repeated behavior and execution cost while preserving task success compared with no guard?
2. Does a hard-stop guard save execution cost but risk success loss when the trigger occurs in productive-risk settings?

## Arms

- `NO_GUARD`: run the agent normally under the fixed budget.
- `WARN_REPLAN_GUARD`: when the primary guard fires, inject one generic warning/replan message and continue.
- `HARD_STOP_GUARD`: when the primary guard fires, stop the run and evaluate the current patch/state if an oracle is available.
- `PROGRESS_AWARE_GUARD`: optional only if deterministic progress signals are straightforward; it must remain separate and cannot replace the primary arms.

## Frozen Guard Rules

- Primary guard: `file_revisit_guard window=5,repeats=3`.
- Secondary guard: `max_step_guard threshold=50`, used only as a secondary budget/safety signal if integration is straightforward.
- The guard rule, prompt, sample, stopping rules, and metrics must not be changed after seeing outcomes.

## Sample

- Prefer 60 runnable tasks; target 90 if budget permits; optionally 120 if runtime/cost remains reasonable.
- Minimum acceptable live task experiment: 30 tasks x 3 arms.
- The sample is fixed before outcome inspection.
- Prefer tasks traceable to Enriched500 and StressFresh100 with public benchmark/oracle availability.
- If historical task IDs cannot be mapped to runnable public tasks, use a public SWE-bench-compatible task subset and report the mismatch as `public_benchmark_fallback`.

## Metrics

Primary metrics:

- Resolved / tests pass / task success when an oracle is available.
- Steps/actions.
- Repeated action/file/tool/error rate.
- Guard trigger count.
- Cost/tokens.
- Wall-clock time.

Secondary metrics:

- Patch generated.
- Failure mode.
- No-trigger fraction.
- Intervention accepted.
- Post-trigger repetition rate.
- Post-trigger new-file/new-test/new-error-change count when measurable.

## Statistical Plan

- Compare arms on the same sampled tasks.
- Use paired bootstrap over tasks.
- For binary success, report paired difference and 95% CI; McNemar test if straightforward.
- For steps/cost/repetition, report mean/median difference and paired bootstrap 95% CI.
- Report all tasks, all arms, and all failures.
- No selective exclusion after outcomes.

## Interpretation

- Strong evidence: `WARN_REPLAN_GUARD` reduces repetition/cost with no large success drop.
- Hard-stop risk evidence: `HARD_STOP_GUARD` saves steps/cost but reduces success or stops productive tasks.
- Negative or mixed results must be reported honestly.
- If no full tool-level runner and oracle are available, the experiment must be reported as replay-lite or infeasible, not as live task-success evidence.
"""


def ensure_preregistration(root: Path | None = None) -> Path:
    path = ensure_output(root) / "LIVE_GUARD_PREREGISTRATION.md"
    if not path.exists():
        path.write_text(preregistration_text(), encoding="utf-8")
    return path


def system_snapshot(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    total, used, free = shutil.disk_usage(root)
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(root),
        "disk_total_gb": round(total / (1024**3), 2),
        "disk_free_gb": round(free / (1024**3), 2),
    }
