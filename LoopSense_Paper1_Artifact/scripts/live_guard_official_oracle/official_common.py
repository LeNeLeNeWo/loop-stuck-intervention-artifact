from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_NAME = "paper1_live_guard_official_oracle"
OUTPUT_SUBDIR = Path("outputs") / EXPERIMENT_NAME
SCRIPT_SUBDIR = Path("scripts") / EXPERIMENT_NAME
PREV_LIVE_SCRIPT_SUBDIR = Path("scripts") / "paper1_live_guard_experiment"
MINI_SWE_PACKAGE = Path(
    "experiments/paper1_loopbench/stage_p1_6_validity_expansion/"
    "closed_loop_mini_validation/backend_acquisition_and_task_reconciliation/python_packages"
)
EXISTING_SWEBENCH_VENV = Path(
    "experiments/paper1_loopbench/stage_p1_6_validity_expansion/"
    "closed_loop_mini_validation/linux_backend_environment_repair/venv"
)

ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
ARMS = ("NO_GUARD", "WARN_REPLAN_GUARD", "HARD_STOP_GUARD")
DATASET_NAME = "princeton-nlp/SWE-Bench_Verified"
DATASET_SPLIT = "test"
SEED = 20270614
WARN_REPLAN_MESSAGE = (
    "You may be repeating. Before taking the next action, summarize the evidence gathered so far, "
    "state the current hypothesis, explain what has changed since the previous attempt, and choose "
    "the next debugging action only if it is justified by new evidence. If no useful new evidence "
    "exists, revise the plan instead of repeating the same action."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / OUTPUT_SUBDIR


def ensure_output(root: Path | None = None) -> Path:
    out = out_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any, n: int = 16) -> str:
    raw = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:n]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def deepseek_env() -> dict[str, str]:
    return {
        ENV_BASE_URL: os.environ.get(ENV_BASE_URL, "").strip(),
        ENV_API_KEY: os.environ.get(ENV_API_KEY, "").strip(),
        ENV_MODEL: os.environ.get(ENV_MODEL, "").strip(),
    }


def apply_deepseek_openai_env() -> None:
    vals = deepseek_env()
    if vals[ENV_API_KEY]:
        os.environ["OPENAI_API_KEY"] = vals[ENV_API_KEY]
    if vals[ENV_BASE_URL]:
        os.environ["OPENAI_BASE_URL"] = vals[ENV_BASE_URL]
        os.environ["OPENAI_API_BASE"] = vals[ENV_BASE_URL]
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


def redactor():
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
        out = re.sub(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;:]*", "<LOCAL_PATH>", out, flags=re.I)
        out = re.sub(r"\\\\" + "wsl" + r"\.localhost[^\s,;:]*", "<LOCAL_PATH>", out, flags=re.I)
        out = re.sub(r"/home/[A-Za-z0-9_.-]+[^\s,;:]*", "<LOCAL_PATH>", out)
        return out

    return _redact


def run_cmd(cmd: list[str], *, timeout: int = 120, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
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


def wsl_path(win_path: Path) -> str:
    resolved = win_path.absolute()
    text = str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    if drive and len(drive) == 1:
        tail = text[len(resolved.drive) :].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{tail}"
    proc = run_cmd(["wsl", "-d", "Ubuntu", "--", "wslpath", "-a", text], timeout=30)
    if proc["returncode"] != 0:
        raise RuntimeError("wslpath failed: " + proc["stdout"][:300])
    return proc["stdout"].strip().splitlines()[-1]


def wsl_bash(command: str, *, timeout: int = 120, cwd: Path | None = None) -> dict[str, Any]:
    if cwd is not None:
        command = "cd " + shlex.quote(wsl_path(cwd)) + " && " + command
    return run_cmd(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], timeout=timeout)


def swebench_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / EXISTING_SWEBENCH_VENV / "bin" / "python"


def mini_env(root: Path | None = None) -> dict[str, str]:
    root = root or repo_root()
    vals = deepseek_env()
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = vals[ENV_API_KEY]
    env["OPENAI_BASE_URL"] = vals[ENV_BASE_URL]
    env["OPENAI_API_BASE"] = vals[ENV_BASE_URL]
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env.setdefault("TQDM_DISABLE", "1")
    pkg = str((root / MINI_SWE_PACKAGE).resolve())
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def docker_image_name(instance_id: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


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


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def zip_single_top_level(zip_path: Path) -> tuple[bool, str]:
    import zipfile

    if not zip_path.exists():
        return False, "zip missing"
    with zipfile.ZipFile(zip_path) as zf:
        tops = sorted({name.split("/")[0] for name in zf.namelist() if name and not name.startswith("/")})
        bad = [name for name in zf.namelist() if "__pycache__" in name or ".git/" in name or ".venv/" in name]
    return tops == ["LoopSense_Paper1_Artifact"] and not bad, f"tops={tops}; bad_count={len(bad)}"


def copy_sanitized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".png", ".pdf"}:
        shutil.copy2(src, dst)
        return
    text = src.read_text(encoding="utf-8", errors="replace")
    dst.write_text(redactor()(text), encoding="utf-8")


def preregistration_text() -> str:
    return f"""# Official-Oracle Live Guard Experiment Preregistration

Experiment: `paper1_live_guard_official_oracle`

Generated: {utc_now()}

This preregistration is written before any full official-oracle live guard run.

## Research Question

Do guard-policy interventions in a real SE-agent loop reduce repetition/cost while preserving task success under an official SWE-bench-style oracle?

## Arms

1. `NO_GUARD`: normal mini-swe-agent runner behavior.
2. `WARN_REPLAN_GUARD`: when the guard fires, inject the frozen warning/replan instruction once.
3. `HARD_STOP_GUARD`: when the guard fires, stop and evaluate the current patch/state.

## Frozen Guard

- Primary guard: `file_revisit_guard window=5,repeats=3`.
- Warning/replan injection cap: at most once per run.
- Secondary safety cap: max steps and token/cost budget are the same for all arms.
- The prompt, guard, sample, metrics, and stopping rules must not be changed after outcomes are inspected.

## Sample

- Primary target: 60 tasks x 3 arms.
- Minimum manuscript-usable threshold: 30 tasks x 3 arms with official oracle results.
- If cost/runtime is high: complete the first 30 frozen tasks, then continue to 60 only if feasible.
- Task source fixed before outcomes: `{DATASET_NAME}:{DATASET_SPLIT}`.
- Fixed seed: `{SEED}`.
- Do not exclude hard tasks after seeing outcomes.
- Infrastructure failures are recorded separately as `infrastructure_failed`.

## Primary Metrics

- Official resolved / tests pass.
- Patch generated.
- Steps/actions.
- Tokens/cost when available.
- Guard triggered and trigger step.
- Post-trigger repetition rate.
- Repeated-file/action/tool/error counts.
- Timeout/failure mode.

## Statistical Plan

- Pair arms by task.
- Use paired bootstrap with 1000 resamples and report 95% confidence intervals.
- For paired binary resolved/tests-pass, report paired differences; McNemar may be added if straightforward.
- Report all tasks and all failed arms.
- No selective exclusion after outcomes.

## Manuscript Integration Rule

Add live task-success claims only if:

- at least 30 tasks have official oracle results for `NO_GUARD` and `WARN_REPLAN_GUARD`;
- `oracle_eval_results.csv` contains resolved/tests-pass fields from a real oracle run;
- no major privacy issue is detected; and
- results are interpretable.

If these conditions are not met, keep the run as artifact-only pilot/feasibility evidence.
"""
