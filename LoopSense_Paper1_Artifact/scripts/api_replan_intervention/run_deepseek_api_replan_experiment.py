from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
OUTPUT_SUBDIR = Path("outputs/paper1_api_replan_intervention")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def masked_endpoint(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_prompts(prompts: list[dict[str, Any]], limit_prefixes: int | None) -> list[dict[str, Any]]:
    if limit_prefixes is None:
        return prompts
    prefix_order: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        pid = str(prompt["released_prefix_id"])
        if pid not in seen:
            seen.add(pid)
            prefix_order.append(pid)
    allowed = set(prefix_order[:limit_prefixes])
    return [p for p in prompts if p["released_prefix_id"] in allowed]


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip().lower()
    if s in {"true", "yes", "1", "y"}:
        return "true"
    if s in {"false", "no", "0", "n"}:
        return "false"
    return ""


def scrub_response_text(text: Any, limit: int = 2500) -> str:
    if text is None:
        return ""
    out = str(text)
    out = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<TOKEN>", out)
    out = re.sub(r"Bearer\s+[A-Za-z0-9._-]{8,}", "Bearer <TOKEN>", out, flags=re.I)
    out = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\s*=\s*(['\"])[^'\"]{4,}\2",
        lambda m: f"{m.group(1)} = {m.group(2)}<REDACTED_SECRET>{m.group(2)}",
        out,
    )
    out = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", out)
    out = re.sub(r"[A-Za-z]:\\[^\s,;:]+", "<LOCAL_PATH>", out)
    out = re.sub(r"\\\\wsl\.localhost\\[^\s,;:]+", "<LOCAL_PATH>", out)
    out = re.sub(r"/home/[^\s,;:]+", "<LOCAL_PATH>", out)
    out = re.sub(r"(?<!:)\/[A-Za-z0-9_.-]+\/([A-Za-z0-9_./-]+)", r"<PROJECT_PATH>/\1", out)
    out = re.sub(r"(?<!:)\/[A-Za-z0-9_.-]+(?=[\s),;]|$)", "<PROJECT_PATH>", out)
    out = re.sub(r"(?:<PROJECT_PATH>)+", "<PROJECT_PATH>", out)
    out = re.sub(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+__train-\d+-of-\d+__row_\d+__[A-Za-z0-9]+", "<TRAJECTORY_ID>", out)
    out = re.sub(r"https?://[^\s)]+", "<URL>", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > limit:
        out = out[: limit - 20].rstrip() + " ... <truncated>"
    return out


def call_api(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        latency = time.perf_counter() - start
        return resp.status, json.loads(raw), latency


def response_content(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0]
    message = choice.get("message") or {}
    return str(message.get("content") or ""), str(choice.get("finish_reason") or "")


def parse_raw_to_csv(raw_path: Path, parsed_path: Path, prompts_path: Path) -> int:
    prompts = {row["prompt_id"]: row for row in read_jsonl(prompts_path)}
    latest_by_prompt: dict[str, dict[str, Any]] = {}
    if not raw_path.exists():
        return 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            latest_by_prompt[str(raw.get("prompt_id", ""))] = raw
    rows: list[dict[str, Any]] = []
    for raw in latest_by_prompt.values():
            prompt = prompts.get(raw.get("prompt_id"), {})
            content = raw.get("content", "")
            parsed = extract_json_object(content) if raw.get("status") == "ok" else None
            rows.append(
                {
                    "prompt_id": raw.get("prompt_id", ""),
                    "released_prefix_id": raw.get("released_prefix_id", ""),
                    "arm": raw.get("arm", ""),
                    "dataset_display": prompt.get("dataset_display", raw.get("dataset_display", "")),
                    "stratum": prompt.get("stratum", raw.get("stratum", "")),
                    "status": raw.get("status", ""),
                    "json_parse_success": bool(parsed),
                    "evidence_summary": scrub_response_text((parsed or {}).get("evidence_summary", "")),
                    "current_hypothesis": scrub_response_text((parsed or {}).get("current_hypothesis", "")),
                    "what_changed_since_previous_attempt": scrub_response_text((parsed or {}).get("what_changed_since_previous_attempt", "")),
                    "next_action_type": scrub_response_text((parsed or {}).get("next_action_type", ""), 120),
                    "next_action_target": scrub_response_text((parsed or {}).get("next_action_target", ""), 400),
                    "next_action_description": scrub_response_text((parsed or {}).get("next_action_description", "")),
                    "why_this_action": scrub_response_text((parsed or {}).get("why_this_action", "")),
                    "uses_new_evidence": normalize_bool((parsed or {}).get("uses_new_evidence", "")),
                    "repeats_recent_pattern": normalize_bool((parsed or {}).get("repeats_recent_pattern", "")),
                    "expected_information_gain": scrub_response_text((parsed or {}).get("expected_information_gain", ""), 80).lower(),
                    "should_continue": normalize_bool((parsed or {}).get("should_continue", "")),
                    "latency_seconds": raw.get("latency_seconds", ""),
                    "prompt_tokens": raw.get("prompt_tokens", ""),
                    "completion_tokens": raw.get("completion_tokens", ""),
                    "total_tokens": raw.get("total_tokens", ""),
                    "finish_reason": raw.get("finish_reason", ""),
                    "error": scrub_response_text(raw.get("error", ""), 600),
                }
            )
    pd.DataFrame(rows).to_csv(parsed_path, index=False, encoding="utf-8-sig")
    return len(rows)


def existing_ok_prompt_ids(raw_path: Path) -> set[str]:
    done: set[str] = set()
    if not raw_path.exists():
        return done
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok" and row.get("prompt_id"):
                done.add(str(row["prompt_id"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepSeek API-assisted prefix replan intervention experiment.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit to the first N prefixes; each prefix still runs both arms.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    root = repo_root()
    out = args.output_dir or root / OUTPUT_SUBDIR
    prompts_path = out / "prefix_replan_prompts.jsonl"
    raw_path = out / "api_replan_raw_responses.jsonl"
    parsed_path = out / "api_replan_parsed_responses.csv"
    log_path = out / "api_call_log.csv"
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    missing = [name for name, value in [(ENV_BASE_URL, base_url), (ENV_API_KEY, api_key), (ENV_MODEL, model)] if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    url = completions_url(base_url)
    endpoint = masked_endpoint(url)

    prompts = select_prompts(read_jsonl(prompts_path), args.limit)
    done = existing_ok_prompt_ids(raw_path) if args.resume else set()
    out.mkdir(parents=True, exist_ok=True)
    log_exists = log_path.exists()
    log_f = log_path.open("a", encoding="utf-8-sig", newline="")
    log_writer = csv.DictWriter(log_f, fieldnames=["prompt_id", "released_prefix_id", "arm", "status", "attempts", "latency_seconds", "prompt_tokens", "completion_tokens", "total_tokens", "error"])
    if not log_exists:
        log_writer.writeheader()

    completed = 0
    failed = 0
    try:
        for idx, prompt in enumerate(prompts, 1):
            prompt_id = str(prompt["prompt_id"])
            if prompt_id in done:
                continue
            payload = {
                "model": model,
                "messages": prompt["messages"],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
            }
            status = "failed"
            error = ""
            response: dict[str, Any] = {}
            latency = 0.0
            finish_reason = ""
            content = ""
            http_status = None
            attempts = 0
            for attempt in range(1, args.retries + 1):
                attempts = attempt
                try:
                    http_status, response, latency = call_api(url, api_key, payload, args.timeout)
                    content, finish_reason = response_content(response)
                    if extract_json_object(content) is None and "response_format" in payload:
                        payload.pop("response_format", None)
                        http_status, response, latency = call_api(url, api_key, payload, args.timeout)
                        content, finish_reason = response_content(response)
                    status = "ok" if extract_json_object(content) is not None else "parse_failed"
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")[:800]
                    error = f"HTTP {exc.code}: {body}"
                    if exc.code in {400, 404} and "response_format" in payload:
                        payload.pop("response_format", None)
                    time.sleep(min(8, 2**attempt))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(min(8, 2**attempt))
            usage = response.get("usage") or {}
            raw_row = {
                "prompt_id": prompt_id,
                "released_prefix_id": prompt["released_prefix_id"],
                "arm": prompt["arm"],
                "dataset_display": prompt.get("dataset_display", ""),
                "stratum": prompt.get("stratum", ""),
                "status": status,
                "http_status": http_status,
                "model": model,
                "endpoint": endpoint,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "latency_seconds": round(latency, 4) if latency else "",
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "finish_reason": finish_reason,
                "content": content,
                "error": error,
                "attempts": attempts,
            }
            append_jsonl(raw_path, raw_row)
            log_writer.writerow({k: raw_row.get(k, "") for k in log_writer.fieldnames or []})
            log_f.flush()
            if status == "ok":
                completed += 1
            else:
                failed += 1
            print(f"[{idx}/{len(prompts)}] {prompt_id} {prompt['arm']} {status}")
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        log_f.close()

    parsed_n = parse_raw_to_csv(raw_path, parsed_path, prompts_path)
    print(f"New ok responses: {completed}; new failed/parse_failed: {failed}; parsed rows: {parsed_n}")
    print(f"Raw responses: {raw_path}")
    print(f"Parsed responses: {parsed_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
