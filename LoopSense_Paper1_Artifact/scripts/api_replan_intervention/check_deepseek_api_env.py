from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_dir(root: Path) -> Path:
    return root / "outputs" / "paper1_api_replan_intervention"


def completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def masked_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    suffix = "/chat/completions" if path.endswith("/chat/completions") else ""
    return f"{parsed.scheme}://{parsed.netloc}/<masked-path>{suffix}"


def post_chat(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any], float]:
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
        elapsed = time.perf_counter() - start
        return resp.status, json.loads(raw), elapsed


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DeepSeek OpenAI-compatible API environment for the API replan experiment.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    root = repo_root()
    out = args.output_dir or output_dir(root)
    out.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    url = completions_url(base_url) if base_url else ""

    missing = [name for name, value in [(ENV_BASE_URL, base_url), (ENV_API_KEY, api_key), (ENV_MODEL, model)] if not value]
    status = "NOT_READY" if missing else "UNKNOWN"
    latency = None
    usage: dict[str, Any] = {}
    error = ""
    content = ""
    http_status: int | None = None
    tried_without_response_format = False

    if not missing:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a JSON-only assistant."},
                {"role": "user", "content": "{\"ping\": true}"},
            ],
            "temperature": 0,
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
        }
        try:
            http_status, response, latency = post_chat(url, api_key, payload, args.timeout)
            usage = response.get("usage") or {}
            content = extract_content(response)
            json.loads(content)
            status = "READY"
        except Exception as exc:
            tried_without_response_format = True
            payload.pop("response_format", None)
            try:
                http_status, response, latency = post_chat(url, api_key, payload, args.timeout)
                usage = response.get("usage") or {}
                content = extract_content(response)
                json.loads(content)
                status = "READY"
            except urllib.error.HTTPError as http_exc:
                error = f"HTTP {http_exc.code}: {http_exc.read().decode('utf-8', errors='replace')[:500]}"
                status = "NOT_READY"
            except Exception as second_exc:
                error = f"{type(second_exc).__name__}: {second_exc}"
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
                status = "NOT_READY"

    lines = [
        "# DeepSeek API Environment Check",
        "",
        f"Status: **{status}**",
        "",
        "## Environment",
        "",
        f"- `{ENV_BASE_URL}`: {'present' if base_url else 'missing'}",
        f"- `{ENV_API_KEY}`: {'present; value not printed' if api_key else 'missing'}",
        f"- `{ENV_MODEL}`: {model if model else 'missing'}",
        f"- Endpoint: `{masked_url(url) if url else 'missing'}`",
        "",
        "## Test Call",
        "",
        f"- HTTP status: {http_status if http_status is not None else 'NA'}",
        f"- Latency seconds: {latency:.3f}" if latency is not None else "- Latency seconds: NA",
        f"- Retried without response_format: {'yes' if tried_without_response_format else 'no'}",
        f"- Token usage: `{json.dumps(usage, sort_keys=True)}`" if usage else "- Token usage: not returned",
        f"- JSON parse: {'success' if status == 'READY' else 'not successful'}",
    ]
    if error:
        lines.extend(["", "## Error", "", "```text", error, "```"])
    if content:
        lines.extend(["", "## Response Content", "", "```json", content[:1000], "```"])
    (out / "api_env_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DeepSeek API env check: {status}")
    print(f"Report: {out / 'api_env_check.md'}")
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
