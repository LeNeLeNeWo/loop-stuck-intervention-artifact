from __future__ import annotations

import argparse
import re
from pathlib import Path

from live_guard_common import repo_root


FORBIDDEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.I),
    re.compile(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;]*", re.I),
    re.compile(r"\\\\" + "wsl" + r"\.localhost", re.I),
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile(r"DEEPSEEK_API_KEY\s*[:=]\s*[^;\n]+", re.I),
]


def scan(root: Path) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    if not root.exists():
        return [(root, 0, "missing live_guard_experiment artifact directory")]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".zip"}:
            continue
        if path.name == "live_guard_release_validation.md":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    hits.append((path, idx, pat.pattern))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sanitized live guard release files.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifact = args.artifact_root or (root / "release_artifacts/LoopSense_Paper1_Artifact")
    live_dir = artifact / "outputs/expected/live_guard_experiment"
    hits = scan(live_dir)
    report = live_dir / "live_guard_release_validation.md"
    lines = ["# Live Guard Release Validation", "", f"Status: **{'FAIL' if hits else 'PASS'}**", ""]
    if hits:
        lines.extend(["## Hits", ""])
        for path, line, pattern in hits[:200]:
            lines.append(f"- `{path.relative_to(artifact)}` line {line}: `{pattern}`")
    else:
        lines.append("No API-key-like strings, authorization-token-like strings, or local paths were found in the live guard artifact directory.")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Live guard release validation: {'FAIL' if hits else 'PASS'}")
    print(f"Report: {report}")
    return 2 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
