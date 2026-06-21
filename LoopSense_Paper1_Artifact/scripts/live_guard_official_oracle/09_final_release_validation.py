from __future__ import annotations

import argparse
import re
from pathlib import Path

from official_common import repo_root


PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}", re.I),
    re.compile(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;]*", re.I),
    re.compile(r"\\\\" + "wsl" + r"\.localhost", re.I),
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile(r"DEEPSEEK_API_KEY\s*[:=]\s*[^;\n]+", re.I),
]


def scan(root: Path) -> list[tuple[Path, int, str]]:
    hits = []
    if not root.exists():
        return [(root, 0, "missing")]
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() in {".png", ".pdf", ".zip"} or p.name.endswith("_validation.md"):
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            for pat in PATTERNS:
                if pat.search(line):
                    hits.append((p, i, pat.pattern))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate official-oracle live guard artifact files.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifact = args.artifact_root or (root / "release_artifacts/LoopSense_Paper1_Artifact")
    live_dir = artifact / "outputs/expected/live_guard_official_oracle"
    hits = scan(live_dir) + scan(artifact / "scripts/live_guard_official_oracle")
    report = live_dir / "live_guard_official_release_validation.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Official-Oracle Live Guard Release Validation", "", f"Status: **{'FAIL' if hits else 'PASS'}**", ""]
    if hits:
        for p, i, pat in hits[:200]:
            lines.append(f"- `{p.relative_to(artifact)}` line {i}: `{pat}`")
    else:
        lines.append("No API-key-like strings, authorization-token-like strings, or local paths were found.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Official-oracle release validation: {'FAIL' if hits else 'PASS'}")
    return 2 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
