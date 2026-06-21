from __future__ import annotations

from pathlib import Path

from official_common import ensure_output, preregistration_text, repo_root


def main() -> int:
    root = repo_root()
    out = ensure_output(root)
    path = out / "OFFICIAL_ORACLE_PREREGISTRATION.md"
    if not path.exists():
        path.write_text(preregistration_text(), encoding="utf-8")
    print(f"Preregistration: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
