from __future__ import annotations

import argparse
from pathlib import Path

from official_common import ensure_output, read_csv_dicts, repo_root, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and sanity-check patches/log manifests.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    rows = read_csv_dicts(out / "patch_manifest.csv")
    checked = []
    for row in rows:
        p = Path(row.get("patch_path_local", ""))
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        generated = str(row.get("patch_generated", "")).lower() in {"true", "1", "yes"}
        checked.append({**row, "patch_file_exists": exists, "patch_file_size": size, "patch_manifest_ok": (exists and size > 0) == generated})
    write_csv(out / "patch_manifest_checked.csv", checked)
    total = len(checked)
    generated_count = sum(1 for r in checked if str(r["patch_generated"]).lower() in {"true", "1", "yes"})
    lines = [
        "# Patch Collection Report",
        "",
        f"- Task-arm rows: {total}",
        f"- Patch generated rows: {generated_count}",
        f"- Manifest mismatches: {sum(1 for r in checked if not r['patch_manifest_ok'])}",
    ]
    (out / "patch_collection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Patch generated rows: {generated_count}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
