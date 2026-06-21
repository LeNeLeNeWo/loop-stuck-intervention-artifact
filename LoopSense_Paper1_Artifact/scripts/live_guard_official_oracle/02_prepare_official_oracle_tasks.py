from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from official_common import DATASET_NAME, DATASET_SPLIT, SEED, docker_image_name, ensure_output, repo_root, sha256_text, stable_hash, write_csv, write_json


FIELDS = [
    "released_task_id",
    "benchmark_instance_id",
    "repo",
    "base_commit",
    "problem_statement_hash",
    "task_source",
    "selected_order",
    "selected_for_30",
    "selected_for_60",
    "traceability_level",
    "docker_image",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare official-oracle live guard task sample.")
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    out.mkdir(parents=True, exist_ok=True)

    prior = root / "outputs/paper1_live_guard_experiment/live_task_sample.csv"
    prior_ids: list[str] = []
    if prior.exists():
        import csv

        with prior.open("r", encoding="utf-8-sig", newline="") as f:
            prior_ids = [row["benchmark_instance_id"] for row in csv.DictReader(f) if row.get("benchmark_instance_id")]

    ds = list(load_dataset(DATASET_NAME, split=DATASET_SPLIT))
    by_id = {str(row["instance_id"]): dict(row) for row in ds}
    ordered_ids = [iid for iid in prior_ids if iid in by_id]
    remaining = sorted([iid for iid in by_id if iid not in set(ordered_ids)])
    rng = random.Random(args.seed)
    rng.shuffle(remaining)
    ordered_ids.extend(remaining)
    selected_ids = ordered_ids[: args.target]
    rows: list[dict[str, Any]] = []
    for idx, iid in enumerate(selected_ids, start=1):
        row = by_id[iid]
        rows.append(
            {
                "released_task_id": "released_task_" + stable_hash(DATASET_NAME, DATASET_SPLIT, iid),
                "benchmark_instance_id": iid,
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
                "problem_statement_hash": sha256_text(str(row.get("problem_statement", ""))),
                "task_source": f"{DATASET_NAME}:{DATASET_SPLIT}",
                "selected_order": idx,
                "selected_for_30": idx <= 30,
                "selected_for_60": idx <= 60,
                "traceability_level": "public_benchmark_fallback",
                "docker_image": row.get("image_name") or row.get("docker_image") or docker_image_name(iid),
                "notes": "Frozen before official-oracle outcomes; prior pilot order reused as prefix when available.",
            }
        )
    write_csv(out / "official_oracle_task_sample.csv", rows, FIELDS)
    write_csv(out / "official_oracle_task_sample_sanitized.csv", rows, FIELDS)
    write_json(
        out / "official_oracle_task_sample_manifest.json",
        {"dataset": DATASET_NAME, "split": DATASET_SPLIT, "seed": args.seed, "target": args.target, "selected": len(rows), "prior_prefix_count": len(prior_ids)},
    )
    print(f"Official-oracle sample selected: {len(rows)}")
    print(f"Sample: {out / 'official_oracle_task_sample.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
