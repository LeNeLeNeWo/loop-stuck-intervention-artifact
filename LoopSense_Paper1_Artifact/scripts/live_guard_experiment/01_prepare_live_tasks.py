from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from live_guard_common import docker_image_name, ensure_output, repo_root, stable_hash, write_csv, write_json


DATASET_PRIORITY = [
    ("verified", "princeton-nlp/SWE-Bench_Verified", "test"),
    ("lite", "princeton-nlp/SWE-Bench_Lite", "test"),
]

SAMPLE_FIELDS = [
    "released_task_id",
    "internal_task_id_local_only",
    "benchmark_instance_id",
    "repo",
    "base_commit",
    "task_source",
    "paper_pool_display",
    "historical_group",
    "traceability_level",
    "selected_for_run",
    "selection_reason",
    "estimated_runtime",
    "notes",
    "docker_image",
]

SANITIZED_FIELDS = [f for f in SAMPLE_FIELDS if f != "internal_task_id_local_only"]


def load_public_dataset() -> tuple[str, str, str, list[dict[str, Any]], str]:
    reasons = []
    for subset, dataset_path, split in DATASET_PRIORITY:
        try:
            ds = list(load_dataset(dataset_path, split=split))
            return subset, dataset_path, split, ds, f"selected_{subset}_because_dataset_loaded"
        except Exception as exc:
            reasons.append(f"{subset}:{type(exc).__name__}:{str(exc)[:180]}")
    raise RuntimeError("no_public_swebench_dataset_loaded|" + "|".join(reasons))


def historical_task_ids(root: Path) -> set[str]:
    hits: set[str] = set()
    candidates = [
        root / "outputs/paper1_audit/normalized_steps.csv",
        root / "outputs/paper1_raw_enrichment/enriched_steps.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for chunk in pd.read_csv(path, usecols=lambda c: c in {"task_id", "dataset"}, chunksize=200_000):
                if "task_id" in chunk.columns:
                    values = chunk["task_id"].dropna().astype(str)
                    hits.update(v for v in values if "__" in v)
        except Exception:
            continue
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed live guard public benchmark task sample.")
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20270614)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir or ensure_output(root)
    out.mkdir(parents=True, exist_ok=True)

    subset, dataset_path, split, ds, reason = load_public_dataset()
    hist = historical_task_ids(root)
    rng = random.Random(args.seed)
    ordered = list(ds)
    ordered.sort(key=lambda row: str(row.get("instance_id", "")))
    rng.shuffle(ordered)

    mapped = [row for row in ordered if str(row.get("instance_id", "")) in hist]
    fallback = [row for row in ordered if str(row.get("instance_id", "")) not in hist]
    selected = (mapped + fallback)[: args.target]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        iid = str(row.get("instance_id", ""))
        traceability = "paper_pool_mapped" if iid in hist else "public_benchmark_fallback"
        group = "mixed/control"
        if traceability == "paper_pool_mapped":
            group = "historical_task_mapped"
        release_id = "released_task_" + stable_hash(dataset_path, split, iid)
        rows.append(
            {
                "released_task_id": release_id,
                "internal_task_id_local_only": iid,
                "benchmark_instance_id": iid,
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
                "task_source": f"{dataset_path}:{split}",
                "paper_pool_display": "Enriched500/StressFresh100 candidate" if traceability == "paper_pool_mapped" else "Public SWE-bench fallback",
                "historical_group": group,
                "traceability_level": traceability,
                "selected_for_run": "yes",
                "selection_reason": reason if idx == 1 else f"deterministic_seed_{args.seed}",
                "estimated_runtime": "variable; bounded by runner timeout",
                "notes": "No historical prefix labels are exposed to the agent.",
                "docker_image": row.get("image_name") or row.get("docker_image") or docker_image_name(iid),
            }
        )

    write_csv(out / "live_task_sample.csv", rows, SAMPLE_FIELDS)
    write_csv(out / "live_task_sample_sanitized.csv", rows, SANITIZED_FIELDS)
    write_json(
        out / "live_task_sample_manifest.json",
        {
            "dataset_subset": subset,
            "dataset_path": dataset_path,
            "split": split,
            "target": args.target,
            "selected": len(rows),
            "seed": args.seed,
            "historical_task_id_count": len(hist),
            "traceability_counts": {k: sum(1 for r in rows if r["traceability_level"] == k) for k in sorted({r["traceability_level"] for r in rows})},
        },
    )
    lines = [
        "# Live Guard Task Sampling Report",
        "",
        f"- Dataset: `{dataset_path}` split `{split}`",
        f"- Requested target: {args.target}",
        f"- Selected tasks: {len(rows)}",
        f"- Seed: {args.seed}",
        f"- Historical task IDs discovered in paper metadata: {len(hist)}",
        f"- Paper-mapped selected tasks: {sum(1 for r in rows if r['traceability_level'] == 'paper_pool_mapped')}",
        f"- Public fallback selected tasks: {sum(1 for r in rows if r['traceability_level'] == 'public_benchmark_fallback')}",
        "",
        "No task outcomes were inspected during sampling. If no paper-mapped runnable tasks are found, the live policy experiment tests the same public SWE-bench task family rather than exact historical prefixes.",
    ]
    (out / "live_task_sampling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected {len(rows)} tasks from {dataset_path}:{split}")
    print(f"Sample: {out / 'live_task_sample.csv'}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
