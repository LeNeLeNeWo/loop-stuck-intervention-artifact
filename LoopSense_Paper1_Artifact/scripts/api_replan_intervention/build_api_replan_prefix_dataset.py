from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_SUBDIR = Path("outputs/paper1_api_replan_intervention")
TRIGGERS = Path("outputs/paper1_intervention_validation/stop_counterfactual_triggers.csv")
NORMALIZED_STEPS = Path("outputs/paper1_audit/normalized_steps.csv")
ENRICHED_STEPS = Path("outputs/paper1_raw_enrichment/enriched_steps.csv")

DISPLAY_DATASET = {"Final500": "Enriched500", "Enriched500": "Enriched500"}
DATASETS = {"Final500", "Enriched500", "StressFresh100"}
PREFERRED = [
    ("file_revisit_guard", "window=5;repeats=3"),
    ("file_revisit_guard", "window=10;repeats=3"),
    ("max_step_guard", "threshold=50"),
    ("tool_repeat_k", "k=3"),
    ("tool_repeat", "k=3"),
    ("repeat_without_progress_guard", None),
]

TEXT_COLUMNS = [
    "raw_action_text",
    "normalized_action_text",
    "tool_name",
    "command_text",
    "thought_text",
    "observation_text",
    "output_text",
    "error_text",
    "error_signature",
    "file_path",
    "mentioned_files",
    "step_type",
    "evidence_text",
]

PUBLIC_SAMPLE_COLUMNS = [
    "released_prefix_id",
    "dataset_display",
    "dataset_internal",
    "prefix_step",
    "trigger_location_type",
    "detector_family",
    "detector_config",
    "recent_context_text",
    "recent_actions_summary",
    "recent_observations_summary",
    "repeated_pattern_summary",
    "original_next_action_if_available",
    "original_suffix_productive_steps",
    "original_suffix_UC_steps",
    "stratum",
    "selection_reason",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def stable_id(*parts: Any) -> str:
    raw = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def scrub_text(text: Any, private_values: list[str] | None = None, limit: int | None = None) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    out = str(text)
    private_values = private_values or []
    for value in private_values:
        if value:
            out = out.replace(str(value), "<ID>")
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
    if limit is not None and len(out) > limit:
        return out[: limit - 20].rstrip() + " ... <truncated>"
    return out


def detector_priority(row: pd.Series) -> int:
    det = str(row.get("detector", ""))
    cfg = str(row.get("detector_config", ""))
    fam = str(row.get("detector_family", ""))
    for idx, (want, want_cfg) in enumerate(PREFERRED):
        if det == want or fam == want:
            if want_cfg is None or cfg == want_cfg:
                return idx
    return len(PREFERRED) + 10


def choose_candidates(candidates: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if candidates.empty or n <= 0:
        return candidates.head(0)
    df = candidates.copy()
    df["_priority"] = df.apply(detector_priority, axis=1)
    df["_rand"] = pd.Series(pd.util.hash_pandas_object(df[["trajectory_id", "step_index", "detector", "detector_config"]], index=False).astype("uint64")) / 2**64
    df["_rand"] = (df["_rand"] + (seed % 997) / 997.0) % 1.0
    df = df.sort_values(["_priority", "_rand"]).reset_index(drop=True)
    diverse = df.drop_duplicates(["trajectory_id"], keep="first")
    selected = diverse.head(n)
    if len(selected) < n:
        used = set(selected.index)
        fill = df.loc[[i for i in df.index if i not in used]].head(n - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)
    return selected.head(n).drop(columns=["_priority", "_rand"], errors="ignore")


def load_trigger_candidates(root: Path) -> pd.DataFrame:
    path = root / TRIGGERS
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = [
        "trigger_id",
        "dataset",
        "trajectory_id",
        "step_index",
        "detector",
        "detector_family",
        "detector_config",
        "trigger_location_type",
        "remaining_productive_steps_after_trigger",
        "remaining_UC_steps_after_trigger",
        "is_first_trigger_for_config_trajectory",
        "evidence_available_flag",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["dataset"].astype(str).isin(DATASETS)].copy()
    df = df[df["is_first_trigger_for_config_trajectory"].map(boolish)].copy()
    df = df[df["evidence_available_flag"].map(boolish)].copy()
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce").astype("Int64")
    df = df[df["step_index"].notna()]
    df["dataset_display"] = df["dataset"].astype(str).map(lambda x: DISPLAY_DATASET.get(x, x))
    return df


def stratum_targets(target: int) -> dict[str, int]:
    productive = max(1, round(target * 0.4))
    uc = max(1, round(target * 0.4))
    control = max(1, target - productive - uc)
    return {"productive_risk": productive, "uc": uc, "matched_control": control}


def load_steps_for_trajectories(root: Path, trajectory_ids: set[str]) -> pd.DataFrame:
    source = root / ENRICHED_STEPS
    if not source.exists():
        source = root / NORMALIZED_STEPS
    if not source.exists():
        raise FileNotFoundError("No enriched or normalized steps file found.")
    wanted = {
        "dataset",
        "dataset_role",
        "trajectory_id",
        "task_id",
        "step_index",
        "label_norm",
        "productive_iteration_flag",
        "unproductive_cycle_flag",
        "hard_negative",
        *TEXT_COLUMNS,
    }
    chunks = []
    for chunk in pd.read_csv(source, usecols=lambda c: c in wanted, chunksize=50_000, low_memory=False):
        hit = chunk[chunk["trajectory_id"].astype(str).isin(trajectory_ids)].copy()
        if not hit.empty:
            chunks.append(hit)
    if not chunks:
        raise RuntimeError("No steps matched selected trajectory ids.")
    steps = pd.concat(chunks, ignore_index=True)
    steps["step_index"] = pd.to_numeric(steps["step_index"], errors="coerce").astype("Int64")
    return steps.sort_values(["trajectory_id", "step_index"])


def concise_action(row: pd.Series, private_values: list[str]) -> str:
    parts = []
    for col in ["step_type", "tool_name", "file_path", "command_text", "normalized_action_text", "raw_action_text"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(f"{col}={scrub_text(row[col], private_values, 180)}")
    return "; ".join(parts[:4]) or "no action text available"


def concise_observation(row: pd.Series, private_values: list[str]) -> str:
    parts = []
    for col in ["error_signature", "error_text", "observation_text", "output_text", "evidence_text"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(f"{col}={scrub_text(row[col], private_values, 220)}")
    return "; ".join(parts[:3]) or "no observation text available"


def repeated_pattern(recent: pd.DataFrame, private_values: list[str], detector: str, config: str) -> str:
    signals = []
    for col in ["tool_name", "file_path", "error_signature", "normalized_action_text", "command_text"]:
        if col in recent.columns:
            vals = [scrub_text(v, private_values, 120) for v in recent[col].dropna().astype(str) if str(v).strip()]
            vals = [v for v in vals if v]
            if vals:
                counts = pd.Series(vals).value_counts()
                if int(counts.iloc[0]) >= 2:
                    signals.append(f"{col} repeated {int(counts.iloc[0])}x: {counts.index[0]}")
    if signals:
        return "; ".join(signals[:3])
    return f"no deterministic repeated text pattern found; trigger detector={detector}, config={config}"


def context_for_prefix(row: pd.Series, steps: pd.DataFrame, max_recent_steps: int, max_chars: int) -> dict[str, str]:
    tid = str(row["trajectory_id"])
    task_id = ""
    traj_steps = steps[steps["trajectory_id"].astype(str).eq(tid)].copy()
    if "task_id" in traj_steps.columns and not traj_steps["task_id"].dropna().empty:
        task_id = str(traj_steps["task_id"].dropna().iloc[0])
    private_values = [tid, task_id]
    prefix_step = int(row["step_index"])
    recent = traj_steps[(traj_steps["step_index"] <= prefix_step) & (traj_steps["step_index"] >= prefix_step - max_recent_steps + 1)].copy()
    recent = recent.sort_values("step_index")
    blocks = []
    action_summaries = []
    obs_summaries = []
    for _, step in recent.iterrows():
        idx = int(step["step_index"])
        action = concise_action(step, private_values)
        obs = concise_observation(step, private_values)
        action_summaries.append(f"{idx}: {action}")
        obs_summaries.append(f"{idx}: {obs}")
        blocks.append(f"Step {idx}\nAction: {action}\nObservation: {obs}")
    pattern = repeated_pattern(recent.tail(6), private_values, str(row["detector_family"]), str(row["detector_config"]))
    next_rows = traj_steps[traj_steps["step_index"] == prefix_step + 1]
    original_next = ""
    if not next_rows.empty:
        original_next = concise_action(next_rows.iloc[0], private_values)
    context = "\n\n".join(blocks)
    context = scrub_text(context, private_values, max_chars)
    return {
        "recent_context_text": context,
        "recent_actions_summary": scrub_text(" | ".join(action_summaries[-6:]), private_values, 1800),
        "recent_observations_summary": scrub_text(" | ".join(obs_summaries[-4:]), private_values, 1800),
        "repeated_pattern_summary": scrub_text(pattern, private_values, 800),
        "original_next_action_if_available": scrub_text(original_next, private_values, 600),
    }


def make_prompts(sample: pd.DataFrame) -> list[dict[str, Any]]:
    system = "You are an autonomous software-engineering debugging agent. You will be given a recent debugging prefix. Return only valid JSON."
    schema = """{
"evidence_summary": "...",
"current_hypothesis": "...",
"next_action_type": "inspect_file|edit_file|run_test|search|reason|stop|other",
"next_action_target": "...",
"next_action_description": "...",
"why_this_action": "...",
"uses_new_evidence": true/false,
"repeats_recent_pattern": true/false,
"expected_information_gain": "low|medium|high",
"should_continue": true/false
}"""
    warn_schema = """{
"evidence_summary": "...",
"current_hypothesis": "...",
"what_changed_since_previous_attempt": "...",
"next_action_type": "inspect_file|edit_file|run_test|search|reason|stop|other",
"next_action_target": "...",
"next_action_description": "...",
"why_this_action": "...",
"uses_new_evidence": true/false,
"repeats_recent_pattern": true/false,
"expected_information_gain": "low|medium|high",
"should_continue": true/false
}"""
    prompts: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        context = (
            f"Dataset: {row['dataset_display']}\n"
            f"Prefix step: {row['prefix_step']}\n"
            f"Recent actions summary: {row['recent_actions_summary']}\n"
            f"Recent observations summary: {row['recent_observations_summary']}\n"
            f"Potential repeated pattern: {row['repeated_pattern_summary']}\n\n"
            f"Recent prefix:\n{row['recent_context_text']}"
        )
        neutral_user = (
            "You are continuing a debugging task from the following recent prefix. Decide the next debugging action. "
            "Do not invent files or test results that are not in the context.\n\n"
            f"{context}\n\nReturn JSON with:\n{schema}"
        )
        warn_user = (
            "You may be repeating. Before taking the next action, summarize the evidence gathered so far, state the current hypothesis, "
            "explain what has changed since the previous attempt, and choose the next debugging action only if it is justified by new evidence. "
            "If no useful new evidence exists, revise the plan instead of repeating the same action.\n\n"
            f"{context}\n\nReturn JSON with:\n{warn_schema}"
        )
        for arm, user in [("NEUTRAL_CONTINUE", neutral_user), ("WARN_REPLAN", warn_user)]:
            prompt_id = "prompt_" + stable_id(row["released_prefix_id"], arm)
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "released_prefix_id": row["released_prefix_id"],
                    "arm": arm,
                    "dataset_display": row["dataset_display"],
                    "stratum": row["stratum"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
            )
    return prompts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.fillna("").astype(str)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real-prefix prompt pairs for the API-assisted warn/replan intervention experiment.")
    parser.add_argument("--target", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20270614)
    parser.add_argument("--max-recent-steps", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=6500)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    out = args.output_dir or root / OUTPUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    triggers = load_trigger_candidates(root)
    targets = stratum_targets(args.target)
    prod = triggers[triggers["trigger_location_type"].astype(str).isin(["PI", "HN"])].copy()
    uc = triggers[triggers["trigger_location_type"].astype(str).eq("UC")].copy()
    ctrl = triggers[~triggers["trigger_location_type"].astype(str).isin(["PI", "HN", "UC"])].copy()

    selected_parts = []
    for name, candidates in [("productive_risk", prod), ("uc", uc), ("matched_control", ctrl)]:
        chosen = choose_candidates(candidates, targets[name], args.seed + len(selected_parts))
        chosen = chosen.copy()
        chosen["stratum"] = name
        chosen["selection_reason"] = f"{name} sampled from first trigger rows with evidence_available_flag=true"
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) < min(args.target, 120):
        raise RuntimeError(f"Only selected {len(selected)} prefixes; minimum for planned experiment is 120.")

    trajectory_ids = set(selected["trajectory_id"].astype(str))
    steps = load_steps_for_trajectories(root, trajectory_ids)

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        released_prefix_id = "released_prefix_" + stable_id(row["trajectory_id"], row["step_index"], row["detector"], row["detector_config"])
        ctx = context_for_prefix(row, steps, args.max_recent_steps, args.max_context_chars)
        public = {
            "released_prefix_id": released_prefix_id,
            "dataset_display": DISPLAY_DATASET.get(str(row["dataset"]), str(row["dataset"])),
            "dataset_internal": str(row["dataset"]),
            "prefix_step": int(row["step_index"]),
            "trigger_location_type": str(row["trigger_location_type"]),
            "detector_family": str(row["detector_family"]),
            "detector_config": str(row["detector_config"]),
            "original_suffix_productive_steps": row.get("remaining_productive_steps_after_trigger", ""),
            "original_suffix_UC_steps": row.get("remaining_UC_steps_after_trigger", ""),
            "stratum": str(row["stratum"]),
            "selection_reason": str(row["selection_reason"]),
            **ctx,
        }
        public_rows.append(public)
        private = public.copy()
        private["internal_trajectory_id"] = str(row["trajectory_id"])
        private["internal_trigger_id"] = str(row["trigger_id"])
        private["detector"] = str(row["detector"])
        private_rows.append(private)

    public_df = pd.DataFrame(public_rows)[PUBLIC_SAMPLE_COLUMNS]
    private_df = pd.DataFrame(private_rows)
    public_df.to_csv(out / "prefix_replan_sample.csv", index=False, encoding="utf-8-sig")
    private_df.to_csv(out / "prefix_replan_sample_private.csv", index=False, encoding="utf-8-sig")
    prompts = make_prompts(public_df)
    write_jsonl(out / "prefix_replan_prompts.jsonl", prompts)

    report = [
        "# API Replan Prefix Sampling Report",
        "",
        f"Target prefixes: {args.target}",
        f"Selected prefixes: {len(public_df)}",
        f"Prompt arms per prefix: 2",
        f"Prompt rows: {len(prompts)}",
        f"Seed: {args.seed}",
        "",
        "## Group Composition",
        "",
        markdown_table(public_df.groupby(["stratum", "dataset_display"]).size().rename("n").reset_index()),
        "",
        "## Detector Composition",
        "",
        markdown_table(public_df.groupby(["stratum", "detector_family", "detector_config"]).size().rename("n").reset_index().head(40)),
        "",
        "## Privacy Boundary",
        "",
        "- `prefix_replan_sample.csv` and `prefix_replan_prompts.jsonl` use released prefix ids only.",
        "- `prefix_replan_sample_private.csv` contains raw trajectory ids for local audit and must not be released.",
        "- Prompts do not include PI/HN/UC labels in the model-visible message text.",
    ]
    (out / "sampling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Selected prefixes: {len(public_df)}")
    print(f"Prompt rows: {len(prompts)}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
