from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "pool", "trajectory_id", "instance_id", "task_id", "prefix_step", "detector_family", "detector_config",
    "trigger_location_type", "original_suffix_length", "original_productive_suffix_length", "original_UC_suffix_length",
    "raw_replay_available", "selected_stratum", "selection_reason",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Select prefixes for a live/replay intervention experiment if feasible.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper1_live_intervention"))
    parser.add_argument("--seed", type=int, default=20270614)
    parser.add_argument("--target", type=int, default=120)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    decision = "NOT_FEASIBLE"
    feas = out / "feasibility_report.json"
    if feas.exists():
        decision = json.loads(feas.read_text(encoding="utf-8")).get("decision", "NOT_FEASIBLE")
    if decision not in {"FULL_LIVE_REPLAY_FEASIBLE", "PARTIAL_REPLAY_FEASIBLE", "PROMPT_ONLY_CONTINUATION_FEASIBLE"}:
        pd.DataFrame(columns=COLUMNS).to_csv(out / "prefix_branch_sample.csv", index=False, encoding="utf-8-sig")
        (out / "sampling_report.md").write_text(
            "# Prefix-Branch Sampling Report\n\nSampling was skipped because live/replay execution is not feasible in the current artifact. See `feasibility_report.md`.\n",
            encoding="utf-8",
        )
        print("Sampling skipped: live/replay not feasible.")
        return 0
    triggers_path = root / "outputs/paper1_intervention_validation/stop_counterfactual_triggers.csv"
    if not triggers_path.exists():
        raise FileNotFoundError(triggers_path)
    usecols = ["dataset", "trajectory_id", "step_index", "detector_family", "detector_config", "trigger_location_type", "remaining_steps_after_trigger", "remaining_productive_steps_after_trigger", "remaining_UC_steps_after_trigger"]
    trig = pd.read_csv(triggers_path, usecols=usecols)
    trig = trig[trig["dataset"].isin(["Final500", "StressFresh100"])]
    trig = trig.sample(frac=1.0, random_state=args.seed)
    rows = []
    per_traj: dict[str, int] = {}
    strata = [
        ("productive-risk", trig[trig["trigger_location_type"].isin(["PI", "HN"]) & trig["remaining_productive_steps_after_trigger"].gt(0)], args.target // 2),
        ("UC", trig[trig["trigger_location_type"].eq("UC") & trig["remaining_UC_steps_after_trigger"].gt(0)], args.target // 2),
    ]
    for stratum, frame, limit in strata:
        for r in frame.itertuples(index=False):
            key = str(r.trajectory_id)
            if per_traj.get(key, 0) >= 2:
                continue
            rows.append({
                "pool": r.dataset, "trajectory_id": r.trajectory_id, "instance_id": "", "task_id": "", "prefix_step": r.step_index,
                "detector_family": r.detector_family, "detector_config": r.detector_config, "trigger_location_type": r.trigger_location_type,
                "original_suffix_length": r.remaining_steps_after_trigger,
                "original_productive_suffix_length": r.remaining_productive_steps_after_trigger,
                "original_UC_suffix_length": r.remaining_UC_steps_after_trigger,
                "raw_replay_available": True, "selected_stratum": stratum,
                "selection_reason": "stratified from first executable trigger table",
            })
            per_traj[key] = per_traj.get(key, 0) + 1
            if sum(1 for x in rows if x["selected_stratum"] == stratum) >= limit:
                break
    sample = pd.DataFrame(rows, columns=COLUMNS)
    sample.to_csv(out / "prefix_branch_sample.csv", index=False, encoding="utf-8-sig")
    (out / "sampling_report.md").write_text(f"# Prefix-Branch Sampling Report\n\nSelected {len(sample)} prefixes with seed {args.seed}.\n", encoding="utf-8")
    print(f"Selected {len(sample)} prefixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
