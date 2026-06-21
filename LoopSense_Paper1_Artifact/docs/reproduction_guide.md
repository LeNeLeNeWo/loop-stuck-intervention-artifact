# Reproduction Guide

Run from artifact root:

```bash
pip install -r requirements.txt
python tests/test_artifact_integrity.py
python tests/test_reproduce_key_outputs.py
python scripts/release/audit_and_test_paper1_artifact.py --artifact-root .
```

Expected outputs are under `outputs/expected/`. Some raw-runtime detector reruns require non-released raw trajectory fields; use included expected intermediate tables and `detector_availability.csv` for partial reproducibility checks.

## API-assisted prefix replan

The audited API-assisted replan outputs are included under `outputs/expected/api_replan_intervention/`. The public release contains sanitized prompts, a sanitized API call log, parsed responses, summary tables, pairwise effects, and API-assisted figure source files. Raw API responses and the private prefix-to-trajectory mapping are not released.

To reproduce the API calls rather than only checking the included outputs, provide your own OpenAI-compatible DeepSeek credentials via:

```bash
export DEEPSEEK_BASE_URL=...
export DEEPSEEK_API_KEY=...
export DEEPSEEK_MODEL=...
python scripts/api_replan_intervention/check_deepseek_api_env.py
python scripts/api_replan_intervention/run_deepseek_api_replan_experiment.py --resume --sleep 1.0
python scripts/api_replan_intervention/analyze_api_replan_results.py
python scripts/api_replan_intervention/build_api_replan_figures.py
```

This rerun is prompt-level only. It does not restore repositories, execute tools, run tests, or estimate final task success.

## Live guard-policy pilot

Sanitized outputs are included under `outputs/expected/live_guard_experiment/`. The pilot can be checked from the included CSV summaries without rerunning the live agent.

To rerun the pilot, provide your own OpenAI-compatible DeepSeek credentials and a Docker-capable environment:

```bash
export DEEPSEEK_BASE_URL=...
export DEEPSEEK_API_KEY=...
export DEEPSEEK_MODEL=...
python scripts/live_guard_experiment/00_check_live_stack.py
python scripts/live_guard_experiment/02_install_or_validate_runner.py
python scripts/live_guard_experiment/01_prepare_live_tasks.py --target 60 --seed 20270614
python scripts/live_guard_experiment/03_run_live_guard_experiment.py --pilot 10 --arms NO_GUARD WARN_REPLAN_GUARD HARD_STOP_GUARD --max-steps 10 --resume
python scripts/live_guard_experiment/04_analyze_live_guard_results.py
python scripts/live_guard_experiment/05_build_live_guard_figures.py
```

This rerun requires external API access, Docker, public benchmark downloads, and local compute. It does not provide official resolved/tests-pass outcomes unless an oracle is separately installed and wired in.

## Official-oracle live guard rerun
Rerunning `scripts/live_guard_official_oracle/` requires Docker, public SWE-bench downloads, a SWE-bench harness, and user-provided DeepSeek credentials. Raw run logs and benchmark workspaces are not included in the release artifact.

The included sanitized outputs can be checked without rerunning the agent. Expected headline values are:

- Resolved/test-pass: NO_GUARD 35/60, WARN_REPLAN_GUARD 42/60, HARD_STOP_GUARD 8/60.
- Patch generation: NO_GUARD 48/60, WARN_REPLAN_GUARD 54/60, HARD_STOP_GUARD 9/60.
- Mean steps: NO_GUARD 42.0, WARN_REPLAN_GUARD 40.2, HARD_STOP_GUARD 8.0.
- Mean tokens: NO_GUARD 758,692, WARN_REPLAN_GUARD 744,589, HARD_STOP_GUARD 67,225.

These values are reproduced by `python tests/test_reproduce_key_outputs.py`.
