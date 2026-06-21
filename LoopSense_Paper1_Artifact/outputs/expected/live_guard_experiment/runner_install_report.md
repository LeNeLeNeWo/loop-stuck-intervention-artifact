# Live Guard Runner Validation

Status: **READY**

## Runner

- Selected runner: mini-swe-agent existing local package.
- Package path: `experiments\paper1_loopbench\stage_p1_6_validity_expansion\closed_loop_mini_validation\backend_acquisition_and_task_reconciliation\python_packages`
- CLI help check: pass

## Model Configuration

- Model: `deepseek-v4-flash`
- LiteLLM model string: `openai/deepseek-v4-flash`
- Endpoint: `https://api.deepseek.com/<masked-path>`
- API key: present; value not printed
- Config: `outputs\paper1_live_guard_experiment\runner_config\deepseek_litellm_swebench.yaml`
- LiteLLM smoke call: pass

## Infrastructure

- Docker daemon: pass
- datasets package: pass

## Notes

- No external runner repository was cloned because an existing mini-swe-agent package is available.
- This validation does not run SWE-bench tasks.
- Official task-success oracle availability is checked separately by the analysis/stack scripts.
