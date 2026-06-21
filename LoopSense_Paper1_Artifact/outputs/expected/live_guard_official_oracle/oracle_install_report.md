# SWE-bench Official Oracle Validation

Decision: **OFFICIAL_ORACLE_READY**

## Harness

- Source: existing local SWE-bench package from prior backend-repair virtual environment.
- Python: `<LOCAL_PATH>
- Import check: pass
- CLI help check: pass
- Docker: pass
- Dataset: `princeton-nlp/SWE-Bench_Verified:test`

## No-op Smoke

- Requested: yes
- Status: pass
- Instance: `pydata__xarray-3095`

## Smoke Output Excerpt

```text
FO - HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=princeton-nlp/SWE-Bench_Verified "HTTP/1.1 404 Not Found"
2026-06-15 12:13:20,019 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/datasets/princeton-nlp/SWE-Bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data?recursive=true&expand=false "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:20,314 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-06-15 12:13:20,753 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/datasets/princeton-nlp/SWE-Bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a?recursive=false&expand=false "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:21,348 - httpx - INFO - HTTP Request: GET https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a?recursive=false&expand=false "HTTP/1.1 200 OK"
2026-06-15 12:13:21,640 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/dataset_infos.json "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:21,928 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/dataset_infos.json "HTTP/1.1 404 Not Found"
2026-06-15 12:13:22,346 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:22,633 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:22,762 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/princeton-nlp/SWE-bench_Verified/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/README.md "HTTP/1.1 200 OK"
2026-06-15 12:13:23,052 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/SWE-Bench_Verified.py "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:23,350 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/SWE-Bench_Verified.py "HTTP/1.1 404 Not Found"
2026-06-15 12:13:24,353 - httpx - INFO - HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/princeton-nlp/SWE-Bench_Verified/princeton-nlp/SWE-Bench_Verified.py "HTTP/1.1 404 Not Found"
2026-06-15 12:13:24,645 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/.huggingface.yaml "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:24,935 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-06-15 12:13:30,617 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/dataset_infos.json "HTTP/1.1 307 Temporary Redirect"
2026-06-15 12:13:30,905 - httpx - INFO - HTTP Request: HEAD https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/dataset_infos.json "HTTP/1.1 404 Not Found"
No instances to run.
Cleaning cached images...
Removed 0 images.
Total instances: 1
Instances submitted: 1
Instances completed: 0
Instances incomplete: 0
Instances resolved: 0
Instances unresolved: 0
Instances with empty patches: 1
Instances with errors: 0
Unstopped containers: 0
Unremoved images: 1
Report written to noop_smoke.official_oracle_noop_smoke.json

```
