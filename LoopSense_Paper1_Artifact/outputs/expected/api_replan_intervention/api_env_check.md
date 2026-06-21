# DeepSeek API Environment Check

Status: **READY**

## Environment

- `DEEPSEEK_BASE_URL`: present
- `DEEPSEEK_API_KEY`: present; value not printed
- `DEEPSEEK_MODEL`: deepseek-v4-flash
- Endpoint: `https://api.deepseek.com/<masked-path>/chat/completions`

## Test Call

- HTTP status: 200
- Latency seconds: 1.337
- Retried without response_format: yes
- Token usage: `{"completion_tokens": 80, "completion_tokens_details": {"reasoning_tokens": 73}, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 16, "prompt_tokens": 16, "prompt_tokens_details": {"cached_tokens": 0}, "total_tokens": 96}`
- JSON parse: success

## Response Content

```json
{"pong": true}
```
