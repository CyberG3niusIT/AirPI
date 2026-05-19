# AirPI API Contract

AirPI intentionally keeps the v0.3 API small. The stable compatibility surface is:

- `GET /live`
- `GET /ready`
- `GET /health`
- `GET /metrics`
- `GET /api/tags`
- `POST /api/generate`

## Error Format

HTTP errors use this shape:

```json
{
  "detail": {
    "error": {
      "code": "MODEL_NOT_FOUND",
      "message": "Model is not available",
      "retryable": false,
      "request_id": "..."
    }
  }
}
```

Streaming errors are sent as the final NDJSON frame:

```json
{"model":"...","response":"","done":true,"error":{"code":"INFERENCE_FAILED","message":"Inference failed","retryable":true,"request_id":"..."}}
```

## Error Codes

| Code | Retryable | Meaning |
| --- | --- | --- |
| `INVALID_REQUEST` | no | The request body failed validation. |
| `MODEL_NOT_FOUND` | no | The requested model file is not available. |
| `QUEUE_FULL` | yes | AirPI rejected the request before inference because the queue is full. |
| `INFERENCE_FAILED` | yes | The model call failed after validation. |
| `MODEL_RECOVERED` | yes | Reserved for explicit recovery reporting. |
| `SERVICE_NOT_READY` | yes | Reserved for readiness failures. |

## AirPI Extensions

`session_id` is an optional AirPI extension for KV-cache reuse. Normal Ollama-compatible clients do not need it.

`preferred_model` is an optional AirPI extension for routing. Supported built-in aliases:

| Alias | Target |
| --- | --- |
| `fast`, `fast-lane`, `airpi-fast` | `AIRPI_FAST_MODEL` |
| `default` | `AIRPI_DEFAULT_MODEL` |
| `large` | `AIRPI_LARGE_MODEL` |

The same aliases are also accepted in the `model` field for AirPI-aware clients.

Structured JSON output is available for non-streaming requests:

```json
{
  "model": "fast",
  "prompt": "Return a route decision",
  "stream": false,
  "format": "json",
  "required_json_keys": ["decision", "risk", "reason"]
}
```

`response_format: "json_object"` is accepted as an AirPI-specific equivalent to `format: "json"`.

When JSON mode is enabled, AirPI prompts the model for a single JSON object, validates the response, retries once with a repair prompt, and returns compact JSON in `response`. If the repaired output is still invalid, AirPI returns `INFERENCE_FAILED` using the normal error contract.
