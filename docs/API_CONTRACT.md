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
