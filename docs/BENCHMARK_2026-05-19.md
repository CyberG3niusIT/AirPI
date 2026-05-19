# AirPI Benchmark 2026-05-19

## Context

| Field | Value |
| --- | --- |
| Host | Raspberry Pi |
| Date | 2026-05-19 |
| Service | `airpi.service` |
| Base URL | `http://127.0.0.1:11435` |
| Model | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` |
| Python | AirPI project virtualenv |
| Command | `.venv/bin/python scripts/airpi_bench.py --base-url http://127.0.0.1:11435 --model qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --format markdown` |

## Result

| Case | OK | Seconds | Tokens | tok/sec |
| --- | --- | ---: | ---: | ---: |
| cold_or_first | yes | 7.971 | 64 | 8.03 |
| warm | yes | 7.517 | 64 | 8.51 |
| session_kv_reuse | yes | 7.702 | 64 | 8.31 |

## Notes

- The service was already warm from the controlled live verification.
- The benchmark uses short deterministic local requests through the HTTP API.
- Metrics before this report were already non-zero because live verification generated two short requests first.
