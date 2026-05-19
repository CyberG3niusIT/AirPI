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

## Baseline Result

| Case | OK | Seconds | Tokens | tok/sec |
| --- | --- | ---: | ---: | ---: |
| cold_or_first | yes | 7.971 | 64 | 8.03 |
| warm | yes | 7.517 | 64 | 8.51 |
| session_kv_reuse | yes | 7.702 | 64 | 8.31 |

## Turbo Profile Result

| Field | Value |
| --- | --- |
| Date | 2026-05-19 |
| Profile | Pi 5 CPU-only turbo |
| Command | `.venv/bin/python scripts/airpi_bench.py --base-url http://127.0.0.1:11435 --model qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --format markdown` |
| Key settings | `AIRPI_N_THREADS=3`, `AIRPI_N_CTX_SMALL=2048`, `AIRPI_N_THREADS_BATCH=4`, `AIRPI_N_BATCH_SMALL=1024`, `AIRPI_N_UBATCH_SMALL=512`, `AIRPI_FLASH_ATTN=true` |

| Case | OK | Seconds | Tokens | tok/sec |
| --- | --- | ---: | ---: | ---: |
| cold_or_first | yes | 5.939 | 64 | 10.78 |
| warm | yes | 6.002 | 64 | 10.66 |
| session_kv_reuse | yes | 5.864 | 64 | 10.91 |

## Comparison

| Case | Baseline tok/sec | Turbo tok/sec | Gain |
| --- | ---: | ---: | ---: |
| cold_or_first | 8.03 | 10.78 | +34.2% |
| warm | 8.51 | 10.66 | +25.3% |
| session_kv_reuse | 8.31 | 10.91 | +31.3% |

## Notes

- The service was already warm from the controlled live verification.
- The benchmark uses short deterministic local requests through the HTTP API.
- Metrics before this report were already non-zero because live verification generated two short requests first.
- The turbo profile improves throughput without changing the model file.
- The 15 tok/sec target likely needs a smaller or more aggressive quantized model, llama.cpp build-level optimization, or both.
