# AirPI Operations

## Local Checks

```bash
scripts/airpi_doctor.py --base-url http://127.0.0.1:11435
scripts/airpi_doctor.py --generate --json
```

## Benchmarks

```bash
scripts/airpi_bench.py --dry-run --format markdown
scripts/airpi_bench.py --model qwen2.5-coder-1.5b-q4_k_m.gguf --format json
```

The benchmark script reports local measurements only. Publish numbers together with model name, hardware, Python version and llama-cpp build flags.

Recorded local benchmark reports:

- [BENCHMARK_2026-05-19.md](BENCHMARK_2026-05-19.md)

## Pi 5 Performance Profile

The default service profile is tuned for Raspberry Pi 5 CPU-only inference with the 1.5B Q4_K_M model:

| Setting | Value | Reason |
| --- | --- | --- |
| `AIRPI_N_THREADS` | `3` | Leaves one core for the server, kernel and paging work. |
| `AIRPI_N_CTX_SMALL` | `2048` | Keeps the KV cache smaller for local assistant workloads. |
| `AIRPI_N_THREADS_BATCH` | `4` | Uses all cores during prompt ingestion. |
| `AIRPI_N_BATCH_SMALL` | `1024` | Improves prompt ingestion without high memory pressure. |
| `AIRPI_N_UBATCH_SMALL` | `512` | Keeps micro batches conservative on 8 GB RAM. |
| `AIRPI_FLASH_ATTN` | `true` | Measured faster with llama-cpp-python 0.3.23 on this Pi. |

To prioritize longer context over speed, override `AIRPI_N_CTX_SMALL` in `/etc/airpi/airpi.env` and restart the service after `systemd-analyze verify`.

## Health Model

- `/live` means the process can answer.
- `/ready` means model storage and queue state look usable.
- `/health` reports runtime state, sessions, loaded models and recovery counters.
- `/metrics` exports lightweight Prometheus text.

## systemd

The repository unit reads optional overrides from `/etc/airpi/airpi.env`. Do not restart the live service after changing the unit until the unit has been checked with:

```bash
systemd-analyze verify systemd/airpi.service
```
