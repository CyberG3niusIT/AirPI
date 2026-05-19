# AirPI Operations

## Local Checks

```bash
scripts/airpi_doctor.py --base-url http://127.0.0.1:11435
scripts/airpi_doctor.py --generate --json
```

## Benchmarks

```bash
scripts/airpi_bench.py --dry-run --format markdown
scripts/airpi_bench.py --model qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --format json
scripts/airpi_guardian_smoke.py --format markdown
```

The benchmark script reports local measurements only. Publish numbers together with model name, hardware, Python version and llama-cpp build flags.

The PI Guardian smoke script is a quality gate. It can intentionally return a non-zero exit code when model output is unsafe for routing.

Recorded local benchmark reports:

- [BENCHMARK_2026-05-19.md](BENCHMARK_2026-05-19.md)
- [MODEL_SHOOTOUT_2026-05-19.md](MODEL_SHOOTOUT_2026-05-19.md)
- [PI_GUARDIAN_SMOKE_2026-05-19.md](PI_GUARDIAN_SMOKE_2026-05-19.md)

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

## Fast Lane

AirPI exposes a configurable fast-lane model through `AIRPI_FAST_MODEL`. AirPI-aware clients can request it with `model="fast"` or `preferred_model="fast"`.

`/ready` requires both `AIRPI_DEFAULT_MODEL` and `AIRPI_FAST_MODEL` to exist. This keeps PI Guardian from seeing the service as ready while its default fast-lane request would fail.

Current measured fast-lane candidate:

| Setting | Value |
| --- | --- |
| `AIRPI_FAST_MODEL` | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` |
| Warm HTTP benchmark | about `28 tok/sec` |
| Intended use | low-latency PI Guardian status, routing and safety smoke tasks |

Keep the higher-quality default model active for coding-heavy work until the PI Guardian smoke test remains acceptable over real prompts.

For structured PI Guardian route decisions, use non-streaming JSON mode with `format="json"` and `required_json_keys`.

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
