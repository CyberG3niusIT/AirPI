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
