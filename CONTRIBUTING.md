# Contributing to AirPI

Thanks for helping make AirPI better.

## Development Checks

```bash
python -m unittest discover -s tests
scripts/airpi_bench.py --dry-run
```

## Contribution Rules

- Keep AirPI local-first and Raspberry-Pi-friendly.
- Avoid new dependencies unless they remove real complexity.
- Do not commit model files, logs, caches, secrets or local benchmark outputs.
- Keep Ollama-compatible behavior stable for `/api/generate` and `/api/tags`.
- Add tests for API contract changes and recovery behavior.

## Releases

AirPI uses SemVer. Release notes should mention compatibility changes, operational changes and benchmark context when performance numbers are updated.
