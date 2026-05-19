#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${AIRPI_BASE_URL:-http://127.0.0.1:11435}"
MODEL="${AIRPI_MODEL:-qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"

curl -sS -N -X POST "${BASE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Write a tiny Python hello world.\",
    \"stream\": true,
    \"max_tokens\": 64
  }"
