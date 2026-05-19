#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
MODEL="${AIRPI_MODEL:-qwen2.5-coder-1.5b-q4_k_m.gguf}"

curl -sS -X POST "${BASE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: pi-guardian-demo" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"You are a local router assistant. Answer with a one line status summary.\",
    \"stream\": false,
    \"max_tokens\": 48,
    \"session_id\": \"pi-guardian-demo\"
  }"
