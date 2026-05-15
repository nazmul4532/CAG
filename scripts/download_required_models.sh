#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OLLAMA_MODELS="${OLLAMA_MODELS:-qwen3:8b qwen3:14b qwen3:4b}" \
HF_MODELS="${HF_MODELS:-albert-base-v2 sentence-transformers/all-MiniLM-L6-v2}" \
  scripts/download_models.sh
