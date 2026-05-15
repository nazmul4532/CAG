#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OLLAMA_MODELS="${OLLAMA_MODELS:-qwen3:8b qwen3:14b gemma3:12b-it-qat mistral-nemo:12b}"
HF_MODELS="${HF_MODELS:-albert-base-v2 sentence-transformers/all-MiniLM-L6-v2}"

echo "Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed or not on PATH." >&2
  exit 1
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama is installed, but the local server is not responding at 127.0.0.1:11434." >&2
  echo "Try: sudo systemctl start ollama" >&2
  exit 1
fi

echo "Pulling Ollama models: $OLLAMA_MODELS"
for model in $OLLAMA_MODELS; do
  echo
  echo "==> ollama pull $model"
  ollama pull "$model"
done

echo
echo "Downloading Hugging Face models into the local cache: $HF_MODELS"
python - "$HF_MODELS" <<'PY'
import sys
from huggingface_hub import snapshot_download

for model_id in sys.argv[1].split():
    print(f"\n==> huggingface snapshot_download {model_id}")
    snapshot_download(repo_id=model_id)
PY

echo
echo "Model download complete."
echo "Installed Ollama models:"
ollama list
