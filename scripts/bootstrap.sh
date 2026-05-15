#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-nlp_game}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

mkdir -p data/raw data/processed outputs runs checkpoints

echo
echo "Bootstrap complete."
echo "Activate with: conda activate $ENV_NAME"
echo "Then run: scripts/download_models.sh"
