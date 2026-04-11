#!/usr/bin/env bash
# Thin launcher: activates venv and runs the bot.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "venv not found. Run ./install.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

exec python run.py "$@"
