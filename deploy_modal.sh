#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHONPATH=src python -m checkpoint_bootstrap
modal deploy modal_app.py "$@"
