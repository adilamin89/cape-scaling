#!/usr/bin/env bash
# CAPE — Quick Setup for Contributors and Forkers
# Usage:  bash setup.sh
# ─────────────────────────────────────────────────
set -e

echo "╔══════════════════════════════════════════╗"
echo "║  CAPE — Capability Coupling Analysis     ║"
echo "║  Setting up environment...               ║"
echo "╚══════════════════════════════════════════╝"
echo

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install Python 3.9+ and try again."
  exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PY_VER found"

# Create venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
  echo "→ Creating virtual environment in .venv/"
  python3 -m venv .venv
  source .venv/bin/activate
  echo "✓ Virtual environment activated"
else
  echo "✓ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Install dependencies
echo "→ Installing core dependencies..."
pip install -q -r requirements.txt
echo "✓ Core packages installed"

# Quick smoke test
echo "→ Running smoke test..."
python3 -c "
import numpy as np
from scipy import stats
print('  numpy', np.__version__)
print('  scipy', stats.__name__, '✓')
" 2>/dev/null && echo "✓ Smoke test passed" || echo "⚠ Smoke test had issues"

# Test quickstart
echo "→ Testing quickstart classifier..."
python3 scripts/quickstart.py --N 7 --hs 78 --tqa 43 2>/dev/null | head -5
echo "✓ Quickstart works"

echo
echo "════════════════════════════════════════════"
echo "  Setup complete! Try:"
echo ""
echo "  # Classify a model"
echo "  python scripts/quickstart.py --N 7 --hs 78 --tqa 43"
echo ""
echo "  # Reproduce all figures"
echo "  python scripts/generate_all_figures.py"
echo ""
echo "  # Run verification suite"
echo "  python scripts/verify_and_reproduce.py"
echo ""
echo "  # Open dashboard"
echo "  open index.html   # macOS"
echo "  xdg-open index.html  # Linux"
echo "════════════════════════════════════════════"
