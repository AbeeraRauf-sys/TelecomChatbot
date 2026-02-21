#!/usr/bin/env bash
# TechFlow Support Chat – single entrypoint for all commands.
# Usage: ./run.sh [install|cli|streamlit|scenarios]

set -e
cd "$(dirname "$0")"

case "${1:-}" in
  install)
    if command -v uv &>/dev/null; then
      uv sync
    else
      pip install -e .
    fi
    echo "Done. Run: ./run.sh cli  or  ./run.sh streamlit  or  ./run.sh scenarios"
    ;;
  cli)
    if command -v uv &>/dev/null; then
      uv run python -m src.main
    else
      python -m src.main
    fi
    ;;
  streamlit)
    if command -v uv &>/dev/null; then
      uv run streamlit run streamlit_app.py
    else
      streamlit run streamlit_app.py
    fi
    ;;
  scenarios)
    if command -v uv &>/dev/null; then
      uv run python scripts/run_scenarios.py
    else
      python scripts/run_scenarios.py
    fi
    ;;
  *)
    echo "Usage: $0 {install|cli|streamlit|scenarios}"
    echo "  install   - Install dependencies (uv sync or pip install -e .)"
    echo "  cli       - Run CLI chat (python -m src.main)"
    echo "  streamlit - Run Streamlit web UI"
    echo "  scenarios - Run the 5 assignment test scenarios"
    exit 1
    ;;
esac
