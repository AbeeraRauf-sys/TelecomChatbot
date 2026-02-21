"""Configuration: paths and env. All paths resolved from repo root."""
import os
from pathlib import Path

# Project root (directory containing pyproject.toml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_resources_dir() -> Path:
    return Path(os.environ.get("RESOURCES_DIR", PROJECT_ROOT / "project_resources"))

def get_customers_path() -> Path:
    return get_resources_dir() / "customers.csv"

def get_rules_path() -> Path:
    return get_resources_dir() / "retention_rules.json"

def get_policy_docs_dir() -> Path:
    return get_resources_dir() / "policy_documents"

def get_playbook_path() -> Path:
    return get_resources_dir() / "retention_playbook.md"

def get_status_log_path() -> Path:
    raw = os.environ.get("STATUS_LOG_PATH", "status_logs/actions.log")
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
