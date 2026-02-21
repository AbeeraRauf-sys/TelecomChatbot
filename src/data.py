"""Load and cache customers.csv and retention_rules.json once at startup."""
import csv
import json
from pathlib import Path
from typing import Any

_customers: list[dict[str, Any]] | None = None
_rules: dict[str, Any] | None = None


def load_resources(customers_path: Path, rules_path: Path) -> None:
    """Load CSV and JSON into module cache. Call once at startup."""
    global _customers, _rules
    _customers = []
    if customers_path.exists():
        with open(customers_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _customers = list(reader)
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as f:
            _rules = json.load(f)
    else:
        _rules = {}


def get_customers() -> list[dict[str, Any]]:
    if _customers is None:
        from src.config import get_customers_path, get_rules_path
        load_resources(get_customers_path(), get_rules_path())
    return _customers or []


def get_rules() -> dict[str, Any]:
    if _rules is None:
        from src.config import get_customers_path, get_rules_path
        load_resources(get_customers_path(), get_rules_path())
    return _rules or {}
