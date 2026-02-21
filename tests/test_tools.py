"""Test the 3 tools with real data from project_resources (no LLM)."""
import pytest
from pathlib import Path

# Ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_customers_path, get_rules_path, get_status_log_path
from src.data import load_resources
from src.tools import get_customer_data, calculate_retention_offer, update_customer_status


@pytest.fixture(scope="module")
def resources_loaded():
    load_resources(get_customers_path(), get_rules_path())


def test_get_customer_data_found(resources_loaded):
    result = get_customer_data.invoke({"email": "sarah.chen@email.com"})
    assert result["found"] is True
    assert result.get("customer_id") == "CUST_001"
    assert result.get("tier") == "premium"


def test_get_customer_data_not_found(resources_loaded):
    result = get_customer_data.invoke({"email": "unknown@example.com"})
    assert result["found"] is False
    assert "message" in result


def test_calculate_retention_offer_financial_premium(resources_loaded):
    result = calculate_retention_offer.invoke({"customer_tier": "premium", "reason": "financial_hardship"})
    assert "offers" in result
    assert len(result["offers"]) >= 1
    assert any("pause" in str(o).lower() or "discount" in str(o).lower() for o in result["offers"])


def test_calculate_retention_offer_invalid_tier(resources_loaded):
    result = calculate_retention_offer.invoke({"customer_tier": "invalid", "reason": "financial"})
    assert result["offers"] == []
    assert "message" in result


def test_update_customer_status(resources_loaded):
    result = update_customer_status.invoke({"customer_id": "CUST_001", "action": "test_cancellation"})
    assert result.get("success") is True
    assert "message" in result
