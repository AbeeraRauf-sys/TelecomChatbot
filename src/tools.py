"""Three required LangChain tools: get_customer_data, calculate_retention_offer, update_customer_status."""
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from src.config import get_status_log_path
from src.data import get_customers, get_rules
from src.utils import extract_email_or_cust_id


# --- Reason mapping for retention_rules.json ---
# reason (user-facing) -> (top_key, tier_key_suffix)
REASON_MAP = {
    "financial_hardship": ("financial_hardship", None),
    "financial": ("financial_hardship", None),
    "money": ("financial_hardship", None),
    "afford": ("financial_hardship", None),
    "overheating": ("product_issues", "overheating"),
    "battery": ("product_issues", "battery_issues"),
    "battery_issues": ("product_issues", "battery_issues"),
    "product_issues": ("product_issues", "overheating"),  # default product to overheating
    "service_value": ("service_value", "care_plus_premium"),
    "value": ("service_value", "care_plus_premium"),
}
TIER_KEY = {"premium": "premium_customers", "regular": "regular_customers", "new": "new_customers"}


@tool
def get_customer_data(email: str) -> dict:
    """Load customer profile from customers.csv. Pass ONLY the customer's email (e.g. sarah.chen@email.com) OR customer_id (e.g. CUST_001). If the user's message contains an email or CUST_ id, extract that value and use it—do not ask again. Do NOT pass long request text; if you only have the full message, the tool will try to extract an email/CUST_ from it. Returns customer_id, name, tier, plan_type, device, etc. If not found returns found=False."""
    raw = (email or "").strip()
    if not raw:
        return {"found": False, "message": "No email or customer_id provided."}
    # If the model passed the whole user message, extract the first email or CUST_ id
    lookup = extract_email_or_cust_id(raw)
    if lookup:
        raw = lookup
    raw_lower = raw.lower()
    # Reject if still no email-like or cust_ value
    if "@" not in raw and not raw_lower.startswith("cust_"):
        return {
            "found": False,
            "message": "Invalid input: pass only an email (e.g. name@email.com) or customer_id (e.g. CUST_001). Do not pass the customer's request or question text. If the customer has not given an email or ID yet, ask them for it.",
        }
    customers = get_customers()
    # Lookup by customer_id (e.g. CUST_001)
    if raw_lower.startswith("cust_"):
        for row in customers:
            if (row.get("customer_id") or "").strip().lower() == raw_lower:
                out = {"found": True, **{k: v for k, v in row.items()}}
                return out
        res = {"found": False, "message": f"No customer found with customer_id {raw}."}
        return res
    # Lookup by email
    for row in customers:
        if (row.get("email") or "").strip().lower() == raw_lower:
            out = {"found": True, **{k: v for k, v in row.items()}}
            return out
    res = {"found": False, "message": f"No customer found with email {raw}. Valid examples: sarah.chen@email.com, jennifer.lee@email.com, or CUST_001."}
    return res


@tool
def calculate_retention_offer(customer_tier: str, reason: str) -> dict:
    """Generate offers using retention_rules.json.
    customer_tier must be one of: premium, regular, new.
    reason describes why they want to cancel: e.g. financial_hardship, overheating, battery_issues, service_value.
    Returns dict with 'offers' (list of offers) and optional 'message' if invalid inputs."""
    tier = (customer_tier or "").strip().lower()
    reason_raw = (reason or "").strip().lower().replace(" ", "_")
    if tier not in TIER_KEY:
        return {
            "offers": [],
            "message": f"Invalid customer_tier: {customer_tier}. Use premium, regular, or new.",
        }
    rules = get_rules()
    if not rules:
        return {"offers": [], "message": "Retention rules not available."}

    # Resolve reason to (top_key, sub_key)
    top_key, sub_key = None, None
    for r, (tk, sk) in REASON_MAP.items():
        if r in reason_raw or r in reason_raw.replace("_", ""):
            top_key, sub_key = tk, sk
            break
    if not top_key:
        top_key, sub_key = "financial_hardship", None  # default

    tier_key = TIER_KEY[tier]
    offers = []
    if top_key == "financial_hardship" and top_key in rules:
        tier_offers = rules[top_key].get(tier_key, [])
        offers = list(tier_offers)
    elif top_key == "product_issues" and top_key in rules:
        sub_key = sub_key or "overheating"
        tier_offers = rules[top_key].get(sub_key, [])
        offers = list(tier_offers)
    elif top_key == "service_value" and top_key in rules:
        sub_key = sub_key or "care_plus_premium"
        tier_offers = rules[top_key].get(sub_key, [])
        offers = list(tier_offers)

    return {"offers": offers, "customer_tier": tier, "reason": reason_raw}


@tool
def update_customer_status(customer_id: str, action: str) -> dict:
    """Process cancellations/changes: log to file. Call when customer confirms cancellation or a status change.
    customer_id: e.g. CUST_001. action: e.g. 'cancellation', 'pause', 'downgrade'."""
    customer_id = (customer_id or "").strip()
    action = (action or "").strip()
    if not customer_id or not action:
        return {"success": False, "message": "customer_id and action are required."}
    log_path = get_status_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            ts = datetime.now(tz=timezone.utc).isoformat()
            f.write(f"{ts}\t{customer_id}\t{action}\n")
        return {"success": True, "message": f"Logged {action} for {customer_id}."}
    except OSError as e:
        return {"success": False, "message": f"Failed to write log: {e}. Please contact support."}
