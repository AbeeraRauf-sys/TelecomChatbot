"""Shared helpers: extract plain text from LLM message content, email/CUST from chat."""
import re
from typing import Any

def extract_email_or_cust_id(raw: str) -> str | None:
    """Extract a single email or CUST_ id from text. Use on user message or tool input."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    m = re.search(r"\bcust_\w+\b", s, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m = re.search(r"\b[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}\b", s)
    if m:
        return m.group(0).strip()
    return None


def _finish_if_chopped(text: str) -> str:
    """If text looks truncated (long, no sentence end), append ellipsis so we don't show a cut-off word."""
    if not text or len(text) < 200:
        return text
    t = text.rstrip()
    if t and t[-1] not in ".?!\"'":
        return t + "…"
    return text


def sanitize_internal_jargon(text: str) -> str:
    """Replace leaked internal phrases so the user never sees 'route has been set' etc. Safe to call on any display string."""
    if not text or not isinstance(text, str):
        return text or ""
    t = text.strip().lower()
    # Catch "Your route has been set to X" and similar
    if "route" in t and ("has been set" in t or "set to " in t):
        return "Is there anything else I can help you with?"
    if "set to end" in t or "set to retention" in t or "set to cancel" in t or "set to billing" in t or "set to tech" in t:
        return "Is there anything else I can help you with?"
    return text


def extract_message_text(content: Any) -> str:
    """Get displayable text from AIMessage.content. Handles string or list-of-blocks (e.g. Gemini)."""
    if content is None:
        return ""
    raw = ""
    if isinstance(content, str):
        raw = _finish_if_chopped(content.strip())
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(str(block["text"]).strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        raw = _finish_if_chopped("\n".join(p for p in parts if p))
    else:
        raw = _finish_if_chopped(str(content).strip())
    return sanitize_internal_jargon(raw)


def fallback_reply_for_route(route: str) -> str:
    """Customer-engaging fallback when the model returns no text. No internal jargon."""
    r = (route or "").strip().lower()
    if r == "billing":
        return "I'm checking your billing details now—I'll have an answer for you in just a moment."
    if r == "retention":
        return "I'm pulling together some options that might work better for you—give me a moment."
    if r == "cancel":
        return "I've got your request. One moment while I take care of that."
    if r == "tech":
        return "I'm looking up the best steps for you—one moment."
    if r == "end":
        return "Thanks for reaching out. We're all set here; reach out anytime if something else comes up."
    return "I'm on it—one moment and I'll get back to you."
