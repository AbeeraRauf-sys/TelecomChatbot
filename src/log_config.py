"""Structured logging for response times and tool calls."""
import logging
import sys
from datetime import datetime

LOG = logging.getLogger("techflow")

def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure techflow logger to stderr with timestamps."""
    if LOG.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(level)

def log_llm_response(api_time_s: float) -> None:
    setup_logging()
    LOG.info("LLM response | api_time=%.2fs", api_time_s)

def log_tool_call(name: str, args: dict) -> None:
    setup_logging()
    a = {k: str(v)[:120] for k, v in (args or {}).items()}
    LOG.info("Tool call | name=%s | args=%s", name, a)

def log_tool_result(name: str, result) -> None:
    setup_logging()
    s = str(result)[:300]
    LOG.info("Tool result | name=%s | result=%s", name, s)

def log_agent_step(step: str, detail: str = "") -> None:
    setup_logging()
    LOG.debug("Agent step | %s%s", step, f" | {detail}" if detail else "")

def log_context(label: str, content: str) -> None:
    setup_logging()
    LOG.debug("Context [%s] | %s", label, content[:500])

def log_reply(source: str, content) -> None:
    setup_logging()
    s = str(content)[:300] if content else "(empty)"
    LOG.info("Reply [%s] | %s", source, s)

def log_turn(cycle_time_s: float, api_time_s: float) -> None:
    setup_logging()
    LOG.info("Turn | cycle_time=%.2fs | api_time=%.2fs", cycle_time_s, api_time_s)

def log_session_summary(turn_count: int, avg_cycle_s: float, avg_api_s: float) -> None:
    setup_logging()
    LOG.info("Session end | turns=%d | avg_cycle_time=%.2fs | avg_api_time=%.2fs", turn_count, avg_cycle_s, avg_api_s)
