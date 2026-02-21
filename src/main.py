"""CLI chat entrypoint. Load env, build graph, run REPL."""
import os
import sys
import time

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph
from src.log_config import log_session_summary, log_turn, setup_logging
from src.utils import extract_message_text, fallback_reply_for_route

load_dotenv()
setup_logging()

BANNER = """
=== TechFlow Customer Support Chat ===

What to type:
  - Your message in plain English (e.g. "I want to cancel my Care+", "My phone won't charge").
  - For personalized offers, include your email (e.g. sarah.chen@email.com). See project_resources/customers.csv for test emails.
  - Type 'quit', 'exit', or 'q' to end the chat.

Examples:
  "Hey, I can't afford the $13/month care+ anymore – sarah.chen@email.com"
  "This phone keeps overheating, I want to return it and cancel everything"
  "My phone won't charge anymore, tried different cables"

---
"""


def _format_error(e: Exception) -> str:
    """Return a clear, user-facing error message with hint if possible."""
    msg = str(e).strip()
    if "GOOGLE_API_KEY" in msg or "GEMINI_API_KEY" in msg or "API" in msg.lower():
        return (
            "[ERROR] Missing or invalid API key.\n"
            "  Fix: Copy .env.example to .env and set GOOGLE_API_KEY=your_key (get one at https://aistudio.google.com/apikey)."
        )
    if "tool()" in msg or "unexpected keyword" in msg:
        return (
            "[ERROR] Internal tool configuration failed.\n"
            "  This is a code bug (tool creation). Please report or check src/graph.py _make_policy_search_tool."
        )
    if "connection" in msg.lower() or "timeout" in msg.lower() or "network" in msg.lower():
        return f"[ERROR] Network or API unreachable.\n  Details: {msg}\n  Check your connection and try again."
    if "404" in msg or "NOT_FOUND" in msg or "model" in msg and "not found" in msg.lower():
        return (
            "[ERROR] Gemini model not found (404).\n"
            "  Try setting in .env: GEMINI_MODEL=gemini-2.0-flash (or gemini-1.5-pro).\n"
            "  List of models: https://ai.google.dev/gemini-api/docs/models"
        )
    return f"[ERROR] {msg}"


def main():
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] No API key set.", file=sys.stderr)
        print("  Copy .env.example to .env and set GOOGLE_API_KEY=your_key (https://aistudio.google.com/apikey).", file=sys.stderr)
        sys.exit(1)
    try:
        graph = build_graph()
    except Exception as e:
        print(_format_error(e), file=sys.stderr)
        sys.exit(1)
    print(BANNER)
    messages = []
    customer_data = None
    timings = []  # (cycle_time_s, api_time_s) per turn
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            if timings:
                n = len(timings)
                avg_cycle = sum(t[0] for t in timings) / n
                avg_api = sum(t[1] for t in timings) / n
                log_session_summary(n, avg_cycle, avg_api)
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            if timings:
                n = len(timings)
                avg_cycle = sum(t[0] for t in timings) / n
                avg_api = sum(t[1] for t in timings) / n
                log_session_summary(n, avg_cycle, avg_api)
            print("Goodbye.")
            break
        messages.append(HumanMessage(content=user_input))
        try:
            t0 = time.perf_counter()
            result = graph.invoke({
                "messages": messages,
                "customer_data": customer_data,
                "next_route": None,
                "_api_time_this_turn": 0.0,
            })
            cycle_s = time.perf_counter() - t0
            api_s = result.get("_api_time_this_turn") or 0.0
            log_turn(cycle_s, api_s)
            timings.append((cycle_s, api_s))
        except Exception as e:
            print(_format_error(e))
            continue
        messages = result["messages"]
        customer_data = result.get("customer_data") or customer_data
        reply = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                reply = extract_message_text(m.content)
                break
        if not reply:
            reply = fallback_reply_for_route(result.get("next_route") or "")
        print("Support:", reply or "(No response)")


if __name__ == "__main__":
    main()
