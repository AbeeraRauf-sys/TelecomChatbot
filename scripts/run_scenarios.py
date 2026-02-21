#!/usr/bin/env python3
"""Run the 5 assignment test scenarios and print Support response (text only)."""
import os
import sys

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from src.graph import build_graph
from src.utils import extract_message_text

SCENARIOS = [
    {
        "name": "Test 1: Money Problems",
        "user_input": "hey can't afford the $13/month care+ anymore, need to cancel",
        "hint": "Try to help with payment pause or discount before canceling; use business rules + customer data.",
    },
    {
        "name": "Test 2: Phone Problems",
        "user_input": "this phone keeps overheating, want to return it and cancel everything",
        "hint": "Offer phone replacement or upgrade before canceling; return policy + tech guide.",
    },
    {
        "name": "Test 3: Questioning Value",
        "user_input": "paying for care+ but never used it, maybe just get rid of it?",
        "hint": "Explain what they get for their money, offer cheaper options; Care+ benefits.",
    },
    {
        "name": "Test 4: Technical Help Needed",
        "user_input": "my phone won't charge anymore, tried different cables",
        "hint": "Send to tech support only; no retention pitch.",
    },
    {
        "name": "Test 5: Billing Question",
        "user_input": "got charged $15.99 but thought care+ was $12.99, what's the extra?",
        "hint": "Send to billing; not cancellation.",
    },
]


def main():
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    graph = build_graph()
    for s in SCENARIOS:
        print("\n" + "=" * 60)
        print(s["name"])
        print("=" * 60)
        print("You:", s["user_input"])
        print("Expected:", s["hint"])
        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=s["user_input"])],
                "customer_data": None,
                "next_route": None,
            })
            messages = result["messages"]
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.content:
                    text = extract_message_text(m.content)
                    print("Support:", text)
                    break
        except Exception as e:
            print("Support: [ERROR]", str(e))
    print("\n" + "=" * 60)
    print("Done. Review responses above against expected behavior.")


if __name__ == "__main__":
    main()
