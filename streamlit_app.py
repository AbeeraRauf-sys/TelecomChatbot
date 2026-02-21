"""Streamlit chat UI for TechFlow Support. Run: streamlit run streamlit_app.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import streamlit as st

load_dotenv()

from src.graph import build_graph
from src.log_config import log_session_summary, log_turn, setup_logging
from src.utils import extract_message_text, fallback_reply_for_route, sanitize_internal_jargon

setup_logging()
st.set_page_config(page_title="TechFlow Support", page_icon="💬", layout="centered")

DEMO_SCENARIOS = [
    {
        "id": "t1",
        "name": "Test 1: Money Problems",
        "expected": "Try payment pause or discount before canceling (business rules + customer data).",
        "turns": [
            "hey can't afford the $13/month care+ anymore, need to cancel – sarah.chen@email.com",
            "how do I get the payment pause?",
            "no, just cancel please.",
        ],
    },
    {
        "id": "t2",
        "name": "Test 2: Phone Problems",
        "expected": "Offer replacement/upgrade before canceling (return policy + tech guide).",
        "turns": [
            "this phone keeps overheating, want to return it and cancel everything – michael.davis@email.com",
            "what does the replacement involve?",
            "no, just cancel everything.",
        ],
    },
    {
        "id": "t3",
        "name": "Test 3: Questioning Value",
        "expected": "Explain Care+ value and offer cheaper options (Care+ benefits).",
        "turns": [
            "paying for care+ but never used it, maybe just get rid of it? – jennifer.lee@email.com",
            "tell me more about the cheaper option",
            "no, just cancel.",
        ],
    },
    {
        "id": "t4",
        "name": "Test 4: Technical Help Needed",
        "expected": "Tech support only; no retention pitch.",
        "turns": [
            "my phone won't charge anymore, tried different cables",
            "still not charging after trying that",
            "ok thanks",
        ],
    },
    {
        "id": "t5",
        "name": "Test 5: Billing Question",
        "expected": "Billing help only; not cancellation.",
        "turns": [
            "got charged $15.99 but thought care+ was $12.99, what's the extra? – sarah.chen@email.com",
            "how do I get that adjusted?",
        ],
    },
]

# First message on screen: from assistant (TechFlow-specific, matches project scope)
ASSISTANT_GREETING = "Hi there! Welcome to TechFlow Electronics support. I can help with your Care+ plan, billing questions, or device issues. How can I help you today?"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_GREETING}]
if "customer_data" not in st.session_state:
    st.session_state.customer_data = None
if "timings" not in st.session_state:
    st.session_state.timings = []
if "demo_results" not in st.session_state:
    st.session_state.demo_results = {}
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"  # "chat" = single chat, "demo" = demo runner + results
# Reset each run so bottom "Demo scenario transcripts" only shows when we didn't just stream
st.session_state.demo_streamed_this_run = False

def _clear_chat():
    if st.session_state.timings:
        n = len(st.session_state.timings)
        avg_c = sum(t[0] for t in st.session_state.timings) / n
        avg_a = sum(t[1] for t in st.session_state.timings) / n
        log_session_summary(n, avg_c, avg_a)
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_GREETING}]
    st.session_state.timings = []
    st.session_state.customer_data = None
    st.rerun()

with st.sidebar:
    st.subheader("Mode")
    view_mode = st.radio(
        "Choose view",
        options=["chat", "demo"],
        format_func=lambda x: "Single chat" if x == "chat" else "Demo runner",
        key="view_mode_radio",
        horizontal=True,
    )
    st.session_state.view_mode = view_mode

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        _clear_chat()

    if view_mode == "demo":
        st.divider()
        st.subheader("Demo runner")
        st.caption("Auto-run the 5 assignment tests (multi-turn) for Loom recording.")
        run_all = st.button("Run all demo scenarios", use_container_width=True)
        chosen = None
        if not run_all:
            for s in DEMO_SCENARIOS:
                if st.button(f"Run {s['name']}", use_container_width=True):
                    chosen = s["id"]
                    break
        if run_all:
            chosen = "all"
        if chosen:
            st.session_state.demo_to_run = chosen
        if st.session_state.demo_results and st.button("Clear demo results", use_container_width=True):
            st.session_state.demo_results = {}
            st.rerun()

st.title("TechFlow Customer Support")
st.caption("Care+ and support. Say what you need (e.g. cancel, phone issue, billing). Include your email for personalized offers.")

# Clear chat button in main area for visibility (single-chat view)
if st.session_state.get("view_mode") == "chat":
    if st.button("Clear chat", key="clear_chat_main"):
        _clear_chat()

@st.cache_resource
def get_graph():
    return build_graph()


# Build graph (and RAG/embeddings) at app load so first message isn't slow
with st.spinner("Loading support assistant..."):
    get_graph()

def _render_scenario_into(placeholder, scenario: dict, transcript: list[dict]) -> None:
    """Write scenario name, expected, and transcript rows into a Streamlit placeholder (for per-turn streaming)."""
    with placeholder.container():
        st.subheader(scenario["name"])
        st.caption(f"Expected: {scenario['expected']}")
        for row in transcript:
            st.markdown(f"**Turn {row['turn']} — User:** {row['user']}")
            st.markdown(f"**Turn {row['turn']} — Support:** {row['assistant']}")
            st.caption(f"timing: cycle={row['cycle_s']:.2f}s api={row['api_s']:.2f}s | route={row['route'] or 'n/a'}")


def _run_demo_scenario(graph, scenario: dict, on_turn_done=None) -> list[dict]:
    """Run a scenario; call on_turn_done(transcript) after each turn for per-turn streaming."""
    messages = []
    customer_data = None
    transcript: list[dict] = []
    for i, user_text in enumerate(scenario["turns"], start=1):
        messages = list(messages) + [HumanMessage(content=user_text)]
        # Turn 2+: inject reminder so model does not re-ask for email when we already have profile
        if i >= 2 and customer_data:
            messages = [SystemMessage(content="You already have the customer profile in context. Do NOT ask for email or customer_id; use it to answer.")] + list(messages)
        t0 = time.perf_counter()
        result = graph.invoke({
            "messages": messages,
            "customer_data": customer_data,
            "next_route": None,
            "_api_time_this_turn": 0.0,
        })
        cycle_s = time.perf_counter() - t0
        api_s = result.get("_api_time_this_turn") or 0.0
        customer_data = result.get("customer_data") or customer_data
        messages = result["messages"]

        reply = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                reply = extract_message_text(m.content)
                break
        if not reply:
            reply = fallback_reply_for_route(result.get("next_route") or "")
        transcript.append({
            "turn": i,
            "user": user_text,
            "assistant": reply or "(No response)",
            "cycle_s": cycle_s,
            "api_s": api_s,
            "route": (result.get("next_route") or ""),
        })
        if on_turn_done:
            on_turn_done(transcript)
    return transcript

if st.session_state.get("demo_to_run"):
    st.session_state.demo_streamed_this_run = True
    graph = get_graph()
    which = st.session_state.demo_to_run
    scenarios_to_run = DEMO_SCENARIOS if which == "all" else [next((s for s in DEMO_SCENARIOS if s["id"] == which), None)]
    scenarios_to_run = [s for s in scenarios_to_run if s is not None]
    if scenarios_to_run:
        st.divider()
        st.header("Demo scenario transcripts")
        # Reserve one placeholder per scenario so we can stream per turn (Loom-friendly)
        placeholders = {s["id"]: st.empty() for s in scenarios_to_run}
        for s in scenarios_to_run:
            def make_callback(pl, sc):
                return lambda tr: _render_scenario_into(pl, sc, tr)
            on_turn = make_callback(placeholders[s["id"]], s)
            with st.spinner(f"Running {s['name']}..."):
                tr = _run_demo_scenario(graph, s, on_turn_done=on_turn)
            st.session_state.demo_results[s["id"]] = tr
    st.session_state.demo_to_run = None

if st.session_state.get("view_mode") == "chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant":
                content = sanitize_internal_jargon(content)
            st.markdown(content)

if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    reply = ""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = get_graph()
                history = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    else:
                        history.append(AIMessage(content=m["content"]))
                t0 = time.perf_counter()
                result = graph.invoke({
                    "messages": history,
                    "customer_data": st.session_state.customer_data,
                    "next_route": None,
                    "_api_time_this_turn": 0.0,
                })
                cycle_s = time.perf_counter() - t0
                api_s = result.get("_api_time_this_turn") or 0.0
                log_turn(cycle_s, api_s)
                st.session_state.timings.append((cycle_s, api_s))
                st.session_state.customer_data = result.get("customer_data") or st.session_state.customer_data
                for m in reversed(result["messages"]):
                    if isinstance(m, AIMessage) and m.content:
                        reply = extract_message_text(m.content)
                        break
                if not reply:
                    reply = fallback_reply_for_route(result.get("next_route") or "")
                reply = sanitize_internal_jargon(reply)
            except Exception as e:
                reply = f"[ERROR] {e}"
        st.markdown(reply or "(No response)")
    st.session_state.messages.append({"role": "assistant", "content": reply or "(No response)"})

# Show transcripts block only when we didn't just stream (avoid duplicate section)
if st.session_state.get("view_mode") == "demo" and st.session_state.demo_results and not st.session_state.get("demo_streamed_this_run"):
    st.divider()
    st.header("Demo scenario transcripts")
    for s in DEMO_SCENARIOS:
        tr = st.session_state.demo_results.get(s["id"])
        if not tr:
            continue
        st.subheader(s["name"])
        st.caption(f"Expected: {s['expected']}")
        for row in tr:
            st.markdown(f"**Turn {row['turn']} — User:** {row['user']}")
            st.markdown(f"**Turn {row['turn']} — Support:** {row['assistant']}")
            st.caption(f"timing: cycle={row['cycle_s']:.2f}s api={row['api_s']:.2f}s | route={row['route'] or 'n/a'}")
