"""LangGraph: Greeter -> (retention -> Problem Solver -> maybe Processor) | (cancel -> Processor) | (tech/billing -> end)."""
import os
import time
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import get_customers_path, get_policy_docs_dir, get_rules_path
from src.data import load_resources
from src.prompts import greeter_prompt, problem_solver_prompt, processor_prompt
from src.rag import build_retriever
from src.log_config import log_llm_response, log_tool_call, log_tool_result, log_agent_step, log_context, log_reply
from src.state import State, ROUTE_CANCEL, ROUTE_END, ROUTE_RETENTION, ROUTE_TECH, ROUTE_BILLING
from src.tools import get_customer_data, calculate_retention_offer, update_customer_status
from src.utils import fallback_reply_for_route, extract_email_or_cust_id


@tool
def set_route(route: str) -> str:
    """Set the next step in the conversation. Call this with one of: retention, cancel, tech, billing, end."""
    return f"Route set to: {route}"


def _make_policy_search_tool(retriever):
    """Build a tool that runs the RAG retriever and returns concatenated content."""
    def policy_search(query: str) -> str:
        log_agent_step("policy_search", f"query={query!r}")
        if not query or not retriever:
            log_agent_step("policy_search", "NO retriever or empty query")
            return "No query or retriever available."
        try:
            docs = retriever.invoke(query)
            if not docs:
                log_agent_step("policy_search", "0 docs returned")
                return "I don't have specific information on that in my policy docs. Suggest the customer contact support for details."
            sources = [d.metadata.get("source", "?") for d in docs]
            log_agent_step("policy_search", f"{len(docs)} docs from: {sources}")
            result = "\n\n".join(d.page_content for d in docs)[:3000]
            log_context("policy_search_result", result)
            return result
        except Exception as e:
            log_agent_step("policy_search", f"ERROR: {e!r}")
            return "Policy search temporarily unavailable."
    return StructuredTool.from_function(
        func=policy_search,
        name="policy_search",
        description="Search company policy documents (return policy, Care+ benefits, tech support, billing and charges). Use for accurate answers about refunds, coverage, troubleshooting, or charge adjustments.",
    )


def _get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    # Default: lite model for lower latency. Override with GEMINI_MODEL in .env if needed.
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=400,  # keep replies short; prompt asks for 2-3 sentences so no mid-sentence chop
        google_api_key=api_key,
    )


def _invoke_agent(state: State, prompt_fn, tools: list, tool_node: ToolNode):
    """Run LLM with tools; process tool calls and return updated state."""
    agent_name = getattr(prompt_fn, '__name__', 'unknown')
    log_agent_step("invoke_agent", f"agent={agent_name} | tools={[t.name for t in tools]}")
    llm = _get_llm().bind_tools(tools)
    prompt = prompt_fn()
    messages = list(state["messages"])
    customer_data = state.get("customer_data")
    customer_data_out = state.get("customer_data")

    log_agent_step("state", f"customer_data={'YES (found)' if (customer_data and isinstance(customer_data, dict) and customer_data.get('found')) else 'NO'} | msg_count={len(messages)} | next_route={state.get('next_route')}")

    # Pre-fetch: scan all user messages for email/CUST_ and load once
    if (not customer_data or not (isinstance(customer_data, dict) and customer_data.get("found"))) and messages:
        log_agent_step("pre-fetch", "scanning messages for email/CUST_")
        for m in messages:
            if isinstance(m, HumanMessage) and m.content:
                content = m.content
                if isinstance(content, list):
                    content = " ".join(
                        (b.get("text") if isinstance(b, dict) and "text" in b else str(b))
                        for b in content
                    )
                else:
                    content = str(content) if content else ""
                extracted = extract_email_or_cust_id(content)
                if extracted:
                    log_agent_step("pre-fetch", f"extracted={extracted!r}")
                    try:
                        result = get_customer_data.invoke({"email": extracted})
                        log_agent_step("pre-fetch", f"result found={result.get('found') if isinstance(result, dict) else '?'}")
                        if isinstance(result, dict) and result.get("found"):
                            customer_data_out = result
                            customer_data = result
                            ctx = "Customer profile: " + str(result)
                            messages = [SystemMessage(content=ctx)] + list(messages)
                            log_agent_step("pre-fetch", "profile injected into context")
                            break
                    except Exception as e:
                        log_agent_step("pre-fetch", f"ERROR: {e!r}")
        else:
            log_agent_step("pre-fetch", "no email/CUST_ found in any message")

    if customer_data and messages and not (isinstance(messages[0], SystemMessage) and messages[0].content.startswith("Customer profile:")):
        ctx = "Customer profile: " + str(customer_data)
        messages = [SystemMessage(content=ctx)] + list(messages)
    api_time_s = 0.0
    next_route = state.get("next_route")
    max_tool_rounds = 6
    tool_rounds = 0
    has_status_update = False
    while True:
        try:
            t0 = time.perf_counter()
            response = llm.invoke(prompt.invoke({"messages": messages}).messages)
            api_time_s += time.perf_counter() - t0
            log_llm_response(time.perf_counter() - t0)
        except Exception as e1:
            try:
                time.sleep(1)
                t0 = time.perf_counter()
                response = llm.invoke(prompt.invoke({"messages": messages}).messages)
                api_time_s += time.perf_counter() - t0
                log_llm_response(time.perf_counter() - t0)
            except Exception as e2:
                raise RuntimeError(f"LLM failed (retry failed): {e1!r}; {e2!r}") from e2
        # Log LLM response
        resp_content = str(response.content)[:200] if response.content else "(empty)"
        resp_tools = [tc.get("name", "?") for tc in (response.tool_calls or [])]
        log_agent_step("llm_response", f"content={resp_content!r} | tool_calls={resp_tools}")

        if not response.tool_calls:
            log_reply("llm_direct", response.content)
            messages = list(messages) + [response]
            break
        for tc in response.tool_calls:
            log_tool_call(tc.get("name", "?"), tc.get("args") or {})
            if tc.get("name") == "set_route":
                args = tc.get("args") or {}
                next_route = (args.get("route") or "end").strip().lower()
            if tc.get("name") == "update_customer_status":
                has_status_update = True
        tool_node_result = tool_node.invoke({"messages": [response]})
        messages = list(messages) + [response] + tool_node_result["messages"]
        # Log tool results
        for tm in tool_node_result.get("messages") or []:
            if isinstance(tm, ToolMessage):
                log_tool_result(tm.name or "?", tm.content)
        # Persist customer_data when get_customer_data succeeds
        try:
            for tm in tool_node_result.get("messages") or []:
                if isinstance(tm, ToolMessage) and isinstance(tm.content, dict):
                    if tm.content.get("found") is True:
                        customer_data_out = dict(tm.content)
                        log_agent_step("customer_data", "persisted from tool result")
                        break
        except Exception:
            pass
        tool_rounds += 1
        only_set_route = all((tc.get("name") or "").strip().lower() == "set_route" for tc in (response.tool_calls or []))
        log_agent_step("after_tools", f"round={tool_rounds} | only_set_route={only_set_route} | next_route={next_route}")
        if only_set_route:
            route_hint = (next_route or "end").strip().lower()
            if route_hint == "end":
                if has_status_update:
                    confirm_system = (
                        "You are a support agent. The customer's request has been processed — check the tool results in the conversation for what was done. "
                        "Confirm the specific action in one sentence (e.g., 'Your Care+ plan has been canceled.'). "
                        "Then ask if there's anything else. Do not call any tools. Do not mention routes or internal terms."
                    )
                    reply_prompt = ChatPromptTemplate.from_messages([
                        ("system", confirm_system),
                        MessagesPlaceholder(variable_name="messages"),
                    ])
                    try:
                        t0 = time.perf_counter()
                        final_response = _get_llm().invoke(reply_prompt.invoke({"messages": messages}).messages)
                        api_time_s += time.perf_counter() - t0
                        fc = final_response.content
                        if fc and isinstance(fc, str) and fc.strip():
                            log_reply("status_confirm", fc)
                            messages = list(messages) + [AIMessage(content=fc)]
                        else:
                            messages = list(messages) + [AIMessage(content="Your request has been processed. Is there anything else I can help with?")]
                    except Exception:
                        messages = list(messages) + [AIMessage(content="Your request has been processed. Is there anything else I can help with?")]
                    break
                last_user = ""
                for m in reversed(messages):
                    if isinstance(m, HumanMessage) and m.content:
                        last_user = str(m.content).lower() if isinstance(m.content, str) else str(m.content)
                        break
                retention_follow_ups = ["payment pause", "cheaper option", "cheaper plan", "basic plan", "replacement involve", "upgrade"]
                if any(rt in last_user for rt in retention_follow_ups):
                    log_agent_step("end_retention_followup", f"answering retention follow-up instead of canned end")
                    followup_system = (
                        "You are a support agent. The customer is asking about a retention option that was discussed earlier in the conversation "
                        "(such as payment pause, discount, cheaper plan, or device replacement). "
                        "Answer their specific question directly in 1-2 sentences based on the conversation context. "
                        "Do not call any tools. Do not use the words 'route', 'set', or 'routing'."
                    )
                    reply_prompt = ChatPromptTemplate.from_messages([
                        ("system", followup_system),
                        MessagesPlaceholder(variable_name="messages"),
                    ])
                    try:
                        t0 = time.perf_counter()
                        final_response = _get_llm().invoke(reply_prompt.invoke({"messages": messages}).messages)
                        api_time_s += time.perf_counter() - t0
                        fc = final_response.content
                        if fc and isinstance(fc, str) and fc.strip():
                            log_reply("retention_followup_llm", fc)
                            messages = list(messages) + [AIMessage(content=fc)]
                        else:
                            messages = list(messages) + [AIMessage(content=fallback_reply_for_route("retention"))]
                    except Exception:
                        messages = list(messages) + [AIMessage(content=fallback_reply_for_route("retention"))]
                    break
                cancel_words = ["cancel", "return", "get rid", "can't afford", "cant afford"]
                if any(w in last_user for w in cancel_words):
                    log_agent_step("end_override", f"user said cancel-like words, doing LLM reply instead of canned end")
                    reply_only_system = (
                        "Reply to the customer in one short, natural sentence. Do not call any tools. "
                        "Do NOT use the words 'route', 'set', 'routing'. Speak only as a support agent."
                    )
                    reply_prompt = ChatPromptTemplate.from_messages([
                        ("system", reply_only_system),
                        MessagesPlaceholder(variable_name="messages"),
                    ])
                    try:
                        t0 = time.perf_counter()
                        final_response = _get_llm().invoke(reply_prompt.invoke({"messages": messages}).messages)
                        api_time_s += time.perf_counter() - t0
                        log_llm_response(time.perf_counter() - t0)
                        fc = final_response.content
                        if fc and (isinstance(fc, str) and fc.strip()):
                            log_reply("end_override_llm", fc)
                            messages = list(messages) + [AIMessage(content=fc)]
                        else:
                            messages = list(messages) + [AIMessage(content=fallback_reply_for_route("end"))]
                    except Exception:
                        messages = list(messages) + [AIMessage(content=fallback_reply_for_route("end"))]
                    break
                fixed_reply = fallback_reply_for_route("end")
                log_reply("fixed_end", fixed_reply)
                messages = list(messages) + [AIMessage(content=fixed_reply)]
                break
            has_profile = customer_data_out and isinstance(customer_data_out, dict) and customer_data_out.get("found")
            log_agent_step("reply_after_route", f"route={route_hint} | has_profile={has_profile}")
            if route_hint == "billing":
                last_user_for_billing = ""
                for m in reversed(messages):
                    if isinstance(m, HumanMessage) and m.content:
                        last_user_for_billing = str(m.content).lower() if isinstance(m.content, str) else str(m.content).lower()
                        break
                retention_follow_ups = ["payment pause", "cheaper option", "cheaper plan", "basic plan"]
                is_retention_followup = any(rt in last_user_for_billing for rt in retention_follow_ups)
                if is_retention_followup:
                    reply_only_system = (
                        "You are a support agent. The customer is asking about a retention option that was discussed earlier in the conversation "
                        "(such as payment pause, discount, cheaper plan, or replacement). "
                        "Answer their question directly in 1-2 sentences based on the conversation context. "
                        "This is about plan options, NOT a billing dispute. Do not escalate to billing. "
                        "Do not call any tools. Do not use 'route', 'set', or 'routing'."
                    )
                    next_route = "end"
                elif has_profile:
                    reply_only_system = (
                        "You are a support agent. Read the full conversation and reply in one short sentence that moves the billing conversation forward. "
                        "Do NOT repeat a previous assistant reply word-for-word. "
                        "IMPORTANT: You do NOT have access to the billing system. You CANNOT review charges, apply credits, correct charges, or process refunds. "
                        "Never say 'I've corrected the charge' or 'the difference is refunded'. Instead, say you'll flag it for the billing team to review (e.g. 'I'll escalate this to the billing team for review'). "
                        "Do not call any tools. Do not use the words 'route', 'set', or 'routing'."
                    )
                else:
                    reply_only_system = (
                        "You are a support agent. Reply in one short sentence: acknowledge their billing concern and ask for their email address so you can look up their account. "
                        "Do NOT repeat a previous assistant reply word-for-word. "
                        "Do not call any tools. Do not use the words 'route', 'set', or 'routing'."
                    )
            else:
                reply_only_system = (
                    "Reply to the customer in one short, natural sentence. "
                    "Do not call any tools. Do NOT use the words 'route', 'set', 'routing', or 'your route has been set'—speak only as a support agent."
                )
            reply_prompt = ChatPromptTemplate.from_messages([
                ("system", reply_only_system),
                MessagesPlaceholder(variable_name="messages"),
            ])
            llm_no_tools = _get_llm()
            try:
                t0 = time.perf_counter()
                final_response = llm_no_tools.invoke(reply_prompt.invoke({"messages": messages}).messages)
                api_time_s += time.perf_counter() - t0
                log_llm_response(time.perf_counter() - t0)
            except Exception as e1:
                try:
                    time.sleep(1)
                    t0 = time.perf_counter()
                    final_response = llm_no_tools.invoke(reply_prompt.invoke({"messages": messages}).messages)
                    api_time_s += time.perf_counter() - t0
                    log_llm_response(time.perf_counter() - t0)
                except Exception as e2:
                    raise RuntimeError(f"LLM reply-after-route failed: {e1!r}; {e2!r}") from e2
            final_content = final_response.content
            empty = (
                not final_content
                or (isinstance(final_content, str) and not final_content.strip())
                or (
                    isinstance(final_content, list)
                    and not any((b.get("text") if isinstance(b, dict) else b) for b in final_content)
                )
            )
            if empty:
                log_agent_step("reply_after_route", "LLM returned empty; using fallback")
                final_content = fallback_reply_for_route(route_hint)
            log_reply("reply_after_route", final_content)
            messages = list(messages) + [AIMessage(content=final_content)]
            break
        if tool_rounds >= max_tool_rounds:
            break
        state = {**state, "messages": messages}
        messages = list(messages)
    # Post-loop: if billing and no profile, do one LLM call to naturally ask for email (contextual, not hardcoded)
    has_profile_final = customer_data_out and isinstance(customer_data_out, dict) and customer_data_out.get("found")
    route_final = (next_route or "end").strip().lower()
    if route_final == "billing" and not has_profile_final:
        from src.utils import extract_message_text as _emt
        last_ai = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and messages[i].content:
                last_ai = i
                break
        if last_ai is not None:
            existing = _emt(messages[last_ai].content)
            if existing and "email" not in existing.lower():
                email_ask_system = (
                    "You are a support agent. The customer has a billing question but we don't have their account yet. "
                    "Rewrite the following reply so it keeps the same meaning AND naturally asks for their email or account ID at the end. "
                    "Keep it to 2 sentences max. Do not use the words 'route' or 'set'. "
                    f"Original reply: \"{existing}\""
                )
                try:
                    t0 = time.perf_counter()
                    rewrite = _get_llm().invoke([SystemMessage(content=email_ask_system)])
                    api_time_s += time.perf_counter() - t0
                    rewritten = _emt(rewrite.content) if rewrite.content else ""
                    if rewritten and "email" in rewritten.lower():
                        messages[last_ai] = AIMessage(content=rewritten)
                        log_reply("email_rewrite", rewritten)
                    else:
                        messages[last_ai] = AIMessage(content=existing + " Could you share your email so I can look up your account?")
                        log_reply("email_fallback", messages[last_ai].content)
                except Exception:
                    messages[last_ai] = AIMessage(content=existing + " Could you share your email so I can look up your account?")
                    log_reply("email_fallback_err", messages[last_ai].content)
    acc = state.get("_api_time_this_turn") or 0.0
    return {
        "messages": messages,
        "customer_data": customer_data_out,
        "next_route": next_route or state.get("next_route"),
        "_api_time_this_turn": acc + api_time_s,
    }


def _greeter_node(state: State, retriever) -> State:
    tools = [get_customer_data, set_route]
    if retriever:
        tools.append(_make_policy_search_tool(retriever))
    tool_node = ToolNode(tools)
    return _invoke_agent(state, greeter_prompt, tools, tool_node)


def _problem_solver_node(state: State, retriever) -> State:
    tools = [get_customer_data, calculate_retention_offer, set_route, update_customer_status]
    if retriever:
        tools.append(_make_policy_search_tool(retriever))
    tool_node = ToolNode(tools)
    return _invoke_agent(state, problem_solver_prompt, tools, tool_node)


def _processor_node(state: State) -> State:
    tools = [update_customer_status, set_route]
    tool_node = ToolNode(tools)
    return _invoke_agent(state, processor_prompt, tools, tool_node)


def _route_after_greeter(state: State) -> Literal["problem_solver", "processor", "end"]:
    r = (state.get("next_route") or "end").strip().lower()
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and m.content:
            last_user = str(m.content).lower() if isinstance(m.content, str) else str(m.content).lower()
            break
    device_words = ["overheat", "broken", "crack", "won't charge", "not charging", "screen flicker", "battery drain"]
    cancel_words = ["cancel", "return", "get rid"]
    if any(w in last_user for w in device_words) and any(w in last_user for w in cancel_words) and r != ROUTE_RETENTION:
        log_agent_step("route_override", f"device+cancel detected, overriding {r} -> retention")
        r = ROUTE_RETENTION
    dest = "problem_solver" if r == ROUTE_RETENTION else ("processor" if r == ROUTE_CANCEL else "end")
    log_agent_step("route_after_greeter", f"next_route={r!r} -> {dest}")
    return dest


def _route_after_problem_solver(state: State) -> Literal["processor", "end"]:
    r = (state.get("next_route") or "end").strip().lower()
    dest = "processor" if r == ROUTE_CANCEL else "end"
    log_agent_step("route_after_problem_solver", f"next_route={r!r} -> {dest}")
    return dest


def build_graph():
    """Build and return the StateGraph. Load resources once."""
    load_resources(get_customers_path(), get_rules_path())
    retriever = build_retriever(get_policy_docs_dir())

    def greeter(s: State):
        return _greeter_node(s, retriever)
    def problem_solver(s: State):
        return _problem_solver_node(s, retriever)

    graph = StateGraph(State)
    graph.add_node("greeter", greeter)
    graph.add_node("problem_solver", problem_solver)
    graph.add_node("processor", _processor_node)
    graph.add_conditional_edges("greeter", _route_after_greeter, {"problem_solver": "problem_solver", "processor": "processor", "end": END})
    graph.add_conditional_edges("problem_solver", _route_after_problem_solver, {"processor": "processor", "end": END})
    graph.add_edge("processor", END)
    graph.set_entry_point("greeter")
    return graph.compile()
