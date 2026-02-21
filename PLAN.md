# TechFlow Customer Support Chat System – Plan

**Source of truth**: [projectPlan.md](projectPlan.md) (AI Engineer Assignment).  
**Time box**: 2 days.

---

## 1. What We're Building

- **Chat system** for TechFlow Electronics: customers who want to cancel phone insurance (Care+).
- **Goals**: Retain when possible; route to tech support or billing when it’s not a cancellation.
- **3 agents**: Greeter & Orchestrator → Problem Solver (retention) → Processor (cancel/update).

---

## 2. Technical Requirements (MUST)

| Requirement | Details |
| ----------- | ------- |
| **Framework** | **LangChain/LangGraph** – orchestration, agent-to-agent communication, state, clear handoffs. |
| **Tools** | `get_customer_data(email)`, `calculate_retention_offer(customer_tier, reason)`, `update_customer_status(customer_id, action)` – backed by customers.csv, retention_rules.json, log to file. |
| **RAG** | Vector search (FAISS / Qdrant / Pinecone) on policy docs; agents query during conversation. |
| **LLM** | Free preferred (Gemini); function calling or structured output; clear prompts. |
| **Architecture** | Workflow + tool calling + RAG + state + **error handling and fallbacks**. |

---

## 3. The 3 Agents

1. **Greeter & Orchestrator** – Identify customer (e.g. by email), classify intent (cancel vs tech vs billing), route to correct agent.
2. **Problem Solver** – Retention only: discounts, pauses, value explanation; use playbook scripts; only hand off to Processor if customer insists on leaving.
3. **Processor** – Execute cancellation/updates via `update_customer_status`, confirm and close.

---

## 4. Acceptance: 5 Test Conversations

| # | Customer says | Expected behavior |
| - | ------------- | ------------------ |
| 1 | "hey can't afford the $13/month care+ anymore, need to cancel" | Payment pause or discount before cancel; use business rules + customer data. |
| 2 | "this phone keeps overheating, want to return it and cancel everything" | Offer replacement/upgrade before cancel; use return policy + tech guide. |
| 3 | "paying for care+ but never used it, maybe just get rid of it?" | Explain value, offer cheaper options; use Care+ benefits doc. |
| 4 | "my phone won't charge anymore, tried different cables" | Route to tech support only; no retention pitch. |
| 5 | "got charged $15.99 but thought care+ was $12.99, what's the extra?" | Route to billing; not cancellation. |

All 5 must show correct routing, tool usage where appropriate, and RAG when answering from policies.

---

## 5. Clean Repo Structure (No Extra Files, No Mess)

**Rule**: One clear place for each thing; no duplicates at root; no spaces in folder names; no ` (1)` in filenames. Remove anything that doesn’t belong in the final tree.

**Final structure (only these):**

```
<repo_root>/
  README.md                 # For viewers (how to run, 5 tests)
  DEVELOPMENT.md            # For you (step-by-step flow)
  PLAN.md                   # This plan
  projectPlan.md            # Assignment spec (source of truth)
  pyproject.toml
  .env.example
  .gitignore
  project_resources/        # All data – nothing else at root
    retention_playbook.md
    retention_rules.json
    customers.csv
    policy_documents/
      return_policy.md
      care_plus_benefits.md
      troubleshooting_guide.md
  src/                      # Code only
    ...
  tests/
    ...
```

**When we set up the repo:**

1. **Create** `project_resources/` and `project_resources/policy_documents/`.
2. **Move and rename** (then delete originals):
   - `retention_playbook (1).md` → `project_resources/retention_playbook.md`
   - `retention_rules (1).json` → `project_resources/retention_rules.json`
   - `customers (1).csv` → `project_resources/customers.csv`
   - `Policy Documents/return_policy.md` → `project_resources/policy_documents/return_policy.md`
   - `Policy Documents/care_plus_benefits.md` → `project_resources/policy_documents/care_plus_benefits.md`
   - `Policy Documents/troubleshooting_guide.md` → `project_resources/policy_documents/troubleshooting_guide.md`
3. **Remove** after the move:
   - Empty `Policy Documents/` folder
   - `projectPlan.docx` (we use `projectPlan.md`; no need for the binary)
4. **Do not keep** any copy of the above data files at repo root or in old paths.

Result: Root has only docs + config; all reference data under `project_resources/` with clean names; code under `src/`; easy to understand and push to GitHub.

---

## 6. Performance and Simplicity (When We Implement)

**Goal**: Good, optimal solution that runs fast; use async or background only where it clearly helps; keep design simple and efficient; handle edge cases.

- **Async where it pays off**
  - Use async for **LLM and RAG** (e.g. `langchain` async invokes, async vector search if the chosen store supports it) so the chat loop isn’t blocked on a single call.
  - Keep **tool calls** async-friendly (e.g. file reads / small logic stay sync; if we add a slow external API later, wrap that in async).
- **Background only if needed**
  - Default: one request = one conversation turn (no background jobs). If we add something slow (e.g. writing to an external system), consider a small background task or fire-and-forget log – only if required by the assignment or to avoid timeouts. Don’t over-engineer.
- **Edge cases to handle**
  - **Missing customer**: email not in customers.csv → graceful message, no crash; optionally route to “guest” or ask for correct email.
  - **Missing/invalid tier or reason**: `calculate_retention_offer` – validate inputs, return clear structure (e.g. empty offers + message) instead of raising.
  - **RAG empty or low score**: fallback to “I don’t have that in my docs, here’s what I can do…” and don’t hallucinate policy.
  - **LLM/API failure**: retry once with backoff if transient; then fallback message and log; never leave user with no response.
  - **File/log errors**: `update_customer_status` – if write fails, log error and return a clear result so the agent can say “we’re recording this; if you don’t get confirmation, please contact support.”
- **Keep it simple but efficient**
  - Load CSV and retention_rules.json **once at startup** (or on first use); cache in memory; no repeated disk reads per message.
  - Build RAG index **once at startup** (or lazy on first query); reuse for all conversations.
  - No extra queues or workers unless we hit a concrete need (e.g. demo timeout). Prefer a single process, clear flow, good error handling.

When we start the project, we’ll apply these rules so the solution is fast, robust, and easy to run/demo.

---

## 7. Evaluation Weights (from assignment)

- LangChain multi-agent: **30%**
- Tool integration & data: **25%**
- RAG & context: **20%**
- Conversation quality & intent: **25%**

---

## 8. Summary

- **Spec**: [projectPlan.md](projectPlan.md).
- **Stack**: LangChain/LangGraph, 3 agents, 3 tools, RAG (vector search on policy docs), free LLM (e.g. Gemini).
- **Data**: All reference files in `project_resources/` only; config points there; no data files at root.
- **Clean repo**: Single layout as in Section 5; move then remove originals; remove `projectPlan.docx`; no duplicates, no messy names or extra files.
- **Docs**: README.md (viewers) + DEVELOPMENT.md (you, step-by-step flow).
- **Performance**: Async for LLM/RAG; background only if needed; handle edge cases; cache data and RAG index; keep flow simple and efficient.

Plan is set. When we start implementation, we follow this for a clean, understandable repo and good performance.
