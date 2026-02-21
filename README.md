# TechFlow Customer Support Chat System

A multi-agent chat system for TechFlow Electronics that helps customers who want to cancel their phone insurance (Care+), tries to retain them with personalized offers, and routes technical or billing questions correctly.

Built with **LangChain/LangGraph**, three required tools, **RAG** (FAISS + HuggingFace embeddings over policy documents), and Google Gemini (free tier).

## Architecture

```
User message
    │
    ▼
┌──────────────────────────┐
│   GREETER & ORCHESTRATOR │  Tools: get_customer_data, set_route, policy_search
│   Classify intent, look  │
│   up customer, route     │
└───────────┬──────────────┘
            │
    ┌───────┼────────┬──────────┐
    ▼       ▼        ▼          ▼
retention  cancel   tech      billing
    │       │        │          │
    ▼       │        ▼          ▼
┌────────┐  │     (Greeter    (Greeter
│PROBLEM │  │      answers     answers
│SOLVER  │  │      + END)      + END)
│        │  │
│Retention│  │    Tools: calculate_retention_offer,
│offers  │  │    get_customer_data, policy_search,
└───┬────┘  │    update_customer_status, set_route
    │       │
    ▼       ▼
┌──────────────┐
│  PROCESSOR   │  Tools: update_customer_status, set_route
│  Cancel/     │
│  pause/      │
│  downgrade   │
└──────────────┘
```

### State Management

State flows between agents via LangGraph's `StateGraph` with `add_messages` annotation:

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation history
    customer_data: dict | None    # customer profile (persisted across agents)
    next_route: str | None        # routing decision: retention/cancel/tech/billing/end
```

Each agent receives the full state, including messages from previous agents. Customer data loaded by the Greeter is automatically available to downstream agents.

## Tool Integration

Three required tools backed by real data:

| Tool | Data Source | What It Does |
|------|------------|-------------|
| `get_customer_data(email)` | `customers.csv` | Loads customer profile: name, tier, plan, device, monthly charge, tenure |
| `calculate_retention_offer(tier, reason)` | `retention_rules.json` | Generates personalized offers based on tier (premium/regular/new) and cancel reason |
| `update_customer_status(id, action)` | `status_logs/actions.log` | Logs cancellations, pauses, and downgrades with timestamps |

Tools are called sequentially during conversations: identify customer → calculate offers → process changes.

## RAG Implementation

- **Vector Store**: FAISS with HuggingFace embeddings (`paraphrase-MiniLM-L3-v2`)
- **Documents**: 4 policy documents indexed at startup:
  - `return_policy.md` — Customer tiers, refund procedures, return windows
  - `care_plus_benefits.md` — Insurance coverage, plan comparison, value propositions
  - `troubleshooting_guide.md` — When to escalate vs handle in chat
  - `billing.md` — Charge explanations, escalation procedures
- **Usage**: Agents call `policy_search(query)` during conversations to retrieve relevant policy chunks for grounded, accurate responses

## How to Run

1. **Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/) (or pip).

2. **Install**:
   ```bash
   uv sync
   # or: pip install -e .
   ```

3. **Configure**: Copy `.env.example` to `.env` and set your Gemini API key:
   ```bash
   cp .env.example .env
   # Edit .env: set GOOGLE_API_KEY=your_key (from https://aistudio.google.com/apikey)
   ```

4. **Run the chat UI**:
   ```bash
   uv run streamlit run streamlit_app.py
   ```

5. **Run the 5 test scenarios** (built-in demo runner):
   - Open the Streamlit app
   - Switch to "Demo runner" mode in the sidebar
   - Click "Run all demo scenarios"

   Or via CLI: `uv run python scripts/run_scenarios.py`

## 5 Test Conversations

Include an email (e.g. `sarah.chen@email.com`) for personalized offers from `customers.csv`.

| # | Customer says | Expected behavior |
|---|--------------|------------------|
| 1 | "hey can't afford the $13/month care+ anymore, need to cancel – sarah.chen@email.com" | Payment pause or discount before cancel; uses retention_rules.json + customer tier |
| 2 | "this phone keeps overheating, want to return it and cancel everything – michael.davis@email.com" | Offer replacement/upgrade before cancel; uses return policy + tech guide via RAG |
| 3 | "paying for care+ but never used it, maybe just get rid of it? – jennifer.lee@email.com" | Explain Care+ value, offer cheaper plan; uses Care+ benefits doc via RAG |
| 4 | "my phone won't charge anymore, tried different cables" | Tech support only — troubleshooting steps, no retention pitch |
| 5 | "got charged $15.99 but thought care+ was $12.99, what's the extra? – sarah.chen@email.com" | Billing inquiry — acknowledge discrepancy, escalate to billing team, no cancellation |

## Project Layout

```
├── streamlit_app.py              # Chat UI + demo runner
├── src/
│   ├── graph.py                  # LangGraph: 3 nodes, routing, agent loop
│   ├── prompts.py                # System prompts: universal + agent-specific guardrails
│   ├── tools.py                  # 3 required tools (customer data, retention offers, status updates)
│   ├── rag.py                    # FAISS vector store + embeddings
│   ├── state.py                  # State TypedDict + route constants
│   ├── data.py                   # Load customers.csv + retention_rules.json
│   ├── config.py                 # File path configuration
│   ├── utils.py                  # Email extraction, jargon sanitization, fallback replies
│   └── log_config.py             # Structured logging
├── project_resources/
│   ├── customers.csv             # 10 customer profiles
│   ├── retention_rules.json      # Tier-based retention offers
│   ├── retention_playbook.md     # Conversation scripts (converted to prompts)
│   └── policy_documents/         # RAG source documents
│       ├── return_policy.md
│       ├── care_plus_benefits.md
│       ├── troubleshooting_guide.md
│       └── billing.md
├── projectPlan.md                # Assignment specification
├── PLAN.md                       # Implementation plan
└── DEVELOPMENT.md                # Developer flow documentation
```

## Error Handling

- **LLM failure**: Retry once with 1s backoff; if retry fails, raise with context
- **Empty LLM response**: Route-specific fallback replies (never shows "(No response)")
- **Missing customer**: Graceful message, asks for email — no crash
- **Invalid tool input**: Validation with clear error messages returned to the agent
- **Internal jargon leak**: `sanitize_internal_jargon()` catches any "route has been set" text before display

## License

For assessment use.
