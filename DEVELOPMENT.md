# Development Guide – Step-by-Step Flow

This document explains how the project works end-to-end and where each file is used.

---

## 1. Startup

1. **main.py** is the entrypoint. It loads `.env` (via `python-dotenv`), checks for `GOOGLE_API_KEY` or `GEMINI_API_KEY`, and calls `build_graph()`.
2. **graph.build_graph()**:
   - Calls **data.load_resources**(customers_path, rules_path) so **customers.csv** and **retention_rules.json** are read once and cached in memory.
   - Calls **rag.build_retriever**(policy_docs_dir) to load all **policy_documents/*.md** files, split them into chunks, build a FAISS index, and return a retriever (used as the `policy_search` tool).
   - Builds the LangGraph (greeter → conditional → problem_solver | processor | end) and returns the compiled graph.
3. **config.py** provides paths: `get_resources_dir()`, `get_customers_path()`, `get_rules_path()`, `get_policy_docs_dir()`, `get_status_log_path()`. Default base is `project_resources/` next to the project root.

---

## 2. Incoming message

1. User types a message in the CLI. **main.py** appends a `HumanMessage` to the message list and invokes the graph with state: `messages`, `customer_data` (from previous turn if any), `next_route` (reset to None for the new turn).
2. The graph **entry point** is the **greeter** node.

---

## 3. Greeter node

1. **prompts.greeter_prompt()** provides the system prompt (from **prompts.py**), which includes the playbook principles (empathy, one option at a time, etc.).
2. Tools available: **get_customer_data**(email), **policy_search**(query) – RAG over policy docs – and **set_route**(route).
3. The LLM (Gemini) is invoked with these tools. It may:
   - Call **get_customer_data** if the user gave an email, to load profile from **customers.csv** (via **data.get_customers()**).
   - Call **policy_search** to cite return policy, Care+ benefits, or tech support guide.
   - Call **set_route** with one of: `retention`, `cancel`, `tech`, `billing`.
4. **graph** reads the **set_route** tool call and sets `state["next_route"]`. Conditional edges then send the flow to:
   - **problem_solver** if route is `retention`,
   - **processor** if route is `cancel`,
   - **END** if route is `tech` or `billing` (support replies and conversation ends for that turn).

---

## 4. Problem Solver node (when route = retention)

1. **prompts.problem_solver_prompt()** defines the retention agent behavior (use playbook, one option at a time).
2. Tools: **get_customer_data**, **calculate_retention_offer**(customer_tier, reason), **policy_search**, **update_customer_status**, **set_route**.
3. **calculate_retention_offer** uses **data.get_rules()** (from **retention_rules.json**) and maps `customer_tier` (premium/regular/new) and `reason` (e.g. financial_hardship, overheating, service_value) to the list of allowed offers. Implemented in **tools.py**.
4. The agent presents offers one at a time. If the customer accepts, it calls **set_route("end")**. If the customer insists on leaving, it calls **set_route("cancel")** so the next node is **processor**.

---

## 5. Processor node (when route = cancel or from problem_solver)

1. **prompts.processor_prompt()** defines the processor: confirm and log the action.
2. Tools: **update_customer_status**(customer_id, action), **set_route**.
3. **update_customer_status** (in **tools.py**) appends a line to the status log file (path from **config.get_status_log_path()**, default `status_logs/actions.log`). It creates the directory if needed and returns success/failure so the agent can confirm or ask the customer to contact support if the log fails.
4. The agent then calls **set_route("end")** and the graph goes to **END**.

---

## 6. Response back to user

1. After **graph.invoke()**, **main.py** takes the updated `messages` from the state.
2. It finds the **last AIMessage** with content and prints it as `Support: <content>`.
3. State (including `messages`) is kept for the next turn; **customer_data** is preserved if the graph returned it (for future use).

---

## Where each project file is used

| File | Role |
|------|------|
| **project_resources/customers.csv** | Loaded once by **data.load_resources**; **get_customer_data** looks up by email. |
| **project_resources/retention_rules.json** | Loaded once by **data.load_resources**; **calculate_retention_offer** uses it by tier and reason. |
| **project_resources/retention_playbook.md** | Content reflected in **prompts.py** (playbook principles and structure); can be read at build time to fill prompts. |
| **project_resources/policy_documents/*.md** | Loaded and chunked by **rag.load_policy_docs**; FAISS index built in **rag.build_retriever**; agents use **policy_search** during the conversation. |
| **config.py** | Paths for CSV, JSON, policy dir, playbook, status log. |
| **data.py** | In-memory cache of customers and rules; used by **tools.py**. |
| **tools.py** | Defines the 3 required tools; used by **graph.py** in each node’s ToolNode. |
| **rag.py** | Builds retriever for policy docs; used in **graph.build_graph** and passed into greeter/problem_solver nodes. |
| **state.py** | State type and route constants for the graph. |
| **prompts.py** | System prompts for Greeter, Problem Solver, Processor. |
| **graph.py** | Builds the LangGraph and wires nodes/edges. |
| **main.py** | CLI loop, env load, graph invoke, print last AI reply. |

---

## Env and secrets

- **GOOGLE_API_KEY** or **GEMINI_API_KEY**: Required for Gemini. Create in [Google AI Studio](https://aistudio.google.com/apikey). Do not commit `.env`.
- **RESOURCES_DIR** (optional): Override base path for `project_resources` (default: repo root).
- **STATUS_LOG_PATH** (optional): Override path for the status log file (default: `status_logs/actions.log`).

---

## Development workflow

- Run chat: `uv run python -m src.main` (or `python -m src.main`).
- Add a new scenario: extend **retention_rules.json** and, if needed, **prompts.py** and **tools.py** (reason mapping).
- Change policy answers: edit markdown in **project_resources/policy_documents/** and restart (RAG index is built at startup).
