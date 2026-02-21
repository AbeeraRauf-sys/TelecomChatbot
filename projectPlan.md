# AI Engineer Assignment - Customer Support Chat System

**Time to Complete**: 2 days

**What you'll build**: A smart chat system that helps customers who want to cancel their phone plans

---

## What You're Building

Create a chat system for **TechFlow Electronics** (a phone company) that:

- Talks to customers who want to cancel their phone insurance
- Tries to help solve their problems instead of losing them as customers
- Knows when someone just needs technical help (not trying to cancel)

Think of it like training 3 different customer service agents who each have different jobs.

---

## Technical Requirements (MUST FOLLOW)

### **Multi-Agent Framework**

- **MUST use LangChain/LangGraph** for agent orchestration
- Demonstrate proper agent-to-agent communication
- Show state management between agents
- Clear workflow definition for agent handoffs

### **Tool Integration (Required)**

Your agents must use these 3 tools to process real data:

```python
@tool
def get_customer_data(email: str) -> dict:
    """Load customer profile from customers.csv"""

@tool
def calculate_retention_offer(customer_tier: str, reason: str) -> dict:
    """Generate offers using retention_rules.json"""

@tool
def update_customer_status(customer_id: str, action: str) -> dict:
    """Process cancellations/changes (log to file)"""
```

### **RAG Implementation (Required)**

- Use provided policy documents for context retrieval
- Implement vector search (FAISS, Qdrant, Pinecone, or similar)
- Agents must query documents during conversations
- Show relevant information retrieval in action

### **LLM Options (Free Preferred)**

- **Recommended**: Google Gemini (free tier)
- **Alternatives**: DeepSeek, Groq, Ollama (local)
- Must demonstrate function calling or structured output
- Show proper prompt engineering

### **System Architecture Must Include**

- LangChain/LangGraph workflow orchestration
- Tool calling with real data processing
- RAG system with document retrieval
- Multi-agent state management
- Error handling and fallbacks

## The 3 Chat Agents You Need

### **Agent 1: The Greeter & Orchestrator**

- Figures out who the customer is
- Decides what kind of help they need
- Sends them to the right specialist

### **Agent 2: The Problem Solver and Communicator**

- Tries to keep customers happy when they want to cancel
- Offers solutions like discounts or pauses
- Only gives up if the customer really wants to leave

### **Agent 3: The Processor**

- Actually cancels services when needed
- Updates customer accounts
- Handles the paperwork

---

## What We Give You

### Customer Information

- A file with 10 fake customers and their details
- Names, emails, what they pay, how long they've been customers
- **Download**: customers.csv

### **Business Rules**

- A guide for what discounts you can offer different customers
- When to offer payment breaks vs. discounts
- **Download**: retention_rules.json

### **Company Policies**

Your agents will need to reference these documents to provide accurate information:

- **Return Policy**: Customer tiers, refund procedures, processing times
- **Care+ Benefits**: Complete insurance coverage details and value explanations
- **Tech Support Guide**: When to escalate technical issues vs handle in chat
- **Download**: Policy Documents Bundle

### **Conversation Templates**

- Structured scripts for handling different cancellation scenarios
- Empathy phrases and discovery questions
- **Note**: Convert these into your agent system prompts
- **Download**: retention_playbook.md

---

## Test Your System With These 5 Conversations

### **Test 1: Money Problems**

Customer says: *"hey can't afford the $13/month care+ anymore, need to cancel"*

**Expected**: Try to help with payment pause or discount before canceling

*Agent should use: Business rules + Customer data to calculate personalized offers*

### **Test 2: Phone Problems**

Customer says: *"this phone keeps overheating, want to return it and cancel everything"*

**Expected**: Offer phone replacement or upgrade before canceling

*Agent should use: Return policy + Tech guide to offer solutions*

### **Test 3: Questioning Value**

Customer says: *"paying for care+ but never used it, maybe just get rid of it?"*

**Expected**: Explain what they get for their money, offer cheaper options

*Agent should use: Care+ benefits document to explain value*

### **Test 4: Technical Help Needed**

Customer says: *"my phone won't charge anymore, tried different cables"*

**Expected**: Send directly to tech support (don't try to sell anything)

*Agent should use: Tech support guide to recognize non-cancellation intent*

### **Test 5: Billing Question**

Customer says: *"got charged $15.99 but thought care+ was $12.99, what's the extra?"*

**Expected**: Send directly to billing department (don't try to sell anything)

*Agent should recognize this as billing inquiry, not cancellation*

---

## What You Need to Show Us

### **Video Demo (5-10 minutes)**

- You can use Loom to record your video
- Show how your system works
- Test all 5 conversations showing proper agent routing
- Demonstrate tool calls and RAG queries in action
- Explain your LangChain architecture

### **Code on GitHub (Public)**

- Clean, working code with LangChain/LangGraph implementation
- Instructions for how to run it
- Proof that all 5 tests work correctly
- Tool integration with real data processing
- RAG system with document retrieval

---

## Evaluation Criteria

### **LangChain Multi-Agent Implementation (30%)**

- Proper agent orchestration and workflow
- Clean agent communication patterns
- State management between conversations
- Professional code structure

### **Tool Integration & Data Processing (25%)**

- Tools fetch real data from provided files
- Sequential tool usage (get customer → calculate offer → update)
- Error handling for missing data
- Business logic implementation

### **RAG & Context Retrieval (20%)**

- Policy documents integrated for accurate responses
- Relevant information retrieval during conversations
- Vector search implementation
- Context-aware agent responses

### **Conversation Quality & Intent Classification (25%)**

- Accurate routing between retention vs non-retention
- Natural conversation flows
- Proper use of provided conversation scripts
- Empathetic customer interactions

---

## What We're Looking For

**Can you build a system that**:

- Correctly identifies when someone wants to cancel vs needs other help?
- Tries reasonable solutions before giving up on customers?
- Uses real customer data to make personalized offers?
- Handles different types of conversations appropriately?

**Technical skills we want to see**:

- Smart conversation flow between different agents
- Using customer data to make decisions
- Finding relevant information from company documents
- Clean, working code that demonstrates your thinking

---

## Keep It Simple

- Use free LLMs - Gemini, Groq, Ollama, OpenAI free versions and SLMs etc.
- No complex databases required - just simple files with LangChain tools
- Focus on making it work well rather than making it fancy
- Show us your problem-solving approach with proper AI engineering practices

**Goal**: Prove you can build practical AI systems that solve real business problems using modern AI engineering frameworks, not just technical demos.
