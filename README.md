# 🛡️ Enterprise-Grade AI Customer Service Platform for American Express

To Read About Project in detail please read **About.md**

## Next-Generation GenAI for Financial Services  
*AI you can trust, at enterprise scale. Built for the real world.*

---

## Why This Project Matters

Most “AI for banking” demos stop at a chatbot.  
**This repo goes further:** it’s a blueprint for how *real* GenAI gets shipped at banks like Amex—safe, scalable, and always compliant.

---

### 🚦 What Sets This System Apart?

#### **Smart Agent Orchestration**
Every user message is routed to the *right* expert:  
- Rewards eligibility? → Custom recommendation engine.
- Policy or compliance? → Hybrid RAG agent (PDF + Notion knowledge).
- Internal ops? → Profile-specific access with strict controls.

#### **Built-In Security & Guardrails**
- **AWS Bedrock Guardrails** filter every prompt and response.
- Two roles: *Customer* (public info only) and *Employee* (expanded access, but still safe).
- Custom denied topics, live PII detection, auto-anonymization, and full audit logs.

#### **Hybrid, Always-Current Knowledge**
- **PDF policy ingestion**: Official docs chunked and vectorized for semantic search.
- **[Notion Help Center](https://verdant-jute-477.notion.site/Amex-Help-Center-Knowledge-Base-2394ba18200f80c3ba0ed88f417c09d7)**: FAQs, toggles, bullet lists—automatically indexed and searchable.
- **Titan Embeddings + FAISS** for fast, accurate retrieval.

#### **Human-First, Brand-Consistent Responses**
- All answers pass through Claude 3.7 Sonnet for *warm*, *friendly*, “Amex-style” language.
- No legalese, no robotic tone—just clear, empathetic customer service.

#### **Real-Time Monitoring & Observability**
- Every security event (blocked word, PII detection, denied topic) is visible and logged via Bedrock’s dashboards.
- Full transparency for compliance and audits.

#### **Production-Ready, Demo-Friendly UI**
- Streamlit interface with live security/KB health, guardrail status, and mode switching.
- Easy to test, easy to trust.

---

## 🏗️ System Architecture

- **Intelligent Router**: Routes every query to either a policy agent (RAG), a rewards agent (rule-based + LLM), or internal assistant, depending on user type and intent.
- **Knowledge Base Fusion**:  
  - Ingests PDFs, cleans, chunks, deduplicates, and embeds  
  - Syncs Notion FAQ data, including toggled and nested content  
  - Unified FAISS vector DB for fast semantic search and retrieval
- **Security First**:  
  - Role-based guardrails applied to both input and output  
  - Real-time PII redaction and topic/word filtering  
  - All events logged and versioned for monitoring and traceability
- **Monitoring and Observability**:  
  - Bedrock dashboard visualizes every guardrail intervention, blocked topic, or masked PII event  
  - Policy versioning, activity logs, and compliance-ready exports
- **NLG Synthesis**:  
  - Every answer rewritten by Claude for tone, clarity, and engagement  
  - Custom system prompts tuned for friendliness and brand voice

---


## 🚀 Quickstart

1. **Clone this repo:**  
   ```bash
   git clone https://github.com/akankshakusf/project-guardrails-banking-ai-assistant.git
   cd project-guardrails-banking-ai-assistant

2. **Install requirements:**
    ```bash
    pip install -r requirements.txt

3. **Configure AWS credentials:**
    ```bash
    Ensure you have AWS credentials for Bedrock and Notion API access.

4. **Launch the UI:**
    ```bash
    streamlit run assistant/ui/amex_streamlit_app.py
