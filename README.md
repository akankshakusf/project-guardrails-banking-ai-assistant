# 🛡️ Enterprise-Grade AI Customer Service Platform for Banking AI ChatBot 

### To read about the Project in detail please read **About.md** 


## Next-Generation GenAI for Financial Services  
Architecture - <img src="https://github.com/akankshakusf/project-guardrails-banking-ai-assistant/blob/master/images/architecture.png" width="70%" />
---

## Video Explaination of the Report
[![alt text](demo.mp4)](https://github.com/user-attachments/assets/46dc8d5f-d6a3-4096-a7f8-9b79dcb90a2f)


---

## 🌟 Why This Project Matters

Most “AI chatbots” for banking are little more than tech demos, missing the real safeguards and business context needed for enterprise deployment. This project stands apart because it’s architected from the ground up for security, compliance, and reliability—making it truly production-ready for financial services and other regulated industries. With dual-profile logic that separates customer and employee access, live guardrail enforcement for topics and PII, and hybrid knowledge retrieval from both official Amex policies and curated Notion FAQs, the assistant always provides brand-consistent, human-quality answers users can trust. Every response is rewritten for clarity, empathy, and warmth, so customers feel genuinely supported—not just processed by a bot. By integrating AWS Bedrock Guardrails, robust logging, and real-time monitoring, this platform doesn’t just show *what* GenAI can do in banking, but *how* to do it the right way. If you want a blueprint for deploying LLMs safely, scalably, and with a customer experience that inspires loyalty, this is it.


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
    uv pip install -e

3. **Configure AWS credentials:**
    ```bash
    Ensure you have AWS credentials for Bedrock and Notion API access added to .env

4. **Launch the UI:**
    ```bash
    streamlit run assistant/ui/amex_streamlit_app.py
