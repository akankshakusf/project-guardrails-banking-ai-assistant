# Enterprise-Grade AI Customer Service Assistant for American Express
### A Case Study in Secure, Scalable, and Responsible Generative AI in Financial Services

---

## Executive Summary

This project is far more than a chatbot—it is a full-stack, production-ready AI customer service platform, designed specifically for the high-stakes regulatory environment of American Express and similar financial institutions. By combining advanced agent orchestration, hybrid knowledge retrieval, rigorous security controls, and brand-consistent language generation, this system demonstrates how modern AI can be deployed safely and effectively in banking and other regulated sectors.

---

## 1. System Architecture: Intelligent Multi-Agent Orchestration

At the core of the solution lies a **multi-agent orchestration engine**—implemented via the `AmexCoordinator` class—responsible for intelligently routing user queries to specialized agents based on semantic understanding and user role.

- **Router Pattern:** Rather than funneling every question through a single monolithic model, the system uses a router to determine whether the query is about rewards, general policy, or internal knowledge.
- **Specialized Engines:**  
  - *Rewards-related queries* go to a dedicated recommendation engine that applies business logic and LLM synthesis to explain eligibility in plain English.  
  - *Policy/FAQ questions* are routed to an assistant backed by a vectorized knowledge base (PDFs + FAQs).
- **Consistent Output:** Each response, regardless of agent, is run through a synthesis layer to ensure brand-appropriate tone and clarity.

**Why this matters:**  
This allows both scalability (agents can evolve independently) and strong separation of concerns (security and compliance can be enforced per agent and per user profile).

---

## 2. Dual-Profile Security: Role-Based Guardrails and Access Control

The system implements robust **Role-Based Access Control (RBAC)** for two user types:

- **External (Customer-facing):**  
  - Strictest security controls.
  - Sensitive topics and risky keywords are blocked.
  - All answers are grounded in approved, public information.

- **Internal (Employee-facing):**  
  - Lighter restrictions, but still robust.
  - Access to additional operational details, with guardrails to prevent sensitive leakage.

**How this is achieved:**  
All input and output are filtered via **AWS Bedrock Guardrails**, which include:
- Dynamic topic/word lists
- Real-time PII detection & anonymization
- Audit logging for compliance risk tracking

---

## 3. Knowledge Base: Hybrid, Multi-Source Retrieval-Augmented Generation (RAG)

A key innovation is the **hybrid knowledge architecture**, unifying both PDF policy documents and Notion FAQs:

- **PDF Pipeline:** Ingests policy docs, splits them into semantic chunks, and creates Titan embeddings.
- **Notion Integration:** Extracts structured FAQ content, including nested items.
- **Semantic Indexing:**  
  - All sources indexed with FAISS for fast, relevant retrieval.  
  - Deduplication & scoring logic ensure answers are accurate and non-redundant.

---

## 4. Advanced Security & Compliance: Bedrock Guardrails and PII Anonymization

Security and compliance are core, not afterthoughts:

- **Content Filtering:**  
  - Automated detection/blocking of hate, violence, sexual, and prompt-attack content, with configurable thresholds.
- **Topic/Keyword Filtering:**  
  - Custom deny-lists for each user role, blocking known prompt injection or policy evasion vectors.
- **PII Detection:**  
  - Real-time scanning for identifiers (SSNs, card numbers, addresses), with auto-anonymization.
- **Auditability:**  
  - All mechanisms and policy settings are fully logged and cached for reliability and cost control.

---

## 5. LLM-Driven Natural Language Synthesis (Claude 3.7 Sonnet)

This system uses a **dedicated synthesis agent** (Claude 3.7 Sonnet via Strands):

- **Brand Voice:** All answers rewritten to match the Amex brand—warm, friendly, accessible, never giving legal/financial advice.
- **Human-like Prompts:** System prompts engineered for empathetic, proactive responses.
- **Content Smoothing:** Removes legalese/repetition for a more engaging experience.

---

## 6. UI/UX and Monitoring: Streamlit as an Enterprise Interface

The interface, built in Streamlit, includes enterprise-grade features:

- **Status & Guardrail Panels:** Users always see current security posture and KB health.
- **Session Management:** Clear chat history and restart system for robust demos and reliability.
- **Safe Rendering:** HTML-escaped and styled for clarity and security.
- **Comprehensive Logging:** Observability into all user actions, agent decisions, and guardrail events.

---

## 7. Production-Ready: Scalability, Cost, and Observability

Designed for real-world, high-volume financial institutions:

- **Caching:** Vector stores and FAQ dumps cached for cost and speed.
- **Resource Management:** Lazy agent initialization, connection pooling, and configurable streaming.
- **Failure Handling:** Error boundaries and graceful degradation if sources are unavailable.

---


## 8. Monitoring, Observability, and Auditability: Bedrock Guardrails in Action

A critical requirement for regulated AI is robust, real-time monitoring and traceability of all content filtering. This project uses **AWS Bedrock Guardrails** for both enforcement and comprehensive monitoring.

**How it works:**

- **Live Enforcement Dashboard:**  
  Every input/output is evaluated in real time; enforcement status (filters, denied topics, custom words) is visible in a centralized dashboard.

- **Alerts & Blocked Message Logs:**  
  Triggered policies (e.g., “internal underwriting” or blocked words) are recorded, and branded notifications are shown to users. Events—including time, phrase, and action—are logged for compliance.

- **Policy Versioning & Change Tracking:**  
  Every guardrail change (new denied topic, updated filter, strength adjustments) is versioned and timestamped for full auditability.

- **PII Handling Audit Trail:**  
  Sensitive info is masked automatically; every instance is tracked for GDPR, CCPA, and privacy compliance.

- **Grounding Score Monitoring:**  
  Each answer’s grounding score is validated and logged, ensuring AI outputs are based on approved information.

- **Compliance-Ready Reporting:**  
  All monitoring data (block events, config history, PII detection) can be exported for audits or incident response, eliminating the need for custom monitoring solutions.

**Why this matters:**  
In regulated financial services, deploying guardrails is not enough—**organizations must prove enforcement and protection in real time**. Bedrock Guardrails monitoring provides the visibility, control, and traceability needed to satisfy legal, risk, and InfoSec teams—making this AI assistant truly enterprise-grade.


---

## 9. Why This Project Matters: Industry and Career Impact

- Tackles real blockers to GenAI in financial services: trustworthy retrieval, regulatory compliance, and consistent customer communication.
- Demonstrates expertise in multi-agent systems, secure RAG, and AWS enterprise AI.
- Transferable to other regulated industries: insurance, legal, healthcare.
- Serves as a proof point for technical depth (vector DBs, prompt engineering, orchestration) and enterprise awareness (compliance, brand safety, UX).

---


## Conclusion

By integrating the best of modern AI—secure RAG, brand-consistent LLM synthesis, enterprise guardrails, and robust multi-agent coordination—this project stands as a blueprint for **safe, scalable GenAI** in regulated industries.

---
