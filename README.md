Enterprise-Grade AI Customer Service Assistant for American Express
A Case Study in Secure, Scalable, and Responsible Generative AI in Financial Services

Executive Summary
This project is far more than a chatbot—it is a full-stack, production-ready AI customer service platform, designed specifically for the high-stakes regulatory environment of American Express and similar financial institutions. By combining advanced agent orchestration, hybrid knowledge retrieval, rigorous security controls, and brand-consistent language generation, this system demonstrates how modern AI can be deployed safely and effectively in banking and other regulated sectors.

1. System Architecture: Intelligent Multi-Agent Orchestration
At the core of the solution lies a multi-agent orchestration engine—implemented via the AmexCoordinator class—responsible for intelligently routing user queries to specialized agents based on semantic understanding and user role.

Rather than funneling every question through a single monolithic model, the system employs a “router” pattern. When a user submits a query, the router first determines whether the question is related to rewards, general policy, or requires internal knowledge.

Rewards-related queries are directed to a dedicated recommendation engine, which uses business logic and LLM synthesis to provide eligibility explanations in plain English.

Policy or FAQ questions are routed to a different assistant, which accesses a vectorized knowledge base containing official policy documents and FAQs.

Each response, regardless of origin, is then run through a synthesis layer that ensures a consistent, brand-appropriate tone and clarity.

Why this matters:
This approach allows for both scalability (different agents can be improved independently) and strong separation of concerns (security and compliance can be enforced per agent and per user profile).

2. Dual-Profile Security: Role-Based Guardrails and Access Control
Understanding the different risk profiles for customers versus employees, the system implements robust role-based access control (RBAC). There are two main user types:

External (Customer-facing):
These users are subject to the strictest security controls. Sensitive topics (like “fraud detection bypass” or “internal approval process”) and risky keywords are explicitly blocked. All answers are strictly grounded in approved, public information.

Internal (Employee-facing):
Employees require access to additional operational details, so their restrictions are lighter but still robust. Certain internal topics are permitted, but guardrails remain in place to prevent leakage of the most sensitive methods or system details.

How this is achieved:
All user input and model output are run through an advanced filtering system built on AWS Bedrock Guardrails. This includes dynamic topic/word lists, real-time PII detection and anonymization, and audit logging to track potential compliance risks.

3. Knowledge Base: Hybrid, Multi-Source Retrieval-Augmented Generation (RAG)
A key innovation is the system’s hybrid knowledge architecture, which unifies information from both official PDF policy documents and live-maintained Notion FAQs:

The PDF pipeline ingests large policy documents, splits them into manageable, semantically coherent chunks, and creates vector embeddings using Amazon Titan.

The Notion integration extracts both simple and nested FAQ content, preserving semantic structure for precise retrieval.

All sources are indexed using FAISS, enabling semantic similarity search and fast, relevant retrieval—even as documents grow or are updated.

Deduplication and scoring logic ensure that users receive accurate, non-redundant answers that combine the best of both official documentation and operational FAQs.

4. Advanced Security & Compliance: Bedrock Guardrails and PII Anonymization
Security and compliance are not afterthoughts—they’re at the core of the system.
Every piece of user input and LLM output passes through Bedrock Guardrails, which enforce a multilayered policy including:

Content filtering: Automated detection and blocking of hate, violence, sexual, and prompt attack content, with configurable thresholds.

Topic and keyword filtering: Custom deny-lists tuned for each user role, blocking known vectors for prompt injection or policy evasion.

PII detection: Real-time scanning for personal identifiers (like SSNs, card numbers, addresses), with automatic anonymization to prevent data leaks.

These mechanisms are fully auditable, and all policy settings are cached for both reliability and cost control.

5. LLM-Driven Natural Language Synthesis (Claude 3.7 Sonnet)
Unlike typical RAG pipelines that merely “paste together” snippets, this system employs a dedicated synthesis agent (powered by Claude 3.7 Sonnet via the Strands framework).

All answers are rewritten to match the American Express brand: warm, friendly, and accessible, but never giving legal or financial advice.

System prompts are engineered to produce human-like, empathetic responses—inviting further questions and providing proactive tips when possible.

The synthesis layer also removes legalese and repetitive text, resulting in a smoother, more engaging user experience.

6. UI/UX and Monitoring: Streamlit as an Enterprise Interface
The user interface is built in Streamlit, but elevated with features expected in enterprise-grade tools:

Real-time status and guardrail panels: Users always see the current security posture and knowledge base health.

Live session management: Supports easy clearing of chat history and system restarts for robust demos and production reliability.

Safe rendering: All messages are HTML-escaped and styled for both clarity and security.

Comprehensive logging provides observability into every user interaction, agent decision, and guardrail intervention.

7. Production-Ready: Scalability, Cost, and Observability
From deployment to day-to-day operation, this system is engineered for the demands of a real-world, high-volume financial institution:

Caching: Vector stores and Notion dumps are locally cached to minimize compute costs and ensure rapid recovery.

Resource management: Lazy agent initialization, connection pooling, and configurable streaming reduce infrastructure overhead.

Failure handling: All critical processes have error boundaries, and the system degrades gracefully if a source becomes unavailable.

8. Why This Project Matters: Industry and Career Impact
This project tackles the real blockers to GenAI in financial services—not just language generation, but trustworthy retrieval, regulatory compliance, and safe, consistent customer communication.

It demonstrates deep expertise in multi-agent systems, secure RAG, and AWS enterprise AI tools.

The architecture is directly transferable to insurance, legal, healthcare, and other regulated industries.

For career advancement, it serves as a proof point for both technical depth (vector databases, prompt engineering, agent orchestration) and enterprise awareness (compliance, brand safety, user experience).

Conclusion
By bringing together the best of modern AI—secure RAG, brand-consistent LLM synthesis, enterprise guardrails, and robust multi-agent coordination—this project stands out as a blueprint for safe, scalable GenAI in any regulated industry.
It doesn’t just showcase what you can build, but how to build it the right way for banks, insurers, and anyone serious about putting AI into production.

