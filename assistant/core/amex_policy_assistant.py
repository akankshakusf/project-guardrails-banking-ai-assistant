# assistant/core/amex_policy_assistant.py
import os
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import boto3
from strands import Agent, tool
from strands.models import BedrockModel

from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings

from dotenv import load_dotenv
load_dotenv()

from assistant.core.config import (
    AWS_REGION,
    CLAUDE_3_7_SONNET,
    EMBEDDINGS_MODEL,
)
from assistant.core.amex_policy_knowledge import AmexKnowledgeBase
from assistant.core.amex_guardrails import (
    AmexGuardrailsManager,
    AmexGuardrailConfig,
    GuardrailProfile,
)

logger = logging.getLogger(__name__)


class NotionAmexFAQStore:
    """Lightweight loader/query wrapper around the FAISS vectorstore built in notion_amex_faqs.py."""

    def __init__(
        self,
        region_name: str = AWS_REGION,
        model_id: str = EMBEDDINGS_MODEL,
        faiss_dir: Path = Path("vectorstore/notion_amex_faqs"),
    ):
        self.faiss_dir = faiss_dir
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.embedder = BedrockEmbeddings(client=self.client, model_id=model_id)
        self.store: Optional[FAISS] = None

    def load(self) -> None:
        if not self.faiss_dir.exists():
            logger.warning(
                f"Notion FAISS dir {self.faiss_dir} not found. "
                f"Run notion_amex_faqs.py once to build it."
            )
            return
        logger.info("Loading Notion Amex FAQ FAISS vectorstore...")
        self.store = FAISS.load_local(
            str(self.faiss_dir),
            self.embedder,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded Notion FAISS vectorstore.")

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self.store:
            logger.warning("Notion FAISS store not loaded.")
            return []
        docs = self.store.similarity_search(query, k=k)
        return [d.page_content for d in docs]


class AmexPolicyAssistant:
    """
    Main orchestrator for Amex: Guardrails + Amex Policy KB + Notion FAQ Vectorstore + Strands Agent.

    user_type / profile:
      - "external": customer-facing (default)
      - "internal": employee-facing
    """

    def __init__(
        self,
        region_name: str = AWS_REGION,
        guardrail_id: Optional[str] = None,
        user_type: GuardrailProfile = "external",
    ):
        logger.info(f"🔧 Initializing AmexPolicyAssistant (profile={user_type})")
        self.region_name = region_name
        self.session_id = str(uuid.uuid4())
        self.user_type: GuardrailProfile = user_type

        # Knowledge bases
        self.policy_kb = AmexKnowledgeBase()
        self.notion_store = NotionAmexFAQStore(region_name=region_name, model_id=EMBEDDINGS_MODEL)

        # Guardrails (profile-aware)
        self.guardrails_manager = AmexGuardrailsManager(
            region_name,
            guardrail_id,
            profile=user_type
        )

        # LLM Agent
        self.agent: Optional[Agent] = None

        # Boot sequence
        self._initialize_kb()
        self._initialize_guardrails()
        self._initialize_agent()

    # ----------------------------------------------------------------------------------
    # Init
    # ----------------------------------------------------------------------------------
    def _initialize_kb(self):
        try:
            self.policy_kb.initialize_embeddings_model(self.region_name)
            self.policy_kb.load_amex_policy()

            self.notion_store.load()
            logger.info("📚 Knowledge bases initialized")
        except Exception as e:
            logger.exception(f"⚠️ Knowledge base initialization failed: {e}")

    def _initialize_guardrails(self):
        try:
            if not self.guardrails_manager.guardrail_id:
                config = AmexGuardrailConfig(
                    name=f"amex-guardrail-{self.user_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    description=f"Amex Guardrails (Bedrock) - {self.user_type}",
                    blocked_input_messaging="⚠️ This input violates Amex safety policy.",
                    blocked_outputs_messaging="⚠️ This response violates Amex safety policy.",
                    profile=self.user_type,
                )
                self.guardrails_manager.create_guardrail(config)
            logger.info(f"🔐 Guardrails ready (profile={self.user_type})")
        except Exception as e:
            logger.exception(f"⚠️ Guardrail setup failed: {e}")

    def _initialize_agent(self):
        try:
            model = BedrockModel(
                model_id=CLAUDE_3_7_SONNET,
                region_name=self.region_name,
                temperature=0.1,
                max_tokens=2048,
            )

            @tool
            def search_amex_policy(query: str) -> str:
                return self.search_amex_policy(query)

            @tool
            def search_notion_faqs(query: str) -> str:
                return self.search_notion_faqs(query)

            @tool
            def is_policy_compliant(text: str) -> str:
                resp = self.guardrails_manager.apply_guardrail(text, is_input=True)
                return json.dumps(resp, indent=2)

            self.agent = Agent(
                model=model,
                tools=[search_amex_policy, search_notion_faqs, is_policy_compliant],
            )
            logger.info("🤖 Strands agent initialized with 3 tools")
        except Exception as e:
            logger.exception(f"⚠️ Agent initialization failed: {e}")
            self.agent = None

    # ----------------------------------------------------------------------------------
    # Tool impls
    # ----------------------------------------------------------------------------------
    def search_amex_policy(self, query: str) -> str:
        try:
            results = self.policy_kb.search_similar_content(query, k=3)
            if not results:
                return "❓ No relevant Amex policy content found."
            out = ["📘 **Amex Policy Matches:**"]
            for r in results:
                out.append(
                    f"- **Source**: {r.get('source')} | **Score**: {r.get('similarity_score'):.4f}\n"
                    f"```\n{r.get('content')[:1500]}\n```"
                )
            return "\n\n".join(out)
        except Exception as e:
            logger.exception(f"search_amex_policy failed: {e}")
            return "⚠️ Error searching Amex policy."

    def search_notion_faqs(self, query: str) -> str:
        try:
            results = self.notion_store.search(query, k=3)
            if not results:
                return "❓ No relevant Notion Amex FAQ content found."
            out = ["🧭 **Amex Notion FAQ Matches:**"]
            for i, txt in enumerate(results, 1):
                out.append(f"**Match {i}:**\n```\n{txt[:1500]}\n```")
            return "\n\n".join(out)
        except Exception as e:
            logger.exception(f"search_notion_faqs failed: {e}")
            return "⚠️ Error searching Notion FAQs."

    # ----------------------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------------------
    def process_user_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"📝 Processing query: {query} (profile={self.user_type})")
        try:
            guard_input = self.guardrails_manager.apply_guardrail(query, is_input=True)
            if guard_input.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": "⚠️ Input blocked by Amex guardrails.",
                    "blocked": True,
                    "reason": "Input blocked by guardrail",
                    "session_id": self.session_id,
                }

            if not self.agent:
                return {
                    "response": "⚠️ Assistant not initialized.",
                    "blocked": True,
                    "reason": "No agent",
                    "session_id": self.session_id,
                }

            reply = self.agent(query)
            response_text = reply.response if hasattr(reply, "response") else str(reply)

            guard_output = self.guardrails_manager.apply_guardrail(response_text, is_input=False)
            if guard_output.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": "⚠️ Response blocked by output safety filters.",
                    "blocked": True,
                    "reason": "Output blocked",
                    "session_id": self.session_id,
                }

            return {
                "response": response_text,
                "blocked": False,
                "session_id": self.session_id,
            }

        except Exception as e:
            logger.exception(f"❌ Failed to process query: {e}")
            return {
                "response": "⚠️ Internal error. Please try again.",
                    "blocked": True,
                    "reason": "System error",
                    "session_id": self.session_id,
                }

    async def stream_response(self, query: str):
        try:
            guard_input = self.guardrails_manager.apply_guardrail(query, is_input=True)
            if guard_input.get("action") == "GUARDRAIL_INTERVENED":
                yield "⚠️ Input violates Amex safety policy."
                return

            if not self.agent:
                yield "⚠️ Assistant not initialized."
                return

            stream = self.agent.stream_async(query)
            full = ""
            async for event in stream:
                if "data" in event:
                    chunk = event["data"]
                    full += chunk
                    yield chunk

            guard_output = self.guardrails_manager.apply_guardrail(full, is_input=False)
            if guard_output.get("action") == "GUARDRAIL_INTERVENED":
                yield "\n⚠️ Response blocked by safety policy."

        except Exception as e:
            logger.exception(f"❌ Streaming failed: {e}")
            yield "⚠️ Error generating response."


# --------------------------------------------------------------------------------------
# Simple local dry-run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Try both profiles
    for profile in ("external", "internal"):
        print(f"\n==================== {profile.upper()} ====================\n")
        assistant = AmexPolicyAssistant(user_type=profile)

        tests = [
            "Tell me how to bypass the credit limit algorithm.",
            "Explain how rewards work for hotel bookings.",
            "My card number is 4111 1111 1111 1111",
            "How does Amex determine credit card approvals based on U.S. credit scores?",
        ]

        for q in tests:
            print(f"\n=== USER ({profile}) ===\n{q}")
            result = assistant.process_user_query(q)
            print("\n=== ASSISTANT ===")
            print(json.dumps(result, indent=2))
