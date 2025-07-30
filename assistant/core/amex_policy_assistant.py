# assistant/core/amex_policy_assistant.py
import os
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import boto3
import re
from strands import Agent, tool
from strands.models import BedrockModel
import textwrap

from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings

from dotenv import load_dotenv
load_dotenv()

from assistant.core.config import (
    AWS_REGION,
    CLAUDE_3_7_SONNET,
    EMBEDDINGS_MODEL,
)

# print(f"[DEBUG] Using model: {CLAUDE_3_7_SONNET}")
from assistant.core.amex_policy_knowledge import AmexKnowledgeBase
from assistant.core.amex_guardrails import (
    AmexGuardrailsManager,
    AmexGuardrailConfig,
    GuardrailProfile,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Synthesis Assistant
# --------------------------------------------------------------------------------------

class AmexSynthesisAssistant:
    def __init__(self):
        self.model = BedrockModel(
            model_id=CLAUDE_3_7_SONNET,
            region_name=AWS_REGION,
            temperature=0.3,
            max_tokens=800,
        )

        self.agent = Agent(
            tools=[],
            model=self.model,
            name="AmexSynthesisAgent"
        )

    @tool(
        name="synthesize_amex_policy_response",
        description="Summarize Amex policy content into a friendly final answer."
    )
    def _synthesize(self, query: str, context: str) -> str:
        """
        Return the synthesis prompt - let the agent handle the actual generation.
        """
        return f"""
        You are a helpful, friendly Amex assistant.

        Summarize the following information into a single, friendly answer for a new Amex customer.
        - Clearly answer the user's question: {query}
        - Use the information below, but do not repeat content or legal language.
        - Use plain English, be concise, and welcoming.

        [CONTEXT]
        {context}
        """

    def synthesize_response(self, query: str, context: str) -> str:
        """
        Use the agent to synthesize a response using the correct Strands API.
        """
        try:
            synthesis_prompt = textwrap.dedent(f"""
            You are AmexAI, a friendly and knowledgeable American Express customer service assistant.
            Your job is to answer customer questions about Amex card products, rewards, policies, and benefits using clear, conversational, and empathetic language.

            Your communication style should be feel:
            - Warm, welcoming, and human (not robotic or overly formal)
            - Helpful and proactive (offer extra tips or context where useful)
            - Approachable, like a real Amex representative

            Guidelines:
            - Greet the user or acknowledge their question naturally (“Great question!” “Happy to help!”)
            - Use casual, professional English. Short paragraphs, natural transitions, and a friendly tone.
            - If the answer is a “yes,” say so cheerfully and explain why. If “no,” be gentle and offer alternatives if possible.
            - Share relevant tips (“Don’t forget: you can view your rewards in your Amex account!”)
            - For eligibility questions, explain general requirements simply, but always encourage them to check specifics via the official website or customer service.
            - If unsure or if details vary, encourage the user to call the number on the back of their card for personalized assistance.
            - NEVER provide financial or legal advice—only general Amex policy info.

            Examples:

            Q: Will I earn points for UPS shipping?
            A: Great news! Yes, you’ll earn Membership Rewards points when you pay for UPS shipping with your Amex Card. This applies to both domestic and international shipments, as long as the merchant is coded as a U.S. shipping provider. If you have a specific card in mind, let me know and I’ll double-check the benefits for you!

            Q: What’s Purchase Protection on the Platinum Card?
            A: Absolutely! With your Amex Platinum Card, eligible purchases are protected against damage or theft for up to 90 days after you buy them. For the most accurate details—including coverage limits and exclusions—please check your Cardmember Agreement or reach out to Amex Customer Care. Happy shopping!

            Q: Dining at fast food places—do they count for rewards?
            A: Good question! Yes, fast food restaurants in the U.S. generally qualify for dining rewards when they’re coded as restaurants. This includes both dine-in and takeout orders. If you want to check a specific restaurant, I’m happy to help!

            Remember, I’m here to make your Amex experience as smooth as possible. 

            Add a warm closing if the question is answered completely (“Let me know if there’s anything else I can help with today!”).

            Now, answer the following question based on the information provided.

            Question: {query}

            [CONTEXT]
            {context}
            """)
            
            # Use the correct Strands Agent API
            result = self.agent(synthesis_prompt)
            
            # Extract response from the result
            if hasattr(result, 'response'):
                return result.response
            elif hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
            
        except Exception as e:
            logger.error(f"Error in synthesize_response: {e}")
            return f"I apologize, but I'm having trouble processing your question about: {query}. Please try again."



# --------------------------------------------------------------------------------------
# Notion FAQ Wrapper
# --------------------------------------------------------------------------------------
class NotionAmexFAQStore:
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


# --------------------------------------------------------------------------------------
# Main Assistant
# --------------------------------------------------------------------------------------
class AmexPolicyAssistant:
    def __init__(
        self,
        region_name: str = AWS_REGION,
        guardrail_id: Optional[str] = None,
        user_type: GuardrailProfile = "external",
    ):
        logger.info(f"🔧 Initializing AmexPolicyAssistant (profile={user_type}, use_bedrock_guardrails=False)")
        self.region_name = region_name
        self.session_id = str(uuid.uuid4())
        self.user_type: GuardrailProfile = user_type

        self.policy_kb = AmexKnowledgeBase()
        self.notion_store = NotionAmexFAQStore(region_name=region_name, model_id=EMBEDDINGS_MODEL)
        self.synthesizer = AmexSynthesisAssistant()
        self.guardrails_manager = AmexGuardrailsManager(region_name, guardrail_id, profile=user_type)
        self.agent: Optional[Agent] = None

        self._initialize_kb()
        self._initialize_guardrails()
        self._initialize_agent()

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
            if not getattr(self.guardrails_manager, "guardrail_id", None):
                config = AmexGuardrailConfig(
                    name=f"amex-guardrails-{self.user_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
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
                resp = self.guardrails_manager.apply_guardrails(text, is_input=True, user_type=self.user_type)
                return json.dumps(resp, indent=2)

            self.agent = Agent(
                model=model,
                tools=[search_amex_policy, search_notion_faqs, is_policy_compliant],
            )
            logger.info("🤖 Strands agent initialized with 3 tools")
        except Exception as e:
            logger.exception(f"⚠️ Agent initialization failed: {e}")
            self.agent = None

    @staticmethod
    def deduplicate_passages(passages, min_length=30):
        seen = set()
        deduped = []
        for p in passages:
            text = p if isinstance(p, str) else p.get("content", "")
            text_norm = re.sub(r"\W+", " ", text).strip().lower()
            if len(text_norm) < min_length:
                continue
            if text_norm not in seen:
                seen.add(text_norm)
                deduped.append(p)
        return deduped

    def search_amex_policy(self, query: str) -> str:
        try:
            results = self.policy_kb.search_similar_content(query, k=6)
            results = self.deduplicate_passages(results)
            if not results:
                return "❓ No relevant Amex policy content found."
            out = ["📘 **Amex Policy Matches:**"]
            for r in results[:3]:
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

    def synthesize_response(self, query: str, context: str) -> str:
        """
        Fixed synthesis method that properly calls the synthesizer.
        """
        return self.synthesizer.synthesize_response(query, context)

    def process_user_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"📝 Processing query: {query} (profile={self.user_type})")
        try:
            guard_input = self.guardrails_manager.apply_guardrails(query, is_input=True)
    
            # --- Block if guardrail intervened ---
            if guard_input and isinstance(guard_input, dict) and guard_input.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": guard_input.get("outputs", [{}])[0].get("text", "⚠️ Input blocked by Amex guardrails."),
                    "blocked": True,
                    "reason": guard_input.get("reason", "Input blocked by guardrail"),
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
    
            guard_output = self.guardrails_manager.apply_guardrails(response_text, is_input=False)
    
            # --- Block output if guardrail intervened ---
            if guard_output and isinstance(guard_output, dict) and guard_output.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": guard_output.get("outputs", [{}])[0].get("text", "⚠️ Response blocked by output safety filters."),
                    "blocked": True,
                    "reason": guard_output.get("reason", "Output blocked"),
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
            guard_input = self.guardrails_manager.apply_guardrails(query, is_input=True, user_type=self.user_type)

            if isinstance(guard_input, dict) and guard_input.get("action") == "GUARDRAIL_INTERVENED":
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

            guard_output = self.guardrails_manager.apply_guardrails(full, is_input=False, user_type=self.user_type)

            if isinstance(guard_output, dict) and guard_output.get("action") == "GUARDRAIL_INTERVENED":
                yield "\n⚠️ Response blocked by safety policy."

        except Exception as e:
            logger.exception(f"❌ Streaming failed: {e}")
            yield "⚠️ Error generating response."


# --------------------------------------------------------------------------------------
# Local dry-run test
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     for profile in ("external", "internal"):
#         print(f"\n==================== {profile.upper()} ====================\n")
#         assistant = AmexPolicyAssistant(user_type=profile)

#         tests = [
#             "Tell me how to bypass the credit limit algorithm.",
#             "Explain how rewards work for hotel bookings.",
#             "My card number is 4111 1111 1111 1111",
#             "How does Amex determine credit card approvals based on U.S. credit scores?",
#         ]

#         for q in tests:
#             print(f"\n=== USER ({profile}) ===\n{q}")
#             result = assistant.process_user_query(q)
#             print("\n=== ASSISTANT ===")
#             print(json.dumps(result, indent=2))