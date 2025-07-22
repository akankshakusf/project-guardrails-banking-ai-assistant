## assistant.py
import uuid
import logging
from strands import Agent, tool
from strands.models import BedrockModel
from datetime import datetime

from assistant.core.config import AWS_REGION, CLAUDE_3_7_SONNET
from assistant.core.guardrails import BedrockGuardrailsManager, GuardrailConfig
from assistant.core.knowledge_base import AwsGenAIKnowledgeBase

logger = logging.getLogger(__name__)


class AwsGenAIAssistant:
    """Main Orchestrator: Combines Guardrails, Knowledge Base, and Strands Agent."""

    def __init__(self, region_name: str = AWS_REGION, guardrail_id: str = None):
        logger.info("🔧 Initializing AWS GenAI Assistant")
        self.region_name = region_name
        self.knowledge_base = AwsGenAIKnowledgeBase()
        self.guardrails_manager = BedrockGuardrailsManager(region_name, guardrail_id)
        self.session_id = str(uuid.uuid4())
        self.agent = None

        self._initialize_knowledge_base()
        self._initialize_guardrails()
        self._initialize_agent()

    def _initialize_knowledge_base(self):
        try:
            self.knowledge_base.initialize_embeddings_model(self.region_name)
            self.knowledge_base.load_aws_genai_guidelines()
            logger.info("📚 Knowledge base loaded")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge base failed: {e}")

    def _initialize_guardrails(self):
        try:
            if not self.guardrails_manager.guardrail_id:
                config = GuardrailConfig(
                    name=f"amex-genai-guardrail-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    description="Guardrails for Banking AI Assistant using Bedrock",
                    blocked_input_messaging="⚠️ Input violates safety policy.",
                    blocked_outputs_messaging="⚠️ Output violates safety policy."
                )
                self.guardrails_manager.create_guardrail(config)
            logger.info("🔐 Guardrails ready")
        except Exception as e:
            logger.warning(f"⚠️ Guardrail setup failed: {e}")

    def _initialize_agent(self):
        try:
            model = BedrockModel(
                model_id=CLAUDE_3_7_SONNET,
                region_name=self.region_name,
                temperature=0.1,
                max_tokens=2048
            )

            @tool
            def search_genai_knowledge(query: str) -> str:
                return self.search_genai_knowledge(query)

            @tool
            def recommend_aws_service(requirements: str) -> str:
                """Recommend AWS GenAI service based on use case."""
                req = requirements.lower()
                if "custom" in req or "training" in req or "ml model" in req:
                    return "🧠 Use **Amazon SageMaker** for building and training custom ML models."
                elif "foundation model" in req or "api" in req:
                    return "⚙️ Use **Amazon Bedrock** to access foundation models via API with built-in guardrails."
                elif "enterprise" in req or "secure internal data" in req:
                    return "🏢 **Amazon Q Business** is ideal for secure GenAI integration in enterprises."
                else:
                    return "🤔 It depends on your needs. SageMaker is best for custom ML, Bedrock for easy APIs, and Q Business for enterprise copilots."

            self.agent = Agent(
                model=model,
                tools=[search_genai_knowledge, recommend_aws_service]
            )
            logger.info("🤖 Strands agent initialized with 2 tools")

        except Exception as e:
            logger.warning(f"⚠️ Agent initialization failed: {e}")
            self.agent = None

    @tool
    def search_genai_knowledge(self, query: str) -> str:
        try:
            results = self.knowledge_base.search_similar_content(query, k=3)
            if not results:
                return "❓ No relevant knowledge found. Please consult AWS documentation."
            response = "📘 Based on AWS docs:\n\n"
            for r in results:
                response += f"• {r['content']}\nSource: {r['source']}\n\n"
            return response
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return "⚠️ Error searching knowledge base."

    def process_user_query(self, query: str) -> dict:
        logger.info(f"📝 Processing query: {query}")
        try:
            guard_input = self.guardrails_manager.apply_guardrail(query, is_input=True)
            if guard_input.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": "⚠️ Input blocked by safety guardrails.",
                    "blocked": True,
                    "reason": "Input blocked by guardrail"
                }

            if not self.agent:
                return {
                    "response": "⚠️ Assistant not initialized.",
                    "blocked": True,
                    "reason": "No agent"
                }

            reply = self.agent(query)
            response_text = reply.response if hasattr(reply, "response") else str(reply)

            guard_output = self.guardrails_manager.apply_guardrail(response_text, is_input=False)
            if guard_output.get("action") == "GUARDRAIL_INTERVENED":
                return {
                    "response": "⚠️ Response blocked by output safety filters.",
                    "blocked": True,
                    "reason": "Output blocked"
                }

            return {
                "response": response_text,
                "blocked": False,
                "session_id": self.session_id
            }

        except Exception as e:
            logger.error(f"❌ Failed to process query: {e}")
            return {
                "response": "⚠️ Internal error. Please try again.",
                "blocked": True,
                "reason": "System error"
            }

    async def stream_response(self, query: str):
        """Async generator for streaming response chunks."""
        try:
            guard_input = self.guardrails_manager.apply_guardrail(query, is_input=True)
            if guard_input.get("action") == "GUARDRAIL_INTERVENED":
                yield "⚠️ Input violates safety policy."
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
            logger.error(f"❌ Streaming failed: {e}")
            yield "⚠️ Error generating response."
