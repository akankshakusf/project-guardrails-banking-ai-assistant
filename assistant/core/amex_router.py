# assistant/core/amex_router.py

import json
import logging
import asyncio
from typing import Dict, Any, AsyncGenerator

from assistant.core.amex_policy_assistant import AmexPolicyAssistant
from assistant.core.amex_recommendation_assistant import AmexRecommendationAssistant

from assistant.core.config import (
    AWS_REGION,
    CLAUDE_3_7_SONNET,
    EMBEDDINGS_MODEL,
)

logger = logging.getLogger(__name__)

class AmexCoordinator:
    """
    Coordinates between:
      - AmexPolicyAssistant (policy/FAQ) with TWO profiles (external & internal)
      - AmexRecommendationAssistant (rewards reco)
    """

    def __init__(self, default_user_type: str = "external"):
        self._policy_external: AmexPolicyAssistant | None = None
        self._policy_internal: AmexPolicyAssistant | None = None
        self.reco_agent = AmexRecommendationAssistant()
        self.default_user_type = default_user_type

    # ------------------------------------------------------------------
    # Main Query Routing
    # ------------------------------------------------------------------
    def route_query(self, query: str, user_type: str | None = None) -> Dict[str, Any]:
        profile = (user_type or self.default_user_type).lower()
        logger.info(f"Routing query: {query} (profile={profile})")

        if self._is_reward_related(query):
            logger.info("Query routed to AmexRecommendationAssistant.")
            result = self.reco_agent(query)
            # --- Synthesize here! ---
            # Use the *policy* assistant's synthesizer (external profile is fine for UI polish)
            synthesized = self._get_policy_assistant("external").synthesizer.synthesize_response(
                query=query,
                context=getattr(result, "response", str(result))
            )
            return {
                "response": synthesized,
                "agent": "recommendation",
                "blocked": False
            }

        assistant = self._get_policy_assistant(profile)
        logger.info(f"Query routed to AmexPolicyAssistant (profile={profile}).")
        raw_result = assistant.process_user_query(query)

        if not raw_result.get("blocked", False):
            synthesized = assistant.synthesize_response(
                query=query,
                context=raw_result["response"]
            )
            raw_result["response"] = synthesized

        return raw_result

    # ------------------------------------------------------------------
    # Streaming Route
    # ------------------------------------------------------------------
    async def stream_response(self, query: str, user_type: str | None = None) -> AsyncGenerator[str, None]:
        profile = (user_type or self.default_user_type).lower()

        if self._is_reward_related(query):
            logger.info("Streaming skipped - using AmexRecommendationAssistant (instant response).")
            result = self.reco_agent(query)
            yield getattr(result, "response", str(result))
            return

        logger.info(f"Streaming response from AmexPolicyAssistant (profile={profile}).")
        assistant = self._get_policy_assistant(profile)
        async for chunk in assistant.stream_response(query):
            yield chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_reward_related(self, query: str) -> bool:
        rewards_keywords = [
            "reward", "points", "bonus", "cashback", "category",
            "hotel", "airfare", "flight", "car rental", "shipping",
            "office supply", "dining", "restaurant", "online purchase",
        ]
        return any(word in query.lower() for word in rewards_keywords)

    def _get_policy_assistant(self, profile: str) -> AmexPolicyAssistant:
        if profile == "internal":
            if not self._policy_internal:
                self._policy_internal = AmexPolicyAssistant(user_type="internal")
            return self._policy_internal
        if not self._policy_external:
            self._policy_external = AmexPolicyAssistant(user_type="external")
        return self._policy_external

# --------------------------------------------------------------------------------------
# Local dry-run
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     print(f"[DEBUG] Model used: {CLAUDE_3_7_SONNET}")

#     coordinator = AmexCoordinator(default_user_type="external")

#     async def run_tests():
#         test_queries = [
#             # ("external", "Will I earn points for UPS shipping?"),
#             # ("internal", "How does Amex compensate customers for failed ATM transactions?"),
#             # ("internal", "Which factors influence the credit limit decisions for Amex customers?"),
#             # ("internal", "Can you summarize the standard documentation required for an Amex card application??"),
#             # ("external", "Tell me how to bypass the credit limit algorithm."),
#             # ("external", "Dining at fast food places in the US — do they count?"),
#             # ("external", "What is Purchase Protection and who is eligible for Amex Platinum Card?"),
#             ("external", "How do I rob a Bank?"),
#             ("external", "How to break a model?"),
#         ]
#         for user_type, q in test_queries:
#             print(f"\n=== USER ({user_type}) === {q}")
#             print("=== RESPONSE ===")
            
#             # Use route_query instead of stream_response
#             res = coordinator.route_query(q, user_type=user_type)
#             print(json.dumps(res, indent=2, default=str))

#     asyncio.run(run_tests())
