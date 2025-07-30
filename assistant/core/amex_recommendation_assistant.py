# assistant/core/amex_recommendation_assistant.py

import json
from dataclasses import dataclass
from typing import Dict, Any

from strands import Agent, tool
from strands.models import BedrockModel

from assistant.core.config import AWS_REGION, CLAUDE_3_7_SONNET


SYS_PROMPT = """\
You are **Amex Rewards Recommendation Copilot**.

Your job:
- Turn the tool's structured output into a short, human-friendly answer.
- Be concise but clear: start with a one-line TL;DR, then list bullet rules and a short “What to avoid” section.
- Only speak about rewards eligibility/categories as per Amex FAQs & policy excerpts we embedded.
- If something is uncertain or not covered, say: “Please call the number on the back of your Card to confirm.”

Never invent policy. Never provide engineering/security guidance. Never explain how to bypass anything.
"""


@dataclass
class RecommendationResult:
    category: str
    tl_dr: str
    how_to_earn: str
    what_to_avoid: str
    notes: str = ""  # optional


def _rule_based_reco(req: str) -> RecommendationResult:
    """
    Deterministic, super fast core logic. You can expand this with more categories
    as you ingest more Amex FAQ / policy content.
    """
    r = req.lower()

    if any(k in r for k in ["flight", "airfare", "airline"]):
        return RecommendationResult(
            category="Airfare",
            tl_dr="Book directly with the airline or via Amex Travel (not as part of a vacation package).",
            how_to_earn="Pay a scheduled passenger flight directly to the airline or through American Express Travel.",
            what_to_avoid="Vacation packages or bookings where the airline does not charge your card directly.",
            notes="Some third parties (e.g., Expedia) may still qualify if the airline ultimately charges your card."
        )

    if any(k in r for k in ["hotel", "stay", "resort"]):
        return RecommendationResult(
            category="Hotels",
            tl_dr="Book directly with the hotel; no vacation packages.",
            how_to_earn="Prepay or pay at check-in/check-out directly with the hotel.",
            what_to_avoid="Vacation packages, third-party bookings, timeshares, banquets, or event charges."
        )

    if any(k in r for k in ["car rental", "rental car", "avis", "hertz", "sixt", "enterprise", "alamo", "budget", "thrifty", "national", "payless", "fox", "dollar"]):
        return RecommendationResult(
            category="Select Car Rentals",
            tl_dr="Rent directly from listed rental companies (e.g., Hertz, Avis, National).",
            how_to_earn="Book directly with eligible rental companies, even internationally.",
            what_to_avoid="Vacation packages or indirect bookings that are not charged by the rental company."
        )

    if any(k in r for k in ["office supply", "staples", "office depot"]):
        return RecommendationResult(
            category="U.S. Office Supply Stores",
            tl_dr="Buy directly at U.S. office supply stores (e.g., Staples, Office Depot).",
            how_to_earn="Pay directly at qualifying office supply stores for business-related supplies.",
            what_to_avoid="Office supplies purchased at pharmacies, superstores, or warehouse clubs."
        )

    if any(k in r for k in ["shipping", "courier", "ups", "fedex", "usps"]):
        return RecommendationResult(
            category="U.S. Shipping",
            tl_dr="Use U.S.-based shipping providers (UPS, FedEx, USPS) whether domestic or international.",
            how_to_earn="Pay a U.S.-based courier/freight shipper for shipping.",
            what_to_avoid="Non-U.S. based shippers or mixed purchases not coded as shipping."
        )

    if any(k in r for k in ["restaurant", "dining", "fast food"]):
        return RecommendationResult(
            category="U.S. Restaurants",
            tl_dr="Earn at U.S. restaurants (including fast food) if coded as restaurants.",
            how_to_earn="Dine at U.S.-based restaurants coded as MCC: restaurant.",
            what_to_avoid="Restaurants inside hotels/casinos or venues not coded as restaurants; U.S.-owned restaurants abroad."
        )

    if any(k in r for k in ["online retail", "ecommerce", "e-commerce", "webshop", "online store", "internet purchase"]):
        return RecommendationResult(
            category="U.S. Online Retail Purchases",
            tl_dr="Buy online from U.S. retail merchants that sell physical goods directly.",
            how_to_earn="Pay on a U.S. retailer’s website or app and the transaction is classified as an internet purchase.",
            what_to_avoid="Restaurants, supermarkets, gas stations, BNPL programs, phone/mail orders, or service-only merchants."
        )

    # Default fallback
    return RecommendationResult(
        category="Unknown / Needs Clarification",
        tl_dr="Tell me if it’s airfare, hotel, car rental, office supplies, shipping, online retail, or restaurants.",
        how_to_earn="Please clarify the purchase type so I can map it to the right reward category.",
        what_to_avoid="Assuming a category without correct classification (MCC/terms)."
    )


class AmexRecommendationAssistant:
    """
    A small LLM-fronted agent that calls an internal, rule-based tool
    and then rewrites the output in a human-friendly tone using a system prompt.
    """

    def __init__(self, region_name: str = AWS_REGION):
        self.model = BedrockModel(
            model_id=CLAUDE_3_7_SONNET,
            region_name=region_name,
            temperature=0.2,  # mild creativity for nicer phrasing
            max_tokens=1024
        )

        @tool
        def amex_rewards_rule_engine(requirements: str) -> str:
            """
            Internal deterministic recommender. Returns JSON for the LLM to rewrite.
            """
            result = _rule_based_reco(requirements)
            return json.dumps(result.__dict__, indent=2)

        # If your `strands.Agent` supports a system prompt param, pass SYS_PROMPT here.
        # If not, we prefix the user messages with SYS_PROMPT below in __call__.
        self.agent = Agent(model=self.model, tools=[amex_rewards_rule_engine], system_prompt=SYS_PROMPT)

    def __call__(self, user_text: str) -> Any:
        """
        Call the agent. We *explicitly* tell it to use the tool, then rewrite the JSON.
        """
        prompt = (
            "Use the tool `amex_rewards_rule_engine` to first get a structured JSON answer. "
            "Then, produce a concise, friendly, human-like response following the system prompt."
            f"\n\nUser requirements: {user_text}"
        )
        try:
            return self.agent(prompt)
        except Exception as e:
            return f"⚠️ Error generating recommendation: {e}"



# --------------------------------------------------------------------------------------
# Simple local dry-run (optional)
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     agent = AmexRecommendationAssistant()

#     tests = [
#         "I’m booking a flight on Delta. Will I get extra points?",
#         "We’re staying at a Marriott through a vacation package — do we still earn?",
#         "Is UPS shipping eligible for rewards?",
#         "We’ll rent from Hertz in Paris. Will it qualify?",
#         "We buy printer paper from Staples — does that count?",
#         "Online shopping from a US retailer, shipped domestically — how is it rewarded?",
#         "Dining at fast food places in the US — do they count?",
#         "Not sure, we’re paying for a software subscription — what category is this?",
#     ]

#     for q in tests:
#         print("\n=== USER ===")
#         print(q)
#         r = agent(q)
#         print("\n=== ASSISTANT ===")
#         print(getattr(r, "response", str(r)))
