import logging
import os
import json
import re
import boto3
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = ".cache"
os.makedirs(CONFIG_DIR, exist_ok=True)

@dataclass
class AmexGuardrailConfig:
    name: str
    description: str
    profile: str
    blocked_input_messaging: str = "⚠️ Input blocked by Amex policy."
    blocked_outputs_messaging: str = "⚠️ Output restricted by Amex policy."

@dataclass
class GuardrailProfile:
    profile: str
    use_bedrock_guardrails: bool = True
    filter_input: bool = True
    filter_output: bool = True
    grounding_required: bool = False
    denied_topics: List[str] = field(default_factory=list)
    blocked_words: List[str] = field(default_factory=list)
    allowed_words: List[str] = field(default_factory=list)
    pii_entities: List[str] = field(default_factory=list)

class BedrockGuardrailWrapper:
    @staticmethod
    def run_guardrails(
        text,
        guardrail_id,
        profile,
        filter_type,
        denied_topics,
        blocked_words,
        allowed_words,
        pii_entities,
        grounding_required,
    ):
        for topic in denied_topics:
            if topic.lower() in text.lower():
                logger.debug(f"Matched denied topic: {topic}")
                return {
                    "action": "GUARDRAIL_INTERVENED",
                    "reason": f"Matched denied topic: {topic}",
                    "outputs": [{"text": f"⚠️ Topic '{topic}' is restricted."}]
                }
        return {"action": "ALLOW"}

class AmexGuardrailsManager:
    def __init__(self, region_name: str, guardrail_id: Optional[str] = None, *, profile: str = "external"):
        self.active_profile = profile
        self.guardrail_id = guardrail_id  # ✅ FIXED
        self.profiles: Dict[str, GuardrailProfile] = {
            "external": GuardrailProfile(
                profile="external",
                use_bedrock_guardrails=True,
                filter_input=True,
                filter_output=True,
                grounding_required=True,
                denied_topics=[
                    "Fraud detection bypass", "Credit risk strategy", "Internal approval process",
                    "Model jailbreaking", "Unauthorized model access", "Bypassing Amex guardrails",
                    "Data exfiltration", "Unauthorized Amex APIs", "Internal underwriting",
                    "Internal collections framework", "Credit limit algorithm", "Chargeback trick",
                    "Unsolicited card exploit"
                ],
                blocked_words=[
                    "hack", "bypass", "exploit", "reverse engineer", "fraud",
                    "unauthorized", "adversarial", "jailbreak", "malicious",
                    "sql injection", "scrape notion", "data exfiltration"
                ],
                allowed_words=[],
                pii_entities=[
                    "NAME", "EMAIL", "PHONE", "ADDRESS",
                    "US_SOCIAL_SECURITY_NUMBER", "CREDIT_DEBIT_CARD_NUMBER",
                    "DRIVER_ID", "US_PASSPORT_NUMBER"
                ]
            ),
            "internal": GuardrailProfile(
                profile="internal",
                use_bedrock_guardrails=True,
                filter_input=True,
                filter_output=True,
                grounding_required=False,
                denied_topics=[
                    "fraud techniques", "how to bypass", "exploit", "internal system flaws"
                ],
                blocked_words=["placeholder"],  # ✅ Added to meet Bedrock requirements
                allowed_words=[
                    "fraud", "risk", "scoring", "credit limit",
                    "chargeback", "delinquency", "underwriting"
                ],
                pii_entities=[
                    "NAME", "EMAIL", "PHONE", "US_SOCIAL_SECURITY_NUMBER"
                ]
            )
        }

    def apply_guardrails(self, text: str, user_type: str = "external", is_input: bool = True) -> Optional[str]:
        profile_key = user_type.lower()
        profile = self.profiles.get(profile_key)

        if not profile:
            logger.warning(f"No guardrail profile found for user_type: {user_type}")
            return None

        if not profile.use_bedrock_guardrails:
            logger.debug(f"Guardrails disabled for profile: {profile_key}")
            return None

        logger.debug(f"Applying Bedrock guardrails (input={is_input}) for profile: {profile_key}")

        result = BedrockGuardrailWrapper.run_guardrails(
            text=text,
            guardrail_id=f"amex-guardrail-{profile_key}",
            profile=profile.profile,
            filter_type="input" if is_input else "output",
            denied_topics=profile.denied_topics,
            blocked_words=profile.blocked_words,
            allowed_words=profile.allowed_words,
            pii_entities=profile.pii_entities,
            grounding_required=profile.grounding_required,
        )

        if result.get("action") == "GUARDRAIL_INTERVENED":
            logger.warning(f"Blocked by Bedrock: {result.get('reason')}")
            return result.get("outputs", [{}])[0].get("text", "⚠️ This content violates policy.")

        return None

    def create_guardrail(self, config: AmexGuardrailConfig) -> str:
        profile = self.profiles.get(config.profile)
        if not profile:
            raise ValueError(f"No profile found for '{config.profile}'")

        bedrock_admin = boto3.client("bedrock")

        logger.info(f"Creating Amex guardrail for profile: {config.profile}")

        topic_policy = {
            "topicsConfig": [
                {
                    "name": re.sub(r"[^a-zA-Z0-9_\-]", "", topic.replace(" ", "_")).lower(),
                    "definition": topic,
                    "examples": [f"What is {topic.lower()}?"],
                    "type": "DENY"
                }
                for topic in profile.denied_topics
            ]
        }

        word_policy = {
            "wordsConfig": [{"text": w} for w in profile.blocked_words or ["placeholder"]]
        }

        pii_policy = {
            "piiEntitiesConfig": [
                {"type": ent, "action": "ANONYMIZE"} for ent in profile.pii_entities
            ]
        }

        content_policy = {
            "filtersConfig": [
                {"type": "HATE", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
            ]
        }

        kwargs = dict(
            name=config.name,
            description=config.description,
            contentPolicyConfig=content_policy,
            topicPolicyConfig=topic_policy,
            wordPolicyConfig=word_policy,
            sensitiveInformationPolicyConfig=pii_policy,
            blockedInputMessaging=config.blocked_input_messaging,
            blockedOutputsMessaging=config.blocked_outputs_messaging,
        )

        if profile.grounding_required:
            kwargs["contextualGroundingPolicyConfig"] = {
                "filtersConfig": [{"type": "GROUNDING", "threshold": 0.75}]
            }

        response = bedrock_admin.create_guardrail(**kwargs)

        guardrail_id = response["guardrailId"]
        version = bedrock_admin.get_guardrail(guardrailIdentifier=guardrail_id)["version"]
        self.guardrail_id = guardrail_id
        self._save_guardrail_config(config.profile, guardrail_id, version)

        logger.info(f"✅ Created guardrail ID: {guardrail_id}, version: {version}")
        return guardrail_id

    def _profiled_config_path(self, base: str, profile: str) -> str:
        name = os.path.splitext(os.path.basename(base))[0]
        return os.path.join(CONFIG_DIR, f"{name}_{profile}.json")

    def _load_guardrail_config(self, profile: str) -> Optional[Dict[str, str]]:
        config_path = self._profiled_config_path("amex_guardrail_config.json", profile)
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load guardrail cache: {e}")
        return None

    def _save_guardrail_config(self, profile: str, guardrail_id: str, version: str) -> None:
        config_path = self._profiled_config_path("amex_guardrail_config.json", profile)
        try:
            data = {
                "guardrail_id": guardrail_id,
                "version": version,
                "profile": profile,
                "saved_at": datetime.now().isoformat()
            }
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"💾 Guardrail config saved: {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save guardrail config: {e}")
