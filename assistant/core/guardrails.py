# guardrail.py
import os
import re
import json
import logging
from datetime import datetime
import boto3
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from assistant.core.config import (
    AWS_REGION,
    GUARDRAIL_CONFIG_FILE,
    GuardrailFilterStrength,
    GuardrailAction,
)

logger = logging.getLogger(__name__)

@dataclass
class AmexGuardrailConfig:
    name: str
    description: str
    profile: str
    blocked_input_messaging: str = "⚠️ Input blocked by Amex policy."
    blocked_outputs_messaging: str = "⚠️ Output restricted by Amex policy."
@dataclass
class GuardrailConfig:
    """Configuration for AWS Bedrock Guardrails"""
    name: str
    description: str
    blocked_input_messaging: str
    blocked_outputs_messaging: str

    hate_filter: GuardrailFilterStrength = GuardrailFilterStrength.MEDIUM
    insults_filter: GuardrailFilterStrength = GuardrailFilterStrength.MEDIUM
    sexual_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    violence_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    misconduct_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    prompt_attack_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH

    denied_topics: List[str] = None
    blocked_words: List[str] = None
    pii_entities: List[str] = None
    pii_action: GuardrailAction = GuardrailAction.ANONYMIZE

    enable_grounding: bool = True
    grounding_threshold: float = 0.75

    def __post_init__(self):
        if self.denied_topics is None:
            self.denied_topics = [
                "Unauthorized model access", "Bypassing guardrails", "Malicious prompt injection",
                "Data exfiltration techniques", "Model jailbreaking", "Adversarial attacks",
                "Unauthorized data access", "Model exploitation", "Financial advice", "Medical advice"
            ]
        if self.blocked_words is None:
            self.blocked_words = [
                "jailbreak", "bypass", "exploit", "hack", "unauthorized", "malicious", "adversarial"
            ]
        if self.pii_entities is None:
            self.pii_entities = [
                "NAME", "EMAIL", "PHONE", "ADDRESS", "US_SOCIAL_SECURITY_NUMBER",
                "CREDIT_DEBIT_CARD_NUMBER", "DRIVER_ID", "US_PASSPORT_NUMBER"
            ]

class BedrockGuardrailsManager:
    """Handles AWS Bedrock Guardrails configuration and enforcement"""

    def __init__(self, region_name: str = AWS_REGION, guardrail_id: Optional[str] = None):
        self.bedrock_client = boto3.client("bedrock", region_name=region_name)
        self.guardrail_id = guardrail_id
        self.guardrail_version = None
        logger.info(f"Initialized BedrockGuardrailsManager for region: {region_name}")

        if not self.guardrail_id:
            self.guardrail_id = self._load_guardrail_config()

    def _load_guardrail_config(self) -> Optional[str]:
        if os.path.exists(GUARDRAIL_CONFIG_FILE):
            try:
                with open(GUARDRAIL_CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    guardrail_id = config.get("guardrail_id")
                    guardrail_version = config.get("guardrail_version")

                    if not guardrail_id or not guardrail_version:
                        raise ValueError("❌ Invalid guardrail config. Please delete the file and reinitialize.")

                    self.guardrail_version = guardrail_version
                    logger.info(f"✅ Loaded guardrail ID: {guardrail_id}, version: {guardrail_version}")
                    return guardrail_id
            except Exception as e:
                logger.warning(f"⚠️ Failed to load guardrail config: {e}")
        return None

    def _save_guardrail_config(self):
        try:
            config = {
                "guardrail_id": self.guardrail_id,
                "guardrail_version": self.guardrail_version,
                "created_at": datetime.now().isoformat()
            }
            with open(GUARDRAIL_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved guardrail configuration to {GUARDRAIL_CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save guardrail config: {e}")

    def create_guardrail(self, config: GuardrailConfig) -> str:
        try:
            logger.info(f"Creating guardrail: {config.name}")

            content_policy = {
                "filtersConfig": [
                    {"type": "HATE", "inputStrength": config.hate_filter.value, "outputStrength": config.hate_filter.value},
                    {"type": "INSULTS", "inputStrength": config.insults_filter.value, "outputStrength": config.insults_filter.value},
                    {"type": "SEXUAL", "inputStrength": config.sexual_filter.value, "outputStrength": config.sexual_filter.value},
                    {"type": "VIOLENCE", "inputStrength": config.violence_filter.value, "outputStrength": config.violence_filter.value},
                    {"type": "MISCONDUCT", "inputStrength": config.misconduct_filter.value, "outputStrength": config.misconduct_filter.value},
                    {"type": "PROMPT_ATTACK", "inputStrength": config.prompt_attack_filter.value, "outputStrength": "NONE"},
                ]
            }

            topic_policy = {
                "topicsConfig": [
                    {
                        "name": re.sub(r"[^0-9a-zA-Z\-_ !?.]", "", topic.replace(" ", "_")).lower(),
                        "definition": topic,
                        "examples": [f"How to {topic.lower()}"],
                        "type": "DENY"
                    } for topic in config.denied_topics
                ]
            }

            word_policy = {
                "wordsConfig": [{"text": word} for word in config.blocked_words]
            }

            pii_policy = {
                "piiEntitiesConfig": [{"type": entity, "action": config.pii_action.value} for entity in config.pii_entities],
                "regexesConfig": [{
                    "name": "insurance_policy",
                    "pattern": r"INS-[0-9]{8}",
                    "action": "ANONYMIZE"
                }]
            }

            grounding_policy = {
                "filtersConfig": [
                    {"type": "GROUNDING", "threshold": config.grounding_threshold}
                ]
            } if config.enable_grounding else {}

            response = self.bedrock_client.create_guardrail(
                name=config.name,
                description=config.description,
                contentPolicyConfig=content_policy,
                topicPolicyConfig=topic_policy,
                wordPolicyConfig=word_policy,
                sensitiveInformationPolicyConfig=pii_policy,
                contextualGroundingPolicyConfig=grounding_policy,
                blockedInputMessaging=config.blocked_input_messaging,
                blockedOutputsMessaging=config.blocked_outputs_messaging,
            )

            self.guardrail_id = response["guardrailId"]

            # ✅ Fix: use correct field name from describe_guardrail
            describe_response = self.bedrock_client.get_guardrail(
                guardrailIdentifier=self.guardrail_id
            )
            self.guardrail_version = describe_response.get("guardrailVersion")

            if not self.guardrail_version:
                raise ValueError(f"❌ 'guardrailVersion' not found in describe_guardrail response: {describe_response}")

            self._save_guardrail_config()
            logger.info(f"Created new guardrail ID: {self.guardrail_id}, Version: {self.guardrail_version}")
            return self.guardrail_id

        except Exception as e:
            logger.error(f"Error creating guardrail: {e}")
            raise

    def apply_guardrail(self, text: str, is_input: bool = True) -> Dict[str, Any]:
        if not self.guardrail_id or not self.guardrail_version:
            raise ValueError("Guardrail not initialized or missing version")

        try:
            bedrock_runtime = boto3.client("bedrock-runtime")
            response = bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,  # ✅ Fix here too
                source="INPUT" if is_input else "OUTPUT",
                content=[{"text": {"text": text}}]
            )
            return response
        except Exception as e:
            logger.error(f"❌ Failed to apply guardrail: {e}")
            return {"action": "NONE", "outputs": [{"text": text}]}
