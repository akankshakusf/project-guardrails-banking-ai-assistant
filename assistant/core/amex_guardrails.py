"""
assistant/core/amex_guardrails.py

Profile-aware (internal vs. external) Bedrock Guardrails manager for the Amex assistant.

What you get in this file:

1) Two *profiles* (external, internal) with different denied-topics lists.
   - external  → customer-facing, very restrictive (blocks internal algo/process talk)
   - internal  → employee-facing, still safe, but allows discussing internal policies

2) Local JSON caching of the created guardrail id/version **per profile**, so you don’t
   recreate them on every run.

3) A single `apply_guardrail()` you can call on both INPUT and OUTPUT text.
   It will:
     - lazily create or load the guardrail for the current profile
     - call Bedrock `apply_guardrail`
     - return the raw Bedrock response (so the caller decides how to act)

If you ever need to rotate / recreate the guardrails:
  - delete the cached JSON files (e.g. bank_guardrail_config_external.json / internal.json)
  - or call `create_guardrail(...)` yourself and then `_save_guardrail_config()`.
"""

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

import boto3

from assistant.core.config import (
    AWS_REGION,
    GUARDRAIL_CONFIG_FILE,
    GuardrailFilterStrength,
    GuardrailAction,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public type for clarity
# -----------------------------------------------------------------------------
GuardrailProfile = Literal["external", "internal"]


# -----------------------------------------------------------------------------
# Dataclass holding the *policy* we send to Bedrock when creating a guardrail
# -----------------------------------------------------------------------------
@dataclass
class AmexGuardrailConfig:
    """
    High-level policy config we’ll turn into a Bedrock Guardrail.
    The *important* part here is `profile`, which lets us change denied_topics,
    grounding strength, etc., per profile.
    """
    name: str
    description: str
    blocked_input_messaging: str
    blocked_outputs_messaging: str

    profile: GuardrailProfile = "external"

    # Core content filters
    hate_filter: GuardrailFilterStrength = GuardrailFilterStrength.MEDIUM
    insults_filter: GuardrailFilterStrength = GuardrailFilterStrength.MEDIUM
    sexual_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    violence_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    misconduct_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH
    prompt_attack_filter: GuardrailFilterStrength = GuardrailFilterStrength.HIGH

    # Topic / word / PII
    denied_topics: List[str] = field(default_factory=list)
    blocked_words: List[str] = field(default_factory=list)
    pii_entities: List[str] = field(default_factory=list)
    pii_action: GuardrailAction = GuardrailAction.ANONYMIZE

    # Grounding
    enable_grounding: bool = True
    grounding_threshold: float = 0.75

    def __post_init__(self):
        # Common blocked word list
        if not self.blocked_words:
            self.blocked_words = [
                "hack", "bypass", "exploit", "reverse engineer", "fraud",
                "unauthorized", "adversarial", "SQL injection", "scrape Notion",
                "malicious", "data exfiltration", "jailbreak",
            ]

        # Common PII list
        if not self.pii_entities:
            self.pii_entities = [
                "NAME", "EMAIL", "PHONE", "ADDRESS",
                "US_SOCIAL_SECURITY_NUMBER", "CREDIT_DEBIT_CARD_NUMBER",
                "DRIVER_ID", "US_PASSPORT_NUMBER"
            ]

        # Profile-specific denied topics
        if not self.denied_topics:
            if self.profile == "external":
                # External: do NOT reveal internal processes/algorithms/etc.
                self.denied_topics = [
                    "Fraud detection bypass", "Credit risk strategy",
                    "Internal approval process", "Model jailbreaking",
                    "Unauthorized model access", "Data exfiltration",
                    "Bypassing Amex guardrails", "Unauthorized Amex APIs",
                    "Internal collections framework", "Internal underwriting",
                    "Credit limit algorithm", "Chargeback trick",
                    "Unsolicited card exploit",
                ]
            else:
                # Internal: we *allow* talking about internal policy, but still block outright abuse/exfil.
                self.denied_topics = [
                    "Fraud detection bypass", "Model jailbreaking",
                    "Unauthorized model access", "Data exfiltration",
                    "Bypassing Amex guardrails", "Unauthorized Amex APIs",
                    "PII data leakage",
                ]


# -----------------------------------------------------------------------------
# The main manager you’ll use
# -----------------------------------------------------------------------------
class AmexGuardrailsManager:
    """
    Wraps Bedrock Guardrails with:
      - Profile awareness (external/internal)
      - Local persistence of (id, version)
      - One-call `apply_guardrail()` that lazy-loads/creates the guardrail

    Typical usage:

        mgr = AmexGuardrailsManager(profile="external")
        res = mgr.apply_guardrail(user_text, is_input=True)
        if res["action"] == "GUARDRAIL_INTERVENED": ...
    """

    def __init__(
        self,
        region_name: str = AWS_REGION,
        guardrail_id: Optional[str] = None,
        profile: GuardrailProfile = "external",
        config_file_path: str = GUARDRAIL_CONFIG_FILE,
    ):
        self.region_name = region_name
        self.profile = profile

        # Dedicated config file per profile so IDs don't clash
        self.config_file_path = self._profiled_config_path(config_file_path, profile)

        # We keep both Admin + Runtime clients
        self._bedrock_admin = boto3.client("bedrock", region_name=region_name)
        self._bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)

        self.guardrail_id: Optional[str] = guardrail_id
        self.guardrail_version: Optional[str] = None

        logger.info(f"Initialized AmexGuardrailsManager for region: {region_name} (profile={profile})")

        # Try to load a cached guardrail id/version
        if not self.guardrail_id:
            self.guardrail_id = self._load_guardrail_config()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def apply_guardrail(self, text: str, is_input: bool = True) -> Dict[str, Any]:
        """
        The main entry-point your agents will call.
        Ensures we have a guardrail (create or load), then applies it to text.
        """
        self._ensure_guardrail_ready()

        try:
            response = self._bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="INPUT" if is_input else "OUTPUT",
                content=[{"text": {"text": text}}]
            )
            return response
        except Exception as e:
            logger.error(f"❌ Failed to apply guardrail (profile={self.profile}): {e}")
            # Fail-open or fail-closed? We keep it explicit & let caller decide.
            return {"action": "ERROR", "error": str(e), "outputs": [{"text": text}]}

    def create_guardrail(self, config: AmexGuardrailConfig) -> str:
        """
        Explicitly create a guardrail on Bedrock from a config.
        You normally don’t call this directly — `_ensure_guardrail_ready()` will
        do it lazily if needed. But it’s here if you want full manual control.
        """
        try:
            logger.info(f"Creating Amex guardrail (profile={config.profile}): {config.name}")

            # Core behavior (hate, violence, etc.)
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

            # Topic blocking
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

            # Words
            word_policy = {"wordsConfig": [{"text": w} for w in config.blocked_words]}

            # PII
            pii_policy = {
                "piiEntitiesConfig": [{"type": t, "action": config.pii_action.value} for t in config.pii_entities],
                "regexesConfig": [
                    {
                        "name": "insurance_policy",
                        "pattern": r"INS-[0-9]{8}",
                        "action": "ANONYMIZE"
                    }
                ]
            }

            # Grounding
            grounding_policy = (
                {
                    "filtersConfig": [
                        {"type": "GROUNDING", "threshold": config.grounding_threshold}
                    ]
                }
                if config.enable_grounding
                else {}
            )

            resp = self._bedrock_admin.create_guardrail(
                name=config.name,
                description=config.description,
                contentPolicyConfig=content_policy,
                topicPolicyConfig=topic_policy,
                wordPolicyConfig=word_policy,
                sensitiveInformationPolicyConfig=pii_policy,
                contextualGroundingPolicyConfig=grounding_policy or None,
                blockedInputMessaging=config.blocked_input_messaging,
                blockedOutputsMessaging=config.blocked_outputs_messaging,
            )

            self.guardrail_id = resp["guardrailId"]

            # Bedrock returns "version" (not guardrailVersion) in get_guardrail
            describe = self._bedrock_admin.get_guardrail(guardrailIdentifier=self.guardrail_id)
            self.guardrail_version = describe.get("version")

            if not self.guardrail_version:
                raise ValueError(f"'version' not found in get_guardrail response: {describe}")

            self._save_guardrail_config()
            logger.info(
                f"Created Amex guardrail (profile={self.profile}) "
                f"ID: {self.guardrail_id}, Version: {self.guardrail_version}"
            )
            return self.guardrail_id
        except Exception as e:
            logger.error(f"Error creating Amex guardrail: {e}")
            raise

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    def _ensure_guardrail_ready(self) -> None:
        """
        Ensures we have a valid guardrail_id + version cached.
        If not present, we’ll create one with profile-aware config.
        """
        if self.guardrail_id and self.guardrail_version:
            return

        # Try to load from disk
        loaded = self._load_guardrail_config()
        if loaded:
            self.guardrail_id = loaded
            if self.guardrail_version:
                return

        # Else create a new one
        cfg = self._default_config_for_profile(self.profile)
        self.create_guardrail(cfg)

    def _default_config_for_profile(self, profile: GuardrailProfile) -> AmexGuardrailConfig:
        """
        Produces a ready-to-use default policy for the given profile.
        This is what you get if you don't pass custom configs in.
        """
        pretty = datetime.now().strftime("%Y%m%d%H%M%S")
        return AmexGuardrailConfig(
            name=f"amex-guardrail-{profile}-{pretty}",
            description=f"Amex Guardrails (Bedrock) - {profile}",
            blocked_input_messaging="⚠️ This input violates Amex safety policy.",
            blocked_outputs_messaging="⚠️ This response violates Amex safety policy.",
            profile=profile,
            # External gets grounding; internal we can choose to disable or leave as True:
            enable_grounding=(profile == "external"),
            grounding_threshold=0.75 if profile == "external" else 0.0,
        )

    def _profiled_config_path(self, base_path: str, profile: GuardrailProfile) -> str:
        """
        Keep separate config cache per profile:
          e.g. bank_guardrail_config_external.json, bank_guardrail_config_internal.json
        """
        root, ext = os.path.splitext(base_path)
        return f"{root}_{profile}{ext or '.json'}"

    def _load_guardrail_config(self) -> Optional[str]:
        """
        If you’ve already created a guardrail for this profile, load id + version, else None.
        """
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r") as f:
                    data = json.load(f)
                guardrail_id = data.get("guardrail_id")
                version = data.get("guardrail_version")
                if not guardrail_id or not version:
                    raise ValueError("Invalid guardrail config file (missing id/version).")
                self.guardrail_version = version
                logger.info(
                    f"✅ Loaded Amex guardrail ({self.profile}) ID: {guardrail_id}, version: {version}"
                )
                return guardrail_id
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to load Amex guardrail config ({self.profile}): {e}"
                )
        return None

    def _save_guardrail_config(self) -> None:
        """
        Cache the created (id, version) so we don't recreate next time.
        """
        try:
            data = {
                "guardrail_id": self.guardrail_id,
                "guardrail_version": self.guardrail_version,
                "profile": self.profile,
                "created_at": datetime.now().isoformat(),
            }
            with open(self.config_file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved Amex guardrail configuration to {self.config_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save Amex guardrail config: {e}")
