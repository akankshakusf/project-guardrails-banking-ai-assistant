# config.py

import boto3
from enum import Enum

# AWS region
AWS_REGION: str = boto3.Session().region_name

# AWS Documentation URL used to generate knowledge base
AWS_KNOWLEDGE: str = "https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html"
AMEX_POLICY = "https://www.americanexpress.com/content/dam/amex/in/legal/our-codes-and-policies/AEBCMasterPolicy_Clean_Website.pdf"


# Embedding and foundation model names
EMBEDDINGS_MODEL: str = "amazon.titan-embed-text-v2:0"
CLAUDE_3_7_SONNET: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# CLAUDE_3_7_SONNET: str = "anthropic.claude-3-sonnet-20240229-v1:0"

# Local file for storing guardrail config
GUARDRAIL_CONFIG_FILE: str = "bank_guardrail_config.json"

# Enum for filter strengths
class GuardrailFilterStrength(Enum):    
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Enum for actions to take on PII detection
class GuardrailAction(Enum):
    BLOCK = "BLOCK"
    ANONYMIZE = "ANONYMIZE"
    NONE = "NONE"
