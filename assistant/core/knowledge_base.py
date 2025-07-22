# knowledgebase.py

import json
import logging
import requests
import numpy as np
import faiss
import boto3
from typing import List, Dict
from assistant.core.config import AWS_REGION, EMBEDDINGS_MODEL, AWS_KNOWLEDGE
logger = logging.getLogger(__name__)


class AwsGenAIKnowledgeBase:
    """FAISS-powered AWS GenAI knowledge base for contextual grounding."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.embeddings_model = None

    def initialize_embeddings_model(self, region_name: str = AWS_REGION):
        """Initialize the Titan embedding model on AWS Bedrock."""
        self.embeddings_model = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        logger.info("Initialized Titan embeddings model.")

    def load_aws_genai_guidelines(self):
        """Download AWS Guardrails documentation and embed it into FAISS."""
        logger.info("Fetching AWS Guardrails documentation...")
        try:
            response = requests.get(AWS_KNOWLEDGE)
            response.raise_for_status()
            content = response.text
            logger.info(f"Fetched AWS knowledge page with {len(content)} characters.")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch AWS knowledge: {e}")
            content = ""

        aws_genai_guidelines = self._default_guidelines()
        if content:
            # Replace this with real parsing if needed
            aws_genai_guidelines += [
                {
                    "id": "fetched_doc",
                    "content": content[:1000],  # truncate for testing
                    "source": "AWS Bedrock Documentation"
                }
            ]

        logger.info(f"Embedding {len(aws_genai_guidelines)} documents...")
        for guideline in aws_genai_guidelines:
            try:
                embedding = self._get_embedding(guideline["content"])
                self.index.add(np.array([embedding]))
                self.documents.append(guideline)
                logger.debug(f"Embedded: {guideline['id']}")
            except Exception as e:
                logger.error(f"Error embedding {guideline['id']}: {e}")

    def _default_guidelines(self) -> List[Dict]:
        """Fallback docs if online fetch fails."""
        return [
            {
                "id": "bedrock_guardrails_overview",
                "content": "Amazon Bedrock Guardrails provides content filtering for categories like hate, insults, and violence.",
                "source": "AWS Bedrock Docs"
            },
            {
                "id": "guardrails_filter_strengths",
                "content": "Guardrails use four strength levels: NONE, LOW, MEDIUM, HIGH.",
                "source": "AWS Bedrock Docs"
            }
        ]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate Titan embedding from text."""
        if not self.embeddings_model:
            raise ValueError("Embeddings model not initialized")

        body = json.dumps({"inputText": text})
        try:
            response = self.embeddings_model.invoke_model(
                body=body,
                modelId=EMBEDDINGS_MODEL,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            embedding = np.array(response_body["embedding"])

            # Ensure proper shape
            if len(embedding) < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
            elif len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.random.rand(self.dimension)  # fallback

    def search_similar_content(self, query: str, k: int = 3) -> List[Dict]:
        """Find top-k similar docs for a user query."""
        if self.index.ntotal == 0:
            logger.warning("Knowledge base is empty.")
            return []

        try:
            query_embedding = self._get_embedding(query)
            scores, indices = self.index.search(np.array([query_embedding]), k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(score)
                    results.append(doc)
            logger.info(f"Found {len(results)} similar documents.")
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
