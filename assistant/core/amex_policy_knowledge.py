# assistant/core/amex_policy_knowledge.py

import json
import logging
import requests
import numpy as np
import faiss
import boto3
from io import BytesIO
from pypdf import PdfReader
from typing import List, Dict
from assistant.core.config import AWS_REGION, EMBEDDINGS_MODEL, AMEX_POLICY

logger = logging.getLogger(__name__)


class AmexKnowledgeBase:
    """
    FAISS-powered Amex knowledge base for contextual grounding.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.embeddings_model = None

    # --------------------------------------------------------------
    # Initialize Titan Embeddings
    # --------------------------------------------------------------
    def initialize_embeddings_model(self, region_name: str = AWS_REGION):
        self.embeddings_model = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        logger.info("Initialized Titan embeddings model for Amex Knowledge Base.")

    # --------------------------------------------------------------
    # PDF Fetch and Parse
    # --------------------------------------------------------------
    def _fetch_pdf_text(self, url: str) -> str:
        """
        Downloads and extracts text from a PDF file.
        Returns concatenated text of all pages.
        """
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download Amex PDF: {e}")
            return ""

        content_type = resp.headers.get("content-type", "").lower()
        # Confirm it's a PDF
        if "application/pdf" not in content_type and not resp.content.startswith(b"%PDF"):
            logger.warning("The URL content does not seem like a PDF, skipping PDF parse.")
            return ""

        try:
            reader = PdfReader(BytesIO(resp.content))
            pages = []
            for i, page in enumerate(reader.pages, start=1):
                try:
                    txt = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Page {i} extract_text() failed: {e}")
                    txt = ""
                if txt.strip():
                    pages.append(f"[Page {i}]\n{txt.strip()}")
            full_text = "\n\n".join(pages)
            logger.info(f"Extracted text from {len(pages)} PDF pages ({len(full_text)} chars).")
            return full_text
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            return ""

    # --------------------------------------------------------------
    # Load Amex Policy
    # --------------------------------------------------------------
    def load_amex_policy(self):
        logger.info("Loading Amex policy knowledge base...")
        content = self._fetch_pdf_text(AMEX_POLICY)

        amex_policy_docs = self._default_policy_guidelines()

        if content:
            # Split the content into chunks of 2000 characters to avoid token overflow
            chunk_size = 2000
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            for idx, chunk in enumerate(chunks):
                amex_policy_docs.append({
                    "id": f"amex_policy_pdf_part_{idx + 1}",
                    "content": chunk,
                    "source": "Amex Official Policy PDF"
                })
            logger.info(f"Amex PDF split into {len(chunks)} chunks and added.")
        else:
            logger.warning("Using only fallback policy snippets (PDF text not found).")

        logger.info(f"Embedding {len(amex_policy_docs)} Amex documents...")
        for policy in amex_policy_docs:
            try:
                embedding = self._get_embedding(policy["content"])
                self.index.add(np.array([embedding]))
                self.documents.append(policy)
            except Exception as e:
                logger.error(f"Error embedding {policy['id']}: {e}")

    # --------------------------------------------------------------
    # Default fallback docs
    # --------------------------------------------------------------
    def _default_policy_guidelines(self) -> List[Dict]:
        return [
            {
                "id": "credit_risk_policy",
                "content": (
                    "CREDIT & RISK POLICY\n"
                    "American Express evaluates credit applications based on U.S. credit bureau reports, "
                    "income verification, payment history, and internal risk assessments. "
                    "Most credit decisions are automated using advanced data models, but final approval "
                    "is subject to regulatory compliance and internal underwriting policies."
                ),
                "source": "Amex U.S. Policy"
            },
            {
                "id": "third_party_risk_management",
                "content": (
                    "THIRD PARTY RISK MANAGEMENT:\n"
                    "American Express partners with select third-party service providers to enhance "
                    "customer experience and operational efficiency. All third parties are required to "
                    "adhere to U.S. federal regulations, Amex data privacy standards, and ongoing risk assessments."
                ),
                "source": "Amex U.S. Policy"
            }
        ]

    # --------------------------------------------------------------
    # Titan Embedding
    # --------------------------------------------------------------
    def _get_embedding(self, text: str) -> np.ndarray:
        if not self.embeddings_model:
            raise ValueError("Embeddings model not initialized. Call initialize_embeddings_model().")

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
            return embedding[:self.dimension] if len(embedding) > self.dimension else \
                np.pad(embedding, (0, self.dimension - len(embedding)))
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.random.rand(self.dimension)

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------
    def search_similar_content(self, query: str, k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            logger.warning("Amex knowledge base is empty.")
            return []

        query_embedding = self._get_embedding(query)
        scores, indices = self.index.search(np.array([query_embedding]), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(score)
                results.append(doc)
        return results
