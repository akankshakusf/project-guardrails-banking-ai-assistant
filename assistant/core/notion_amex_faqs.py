# assistant/core/notion_amex_faqs.py

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from notion_client import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
import boto3

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths for caching
# ---------------------------------------------------------------------------
FAISS_DIR = Path("vectorstore/notion_amex_faqs")
FAQ_JSON = Path("data/notion_faq_dump.json")

# ---------------------------------------------------------------------------
# Helper Functions for Notion Content
# ---------------------------------------------------------------------------
def extract_block_text(block):
    block_type = block["type"]
    rich_parts = block.get(block_type, {}).get("rich_text", [])
    return "".join(part.get("plain_text", "") for part in rich_parts)


def extract_children_recursive(block_id, notion):
    """Recursively fetches children (like nested answers)."""
    children = []
    response = notion.blocks.children.list(block_id)
    for child in response.get("results", []):
        text = extract_block_text(child)
        if text:
            if child["type"] == "bulleted_list_item":
                children.append(f"- {text}")
            else:
                children.append(text)
        # Recurse further if this child has its own children
        if child.get("has_children"):
            children.extend(extract_children_recursive(child["id"], notion))
    return children


def convert_page_to_doc_full(page_title, page_id, notion):
    """Converts a Notion page into a structured text document."""
    content_lines = []

    def walk_blocks(blocks):
        for block in blocks:
            block_type = block["type"]
            text = extract_block_text(block)

            if block_type.startswith("heading") and text:
                content_lines.append(f"\n## {text}")
            elif block_type == "paragraph" and text:
                content_lines.append(text)
            elif block_type == "callout" and text:
                content_lines.append(f"💬 {text}")
            elif block_type == "bulleted_list_item" and text:
                content_lines.append(f"- {text}")
            elif block_type == "toggle" and text:
                content_lines.append(f"\n### ❓ {text}")
                children = extract_children_recursive(block["id"], notion)
                content_lines.extend(children)

            # Recurse into all child blocks
            if block.get("has_children") and block_type != "toggle":
                sub_blocks = notion.blocks.children.list(block["id"])["results"]
                walk_blocks(sub_blocks)

    root_blocks = notion.blocks.children.list(page_id)["results"]
    walk_blocks(root_blocks)

    full_text = "\n".join(content_lines).strip()
    return {"page_content": full_text, "metadata": {"source": page_title}}


# ---------------------------------------------------------------------------
# JSON Cache Helpers
# ---------------------------------------------------------------------------
def save_doc_cache(doc: dict):
    FAQ_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FAQ_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info(f"Cached Notion content to {FAQ_JSON}")


def load_doc_cache() -> dict | None:
    if FAQ_JSON.exists():
        with open(FAQ_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    notion_token = os.getenv("NOTION_API_KEY")
    if not notion_token:
        raise RuntimeError("NOTION_API_KEY is not set in your environment (.env).")

    # Titan Embedder
    logger.info("Building Titan embedder...")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    titan_embedder = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    # Use cache if available
    if FAISS_DIR.exists() and FAQ_JSON.exists():
        logger.info("⚡ Using cached FAISS vectorstore and notion_faq_dump.json...")
        doc = load_doc_cache()
        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            titan_embedder,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("Pulling content from Notion...")
        notion = Client(auth=notion_token)

        PAGE_ID = "2394ba18-200f-80c3-ba0e-d88f417c09d7"
        PAGE_TITLE = "Amex Knowledge Base"

        doc = convert_page_to_doc_full(PAGE_TITLE, PAGE_ID, notion)

        # Save the raw Notion content to cache
        save_doc_cache(doc)

        # Split and create FAISS vectorstore
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.create_documents([doc["page_content"]])

        logger.info("Embedding & saving FAISS store...")
        vectorstore = FAISS.from_documents(chunks, titan_embedder)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_DIR))

    # Preview query test
    query = "How do I earn rewards for booking flights with Amex?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n🔍 Query: {query}\n")
    for i, d in enumerate(results, 1):
        print(f"--- Match {i} ---\n{d.page_content}\n")
