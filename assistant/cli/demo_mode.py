# demo_mode.py

import logging
from assistant.core.assistant import AwsGenAIAssistant

logger = logging.getLogger(__name__)

def demo_mode():
    print("🤖 Launching demo mode for AWS GenAI Decision Assistant...")

    try:
        assistant = AwsGenAIAssistant()
        print("✅ Assistant initialized in demo mode.")

        test_queries = [
            "What is the difference between Amazon Bedrock and SageMaker?",
            "How can I build a custom ML model for my business?",
            "What is Amazon Q Business used for?",
            "My name is John Doe. Can you help me with generative AI integration?"
        ]

        for q in test_queries:
            print(f"\n👤 User: {q}")
            result = assistant.process_user_query(q)
            print(f"🤖 Assistant: {result['response']}")
            if result.get("blocked"):
                print(f"🚫 [Blocked] Reason: {result['reason']}")
            else:
                print("✅ [Safe Response]")

        print("\n🎯 Demo complete. All core features tested.")

    except Exception as e:
        logger.exception(f"❌ Error in demo_mode: {e}")
        print("❌ Failed to run demo mode.")
