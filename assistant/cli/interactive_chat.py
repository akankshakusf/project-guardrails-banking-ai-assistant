# assistant/cli/interactive_chat.py

import logging
from assistant.core.assistant import AwsGenAIAssistant

logger = logging.getLogger(__name__)

def interactive_chat():
    print("\n🧠 AWS Generative AI Assistant - Terminal Chat\n")
    assistant = AwsGenAIAssistant()

    print("✅ Assistant ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting assistant.")
                break

            result = assistant.process_user_query(user_input)

            if result.get("blocked"):
                print(f"\n🤖 Assistant: {result['response']} ❌ (Reason: {result.get('reason')})\n")
            else:
                response = result.get("response")
                print(f"\n🤖 Assistant: {str(response)}\n")  # Convert AgentResult or other types to string

        except KeyboardInterrupt:
            print("\n👋 Exiting assistant.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("\n⚠️ Unexpected error. Please try again.\n")
