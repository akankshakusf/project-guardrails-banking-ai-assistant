# main.py

import sys
from assistant.cli.interactive_chat import interactive_chat
from assistant.cli.demo_mode import demo_mode
from assistant.ui.streamlit_app import streamlit_app

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "--streamlit":
            streamlit_app()
        elif mode == "--demo":
            demo_mode()
        elif mode == "--help":
            print("\n🧠 Usage Options:")
            print("  streamlit run main.py -- --streamlit   # Launch Streamlit web interface")
            print("  python main.py --demo                  # Run in demo (pre-set queries) mode")
            print("  python main.py                         # Start interactive CLI chat")
            print("  python main.py --help                  # Show help\n")
        else:
            print(f"\n❌ Unknown option: {mode}")
            print("Run with '--help' to see available options.\n")
    else:
        interactive_chat()

if __name__ == "__main__":
    main()
