import os
import html
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from assistant.core.assistant import AwsGenAIAssistant
from assistant.core.config import GuardrailFilterStrength, GuardrailAction
from assistant.core.guardrails import GuardrailConfig

# ✅ Load environment variables
load_dotenv()

# === Helper display functions ===

def display_status():
    status = st.session_state.initialization_status
    if status == "success":
        st.markdown("""
        <div style="background-color:#e6f4ea; padding:10px; border-radius:10px; margin:10px 0; border-left:5px solid #34a853;">
            ✅ <strong>Assistant is ready</strong> – All guardrails enabled.
        </div>
        """, unsafe_allow_html=True)
    elif status == "loading":
        st.markdown("""
        <div style="background-color:#fff8e1; padding:10px; border-radius:10px; margin:10px 0; border-left:5px solid #fbbc04;">
            ⏳ <strong>Initializing</strong> – Please wait...
        </div>
        """, unsafe_allow_html=True)
    elif status == "error":
        st.markdown("""
        <div style="background-color:#fdecea; padding:10px; border-radius:10px; margin:10px 0; border-left:5px solid #ea4335;">
            ❌ <strong>Initialization failed</strong> – Check configuration.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#e3f2fd; padding:10px; border-radius:10px; margin:10px 0; border-left:5px solid #4285f4;">
            ⚙️ <strong>Please Configure the assistant</strong>.
        </div>
        """, unsafe_allow_html=True)

def display_message(role: str, content: str, blocked: bool = False):
    avatar = "👤" if role == "user" else "🤖"
    bg_color = "#e3f2fd" if role == "user" else "#fff3e0"
    border = "#90caf9" if role == "user" else "#ffb74d"
    safe_content = html.escape(content)
    if blocked:
        safe_content = f"🚫 {safe_content}"
    st.markdown(f"""
    <div style="background-color:{bg_color}; border-left:5px solid {border}; padding:10px; border-radius:10px; margin:10px 0">
        <strong>{avatar} {role.capitalize()}:</strong><br>{safe_content}</div>""", unsafe_allow_html=True)

# === Main App ===

def initialize_assistant(hate, sexual, violence, pii, grounding, threshold, topics):
    st.session_state.initialization_status = "loading"
    try:
        config = GuardrailConfig(
            name=f"amex-genai-guardrail-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description="Banking LLM assistant with secure guardrails",
            blocked_input_messaging="⚠️ This banking query violates our policy.",
            blocked_outputs_messaging="⚠️ Response filtered due to compliance guidelines.",
            hate_filter=hate,
            sexual_filter=sexual,
            violence_filter=violence,
            pii_action=pii,
            enable_grounding=grounding,
            grounding_threshold=threshold,
            denied_topics=topics
        )

        st.session_state.assistant = AwsGenAIAssistant()
        if not st.session_state.assistant.guardrails_manager.guardrail_id:
            st.session_state.assistant.guardrails_manager.create_guardrail(config)
        st.session_state.initialization_status = "success"
    except Exception as e:
        st.session_state.initialization_status = "error"
        st.error(f"Initialization failed: {e}")

def streamlit_app():
    st.set_page_config(page_title="AWS GenAI Assistant", layout="wide")
    st.title("🤖 AWS Generative AI Decision Assistant")

    # === Init session state ===
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "initialization_status" not in st.session_state:
        st.session_state.initialization_status = "not_started"

    # === Sidebar Configuration ===
    with st.sidebar:
        st.header("🔧 Guardrail Configuration")
        hate_filter = st.selectbox("Hate Filter", list(GuardrailFilterStrength), index=2)
        sexual_filter = st.selectbox("Sexual Filter", list(GuardrailFilterStrength), index=3)
        violence_filter = st.selectbox("Violence Filter", list(GuardrailFilterStrength), index=3)
        pii_action = st.selectbox("PII Action", list(GuardrailAction), index=1)

        enable_grounding = st.checkbox("Enable Grounding", value=True)
        grounding_threshold = st.slider("Grounding Threshold", 0.0, 1.0, 0.75)

        denied_topics_text = st.text_area("Denied Topics", "\n".join([
            "Unauthorized model access", "Model jailbreaking", "Medical advice", "Financial advice"
        ]))

        if st.button("🚀 Initialize Assistant", type="primary"):
            denied_topics = [t.strip() for t in denied_topics_text.split("\n") if t.strip()]
            initialize_assistant(
                hate_filter, sexual_filter, violence_filter,
                pii_action, enable_grounding, grounding_threshold, denied_topics
            )

    # === Layout: Chat + Status Panel ===
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat Interface")
        display_status()

        with st.container():
            for message in st.session_state.chat_history:
                display_message(message['role'], message['content'], message.get('blocked', False))

        if st.session_state.initialization_status == "success":
            user_input = st.chat_input("Ask me about AWS generative AI services...")
            if user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'blocked': False
                })
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.assistant.process_user_query(user_input)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result['response'],
                            'blocked': result['blocked']
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        else:
            st.info("Please initialize the assistant using the sidebar configuration.")

    with col2:
        st.header("📊 Status Panel")
        if st.session_state.initialization_status == "success":
            st.success("✅ Assistant Ready")
            st.success("🔒 Guardrails Active")
            st.success("📚 Knowledge Base Loaded")
            st.success("🤖 Agent Initialized")
        elif st.session_state.initialization_status == "loading":
            st.warning("⏳ Initializing...")
        elif st.session_state.initialization_status == "error":
            st.error("❌ Initialization Failed")
        else:
            st.info("⚙️ Not Initialized")

        st.subheader("🚀 Quick Actions")
        if st.button("🔄 Restart Assistant"):
            st.session_state.assistant = None
            st.session_state.initialization_status = "not_started"
            st.rerun()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.subheader("💡 Example Queries")
        for query in [
            "What's the difference between Bedrock and SageMaker?",
            "How do I add AI to my business app?",
            "Can I fine-tune a model in Bedrock?",
            "Which AWS service supports RAG pipelines?"
        ]:
            if st.button(f"📝 {query[:30]}...", key=f"ex_{hash(query)}"):
                if st.session_state.initialization_status == "success":
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': query,
                        'blocked': False
                    })
                    st.rerun()

if __name__ == "__main__":
    streamlit_app()
