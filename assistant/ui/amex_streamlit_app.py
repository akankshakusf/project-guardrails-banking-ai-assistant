# amex_streamlit_app.py

import os
import html
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from assistant.core.config import (
    AWS_REGION,
    GuardrailFilterStrength,
    GuardrailAction,
)
from assistant.core.amex_router import AmexCoordinator
from assistant.core.amex_guardrails import AmexGuardrailsManager

# ========== Session & State Helpers ==========

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
            ⚙️ <strong>Please configure the assistant.</strong>
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

# ========== NEW: Guardrail Policy Box ==========

def display_guardrail_policy(profile="external"):
    guardrails_manager = AmexGuardrailsManager(region_name=AWS_REGION, profile=profile)
    profile_config = guardrails_manager.profiles.get(profile, None)

    st.subheader("🔒 Guardrail Policy (Read Only)")
    if not profile_config:
        st.error("Guardrail config not found.")
        return

    st.markdown(f"**Profile:**  `{profile_config.profile.capitalize()}`")
    st.markdown("**Filters Enabled:**")
    st.markdown(
        f"- Hate Speech: MEDIUM  \n"
        f"- Sexual: HIGH  \n"
        f"- Violence: HIGH  \n"
        f"- PII Detection: ANONYMIZE  \n"
        f"- Grounding: {'Enabled' if profile_config.grounding_required else 'Disabled'}"
    )
    st.markdown("**Denied Topics:**")
    denied_topics_md = "\n".join(f"- {topic}" for topic in profile_config.denied_topics)
    st.code(denied_topics_md, language="text")

# ========== Streamlit App Logic ==========

def initialize_assistant(user_type: str):
    st.session_state.initialization_status = "loading"
    try:
        # Optionally: Add any config/guardrail params you want to make dynamic
        st.session_state.amex_coordinator = AmexCoordinator(default_user_type=user_type)
        st.session_state.initialization_status = "success"
    except Exception as e:
        st.session_state.initialization_status = "error"
        st.error(f"Initialization failed: {e}")

def amex_streamlit_app():
    load_dotenv()
    st.set_page_config(page_title="Amex Banking AI Compliance Assistant", layout="wide")
    st.title("💳 Amex Banking AI Compliance Assistant")

    # ====== Init session state ======
    if "amex_coordinator" not in st.session_state:
        st.session_state.amex_coordinator = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "initialization_status" not in st.session_state:
        st.session_state.initialization_status = "not_started"
    if "user_type" not in st.session_state:
        st.session_state.user_type = "external"

    # ====== Sidebar ======
    with st.sidebar:
        st.header("🔧 Assistant Configuration")
        user_type = st.selectbox("User Type (Profile)", ["external", "internal"], index=0)
        st.session_state.user_type = user_type
        if st.button("🚀 Initialize Assistant", type="primary"):
            initialize_assistant(user_type=user_type)
        # NEW: Show current guardrails for the selected profile
        display_guardrail_policy(profile=user_type)
        

    # ====== Main Layout ======
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("💬 Chat Interface")
        display_status()
        with st.container():
            for message in st.session_state.chat_history:
                display_message(message['role'], message['content'], message.get('blocked', False))

        if st.session_state.initialization_status == "success":
            user_input = st.chat_input("Ask your Amex banking or rewards question...")
            if user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'blocked': False
                })
                with st.spinner("Thinking..."):
                    try:
                        agent = st.session_state.amex_coordinator
                        if agent:
                            result = agent.route_query(user_input, user_type=st.session_state.user_type)
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': result['response'],
                                'blocked': result.get('blocked', False)
                            })
                            st.rerun()
                        else:
                            st.error("Assistant is not initialized. Please initialize in the sidebar.")
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        else:
            st.info("Please initialize the assistant using the sidebar configuration.")

    with col2:
        st.header("📊 App Status")
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

        st.markdown("---")
        st.subheader("💡 Example Queries")
        for query in [
            "What is Purchase Protection and who is eligible for Amex Platinum Card?",
            "How does Amex determine credit card approvals based on U.S. credit scores?",
            "Will I earn points for UPS shipping?",
            "Dining at fast food places in the US — do they count?",
        ]:
            if st.button(f"📝 {query[:30]}...", key=f"ex_{hash(query)}"):
                if st.session_state.initialization_status == "success":
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': query,
                        'blocked': False
                    })
                    st.rerun()
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("🔄 Restart Assistant"):
            st.session_state.amex_coordinator = None
            st.session_state.initialization_status = "not_started"
            st.rerun()

# ========== Entrypoint ==========
if __name__ == "__main__":
    amex_streamlit_app()
