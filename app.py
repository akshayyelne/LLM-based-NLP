import asyncio
import uuid
import streamlit as st

from agent.graph import docs_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="LangChain Docs Assistant",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>
/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] * { color: #e0e0e0; }

/* Sidebar header */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 4px 20px 4px;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 0.5px;
}

/* Thread list items */
.thread-item {
    background: #2d2d4e;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    color: #c8c8e8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid transparent;
    transition: all 0.15s;
}
.thread-item:hover { border-color: #4f4f7f; background: #353560; }
.thread-item.active { background: #3d3d6b; border-color: #6c6cff; color: #ffffff; }

/* Bottom links */
.sidebar-links {
    position: fixed;
    bottom: 1.5rem;
    left: 0;
    width: 17rem;
    padding: 0 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.sidebar-link {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #9090b8 !important;
    text-decoration: none !important;
    font-size: 0.82rem;
    padding: 6px 8px;
    border-radius: 6px;
    transition: background 0.15s;
}
.sidebar-link:hover { background: #2d2d4e; color: #c8c8e8 !important; }
.sidebar-link svg { flex-shrink: 0; }

/* Main area top bar */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #e8e8f0;
    margin-bottom: 1rem;
}
.topbar-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a2e;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* New conversation button styling */
div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 500;
}

/* Search input */
[data-testid="stTextInput"] input {
    background-color: #252540 !important;
    color: #e0e0e0 !important;
    border: 1px solid #3d3d6b !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input::placeholder { color: #7070a0 !important; }

/* Thinking spinner */
.stSpinner > div { border-top-color: #6c6cff !important; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #8888aa;
}
.empty-state h2 { color: #1a1a2e; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State
# -------------------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_id" not in st.session_state:
    st.session_state.current_id = None


def new_conversation():
    cid = str(uuid.uuid4())[:8]
    st.session_state.conversations[cid] = {
        "title": "New conversation",
        "messages": [],
    }
    st.session_state.current_id = cid
    return cid


if not st.session_state.current_id:
    new_conversation()


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🦜 LangChain Assistant</div>', unsafe_allow_html=True)

    if st.button("＋  New Conversation", use_container_width=True):
        new_conversation()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    search = st.text_input(
        "search",
        placeholder="Search threads...",
        label_visibility="collapsed",
        key="thread_search",
    )

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    convs = list(st.session_state.conversations.items())
    convs_filtered = [
        (cid, conv) for cid, conv in convs
        if not search or search.lower() in conv["title"].lower()
    ]

    for cid, conv in reversed(convs_filtered):
        is_active = cid == st.session_state.current_id
        label = conv["title"]
        btn_style = "primary" if is_active else "secondary"
        if st.button(
            label,
            key=f"thread_{cid}",
            use_container_width=True,
            type=btn_style,
        ):
            st.session_state.current_id = cid
            st.rerun()

    st.markdown("""
    <div class="sidebar-links">
        <hr style="border-color:#2d2d4e; margin-bottom:10px"/>
        <a class="sidebar-link" href="https://smith.langchain.com/" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 3H6a3 3 0 0 0-3 3v12a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3V6a3 3 0 0 0-3-3z"/><path d="M8 12h8M12 8v8"/></svg>
            LangSmith
        </a>
        <a class="sidebar-link" href="https://forum.langchain.com/" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            Forum
        </a>
        <a class="sidebar-link" href="https://docs.langchain.com/oss/python/langchain/overview" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            Documentation
        </a>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Response Handler
# -------------------------

async def get_response(query: str) -> str:
    try:
        result = await docs_agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        messages = result.get("messages", [])
        if not messages:
            return "I couldn't generate a response. Please try again."

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
            if isinstance(msg, ToolMessage) and msg.content:
                return str(msg.content)

        return "I couldn't generate a response. Please try again."

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# Main Chat Area
# -------------------------

current_conv = st.session_state.conversations.get(st.session_state.current_id)
if not current_conv:
    new_conversation()
    st.rerun()

messages = current_conv["messages"]

col1, col2 = st.columns([6, 1])
with col1:
    title = current_conv["title"]
    st.markdown(f'<div class="topbar-title">{title}</div>', unsafe_allow_html=True)
with col2:
    if st.button("New Chat", type="primary"):
        new_conversation()
        st.rerun()

if not messages:
    st.markdown("""
    <div class="empty-state">
        <h2>🦜 LangChain Docs Assistant</h2>
        <p>Ask me anything about LangChain, LangGraph, and LangSmith.<br>
        I can search the documentation, explain concepts, and validate links.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about LangChain, LangGraph, or LangSmith..."):

    if len(messages) == 0:
        short = prompt.strip()
        current_conv["title"] = (short[:45] + "...") if len(short) > 45 else short

    current_conv["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            try:
                response = asyncio.run(get_response(prompt))
            except Exception as e:
                response = f"Error: {str(e)}"

        if not response or not response.strip():
            response = "I couldn't generate a response. Please try again."

        st.markdown(response)
        current_conv["messages"].append({"role": "assistant", "content": response})
