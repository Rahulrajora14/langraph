import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_database_backend import chatbot, retrieve_all_threads

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("LangGraph Chatbot")

# ── Utility Functions ──────────────────────────────────────────────────────────
def generate_thread_id():
    return str(uuid.uuid4())     # random unique ID every time

def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []   # clears the screen
    add_thread(new_id)

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state  = chatbot.get_state(config)
    history = []
    if state and state.values:
        for msg in state.values.get("messages", []):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})
    return history

# ── Session Initialisation ─────────────────────────────────────────────────────
if "chat_threads" not in st.session_state:
    # On first load, pull all existing threads from the database!
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"]:
    if st.sidebar.button(str(thread_id)[:12] + "..."):   # truncate for readability
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()

# ── Main Chat Area ─────────────────────────────────────────────────────────────
CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type a message...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG, stream_mode="messages"
            )
            if message_chunk.content
        )
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})