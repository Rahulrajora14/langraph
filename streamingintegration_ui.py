import streamlit as st
from langchain_core.messages import HumanMessage

# Import the compiled chatbot from your backend file
# from langgraph_backend import chatbot, CONFIG

st.title("LangGraph Chatbot with Streaming")

# Session state stores message history for UI display
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display past messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input at the bottom
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Stream the AI response — st.write_stream creates the typewriter effect automatically
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            # Generator expression: extracts content from each streaming chunk
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},  # NOTE: use user_input, not hardcoded text!
                config=CONFIG,
                stream_mode="messages"
            )
            if message_chunk.content  # skip empty chunks
        )
    # st.write_stream returns the fully stitched string — save it to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})