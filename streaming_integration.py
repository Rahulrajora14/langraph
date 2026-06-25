# Video 10 — Streaming in LangGraph.
# Part 1: backend streaming demo (console).
# Part 2: Streamlit frontend with st.write_stream typewriter effect.
# Run the console demo with "python this_file.py"
# For Streamlit: "streamlit run this_file.py" (uncomment the Streamlit section)

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# ─────────────────────────────────────────────
# 1. SAME CHATBOT BACKEND AS VIDEO 9
# ─────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

memory  = MemorySaver()
chatbot = graph.compile(checkpointer=memory)

CONFIG = {"configurable": {"thread_id": "stream_demo"}}

# ─────────────────────────────────────────────
# 2. STREAMING IN THE CONSOLE — the key change: .stream() not .invoke()
# ─────────────────────────────────────────────
def stream_to_console(user_input: str):
    print("Bot: ", end="", flush=True)
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages"   # this mode streams LLM tokens; others stream state updates
    ):
        # Print each token as it arrives — end="" prevents newlines between tokens
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print()  # newline after the full response is done


# 4. CONSOLE DEMO ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Streaming chatbot demo. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        stream_to_console(user_input)
        print()