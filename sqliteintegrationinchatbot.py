# Video 11 — Permanent Storage with SqliteSaver.
# Backend: langgraph_database_backend.py
# Shows: SqliteSaver setup, retrieve_all_threads, frontend initialisation.

# ─────────────────────────────────────────────
# BACKEND: langgraph_database_backend.py
# ─────────────────────────────────────────────
import os
import sqlite3
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver      # pip install langgraph-checkpoint-sqlite
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 1. Create a persistent SQLite database connection
#    check_same_thread=False is required for Streamlit's multi-threaded environment
conn   = sqlite3.connect("chatbot.db", check_same_thread=False)
memory = SqliteSaver.from_conn_string(conn)  # checkpointer backed by the .db file

llm = ChatOpenAI(model="gpt-4o-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# 2. Compile with SqliteSaver instead of MemorySaver — everything else is identical!
chatbot = graph.compile(checkpointer=memory)

# 3. Utility: fetch every unique thread_id from the database
def retrieve_all_threads() -> list[str]:
    all_threads = set()                           # set ignores duplicates automatically
    for checkpoint in memory.list(None):          # None = retrieve ALL checkpoints
        thread_id = checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)
    return list(all_threads)


if __name__ == "__main__":
    # Quick test: run two threads and check they are saved independently
    c1 = {"configurable": {"thread_id": "nitish"}}
    c2 = {"configurable": {"thread_id": "rahul"}}
    chatbot.invoke({"messages": [HumanMessage(content="Hi! My name is Nitish.")]}, config=c1)
    chatbot.invoke({"messages": [HumanMessage(content="Hi! My name is Rahul.")]},  config=c2)

    print("All threads in DB:", retrieve_all_threads())
    # Close app, reopen, and threads will still be there!