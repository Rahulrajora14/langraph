# Final Lecture — Tools, ToolNode, and the crucial looping architecture.
# Combines: custom tools (@tool), DuckDuckGo search, calculator, stock price API,
# tools_condition routing, the ToolNode loop, and streaming with AIMessage filtering.

import os
import math
import requests
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition  # pre-built helpers
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# ─────────────────────────────────────────────
# 1. DEFINE TOOLS — @tool decorator + clear docstring (LLM reads the docstring!)
# ─────────────────────────────────────────────
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for any multiplication calculation."""
    return a * b

@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a given ticker symbol (e.g., AAPL, TSLA, GOOGL).
    Returns a dict with the ticker and price."""
    # Using a free API — replace with a real financial API for production
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
    try:
        resp  = requests.get(url, timeout=5)
        data  = resp.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return {"ticker": ticker.upper(), "price": round(price, 2)}
    except Exception:
        return {"ticker": ticker.upper(), "price": "unavailable"}

# DuckDuckGo search: no API key needed
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for recent news, facts, or anything
    the LLM might not know (post 2023 events, live data)."""
    return search.run(query)

# Collect all tools into a list
tools = [multiply, get_stock_price, web_search]

# ─────────────────────────────────────────────
# 2. LLM — bind tools so the LLM knows it has these abilities
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)  # LLM can now output structured tool-call requests

# ─────────────────────────────────────────────
# 3. STATE
# ─────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ─────────────────────────────────────────────
# 4. CHAT NODE — uses llm_with_tools (not plain llm)
# ─────────────────────────────────────────────
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ─────────────────────────────────────────────
# 5. TOOL NODE — pre-built; auto-executes whatever tool the LLM requested
# ─────────────────────────────────────────────
tool_node = ToolNode(tools)  # receives the LLM's tool-call message, runs the function

# ─────────────────────────────────────────────
# 6. BUILD THE GRAPH WITH THE CRUCIAL LOOP
# ─────────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "chat_node")

# tools_condition: reads LLM output -> routes to "tool_node" OR END automatically
graph.add_conditional_edges("chat_node", tools_condition)

# THE LOOP — without this, raw JSON is shown to the user and multi-step tools fail!
# chat_node -> tool_node (executes tool, gets raw JSON) -> chat_node (polishes answer) -> END
graph.add_edge("tool_node", "chat_node")

# ─────────────────────────────────────────────
# 7. COMPILE WITH PERSISTENCE
# ─────────────────────────────────────────────
conn    = sqlite3.connect("tools_chatbot.db", check_same_thread=False)
memory  = SqliteSaver.from_conn_string(conn)
chatbot = graph.compile(checkpointer=memory)

# ─────────────────────────────────────────────
# 8. CHAT LOOP WITH AIMAESSAGE FILTER (mimics the Streamlit streaming fix)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "tools_demo"}}
    print("Tools chatbot ready. Try: 'What is 15 * 23?', 'What is Apple's stock price?', 'quit'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        final_ai_message = ""
        for msg, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            # FILTER: only print AIMessage tokens; skip raw ToolMessage JSON
            if isinstance(msg, AIMessage) and msg.content:
                print(msg.content, end="", flush=True)
                final_ai_message += msg.content
        print()  # newline after streaming finishes