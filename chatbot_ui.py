from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================
# LLM
# ==========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ==========================
# Memory
# ==========================

memory = MemorySaver()

# ==========================
# State
# ==========================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

# ==========================
# Node
# ==========================

def chat_node(state: ChatState):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }

# ==========================
# Graph
# ==========================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_edge(
    "chat_node",
    END
)

# ==========================
# Compile
# ==========================

chatbot = graph.compile(
    checkpointer=memory
)