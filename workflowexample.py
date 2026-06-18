from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------
# LLM
# ------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ------------------
# Shared State
# ------------------

class State(TypedDict):
    text: str
    summary: str
    keywords: str
    title: str

# ------------------
# Nodes
# ------------------

def summarize(state: State):
    response = llm.invoke(
        f"Summarize this in one sentence:\n\n{state['text']}"
    )

    return {"summary": response.content}


def extract_keywords(state: State):
    response = llm.invoke(
        f"Extract 5 keywords from:\n\n{state['summary']}"
    )

    return {"keywords": response.content}


def generate_title(state: State):
    response = llm.invoke(
        f"Create a catchy blog title using:\n\n{state['keywords']}"
    )

    return {"title": response.content}

# ------------------
# Graph
# ------------------

graph = StateGraph(State)

graph.add_node("summarize", summarize)
graph.add_node("keywords", extract_keywords)
graph.add_node("title", generate_title)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "keywords")
graph.add_edge("keywords", "title")
graph.add_edge("title", END)

app = graph.compile()

# ------------------
# Run
# ------------------

result = app.invoke({
    "text": """
    LangGraph is a framework for building
    stateful AI workflows with nodes and edges.
    """
})

print("\nFINAL STATE:")
print(result)