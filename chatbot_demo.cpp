from typing import TypedDict
from statistics import mean
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ==========================
# Gemini 2.5 Flash Model
# ==========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ==========================
# Structured Output Schema
# ==========================

class Evaluation(BaseModel):
    score: float = Field(description="Score out of 10")
    feedback: str = Field(description="Feedback")


evaluator = llm.with_structured_output(Evaluation)

# ==========================
# State Definition
# ==========================

class EssayState(TypedDict):
    essay: str

    language_score: float
    language_feedback: str

    analysis_score: float
    analysis_feedback: str

    clarity_score: float
    clarity_feedback: str

    average_score: float
    overall_feedback: str

# ==========================
# Language Evaluator Node
# ==========================

def language_node(state: EssayState):

    result = evaluator.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY Language Quality.

        Consider:
        - Grammar
        - Vocabulary
        - Sentence Formation
        - Readability

        Essay:
        {state["essay"]}
        """
    )

    return {
        "language_score": result.score,
        "language_feedback": result.feedback
    }

# ==========================
# Analysis Evaluator Node
# ==========================

def analysis_node(state: EssayState):

    result = evaluator.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY Analytical Depth.

        Consider:
        - Critical Thinking
        - Logical Arguments
        - Multi-dimensional Analysis
        - Depth of Ideas

        Essay:
        {state["essay"]}
        """
    )

    return {
        "analysis_score": result.score,
        "analysis_feedback": result.feedback
    }

# ==========================
# Clarity Evaluator Node
# ==========================

def clarity_node(state: EssayState):

    result = evaluator.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY Clarity and Structure.

        Consider:
        - Flow
        - Organization
        - Introduction
        - Conclusion
        - Coherence

        Essay:
        {state["essay"]}
        """
    )

    return {
        "clarity_score": result.score,
        "clarity_feedback": result.feedback
    }

# ==========================
# Aggregator Node
# ==========================

def aggregate_node(state: EssayState):

    avg = mean([
        state["language_score"],
        state["analysis_score"],
        state["clarity_score"]
    ])

    feedback = f"""
LANGUAGE ({state['language_score']}/10)
{state['language_feedback']}

-----------------------------------

ANALYSIS ({state['analysis_score']}/10)
{state['analysis_feedback']}

-----------------------------------

CLARITY ({state['clarity_score']}/10)
{state['clarity_feedback']}

-----------------------------------

FINAL SCORE : {avg:.2f}/10
"""

    return {
        "average_score": avg,
        "overall_feedback": feedback
    }

# ==========================
# Build LangGraph
# ==========================

graph = StateGraph(EssayState)

graph.add_node("language", language_node)
graph.add_node("analysis", analysis_node)
graph.add_node("clarity", clarity_node)
graph.add_node("aggregate", aggregate_node)

# Parallel Execution

graph.add_edge(START, "language")
graph.add_edge(START, "analysis")
graph.add_edge(START, "clarity")

# Join

graph.add_edge("language", "aggregate")
graph.add_edge("analysis", "aggregate")
graph.add_edge("clarity", "aggregate")

graph.add_edge("aggregate", END)

app = graph.compile()

# ==========================
# Run
# ==========================

essay = """
Artificial Intelligence is transforming every sector of society.
It improves efficiency, productivity and decision making.

However, concerns related to privacy, bias and job displacement
must also be addressed through responsible governance.

Therefore, AI should be developed with innovation as well as ethics.
"""

result = app.invoke({
    "essay": essay
})

print("\n==========================")
print("UPSC ESSAY REPORT")
print("==========================")

print(f"\nAverage Score: {result['average_score']:.2f}/10\n")
print(result["overall_feedback"])