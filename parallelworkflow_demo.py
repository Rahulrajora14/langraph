```python
from typing import TypedDict
from statistics import mean
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ==========================
# Load API Key
# ==========================

load_dotenv()

# ==========================
# Gemini 2.5 Flash
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


structured_llm = llm.with_structured_output(Evaluation)

# ==========================
# LangGraph State
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
# Language Evaluator
# ==========================

def language_node(state: EssayState):

    result = structured_llm.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY the language quality.

        Consider:
        - Grammar
        - Vocabulary
        - Sentence construction
        - Readability

        Give score out of 10 and feedback.

        Essay:
        {state['essay']}
        """
    )

    return {
        "language_score": result.score,
        "language_feedback": result.feedback
    }


# ==========================
# Analysis Evaluator
# ==========================

def analysis_node(state: EssayState):

    result = structured_llm.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY analytical depth.

        Consider:
        - Critical thinking
        - Multi-dimensional analysis
        - Arguments
        - Logical reasoning

        Give score out of 10 and feedback.

        Essay:
        {state['essay']}
        """
    )

    return {
        "analysis_score": result.score,
        "analysis_feedback": result.feedback
    }


# ==========================
# Clarity Evaluator
# ==========================

def clarity_node(state: EssayState):

    result = structured_llm.invoke(
        f"""
        You are a UPSC essay examiner.

        Evaluate ONLY clarity and structure.

        Consider:
        - Flow of ideas
        - Organization
        - Introduction
        - Conclusion
        - Paragraph transitions

        Give score out of 10 and feedback.

        Essay:
        {state['essay']}
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

    avg_score = mean([
        state["language_score"],
        state["analysis_score"],
        state["clarity_score"]
    ])

    overall_feedback = f"""
==============================
UPSC ESSAY EVALUATION REPORT
==============================

LANGUAGE
Score: {state['language_score']}/10

Feedback:
{state['language_feedback']}

--------------------------------

ANALYSIS
Score: {state['analysis_score']}/10

Feedback:
{state['analysis_feedback']}

--------------------------------

CLARITY
Score: {state['clarity_score']}/10

Feedback:
{state['clarity_feedback']}

--------------------------------

FINAL SCORE

Average Score: {avg_score:.2f}/10

Overall Summary:

The essay has been evaluated independently on
language quality, analytical depth, and clarity.
Focus on the weakest scoring dimension to improve
overall UPSC essay performance.
"""

    return {
        "average_score": avg_score,
        "overall_feedback": overall_feedback
    }


# ==========================
# Build Graph
# ==========================

builder = StateGraph(EssayState)

builder.add_node("language", language_node)
builder.add_node("analysis", analysis_node)
builder.add_node("clarity", clarity_node)
builder.add_node("aggregate", aggregate_node)

# Parallel Branches

builder.add_edge(START, "language")
builder.add_edge(START, "analysis")
builder.add_edge(START, "clarity")

# Join Branches

builder.add_edge("language", "aggregate")
builder.add_edge("analysis", "aggregate")
builder.add_edge("clarity", "aggregate")

builder.add_edge("aggregate", END)

graph = builder.compile()

# ==========================
# Sample Essay
# ==========================

essay = """
Technology has transformed human civilization in
unprecedented ways. It has improved healthcare,
education, governance, and communication.

However, technological advancement also brings
challenges such as unemployment, privacy concerns,
and ethical issues related to artificial intelligence.

Therefore, society must balance innovation with
responsible regulation to ensure sustainable growth.
"""

# ==========================
# Run Workflow
# ==========================

result = graph.invoke({
    "essay": essay
})

print("\n")
print("=" * 50)
print("FINAL RESULT")
print("=" * 50)

print(f"\nAverage Score: {result['average_score']:.2f}/10\n")

print(result["overall_feedback"])
```
