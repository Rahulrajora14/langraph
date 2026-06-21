from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# ==========================
# State
# ==========================

class TweetState(TypedDict):
    topic: str
    tweet: str
    feedback: str
    evaluation: Literal["Approved", "Needs Improvement"]
    iteration: int
    max_iterations: int


# ==========================
# Generator Node
# ==========================

def generate_tweet(state: TweetState):

    topic = state["topic"]

    tweet = f"Generated tweet about {topic}"

    return {
        "tweet": tweet
    }


# ==========================
# Evaluator Node
# ==========================

def evaluate_tweet(state: TweetState):

    tweet = state["tweet"]

    if len(tweet) > 20:
        evaluation = "Approved"
        feedback = "Good tweet."
    else:
        evaluation = "Needs Improvement"
        feedback = "Make it more engaging."

    return {
        "evaluation": evaluation,
        "feedback": feedback
    }


# ==========================
# Optimizer Node
# ==========================

def optimize_tweet(state: TweetState):

    old_tweet = state["tweet"]
    feedback = state["feedback"]

    improved_tweet = old_tweet + " 🚀"

    return {
        "tweet": improved_tweet,
        "iteration": state["iteration"] + 1
    }


# ==========================
# Routing Function
# ==========================

def route_evaluation(state: TweetState):

    is_approved = state["evaluation"] == "Approved"

    reached_limit = (
        state["iteration"] >= state["max_iterations"]
    )

    if is_approved or reached_limit:
        return "END"

    return "OPTIMIZE"


# ==========================
# Graph
# ==========================

graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")

graph.add_edge(
    "generate",
    "evaluate"
)

graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {
        "END": END,
        "OPTIMIZE": "optimize"
    }
)

graph.add_edge(
    "optimize",
    "evaluate"
)

workflow = graph.compile()


# ==========================
# Run
# ==========================

initial_state = {
    "topic": "AI in India",
    "tweet": "",
    "feedback": "",
    "evaluation": "Needs Improvement",
    "iteration": 1,
    "max_iterations": 5
}

result = workflow.invoke(initial_state)

print("\nFinal Tweet:")
print(result["tweet"])

print("\nEvaluation:")
print(result["evaluation"])

print("\nIterations:")
print(result["iteration"])