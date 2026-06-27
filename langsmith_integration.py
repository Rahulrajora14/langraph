# Video 12 — LangSmith Observability.
# Shows: automatic tracing, custom config (tags/metadata), and manual @traceable decorator.
# Prerequisite: pip install langsmith, create account at smith.langchain.com

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# 1. ENVIRONMENT VARIABLES (.env file)
#    LANGCHAIN_TRACING_V2=true is the only mandatory addition — it turns on auto-logging.
# ─────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]     = "your-langsmith-api-key"   # from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"]     = "langgraph_demo"            # project/folder name in UI
os.environ["OPENAI_API_KEY"]        = "your-openai-api-key"

# ─────────────────────────────────────────────
# 2. EXAMPLE 1 — Simple chain: auto-traced with zero extra code
# ─────────────────────────────────────────────
model  = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Prompt -> LLM -> Parser chain (standard LangChain LCEL)
simple_chain = ChatPromptTemplate.from_template("Answer this: {question}") | model | parser

# Just invoke normally — LangSmith logs it automatically
answer = simple_chain.invoke({"question": "What is the capital of Peru?"})
print("Simple chain answer:", answer)
# -> In LangSmith UI: see 1 Trace with 3 Runs (PromptTemplate, ChatOpenAI, StrOutputParser)

# ─────────────────────────────────────────────
# 3. EXAMPLE 2 — Sequential chain with custom config (tags + metadata + run_name)
# ─────────────────────────────────────────────

# Change the project folder dynamically from code (overrides the env var)
os.environ["LANGCHAIN_PROJECT"] = "sequential_llm_app"

report_prompt = ChatPromptTemplate.from_template(
    "Write a 3-paragraph report on: {topic}"
)
summary_prompt = ChatPromptTemplate.from_template(
    "Summarise this report in 5 bullet points:\n{report}"
)

report_chain  = report_prompt  | model | parser  # step 1: generate report
summary_chain = summary_prompt | model | parser  # step 2: summarise report

# Custom config: add tags, metadata, and a human-readable run name
config = {
    "tags":     ["report_generation", "summarization", "sequential"],
    "metadata": {
        "model":       "gpt-4o-mini",
        "temperature": 0.7,
        "parser":      "StrOutputParser"
    },
    "run_name": "Report + Summary Chain"  # shows in LangSmith instead of raw class name
}

report  = report_chain.invoke({"topic": "AI in Indian Healthcare"}, config=config)
summary = summary_chain.invoke({"report": report},                   config=config)
print("Summary:", summary)
# -> In LangSmith UI: see "Report + Summary Chain" with your custom tags and metadata

# ─────────────────────────────────────────────
# 4. EXAMPLE 3 — Manual tracing with @traceable
#    Use this when your custom Python functions (not LangChain objects) should appear in traces.
# ─────────────────────────────────────────────
from langsmith import traceable

@traceable(name="Resume-JD Matcher", tags=["matching", "hr"], run_type="chain")
def match_resume_to_jd(resume: str, job_description: str) -> str:
    # This custom Python function now appears in LangSmith as a named Run!
    match_prompt = ChatPromptTemplate.from_template(
        "Resume: {resume}\nJD: {jd}\nHow well does this resume match the JD? Score 1-10."
    )
    chain = match_prompt | model | parser
    return chain.invoke({"resume": resume, "jd": job_description})

if __name__ == "__main__":
    result = match_resume_to_jd(
        resume="5 years Python, Django, AWS, REST APIs",
        job_description="Backend Engineer: Django + AWS, 3-5 years exp"
    )
    print("Match result:", result)
    # -> In LangSmith: see a Run called "Resume-JD Matcher" with latency and token usage