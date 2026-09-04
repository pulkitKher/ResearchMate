from langgraph.graph import StateGraph, START, END
from state import ResearchState
from agents.research_agent import research_node
from agents.verification_agent import verify_node
from agents.summarization_agent import summarization_agent
from agents.report_generation_agent import report_generation_agent


def route_after_verification(state: ResearchState) -> str:
    status = state.get("verification_status")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if status == "passed":
        return "done"

    if retry_count < max_retries:
        return "retry"

    print(f"[Router] Max retries ({max_retries}) reached. Stopping with best-effort results.")
    return "done"


def increment_retry(state: ResearchState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("research", research_node)
    graph.add_node("verify", verify_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("summarize", summarization_agent)
    graph.add_node("generate_report", report_generation_agent)

    graph.add_edge(START, "research")
    graph.add_edge("research", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "retry": "increment_retry",
            "done": "summarize",
        },
    )

    graph.add_edge("increment_retry", "research")
    graph.add_edge("summarize", "generate_report")   # was END
    graph.add_edge("generate_report", END)             # new true exit point

    return graph.compile()