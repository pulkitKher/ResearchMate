import os
from tavily import TavilyClient
from state import ResearchState

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def research_node(state: ResearchState) -> dict:
    topic = state["topic"]
    retry_count = state.get("retry_count", 0)
    reason = state.get("verification_reason")

    # If this is a re-search (verification failed last time),
    # refine the query using the reason instead of repeating the same search
    if retry_count > 0 and reason:
        query = f"{topic} — focus on: {reason}"
        print(f"[Research Agent] Retry #{retry_count} | Refined query: {query}")
    else:
        query = topic
        print(f"[Research Agent] Initial search: {query}")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
    )

    results = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content"),
        }
        for r in response.get("results", [])
    ]

    print(f"[Research Agent] Found {len(results)} results")

    return {
        "search_results": results,
    }