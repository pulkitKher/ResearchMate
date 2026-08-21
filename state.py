from typing import TypedDict,List,Optional

class ResearchState(TypedDict):
    topic: str                      # the user's research query
    search_results: List[dict]      # raw results from Tavily (this round)
    all_findings: List[dict]        # accumulated verified findings across rounds
    verification_status: str        # "pending" | "passed" | "failed"
    verification_reason: Optional[str]  # why it failed, if it did
    retry_count: int                # how many re-search attempts so far
    max_retries: int                # cap, so we don't loop forever