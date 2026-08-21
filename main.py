from dotenv import load_dotenv
load_dotenv()

from graph import build_graph

def main():
    app = build_graph()

    initial_state = {
        "topic": "latest advancements in solid-state batteries",
        "search_results": [],
        "all_findings": [],
        "verification_status": "pending",
        "verification_reason": None,
        "retry_count": 0,
        "max_retries": 2,
    }

    print(f"\n{'='*60}")
    print(f"Starting research on: {initial_state['topic']}")
    print(f"{'='*60}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Status: {final_state['verification_status']}")
    print(f"Retries used: {final_state['retry_count']}")
    print(f"Verified findings: {len(final_state.get('all_findings', []))}\n")

    for i, finding in enumerate(final_state.get("all_findings", []), 1):
        print(f"{i}. {finding.get('title')}")
        print(f"   {finding.get('url')}")
        print(f"   {finding.get('key_point')}\n")


if __name__ == "__main__":
    main()