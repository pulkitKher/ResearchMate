from langgraph.graph import StateGraph, END
from state import ResearchState

def node_a(state):
    return {"verification_history": [{"attempt": 0, "status": "failed"}]}

def node_b(state):
    return {"verification_history": [{"attempt": 1, "status": "passed"}]}

g = StateGraph(ResearchState)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.set_entry_point("a")
g.add_edge("a", "b")
g.add_edge("b", END)

out = g.compile().invoke({"verification_history": []})
print(out["verification_history"])