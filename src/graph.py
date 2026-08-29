"""
Graph wiring: ingest -> academic -> grounding -> skeptic -> host -> review -> finalize

The `review` node is where human-in-the-loop lives, and it is OPTIONAL,
controlled by `state["human_in_loop"]` (bool), set per-run in main.py.

  - human_in_loop=True  -> the graph pauses at `review` via LangGraph's
    `interrupt()`, exposing the draft script for editing (e.g. in the
    Streamlit dashboard). The graph resumes only once a human approves
    or supplies an edited script.
  - human_in_loop=False -> `review` passes the draft straight through as
    final_script with no pause. Useful for unattended/scheduled runs where
    you trust the pipeline and just want output.

A checkpointer (MemorySaver) is required for `interrupt()` to work at all -
without it, LangGraph has nowhere to persist state across the pause.
"""
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import PipelineState, DebateScript
from .ingestion import fetch_papers
from .cost_tracker import track_node
from .agents.academic import academic_node
from .agents.grounding import grounding_node
from .agents.rag import rag_node
from .agents.skeptic import skeptic_node
from .agents.host import host_node


def ingest_node(state: PipelineState) -> dict:
    run_log = state.setdefault("run_log", [])
    with track_node("ingestion", run_log) as ctx:
        papers = fetch_papers(state["input_sources"])
        ctx["input_tokens"] = 0
        ctx["output_tokens"] = 0
    return {"papers": papers, "run_log": run_log}


def review_node(state: PipelineState) -> dict:
    draft = state["draft_script"]

    if not state.get("human_in_loop", False):
        # Toggle is off: pass straight through, no pause.
        return {"final_script": draft, "human_approved": True}

    # Toggle is on: pause here. `interrupt()` halts execution and surfaces
    # this payload to whatever is driving the graph (CLI prompt, Streamlit
    # dashboard, etc). Execution resumes when the caller invokes the graph
    # again with a Command(resume=<feedback>).
    feedback = interrupt(
        {
            "type": "human_review_required",
            "draft_script": [line.model_dump() for line in draft.lines],
            "grounding_report": state.get("grounding_report", {}),
            "instructions": "Return either {'approved': true} to accept as-is, "
            "or {'approved': true, 'edited_lines': [...]} with a replacement "
            "line list in the same shape as draft_script.",
        }
    )

    if feedback and feedback.get("edited_lines"):
        from .models import DialogueLine
        final = DebateScript(lines=[DialogueLine(**l) for l in feedback["edited_lines"]])
    else:
        final = draft

    return {"final_script": final, "human_approved": True, "human_feedback": str(feedback)}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("academic_agent", academic_node)
    graph.add_node("grounding", grounding_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("skeptic_agent", skeptic_node)
    graph.add_node("host_agent", host_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "academic_agent")
    graph.add_edge("academic_agent", "grounding")
    graph.add_edge("grounding", "rag_lookup")
    graph.add_edge("rag_lookup", "skeptic_agent")
    graph.add_edge("skeptic_agent", "host_agent")
    graph.add_edge("host_agent", "review")
    graph.add_edge("review", END)

    # Explicitly allow our Pydantic model classes through the checkpoint
    # serializer. Without this, LangGraph still works today but emits a
    # deprecation warning on every checkpoint write/read - future LangGraph
    # versions will refuse to (de)serialize unregistered types entirely.
    _model_classes = [
        "PaperData", "Citation", "VerifiedCitation", "PaperSummary",
        "SkepticCritique", "DialogueLine", "DebateScript", "NodeRunLog", "RAGMatch",
    ]
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("src.models", name) for name in _model_classes],
    )
    checkpointer = MemorySaver(serde=serde)
    return graph.compile(checkpointer=checkpointer)
