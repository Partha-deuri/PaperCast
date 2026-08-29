"""
Streamlit dashboard - three tabs:
  1. "Run Pipeline" - kick off a run with a toggle for human-in-loop, and if
     it pauses for review, edit/approve the draft script right here. Once a
     final script exists, generate audio (Kokoro TTS) and play it inline.
  2. "Cost & Latency" - reads logs/run_log.jsonl (persisted across every run,
     not just the current session) and renders per-node cost/latency charts.
  3. Sidebar - shows which LLM provider/model is active and whether you're
     in MOCK_MODE, so it's obvious at a glance why output looks like [MOCK].

Run with: streamlit run dashboard/app.py
"""
import sys
import os
import uuid
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph
from src.cost_tracker import load_all_runs
from src import config as podcast_config
from langgraph.types import Command

st.set_page_config(page_title="ArXiv Podcast Debate Generator", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "result" not in st.session_state:
    st.session_state.result = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "audio_error" not in st.session_state:
    st.session_state.audio_error = None

# ---------------------------------------------------------------------------
# SIDEBAR: current provider/model/mock-mode status - always visible so
# "why is this [MOCK]" or "why is this free/paid" is never a mystery.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Current Configuration")
    st.text(f"Provider: {podcast_config.PROVIDER}")
    st.text(f"Model: {podcast_config.MODEL_NAME}")
    if podcast_config.MOCK_MODE:
        st.warning("MOCK_MODE is ON - no real LLM calls will be made. "
                    "Output will contain '[MOCK]' placeholder text.\n\n"
                    "Set the matching API key in your .env file to use a real model.")
    else:
        st.success("Using a real LLM - MOCK_MODE is off.")

    st.divider()
    st.subheader("🔊 Audio Generation")
    tts_engine = st.selectbox("TTS Engine", ["kokoro", "edge_tts"])
    
    st.caption("Kokoro Model Files (for offline TTS)")
    model_path = st.text_input("kokoro model path", value="kokoro-v1.0.int8.onnx", key="model_path_input")
    voices_path = st.text_input("voices path", value="voices-v1.0.bin", key="voices_path_input")
    
    if tts_engine == "kokoro":
        if os.path.exists(model_path) and os.path.exists(voices_path):
            st.success("Model files found - audio generation is available.")
        else:
            st.warning(
                "Model files not found in the project root. Audio generation will fail until "
                "you download them (see README.md for the download commands)."
            )

tab_run, tab_dashboard, tab_library = st.tabs(["🎙️ Run Pipeline", "📊 Cost & Latency", "📚 Library Management"])

# ---------------------------------------------------------------------------
# TAB 1: Run pipeline + human-in-the-loop review + audio generation
# ---------------------------------------------------------------------------
with tab_run:
    st.header("Generate a Debate Episode")

    col1, col2 = st.columns([3, 1])
    with col1:
        papers_input = st.text_input(
            "ArXiv IDs or PDF paths (comma-separated)", placeholder="2306.07691, ./paper.pdf"
        )
        uploaded_files = st.file_uploader("Or upload PDF files", type=["pdf"], accept_multiple_files=True)
    with col2:
        human_in_loop = st.toggle(
            "Human-in-the-loop review",
            value=False,
            help="When ON, the pipeline pauses before finalizing so you can edit or "
            "approve the script here. When OFF, it runs straight through unattended.",
        )

    if st.button("Run Pipeline", type="primary", disabled=not (papers_input.strip() or uploaded_files)):
        import tempfile
        input_sources = [p.strip() for p in papers_input.split(",") if p.strip()]
        if uploaded_files:
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_sources.append(temp_path)
        input_sources = input_sources[:3]
        
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.audio_path = None  # new run - clear any previous episode's audio
        st.session_state.audio_error = None
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.spinner("Running: ingest → academic → grounding → skeptic → host..."):
            try:
                result = st.session_state.app.invoke(
                    {"input_sources": input_sources, "human_in_loop": human_in_loop}, config=config
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                result = None
        if result is not None:
            if "__interrupt__" in result:
                st.session_state.pending_interrupt = result["__interrupt__"][0].value
                st.session_state.result = None
            else:
                st.session_state.pending_interrupt = None
                st.session_state.result = result

    # --- Human review UI (only shown when the graph is actually paused) ---
    if st.session_state.pending_interrupt:
        payload = st.session_state.pending_interrupt
        st.warning("⏸️ Pipeline paused for human review.")

        gr = payload["grounding_report"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Citations verified", f"{gr['verified_citations']}/{gr['total_citations']}")
        c2.metric("Grounding rate", f"{gr['grounding_rate']:.0%}")
        c3.metric("Flagged (possibly hallucinated)", gr["hallucinated_citations"])
        if gr["flagged"]:
            st.error("⚠️ Some citations could not be verified against the source paper:")
            for f in gr["flagged"]:
                st.text(f"  [{f['paper_id']}] \"{f['quote']}\" (match score {f['match_score']})")

        st.subheader("Edit the script")
        edited_lines = []
        for i, line in enumerate(payload["draft_script"]):
            cols = st.columns([1, 5])
            speaker = cols[0].selectbox(
                "Speaker", ["Host", "Academic", "Skeptic"],
                index=["Host", "Academic", "Skeptic"].index(line["speaker"]),
                key=f"speaker_{i}", label_visibility="collapsed",
            )
            text = cols[1].text_area("Line", value=line["text"], key=f"text_{i}", label_visibility="collapsed", height=68)
            edited_lines.append({"speaker": speaker, "text": text, "cites_paper_ids": line.get("cites_paper_ids", [])})

        col_a, col_r = st.columns(2)
        if col_a.button("✅ Approve & Finalize", type="primary"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.app.invoke(
                Command(resume={"approved": True, "edited_lines": edited_lines}), config=config
            )
            st.session_state.pending_interrupt = None
            st.session_state.result = result
            st.rerun()
        if col_r.button("❌ Reject (use original draft, no edits)"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.app.invoke(Command(resume={"approved": True}), config=config)
            st.session_state.pending_interrupt = None
            st.session_state.result = result
            st.rerun()

    # --- Final result ---
    if st.session_state.result:
        result = st.session_state.result
        st.subheader("🎬 Final Script")
        for line in result["final_script"].lines:
            refs = f"  _(refs: {', '.join(line.cites_paper_ids)})_" if line.cites_paper_ids else ""
            st.markdown(f"**{line.speaker}:** {line.text}{refs}")

        total_cost = sum(r.estimated_cost_usd for r in result["run_log"])
        total_time = sum(r.duration_s for r in result["run_log"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cost (this run)", f"${total_cost:.4f}")
        m2.metric("Total latency (this run)", f"{total_time:.2f}s")
        m3.metric("Grounding rate", f"{result['grounding_report']['grounding_rate']:.0%}")

        # --- Audio generation ---
        st.divider()
        st.subheader("🔊 Audio Episode")

        files_ready = True
        if tts_engine == "kokoro":
            files_ready = os.path.exists(model_path) and os.path.exists(voices_path)
            
        gen_col, status_col = st.columns([1, 3])
        with gen_col:
            generate_clicked = st.button(
                "🎧 Generate Audio", disabled=not files_ready, type="primary"
            )
        with status_col:
            if not files_ready:
                st.caption("⚠️ Model files not found - see sidebar / README to download them first.")

        if generate_clicked:
            from src.tts import render_script_to_audio
            out_path = os.path.join("outputs", f"episode_{st.session_state.thread_id}.wav")
            with st.spinner(f"Rendering audio with {tts_engine}... (a few seconds per line)"):
                try:
                    render_script_to_audio(
                        result["final_script"], out_path,
                        model_path=model_path, voices_path=voices_path,
                        tts_engine=tts_engine
                    )
                    st.session_state.audio_path = out_path
                    st.session_state.audio_error = None
                except Exception as e:
                    st.session_state.audio_path = None
                    st.session_state.audio_error = str(e)

        if st.session_state.audio_error:
            st.error(f"Audio generation failed: {st.session_state.audio_error}")

        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            st.success(f"Episode ready: `{st.session_state.audio_path}`")
            with open(st.session_state.audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download .wav", data=audio_bytes,
                file_name=os.path.basename(st.session_state.audio_path), mime="audio/wav",
            )

# ---------------------------------------------------------------------------
# TAB 2: Cost & latency dashboard, across ALL runs ever logged
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.header("Cost & Latency, All Runs")
    runs = load_all_runs()

    if not runs:
        st.info("No runs logged yet - run the pipeline in the first tab to populate this dashboard.")
    else:
        df = pd.DataFrame(runs)
        # Strip the per-paper suffix (e.g. "academic_agent[1234.5678]") so
        # multiple papers in one run group under one node category.
        df["node_group"] = df["node_name"].str.replace(r"\[.*\]", "", regex=True)

        total_cost = df["estimated_cost_usd"].sum()
        total_calls = len(df)
        avg_latency = df["duration_s"].mean()
        error_rate = (df["error"].notna()).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total spend (all time)", f"${total_cost:.4f}")
        m2.metric("Total node calls", total_calls)
        m3.metric("Avg latency/node", f"{avg_latency:.2f}s")
        m4.metric("Node error rate", f"{error_rate:.1%}")

        col_a, col_b = st.columns(2)
        with col_a:
            cost_by_node = df.groupby("node_group")["estimated_cost_usd"].sum().reset_index()
            fig1 = px.bar(cost_by_node, x="node_group", y="estimated_cost_usd",
                          title="Total Cost by Node", labels={"estimated_cost_usd": "USD", "node_group": "Node"})
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            latency_by_node = df.groupby("node_group")["duration_s"].mean().reset_index()
            fig2 = px.bar(latency_by_node, x="node_group", y="duration_s",
                          title="Avg Latency by Node", labels={"duration_s": "seconds", "node_group": "Node"})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Raw run log")
        st.dataframe(df[["node_name", "duration_s", "input_tokens", "output_tokens", "estimated_cost_usd", "error"]],
                     use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Library Management
# ---------------------------------------------------------------------------
with tab_library:
    st.header("RAG Library Management")
    st.markdown("Manage the local Chroma database of previously processed papers.")
    
    from src.rag import list_library, delete_from_library, clear_library
    
    papers = list_library()
    if not papers:
        st.info("The RAG library is currently empty.")
    else:
        st.metric("Total Papers in Library", len(papers))
        
        st.subheader("Papers")
        for paper in papers:
            col1, col2 = st.columns([5, 1])
            col1.text(f"[{paper['id']}] {paper['title']}")
            if col2.button("Delete", key=f"del_{paper['id']}"):
                if delete_from_library(paper['id']):
                    st.success(f"Deleted {paper['id']}")
                    st.rerun()
                else:
                    st.error("Failed to delete paper.")
                    
        st.divider()
        if st.button("🗑️ Clear Entire Library", type="primary"):
            if clear_library():
                st.success("Library cleared.")
                st.rerun()
            else:
                st.error("Failed to clear library.")
