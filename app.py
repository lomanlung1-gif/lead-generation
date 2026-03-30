from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from main import (
    LeadGenConfig,
    analyze_node,
    dedupe_and_sort,
    discover_rules,
    init_llm_client,
    load_artifacts,
    retrieve_candidates,
    score_batch,
    with_topology,
)


DEFAULT_GOAL = "Increase financial product sales revenue by identifying high-value prospective clients"
DEFAULT_EXCEL_PATH = "COI_Template.xlsx"
DEFAULT_TOPOLOGY_PATH = "topology.md"


st.set_page_config(page_title="Lead Generation", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #f4f7fb 0%, #ffffff 45%);
    }
    .block-container {
        max-width: 920px;
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
    }
    .hero {
        padding: 0.9rem 1rem;
        border: 1px solid #e5e9f0;
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 0.8rem;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .step-hint {
        color: #5b6470;
        margin-bottom: 0.7rem;
        font-size: 0.93rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #dce3ec;
        background: #f7f9fc;
        color: #334155;
        margin-right: 0.4rem;
    }
    .step-state {
        display: inline-block;
        margin-left: 0.5rem;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        border: 1px solid;
        vertical-align: middle;
    }
    .step-state.done {
        color: #14532d;
        background: #dcfce7;
        border-color: #86efac;
    }
    .step-state.waiting {
        color: #374151;
        background: #f3f4f6;
        border-color: #d1d5db;
    }

    .stButton > button {
        background-color: #0f6b45;
        color: #ffffff;
        border: 1px solid #0c5a3a;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #0b5a3a;
        border-color: #0a4f33;
        color: #ffffff;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 0.2rem rgba(15, 107, 69, 0.25);
    }
    .narrative-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem 1.2rem 1.6rem;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }
    .narrative-title {
        font-size: 1.05rem;
        font-weight: 700;
        padding-bottom: 0.7rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #f1f5f9;
        color: #0f172a;
    }
    .narrative-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #0f6b45;
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .narrative-body {
        font-size: 0.92rem;
        color: #374151;
        line-height: 1.65;
    }
    .narrative-empty {
        font-size: 0.9rem;
        color: #94a3b8;
        font-style: italic;
        padding: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Lead Generation")
st.markdown(
    '<div class="hero"><b>Clean 4-step workflow</b><br/>Load graph, refine topology, run discovery, and review results in one flow.</div>',
    unsafe_allow_html=True,
)


def topology_options(topology_text: str) -> list[str]:
    lines = topology_text.splitlines()
    options = [line.strip()[2:].strip() for line in lines if line.strip().startswith("-")]
    if not options:
        options = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    deduped: list[str] = []
    seen: set[str] = set()
    for option in options:
        if option and option not in seen:
            deduped.append(option)
            seen.add(option)
    return deduped


def ensure_state() -> None:
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = None
    if "artifacts_ready" not in st.session_state:
        st.session_state.artifacts_ready = False
    if "targets" not in st.session_state:
        st.session_state.targets = []
    if "proposed_rules" not in st.session_state:
        st.session_state.proposed_rules = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "topology_text" not in st.session_state:
        topo_path = Path(DEFAULT_TOPOLOGY_PATH)
        st.session_state.topology_text = topo_path.read_text(encoding="utf-8") if topo_path.exists() else ""
    if "node_analysis" not in st.session_state:
        st.session_state.node_analysis = {}
    if "selected_analysis_node" not in st.session_state:
        st.session_state.selected_analysis_node = ""
    if "selected_topologies" not in st.session_state:
        st.session_state.selected_topologies = topology_options(st.session_state.topology_text)


def step_badge(state: str) -> str:
    css = "done" if state == "DONE" else "waiting"
    return f'<span class="step-state {css}">{state}</span>'


def active_artifacts() -> Any:
    artifacts = st.session_state.artifacts
    if artifacts is None:
        return None
    selected = st.session_state.selected_topologies
    selected_topology_text = "\n".join(f"- {rule}" for rule in selected)
    return with_topology(artifacts, selected_topology_text)


def run_pipeline(
    goal: str,
    cfg: LeadGenConfig,
    status_callback: Any | None = None,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts = active_artifacts()
    if artifacts is None:
        return [], []

    if status_callback:
        status_callback("Retrieving candidates from selected topology rules...")
    if progress_bar:
        progress_bar.progress(10)
    candidates = retrieve_candidates(artifacts, cfg)
    if not candidates:
        if status_callback:
            status_callback("No candidates matched the current topology and goal.")
        if progress_bar:
            progress_bar.progress(100)
        return [], []

    all_results: list[dict[str, Any]] = []
    batch_size = max(1, cfg.llm_batch_size)
    llm = init_llm_client()
    total_batches = (len(candidates) + batch_size - 1) // batch_size

    if status_callback:
        status_callback(f"Scoring {len(candidates)} candidates across {total_batches} batches...")
    if progress_bar:
        progress_bar.progress(25)

    for batch_index, start in enumerate(range(0, len(candidates), batch_size), start=1):
        batch = candidates[start : start + batch_size]
        if status_callback:
            status_callback(f"Scoring batch {batch_index}/{total_batches} ({len(batch)} nodes)...")
        all_results.extend(score_batch(batch, artifacts, goal, cfg, llm))
        if progress_bar:
            progress_bar.progress(min(80, 25 + int(55 * batch_index / max(total_batches, 1))))

    if status_callback:
        status_callback("Ranking final targets...")
    final_results = dedupe_and_sort(all_results)

    if status_callback:
        status_callback("Discovering new rules from top targets...")
    if progress_bar:
        progress_bar.progress(90)
    new_rules = discover_rules(final_results, artifacts, goal, cfg, llm)
    if status_callback:
        status_callback("Run complete.")
    if progress_bar:
        progress_bar.progress(100)
    return final_results, new_rules


ensure_state()

pill_1 = "Ready" if st.session_state.artifacts_ready else "Not Ready"
pill_2 = "Available" if st.session_state.targets else "Empty"
st.markdown(
    f'<span class="status-pill">Artifacts: {pill_1}</span><span class="status-pill">Targets: {pill_2}</span>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        excel_path = st.text_input("Excel file", value=DEFAULT_EXCEL_PATH)
    with col_b:
        topology_path = st.text_input("Topology file", value=DEFAULT_TOPOLOGY_PATH)
    goal = st.text_area("Goal", value=DEFAULT_GOAL, height=90)

st.divider()

step_1_state = "DONE" if st.session_state.artifacts_ready else "WAITING"
st.markdown(
    f'<div class="step-title">1) Load Graph and Embeddings {step_badge(step_1_state)}</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="step-hint">Prepare graph + embedding artifacts for querying.</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption("This can take a while when the embedding model is initialized for the first time.")
    if st.button("Load Graph and Embeddings", use_container_width=True):
        try:
            cfg = LeadGenConfig(debug=False, auto_write_rules=False)
            progress_bar = st.progress(0, text="Starting load...")
            with st.status("Loading graph and embeddings...", expanded=True) as status:
                stage_progress = {
                    "Reading topology and graph data...": 15,
                    "Initializing embedding model...": 35,
                    "Artifacts ready.": 100,
                }

                def report(message: str) -> None:
                    pct = stage_progress.get(message, 65 if message.startswith("Encoding") else 50)
                    progress_bar.progress(pct, text=message)
                    status.write(message)

                st.session_state.artifacts = load_artifacts(excel_path, topology_path, cfg, status_callback=report)
                st.session_state.artifacts_ready = True
                g = st.session_state.artifacts.graph
                progress_bar.progress(100, text="Graph and embeddings ready.")
                status.update(label="Graph and embeddings ready", state="complete", expanded=False)
            st.success(f"Ready: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Load failed: {exc}")

st.divider()

step_2_state = "DONE" if st.session_state.topology_text.strip() else "WAITING"
st.markdown(
    f'<div class="step-title">2) Topology Rules {step_badge(step_2_state)}</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="step-hint">Review and edit current rules in topology.md.</div>', unsafe_allow_html=True)
with st.container(border=True):
    editor_value = st.text_area(
        "Edit topology.md",
        value=st.session_state.topology_text,
        height=280,
        label_visibility="collapsed",
    )

    options = topology_options(editor_value)
    current_selected = [rule for rule in st.session_state.selected_topologies if rule in options]
    if not current_selected and options:
        current_selected = options
    st.session_state.selected_topologies = st.multiselect(
        "Select topologies for retrieval",
        options=options,
        default=current_selected,
        help="Only selected topology rules are used during retrieval and scoring.",
    )

    save_col, state_col = st.columns([1, 2])
    with save_col:
        if st.button("Save Topology", use_container_width=True):
            try:
                Path(topology_path).write_text(editor_value, encoding="utf-8")
                st.session_state.topology_text = editor_value
                new_options = topology_options(editor_value)
                kept = [rule for rule in st.session_state.selected_topologies if rule in new_options]
                st.session_state.selected_topologies = kept if kept else new_options
                st.session_state.targets = []
                st.session_state.proposed_rules = []
                st.session_state.node_analysis = {}
                st.session_state.selected_analysis_node = ""
                if st.session_state.artifacts_ready:
                    st.info("Topology saved. Graph and embeddings stay loaded. Re-run step 3 to refresh results.")
                else:
                    st.success("Topology saved.")
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    with state_col:
        if st.session_state.artifacts_ready:
            st.caption("Artifacts are loaded and ready for query.")
        else:
            st.caption("Artifacts are not loaded yet.")

st.divider()

step_3_state = "DONE" if st.session_state.targets else "WAITING"
st.markdown(
    f'<div class="step-title">3) Run Target List and Discover New Rules {step_badge(step_3_state)}</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="step-hint">Execute retrieval + scoring + rule discovery.</div>', unsafe_allow_html=True)
with st.container(border=True):
    selected_count = len(st.session_state.selected_topologies)
    st.caption(
        f"This step runs retrieval, LLM scoring by batch, and rule discovery using {selected_count} selected topology rule(s)."
    )
    if st.button("Run Target List and Discover New Rules", use_container_width=True):
        if not st.session_state.artifacts_ready:
            st.warning("Please load graph and embeddings first.")
        else:
            try:
                cfg = LeadGenConfig(debug=False, auto_write_rules=False)
                progress_bar = st.progress(0, text="Starting run...")
                with st.status("Running target discovery...", expanded=True) as status:
                    targets, rules = run_pipeline(
                        goal,
                        cfg,
                        status_callback=lambda message: status.write(message),
                        progress_bar=progress_bar,
                    )
                    status.update(label="Target discovery complete", state="complete", expanded=False)
                st.session_state.targets = targets
                st.session_state.proposed_rules = rules
                st.session_state.selected_analysis_node = ""
                st.success(f"Done: {len(targets)} targets, {len(rules)} proposed rules")
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.error(f"Run failed: {exc}")

st.divider()

step_4_state = "DONE" if st.session_state.targets else "WAITING"
st.markdown(
    f'<div class="step-title">4) Results {step_badge(step_4_state)}</div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    tab1, tab2 = st.tabs(["🎯 Target List", "📋 Client Narrative"])

    with tab1:
        st.markdown("**Target List**")
        if st.session_state.targets:
            results_df = pd.DataFrame(st.session_state.targets)
            if not results_df.empty:
                results_df = results_df.copy()
                results_df["score"] = pd.to_numeric(results_df["score"], errors="coerce").fillna(0)
                results_df["score_pct"] = results_df["score"].clip(lower=0, upper=100)
                display_cols = [col for col in ["node_name", "score_pct", "score", "reason"] if col in results_df.columns]
                selection_event = st.dataframe(
                    results_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    height=360,
                    column_config={
                        "score_pct": st.column_config.ProgressColumn(
                            "Score (0-100)",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),
                        "score": st.column_config.NumberColumn("Raw Score", format="%.2f"),
                    },
                )

                selected_rows = selection_event.selection.rows if selection_event else []
                if selected_rows:
                    selected_idx = selected_rows[0]
                    selected_row = results_df.iloc[selected_idx]
                    selected_node = str(selected_row["node_name"])
                    st.session_state.selected_analysis_node = selected_node
                    if selected_node not in st.session_state.node_analysis:
                        runtime_artifacts = active_artifacts()
                        if not st.session_state.artifacts_ready or runtime_artifacts is None:
                            st.warning("Please load graph and embeddings first.")
                        else:
                            try:
                                llm = init_llm_client()
                                cfg = LeadGenConfig(debug=False, auto_write_rules=False)
                                with st.spinner(f"Analyzing {selected_node}..."):
                                    analysis = analyze_node(
                                        node=selected_node,
                                        artifacts=runtime_artifacts,
                                        final_goal=goal,
                                        config=cfg,
                                        llm=llm,
                                        score_reason=str(selected_row.get("reason", "")),
                                    )
                                st.session_state.node_analysis[selected_node] = analysis
                                st.info("Analysis ready — open the **Client Narrative** tab.")
                            except Exception as exc:
                                st.session_state.last_error = str(exc)
                                st.error(f"Analysis failed: {exc}")
                    else:
                        st.info("Analysis ready — open the **Client Narrative** tab.")
                else:
                    st.caption("Select a target row to generate client insight and recommended action.")
        else:
            st.caption("Target list will appear here after step 3.")

        st.markdown("**Proposed New Rules**")
        if st.session_state.proposed_rules:
            for idx, rule in enumerate(st.session_state.proposed_rules, start=1):
                st.write(f"{idx}. {rule}")
        else:
            st.caption("Proposed rules will appear here after step 3.")

    with tab2:
        selected = st.session_state.selected_analysis_node
        if selected and selected in st.session_state.node_analysis:
            detail = st.session_state.node_analysis[selected]
            insight = detail.get("insight", "").replace("\n", "<br>")
            action = detail.get("recommended_action", "").replace("\n", "<br>")
            st.markdown(
                f'<div class="narrative-card">'
                f'<div class="narrative-title">Client Narrative: {selected}</div>'
                f'<div class="narrative-label">Insight</div>'
                f'<div class="narrative-body">{insight}</div>'
                f'<div class="narrative-label">Recommended Action</div>'
                f'<div class="narrative-body">{action}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="narrative-card">'
                '<div class="narrative-empty">Select a target from the Target List tab to generate a detailed insight and recommended action.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

if st.session_state.last_error:
    st.divider()
    st.caption(f"Last error: {st.session_state.last_error}")
