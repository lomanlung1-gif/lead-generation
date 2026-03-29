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
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Lead Generation")
st.markdown(
    '<div class="hero"><b>Clean 4-step workflow</b><br/>Load graph, refine topology, run discovery, and review results in one flow.</div>',
    unsafe_allow_html=True,
)


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
    if "last_loaded_topology" not in st.session_state:
        st.session_state.last_loaded_topology = ""
    if "topology_text" not in st.session_state:
        topo_path = Path(DEFAULT_TOPOLOGY_PATH)
        st.session_state.topology_text = topo_path.read_text(encoding="utf-8") if topo_path.exists() else ""
    if "node_analysis" not in st.session_state:
        st.session_state.node_analysis = {}
    if "selected_analysis_node" not in st.session_state:
        st.session_state.selected_analysis_node = ""


def step_badge(state: str) -> str:
    css = "done" if state == "DONE" else "waiting"
    return f'<span class="step-state {css}">{state}</span>'


def run_pipeline(goal: str, cfg: LeadGenConfig) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts = st.session_state.artifacts
    if artifacts is None:
        return [], []

    candidates = retrieve_candidates(artifacts, goal, cfg)
    if not candidates:
        return [], []

    all_results: list[dict[str, Any]] = []
    batch_size = max(1, cfg.llm_batch_size)
    llm = init_llm_client()

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        all_results.extend(score_batch(batch, artifacts, goal, cfg, llm))

    final_results = dedupe_and_sort(all_results)
    new_rules = discover_rules(final_results, artifacts, goal, cfg, llm)
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
    if st.button("Load Graph and Embeddings", use_container_width=True):
        try:
            cfg = LeadGenConfig(debug=False, auto_write_rules=False)
            with st.spinner("Loading graph and embeddings..."):
                st.session_state.artifacts = load_artifacts(excel_path, topology_path, cfg)
            st.session_state.artifacts_ready = True
            st.session_state.last_loaded_topology = st.session_state.topology_text
            g = st.session_state.artifacts.graph
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

    save_col, state_col = st.columns([1, 2])
    with save_col:
        if st.button("Save Topology", use_container_width=True):
            try:
                Path(topology_path).write_text(editor_value, encoding="utf-8")
                st.session_state.topology_text = editor_value
                if st.session_state.artifacts_ready and editor_value != st.session_state.last_loaded_topology:
                    st.session_state.artifacts_ready = False
                    st.session_state.artifacts = None
                    st.info("Topology changed. Please reload graph and embeddings.")
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
    if st.button("Run Target List and Discover New Rules", use_container_width=True):
        if not st.session_state.artifacts_ready:
            st.warning("Please load graph and embeddings first.")
        else:
            try:
                cfg = LeadGenConfig(debug=False, auto_write_rules=False)
                with st.spinner("Scoring targets and discovering rules..."):
                    targets, rules = run_pipeline(goal, cfg)
                st.session_state.targets = targets
                st.session_state.proposed_rules = rules
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
    st.markdown("**Target List**")
    if st.session_state.targets:
        results_df = pd.DataFrame(st.session_state.targets)
        if not results_df.empty:
            results_df = results_df.copy()
            results_df["score"] = pd.to_numeric(results_df["score"], errors="coerce").fillna(0)
            results_df["score_pct"] = results_df["score"].clip(lower=0, upper=100)
            display_cols = [col for col in ["node_name", "score_pct", "score", "reason"] if col in results_df.columns]
            st.dataframe(
                results_df[display_cols],
                use_container_width=True,
                hide_index=True,
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

            st.markdown("**Click Target Name to Generate Insight**")
            top_rows = results_df[["node_name", "reason"]].head(12).to_dict(orient="records")
            for item in top_rows:
                node_name = item.get("node_name", "")
                if not node_name:
                    continue
                btn_key = f"analyze_{node_name}"
                if st.button(f"Analyze: {node_name}", key=btn_key, use_container_width=True):
                    if not st.session_state.artifacts_ready or st.session_state.artifacts is None:
                        st.warning("Please load graph and embeddings first.")
                    else:
                        try:
                            llm = init_llm_client()
                            cfg = LeadGenConfig(debug=False, auto_write_rules=False)
                            with st.spinner(f"Analyzing {node_name}..."):
                                analysis = analyze_node(
                                    node=node_name,
                                    artifacts=st.session_state.artifacts,
                                    final_goal=goal,
                                    config=cfg,
                                    llm=llm,
                                    score_reason=str(item.get("reason", "")),
                                )
                            st.session_state.node_analysis[node_name] = analysis
                            st.session_state.selected_analysis_node = node_name
                        except Exception as exc:
                            st.session_state.last_error = str(exc)
                            st.error(f"Analysis failed: {exc}")

            selected = st.session_state.selected_analysis_node
            if selected and selected in st.session_state.node_analysis:
                detail = st.session_state.node_analysis[selected]
                st.markdown("---")
                st.markdown(f"### Client Narrative: {selected}")
                st.markdown("**Insight**")
                st.write(detail.get("insight", ""))
                st.markdown("**Recommended Action**")
                st.write(detail.get("recommended_action", ""))
    else:
        st.caption("Target list will appear here after step 3.")

    st.markdown("**Proposed New Rules**")
    if st.session_state.proposed_rules:
        for idx, rule in enumerate(st.session_state.proposed_rules, start=1):
            st.write(f"{idx}. {rule}")
    else:
        st.caption("Proposed rules will appear here after step 3.")

if st.session_state.last_error:
    st.divider()
    st.caption(f"Last error: {st.session_state.last_error}")
