import streamlit as st
import pandas as pd
from main import GraphRAGLeadGenerator

st.set_page_config(page_title="Lead Generator", page_icon="🎯", layout="wide")
st.title("🎯 GraphRAG Lead Generator")


@st.cache_resource
def load_generator(excel_path, goal, topology_path):
    return GraphRAGLeadGenerator(
        excel_path=excel_path,
        final_goal=goal,
        topology_path=topology_path,
    )


# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Settings")
    excel_path = st.text_input("Excel file", value="COI_Template.xlsx")
    goal = st.text_area(
        "Goal",
        value="Increase financial product sales revenue by identifying high-value prospective clients",
        height=80,
    )
    topology_path = st.text_input("Topology file", value="topology.md")

    if st.button("Load Graph", type="primary", use_container_width=True):
        st.cache_resource.clear()

# --- Load generator ---
try:
    gen = load_generator(excel_path, goal, topology_path)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

st.sidebar.success(f"Graph loaded — {gen.G.number_of_nodes()} nodes, {gen.G.number_of_edges()} edges")

# --- Tabs ---
tab_targets, tab_ask = st.tabs(["🔍 Generate Targets", "💬 Ask Graph"])

# --- Tab 1: Generate Targets ---
with tab_targets:
    if st.button("Generate Targets", type="primary"):
        with st.spinner("Analyzing nodes with LLM..."):
            targets = gen.generate_targets()

        if targets:
            df = pd.DataFrame(targets)
            df.columns = ["Node", "Score", "Reason"]
            df = df.sort_values("Score", ascending=False).reset_index(drop=True)
            df.index += 1

            st.subheader(f"Found {len(df)} high-priority targets")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No targets found above the score threshold.")

# --- Tab 2: Ask the Graph ---
with tab_ask:
    question = st.text_input("Ask a question about the graph")
    if question:
        with st.spinner("Querying..."):
            answer = gen.ask(question)
        st.markdown(answer)
