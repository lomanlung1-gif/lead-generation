import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".secrets" / "deepseek.env")
load_dotenv(ROOT / ".env")


_cross_encoder_model: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _cross_encoder_model


def cross_semantic_search(query_text: str, corpus_texts: list[str], top_k: int) -> list[dict[str, float]]:
    if not corpus_texts or top_k <= 0:
        return []

    model = get_cross_encoder()
    pairs = [[query_text, text] for text in corpus_texts]
    scores = model.predict(pairs)

    ranked = sorted(
        ((idx, float(score)) for idx, score in enumerate(scores)),
        key=lambda item: item[1],
        reverse=True,
    )[: min(top_k, len(corpus_texts))]
    return [{"corpus_id": idx, "score": score} for idx, score in ranked]


@dataclass(frozen=True)
class LeadGenConfig:
    node_sheet: str = "Node"
    edge_sheet: str = "Edge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_k: int = 50
    candidate_pool_size: int = 50
    edge_score_weight: float = 0.7
    node_score_weight: float = 0.3
    rule_type_bonus: float = 0.35
    deterministic_match_bonus: float = 0.6
    score_threshold: int = 60
    llm_batch_size: int = 10
    auto_write_rules: bool = False
    debug: bool = False
    llm_model: str = "deepseek-chat"


@dataclass(frozen=True)
class GraphArtifacts:
    graph: nx.DiGraph
    topology: str
    encoder: SentenceTransformer
    node_names: list[str]
    node_embs: Any
    edges: list[tuple[str, str, dict[str, Any]]]
    edge_embs: Any | None


def with_topology(artifacts: GraphArtifacts, topology: str) -> GraphArtifacts:
    return replace(artifacts, topology=topology)


def clean(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("\u200b", "").replace("\ufeff", "").strip()


def build_graph(excel_path: str, node_sheet: str, edge_sheet: str) -> nx.DiGraph:
    nodes_df = pd.read_excel(excel_path, sheet_name=node_sheet)
    edges_df = pd.read_excel(excel_path, sheet_name=edge_sheet)

    graph = nx.DiGraph()

    for row in nodes_df.to_dict(orient="records"):
        name = clean(row["Name"])
        graph.add_node(name, **{clean(k): clean(v) for k, v in row.items()})

    for row in edges_df.to_dict(orient="records"):
        graph.add_edge(
            clean(row["Source Node"]),
            clean(row["Target Node"]),
            edge_type=clean(row["Edge Type"]),
            edge_info=clean(row.get("Other Info Object (JSON)", "")),
        )

    return graph


def node_context_text(graph: nx.DiGraph, node: str) -> str:
    attrs = graph.nodes[node]
    attr_text = "; ".join(f"{k}={v}" for k, v in attrs.items() if not pd.isna(v))[:600]

    out_edges = [graph[node][nb]["edge_type"] for nb in graph.neighbors(node)]
    in_edges = [graph[parent][node]["edge_type"] for parent in graph.predecessors(node)]

    out_top = ", ".join(sorted(set(out_edges))[:10])
    in_top = ", ".join(sorted(set(in_edges))[:10])
    out_neighbors = ", ".join(list(graph.neighbors(node))[:6])
    in_neighbors = ", ".join(list(graph.predecessors(node))[:6])

    return (
        f"node={node} | attrs={attr_text} | "
        f"out_edge_types={out_top} | in_edge_types={in_top} | "
        f"out_neighbors={out_neighbors} | in_neighbors={in_neighbors}"
    )


def edge_context_text(graph: nx.DiGraph, src: str, dst: str, data: dict[str, Any]) -> str:
    src_attrs = graph.nodes[src]
    dst_attrs = graph.nodes[dst]

    src_text = "; ".join(f"{k}={v}" for k, v in src_attrs.items() if not pd.isna(v))[:300]
    dst_text = "; ".join(f"{k}={v}" for k, v in dst_attrs.items() if not pd.isna(v))[:300]

    return f"src={src} [{src_text}] --[{data['edge_type']}]--> dst={dst} [{dst_text}]"


def load_artifacts(
    excel_path: str,
    topology_path: str,
    config: LeadGenConfig,
    status_callback: Callable[[str], None] | None = None,
) -> GraphArtifacts:
    if status_callback:
        status_callback("Reading topology and graph data...")
    topology = Path(topology_path).read_text(encoding="utf-8")
    graph = build_graph(excel_path, config.node_sheet, config.edge_sheet)

    if status_callback:
        status_callback("Initializing embedding model...")
    print("Initializing embeddings...")
    encoder = SentenceTransformer(config.embedding_model)

    node_names = list(graph.nodes)
    if status_callback:
        status_callback(f"Encoding {len(node_names)} nodes...")
    node_texts = [node_context_text(graph, node) for node in node_names]
    node_embs = encoder.encode(node_texts, convert_to_tensor=True)

    edges = [(u, v, d) for u, v, d in graph.edges(data=True)]
    edge_embs = None
    if edges:
        if status_callback:
            status_callback(f"Encoding {len(edges)} edges...")
        edge_texts = [edge_context_text(graph, u, v, d) for u, v, d in edges]

        for u, v, d in edges:
            if u == 'Fredrick Chang' or v == 'Fredrick Chang':
                print("Fredrick Chang:  "+edge_context_text(graph, u, v, d))

        edge_embs = encoder.encode(edge_texts, convert_to_tensor=True)

    if status_callback:
        status_callback("Artifacts ready.")
    print(f"Ready | {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return GraphArtifacts(
        graph=graph,
        topology=topology,
        encoder=encoder,
        node_names=node_names,
        node_embs=node_embs,
        edges=edges,
        edge_embs=edge_embs,
    )


def init_llm_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def retrieve_candidates(
    artifacts: GraphArtifacts,
    config: LeadGenConfig,
) -> list[str]:
    candidate_scores: dict[str, float] = {}

    def get_max(name: str, score: float) -> None:
        candidate_scores[name] = max(candidate_scores.get(name, -99), score)


    edge_texts = [edge_context_text(artifacts.graph, u, v, d) for u, v, d in artifacts.edges]
    node_texts = [node_context_text(artifacts.graph, node) for node in artifacts.node_names]

    if edge_texts:
        edge_k = min(config.retrieval_k, len(artifacts.edges))
        edge_hits_topology = cross_semantic_search(artifacts.topology, edge_texts, top_k=edge_k)

        for hit in edge_hits_topology:
            idx = hit["corpus_id"]
            score = float(hit["score"])
            get_max( artifacts.edges[idx][0], score)
            get_max( artifacts.edges[idx][1], score)

    node_k = min(config.retrieval_k, len(artifacts.node_names))
    node_hits_topology = cross_semantic_search(artifacts.topology, node_texts, top_k=node_k)

    for hit in node_hits_topology:
        idx = hit["corpus_id"]
        score = float(hit["score"])
        get_max(artifacts.node_names[idx], score)

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    candidates = [name for name, _ in ranked[: config.candidate_pool_size]]

    if config.debug:
        print(
            "DEBUG | "
            f"scored_candidates={len(candidate_scores)} "
            f"final_candidates={len(candidates)}"
        )
        if ranked:
            print(f"DEBUG | top_candidate={ranked[0][0]} score={ranked[0][1]:.4f}")

    return candidates


def strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped


def neighbor_summary(graph: nx.DiGraph, nb: str) -> str:
    items = [(str(k), str(v)) for k, v in graph.nodes[nb].items() if not pd.isna(v)]
    return "; ".join(f"{k}={v[:100]}" for k, v in items[:3])


def relation_text(src: str, edge_type: Any, dst: str) -> str:
    label = str(edge_type).strip()
    key = " ".join(label.replace("_", " ").replace("-", " ").split()).casefold()

    templates = {
        "asset contributor": "{src} is an asset contributor to {dst}",
        "business ownership": "{src} has business ownership in {dst}",
        "business profit": "{src} has business profit linked to {dst}",
        "father": "{src} is the father of {dst}",
        "family relationship": "{src} has a family relationship with {dst}",
        "has account": "{src} has an account relationship with {dst}",
        "mentions": "{src} is mentioned with {dst}",
        "mother": "{src} is the mother of {dst}",
        "parent": "{src} is a parent of {dst}",
        "revoked": "{src} has a revoked relationship involving {dst}",
        "sale of business": "{src} is involved in a sale of business related to {dst}",
        "shared address": "{src} shares an address with {dst}",
        "shared email": "{src} shares an email with {dst}",
        "shared phone": "{src} shares a phone number with {dst}",
        "child": "{src} is a child of {dst}",
        "transaction": "{src} has a transaction with {dst}",
        "son": "{src} is the son of {dst}",
        "daughter": "{src} is the daughter of {dst}",
        "spouse": "{src} is the spouse of {dst}",
        "husband": "{src} is the husband of {dst}",
        "wife": "{src} is the wife of {dst}",
        "brother": "{src} is the brother of {dst}",
        "sister": "{src} is the sister of {dst}",
        "sibling": "{src} is a sibling of {dst}",
        "owns": "{src} owns {dst}",
        "owner": "{src} is an owner of {dst}",
        "shareholder": "{src} is a shareholder of {dst}",
        "partner": "{src} is a partner of {dst}",
        "founder": "{src} is the founder of {dst}",
        "director": "{src} is a director of {dst}",
        "employee": "{src} is an employee of {dst}",
        "employer": "{src} is the employer of {dst}",
        "works at": "{src} works at {dst}",
        "works for": "{src} works for {dst}",
        "beneficiary": "{src} is a beneficiary of {dst}",
        "trustee": "{src} is a trustee of {dst}",
        "settlor": "{src} is the settlor of {dst}",
    }

    template = templates.get(key)
    if template:
        return template.format(src=src, dst=dst)
    return f"{src} has a '{label}' relationship to {dst}"


def score_context(graph: nx.DiGraph, node: str) -> dict[str, Any]:
    attrs = {str(k): str(v)[:200] for k, v in graph.nodes[node].items() if not pd.isna(v)}

    out_rels = []
    for nb in list(graph.neighbors(node))[:8]:
        d = graph[node][nb]
        rel: dict[str, str] = {
            "dir": "out",
            "edge_type": d["edge_type"],
            "target": nb,
            "target_attrs": neighbor_summary(graph, nb),
            "relation_text": relation_text(node, d["edge_type"], nb),
        }
        if d.get("edge_info"):
            rel["edge_info"] = str(d["edge_info"])[:200]
        out_rels.append(rel)

    in_rels = []
    for parent in list(graph.predecessors(node))[:7]:
        d = graph[parent][node]
        rel = {
            "dir": "in",
            "edge_type": d["edge_type"],
            "source": parent,
            "source_attrs": neighbor_summary(graph, parent),
            "relation_text": relation_text(parent, d["edge_type"], node),
        }
        if d.get("edge_info"):
            rel["edge_info"] = str(d["edge_info"])[:200]
        in_rels.append(rel)

    return {"node": node, "attrs": attrs, "relations": out_rels + in_rels}



def deep_node_context(graph: nx.DiGraph, node: str) -> dict[str, Any]:
    attrs = {str(k): str(v)[:400] for k, v in graph.nodes[node].items() if not pd.isna(v)}

    out_rels = []
    for nb in list(graph.neighbors(node))[:15]:
        d = graph[node][nb]
        nb_attrs = {str(k): str(v)[:200] for k, v in graph.nodes[nb].items() if not pd.isna(v)}
        rel: dict[str, Any] = {
            "dir": "out",
            "edge_type": d["edge_type"],
            "target": nb,
            "target_attrs": nb_attrs,
            "relation_text": relation_text(node, d["edge_type"], nb),
        }
        if d.get("edge_info"):
            rel["edge_info"] = str(d["edge_info"])[:400]
        out_rels.append(rel)

    in_rels = []
    for parent in list(graph.predecessors(node))[:15]:
        d = graph[parent][node]
        parent_attrs = {
            str(k): str(v)[:200] for k, v in graph.nodes[parent].items() if not pd.isna(v)
        }
        rel: dict[str, Any] = {
            "dir": "in",
            "edge_type": d["edge_type"],
            "source": parent,
            "source_attrs": parent_attrs,
            "relation_text": relation_text(parent, d["edge_type"], node),
        }
        if d.get("edge_info"):
            rel["edge_info"] = str(d["edge_info"])[:400]
        in_rels.append(rel)

    second_hop = []
    for nb in list(graph.neighbors(node))[:10]:
        for nb2 in list(graph.neighbors(nb))[:5]:
            if nb2 == node:
                continue
            d2 = graph[nb][nb2]
            second_hop.append(
                {
                    "path": f"{node} -> {nb} -> {nb2}",
                    "edge_type": d2["edge_type"],
                    "relation_text": relation_text(nb, d2["edge_type"], nb2),
                }
            )

    third_hop = []
    visited_2hop = {(nb, nb2) for item in second_hop for nb, nb2 in [(item["path"].split(" -> ")[1], item["path"].split(" -> ")[2])]}
    for nb in list(graph.neighbors(node))[:8]:
        for nb2 in list(graph.neighbors(nb))[:4]:
            if nb2 == node:
                continue
            for nb3 in list(graph.neighbors(nb2))[:3]:
                if nb3 == node or nb3 == nb:
                    continue
                d3 = graph[nb2][nb3]
                third_hop.append(
                    {
                        "path": f"{node} -> {nb} -> {nb2} -> {nb3}",
                        "hops": [
                            relation_text(node, graph[node][nb]["edge_type"], nb),
                            relation_text(nb, graph[nb][nb2]["edge_type"], nb2),
                            relation_text(nb2, d3["edge_type"], nb3),
                        ],
                    }
                )

    return {
        "node": node,
        "attrs": attrs,
        "relations": out_rels + in_rels,
        "second_hop": second_hop[:20],
        "third_hop": third_hop[:15],
    }


def analyze_node(
    node: str,
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
    llm: OpenAI,
    score_reason: str = "",
) -> dict[str, str]:
    context = deep_node_context(artifacts.graph, node)
    context_json = json.dumps(context, ensure_ascii=False, default=str)

    prompt = f"""
You are a senior wealth management advisor. Write a client briefing that reads like a clear, logical story — not a data dump.

### GOAL:
{final_goal}

### NODE CONTEXT (raw graph data — synthesize, do not echo field names):
{context_json}

### WHY THIS CLIENT WAS FLAGGED:
{score_reason}

### HOW TO WRITE THE INSIGHT:
Write ONE continuous narrative (3-4 paragraphs) that flows like this:

Paragraph 1 — WHO: Introduce the client. State their name, type (person/org), and the single most important fact about them (e.g., role, net worth, recent life event). One or two sentences.

Paragraph 2 — NETWORK STORY: This is the core. Tell the story of how this client is connected to money, influence, or opportunity by tracing relationship chains end-to-end. Do NOT list relationships one by one. Instead, weave them: "A is the father of B, who owns C, which recently transacted with D — meaning A sits at the center of a family wealth structure spanning multiple entities." Use the third_hop data to extend chains. Every chain must end with a "so what" — what does this chain mean for the client's wealth or opportunity?

Paragraph 3 — OPPORTUNITY SIGNAL: Based on the chains above, explain the specific financial signal or life event that makes this client actionable RIGHT NOW. Connect it back to the goal. Be explicit: "Because X happened (cite evidence), this client likely needs Y."

### HOW TO WRITE THE RECOMMENDED ACTIONS:
Write 3 numbered actions. Each action must follow this exact logic:

"[Number]. [What to do] — because [cite the specific relationship chain or data point from the insight above that justifies this action], this client likely [explain the client need this addresses], which directly supports [tie back to the goal]."

Each action must reference something already stated in the insight — if you can't trace it back, don't include it.

### RULES:
- Plain English only. Never output JSON keys, field names, or technical graph terms.
- Never list relationships as bullet points or isolated facts. Always connect them into chains.
- Every claim needs a specific name or data point behind it.
- The reader should finish the insight and think: "I understand exactly who this person is, why they matter, and what to do next."

Return ONLY this JSON (no other text):
{{{{"insight": "...", "recommended_action": "..."}}}}
"""

    response = llm.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    result = json.loads(strip_fences(response.choices[0].message.content))
    return {
        "insight": result.get("insight", ""),
        "recommended_action": result.get("recommended_action", ""),
    }


def score_batch(
    nodes: list[str],
    artifacts: GraphArtifacts,
    config: LeadGenConfig,
    llm: OpenAI,
) -> list[dict[str, Any]]:
    payload = [score_context(artifacts.graph, node) for node in nodes]

    prompt = (
        f"""
        Your task is to evaluate how well each node in the provided list of candidates aligns with the TOPOLOGY RULES and contributes to achieving the GOAL.
     
        ### TOPOLOGY RULES (primary optimization target):
        {artifacts.topology}

        ### NODES:
        {json.dumps(payload, ensure_ascii=False, default=str)}

        ### RULES FOR SCORING:
        - Score each node 0-100 on how strongly it aligns with the TOPOLOGY RULES.
        - If Topology Rules apply to Person ONLY and the node is also the same type -> HIGHER score. 
        - Use topology rules AND node/edge data as evidence for your scoring.
        - Read relation_text first because it gives the plain-English meaning of each edge.
        - Use edge_type, dir, source, and target as the structured ground truth for validation.
        - For nodes scoring >= {config.score_threshold}, provide a concise reason explaining how this node aligns with the topology rules.
        - In your reasoning, prefer natural relationship wording such as "A is the father of B" instead of repeating raw JSON field names.
        

        ### Return ONLY JSON:
        {{"targets": [{{"node_name": "...", "score": N, "reason": "..."}}]}}
        """
    )

    try:
        response = llm.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result = json.loads(strip_fences(response.choices[0].message.content))
    except Exception as exc:
        if "context length" in str(exc).lower() and len(nodes) > 1:
            mid = len(nodes) // 2
            left = score_batch(nodes[:mid], artifacts, config, llm)
            right = score_batch(nodes[mid:], artifacts, config, llm)
            return left + right
        raise

    valid_nodes = set(nodes)
    targets = [
        target
        for target in result.get("targets", [])
        if target.get("node_name") in valid_nodes
        and target.get("score", 0) >= config.score_threshold
    ]

    if config.debug:
        raw = result.get("targets", [])
        print(f"DEBUG | batch={len(nodes)} returned={len(raw)} kept={len(targets)}")

    return targets


def dedupe_and_sort(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}

    for result in results:
        name = result.get("node_name")
        if not name:
            continue
        if name not in best_by_name or result["score"] > best_by_name[name]["score"]:
            best_by_name[name] = result

    return sorted(best_by_name.values(), key=lambda item: item["score"], reverse=True)


def graph_stats(artifacts: GraphArtifacts) -> str:
    from collections import Counter
    edge_counts = Counter(d["edge_type"] for _, _, d in artifacts.edges)
    return "; ".join(f"{etype}({count})" for etype, count in edge_counts.most_common(15))


def discover_rules(
    scored_targets: list[dict[str, Any]],
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
    llm: OpenAI,
) -> list[str]:
    top_evidence = scored_targets[:30]
    if not top_evidence:
        return []

    evidence_text = json.dumps(top_evidence, ensure_ascii=False, default=str)
    stats = graph_stats(artifacts)
    existing_edge_types = sorted({d["edge_type"] for _, _, d in artifacts.edges})

    prompt = (
        "You are a topology-rule analyst for financial services.\n\n"
        f"GOAL (optimize all rules for this): {final_goal}\n\n"
        f"EXISTING RULES (do NOT repeat these):\n{artifacts.topology}\n\n"
        f"GRAPH EDGE TYPES & FREQUENCIES: {stats}\n\n"
        f"HIGH-SCORING NODES (evidence):\n{evidence_text}\n\n"
        "Discover NEW graph-topology rules that predict which nodes are most "
        "valuable for achieving the GOAL above.\n\n"
        "Requirements for each rule:\n"
        "- Must explain a specific edge pattern (e.g., 'A --[edge_type]--> B means ...')\n"
        "- Must cite at least 2 node names from the evidence as support\n"
        "- Must explain WHY this pattern indicates the node can help achieve the GOAL\n"
        "- Must use only edge types that exist: " + json.dumps(existing_edge_types) + "\n"
        "- Do NOT restate existing rules in any form\n\n"
        "Return ONLY JSON:\n"
        '{"rules": [{"rule": "...", "supporting_nodes": ["node1", "node2"], '
        '"edge_type": "..."}]}'
    )

    try:
        response = llm.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result = json.loads(strip_fences(response.choices[0].message.content))
    except Exception as exc:
        if config.debug:
            print(f"DEBUG | rule discovery failed: {exc}")
        return []

    valid_edge_types = set(existing_edge_types)
    validated = []
    for entry in result.get("rules", []):
        rule_text = entry.get("rule", "")
        cited_type = entry.get("edge_type", "")
        if not rule_text:
            continue
        if cited_type and cited_type not in valid_edge_types:
            if config.debug:
                print(f"DEBUG | rejected rule (bad edge_type={cited_type}): {rule_text}")
            continue
        validated.append(rule_text)

    if config.debug:
        print(f"DEBUG | rules proposed={len(result.get('rules', []))} validated={len(validated)}")

    return validated


def append_rules(topology_path: str, rules: list[str]) -> int:
    content = Path(topology_path).read_text(encoding="utf-8")
    new_rules = [rule for rule in rules if rule not in content]
    if not new_rules:
        return 0

    with open(topology_path, "a", encoding="utf-8") as file:
        for rule in new_rules:
            file.write(f"\n- {rule}")

    print(f"Saved {len(new_rules)} new rules to {topology_path}")
    return len(new_rules)


def generate_targets(
    excel_path: str,
    final_goal: str,
    topology_path: str = "topology.md",
    config: LeadGenConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or LeadGenConfig()
    artifacts = load_artifacts(excel_path, topology_path, cfg)
    llm = init_llm_client()

    candidates = retrieve_candidates(artifacts, cfg)
    if not candidates:
        return []

    print(f"Scoring {len(candidates)} candidates...")
    all_results: list[dict[str, Any]] = []
    batch_size = max(1, cfg.llm_batch_size)

    try:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            targets = score_batch(batch, artifacts, cfg, llm)
            all_results.extend(targets)
    except Exception as exc:
        print(f"LLM error: {exc}")
        return []

    final_results = dedupe_and_sort(all_results)

    if cfg.auto_write_rules:
        print("Discovering topology rules...")
        new_rules = discover_rules(final_results, artifacts, final_goal, cfg, llm)
        if new_rules:
            append_rules(topology_path, new_rules)

    return final_results


if __name__ == "__main__":
    config = LeadGenConfig(debug=True, auto_write_rules=False)

    targets = generate_targets(
        excel_path="COI_Template.xlsx",
        final_goal="Increase financial product sales revenue by identifying high-value prospective clients",
        topology_path="topology.md",
        config=config
    )

    print("\n=== High-Priority Target List ===")
    for target in targets[:10]:
        print(f"  {target['node_name']} | {target['score']} | {target['reason']}")
