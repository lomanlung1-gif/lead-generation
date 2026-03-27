import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from altair import value
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".secrets" / "deepseek.env")
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class LeadGenConfig:
    node_sheet: str = "Node"
    edge_sheet: str = "Edge"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_k: int = 50
    candidate_pool_size: int = 120
    topology_query_weight: float = 0.7
    goal_query_weight: float = 0.3
    edge_score_weight: float = 0.7
    node_score_weight: float = 0.3
    rule_type_bonus: float = 0.35
    deterministic_match_bonus: float = 0.6
    score_threshold: int = 20
    llm_batch_size: int = 20
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
    attr_text = "; ".join(f"{k}={v}" for k, v in attrs.items() if not pd.isna(v))[:450]

    out_edges = [graph[node][nb]["edge_type"] for nb in graph.neighbors(node)]
    in_edges = [graph[parent][node]["edge_type"] for parent in graph.predecessors(node)]

    out_top = ", ".join(sorted(set(out_edges))[:8])
    in_top = ", ".join(sorted(set(in_edges))[:8])
    neighbors = ", ".join(list(graph.neighbors(node))[:5])

    return (
        f"node={node} | attrs={attr_text} | "
        f"out_edge_types={out_top} | in_edge_types={in_top} | "
        f"neighbors={neighbors}"
    )


def edge_context_text(graph: nx.DiGraph, src: str, dst: str, data: dict[str, Any]) -> str:
    src_attrs = graph.nodes[src]
    dst_attrs = graph.nodes[dst]

    src_text = "; ".join(f"{k}={v}" for k, v in src_attrs.items() if not pd.isna(v))[:220]
    dst_text = "; ".join(f"{k}={v}" for k, v in dst_attrs.items() if not pd.isna(v))[:220]

    return f"src={src} [{src_text}] --[{data['edge_type']}]--> dst={dst} [{dst_text}]"


def load_artifacts(excel_path: str, topology_path: str, config: LeadGenConfig) -> GraphArtifacts:
    topology = Path(topology_path).read_text(encoding="utf-8")
    graph = build_graph(excel_path, config.node_sheet, config.edge_sheet)

    print("Initializing embeddings...")
    encoder = SentenceTransformer(config.embedding_model)

    node_names = list(graph.nodes)
    node_texts = [node_context_text(graph, node) for node in node_names]
    node_embs = encoder.encode(node_texts, convert_to_tensor=True)

    edges = [(u, v, d) for u, v, d in graph.edges(data=True)]
    edge_embs = None
    if edges:
        edge_texts = [edge_context_text(graph, u, v, d) for u, v, d in edges]
        edge_embs = encoder.encode(edge_texts, convert_to_tensor=True)

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


def match_edge_types(artifacts: GraphArtifacts, debug: bool = False) -> set[str]:
    edge_types = sorted({data["edge_type"] for _, _, data in artifacts.edges})
    if not edge_types:
        return set()

    rule_emb = artifacts.encoder.encode(artifacts.topology, convert_to_tensor=True)
    type_embs = artifacts.encoder.encode(edge_types, convert_to_tensor=True)

    hits = util.semantic_search(rule_emb, type_embs, top_k=min(5, len(edge_types)))[0]
    matched = {edge_types[hit["corpus_id"]] for hit in hits if hit["score"] > 0.25}

    if debug:
        print(f"DEBUG | matched_edge_types={matched}")

    return matched


def retrieve_candidates(
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
) -> list[str]:
    candidate_scores: dict[str, float] = {}

    def add(name: str, score: float) -> None:
        candidate_scores[name] = candidate_scores.get(name, 0.0) + score

    matched_types = match_edge_types(artifacts, config.debug)

    for src, _, data in artifacts.edges:
        if data["edge_type"] in matched_types:
            add(src, config.deterministic_match_bonus)

    topology_q = artifacts.encoder.encode(artifacts.topology, convert_to_tensor=True)
    goal_q = artifacts.encoder.encode(final_goal, convert_to_tensor=True)

    if artifacts.edge_embs is not None:
        edge_k = min(config.retrieval_k, len(artifacts.edges))
        edge_hits_topology = util.semantic_search(topology_q, artifacts.edge_embs, top_k=edge_k)[0]
        edge_hits_goal = util.semantic_search(goal_q, artifacts.edge_embs, top_k=edge_k)[0]

        for hit in edge_hits_topology:
            idx = hit["corpus_id"]
            score = float(hit["score"])
            bonus = config.rule_type_bonus if artifacts.edges[idx][2]["edge_type"] in matched_types else 0.0
            add(
                artifacts.edges[idx][0],
                config.edge_score_weight * config.topology_query_weight * score + bonus,
            )

        for hit in edge_hits_goal:
            idx = hit["corpus_id"]
            score = float(hit["score"])
            add(
                artifacts.edges[idx][0],
                config.edge_score_weight * config.goal_query_weight * score,
            )

    node_k = min(config.retrieval_k, len(artifacts.node_names))
    node_hits_topology = util.semantic_search(topology_q, artifacts.node_embs, top_k=node_k)[0]
    node_hits_goal = util.semantic_search(goal_q, artifacts.node_embs, top_k=node_k)[0]

    for hit in node_hits_topology:
        idx = hit["corpus_id"]
        score = float(hit["score"])
        add(
            artifacts.node_names[idx],
            config.node_score_weight * config.topology_query_weight * score,
        )

    for hit in node_hits_goal:
        idx = hit["corpus_id"]
        score = float(hit["score"])
        add(
            artifacts.node_names[idx],
            config.node_score_weight * config.goal_query_weight * score,
        )

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    candidates = [name for name, _ in ranked[: config.candidate_pool_size]]

    if config.debug:
        print(f"DEBUG | matched_edge_types={matched_types}")
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


def node_payload(graph: nx.DiGraph, node: str) -> dict[str, Any]:
    attrs = {str(k): str(v)[:200] for k, v in graph.nodes[node].items() if not pd.isna(v)}
    rels = [f"--[{graph[node][nb]['edge_type']}]--> {nb}" for nb in graph.neighbors(node)][:15]
    return {"node": node, "attrs": attrs, "relations": rels}


def score_batch(
    nodes: list[str],
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
    llm: OpenAI,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = [node_payload(artifacts.graph, node) for node in nodes]

    prompt = (
        "You are a lead-scoring analyst for financial services.\n\n"
        f"GOAL: {final_goal}\n\n"
        f"TOPOLOGY RULES (apply strictly):\n{artifacts.topology}\n\n"
        f"NODES:\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        "Score each node 0-100 based on topology rules and data only.\n"
        f"For nodes >= {config.score_threshold}, give a concise reason.\n"
        "Also list any new topology rules you can confidently derive.\n\n"
        "Return ONLY JSON:\n"
        '{"targets": [{"node_name": "...", "score": N, "reason": "..."}], '
        '"new_discovered_rules": ["..."]}'
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
            left_targets, left_rules = score_batch(nodes[:mid], artifacts, final_goal, config, llm)
            right_targets, right_rules = score_batch(nodes[mid:], artifacts, final_goal, config, llm)
            return left_targets + right_targets, left_rules + right_rules
        raise

    valid_nodes = set(nodes)
    targets = [
        target
        for target in result.get("targets", [])
        if target.get("node_name") in valid_nodes
        and target.get("score", 0) >= config.score_threshold
    ]
    rules = result.get("new_discovered_rules", [])

    if config.debug:
        raw = result.get("targets", [])
        print(f"DEBUG | batch={len(nodes)} returned={len(raw)} kept={len(targets)}")

    return targets, rules


def dedupe_and_sort(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}

    for result in results:
        name = result.get("node_name")
        if not name:
            continue
        if name not in best_by_name or result["score"] > best_by_name[name]["score"]:
            best_by_name[name] = result

    return sorted(best_by_name.values(), key=lambda item: item["score"], reverse=True)


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

    candidates = retrieve_candidates(artifacts, final_goal, cfg)
    if not candidates:
        return []

    print(f"Scoring {len(candidates)} candidates...")
    all_results: list[dict[str, Any]] = []
    all_rules: list[str] = []
    batch_size = max(1, cfg.llm_batch_size)

    try:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            targets, rules = score_batch(batch, artifacts, final_goal, cfg, llm)
            all_results.extend(targets)
            all_rules.extend(rules)
    except Exception as exc:
        print(f"LLM error: {exc}")
        return []

    final_results = dedupe_and_sort(all_results)

    if cfg.auto_write_rules and all_rules:
        unique_rules = list(dict.fromkeys(all_rules))
        append_rules(topology_path, unique_rules)

    return final_results


if __name__ == "__main__":
    config = LeadGenConfig(debug=True, auto_write_rules=False)

    targets = generate_targets(
        excel_path="COI_Template.xlsx",
        final_goal="Increase financial product sales revenue by identifying high-value prospective clients",
        topology_path="topology.md",
        config=config,
    )

    print("\n=== High-Priority Target List ===")
    for target in targets[:10]:
        print(f"  {target['node_name']} | {target['score']} | {target['reason']}")
