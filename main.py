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

    out_top = ", ".join(sorted(set(out_edges)))
    in_top = ", ".join(sorted(set(in_edges)))
    out_neighbors = ", ".join(graph.neighbors(node))
    in_neighbors = ", ".join(graph.predecessors(node))

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
    for nb in graph.neighbors(node):
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
    for parent in graph.predecessors(node):
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



def _node_attrs(graph: nx.DiGraph, n: str, limit: int = 200) -> dict[str, str]:
    return {str(k): str(v)[:limit] for k, v in graph.nodes[n].items() if not pd.isna(v)}


def _edge_rel(graph: nx.DiGraph, src: str, dst: str) -> dict[str, Any]:
    d = graph[src][dst]
    rel: dict[str, Any] = {
        "edge_type": d["edge_type"],
        "relation_text": relation_text(src, d["edge_type"], dst),
    }
    if d.get("edge_info"):
        rel["edge_info"] = str(d["edge_info"])[:300]
    return rel


_HIGH_IMPACT = frozenset({
    "owner", "owns", "director", "shareholder", "founder", "settlor",
    "beneficiary", "trustee", "business ownership", "business profit",
    "asset contributor", "has account", "sale of business",
    "father", "mother", "spouse", "husband", "wife",
    "son", "daughter", "sibling", "brother", "sister", "parent", "child", "family relationship",
})


def _edge_priority(edge_type: str) -> int:
    return 0 if str(edge_type).strip().casefold() in _HIGH_IMPACT else 1


def deep_node_context(graph: nx.DiGraph, node: str, max_hops: int = 1) -> dict[str, Any]:
    attrs = _node_attrs(graph, node, limit=400)
    relations = sorted(
        [{"dir": "out", "target": nb, "target_attrs": _node_attrs(graph, nb), **_edge_rel(graph, node, nb)}
         for nb in graph.neighbors(node)]
        + [{"dir": "in", "source": p, "source_attrs": _node_attrs(graph, p), **_edge_rel(graph, p, node)}
           for p in graph.predecessors(node)],
        key=lambda r: _edge_priority(r["edge_type"]),
    )

    hops: dict[int, list[dict[str, Any]]] = {}
    if max_hops >= 2:
        visited = {node}
        frontier: list = []
        for nb in graph.neighbors(node):
            visited.add(nb)
            frontier.append(([node, nb], [relation_text(node, graph[node][nb]["edge_type"], nb)]))
        for p in graph.predecessors(node):
            if p not in visited:
                visited.add(p)
                frontier.append(([node, p], [relation_text(p, graph[p][node]["edge_type"], node)]))

        for hop in range(2, max_hops + 1):
            next_frontier = []
            for path, chain in frontier:
                tail = path[-1]
                for nb in graph.neighbors(tail):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append((path + [nb], chain + [relation_text(tail, graph[tail][nb]["edge_type"], nb)]))
                for nb in graph.predecessors(tail):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append((path + [nb], chain + [relation_text(nb, graph[nb][tail]["edge_type"], tail)]))
            if next_frontier:
                next_frontier.sort(key=lambda pc: min(
                    _edge_priority(graph[pc[0][i]][pc[0][i+1]]["edge_type"]) if graph.has_edge(pc[0][i], pc[0][i+1])
                    else _edge_priority(graph[pc[0][i+1]][pc[0][i]]["edge_type"])
                    for i in range(len(pc[0]) - 1)
                ))
                hops[hop] = [
                    {"path": " -> ".join(p), "relation_chain": c, "end_node_attrs": _node_attrs(graph, p[-1])}
                    for p, c in next_frontier
                ]
            frontier = next_frontier
            if not frontier:
                break

    result: dict[str, Any] = {"node": node, "attrs": attrs, "relations": relations}
    if hops:
        result["hops"] = hops
    return result


def analyze_node(
    node: str,
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
    llm: OpenAI,
    score_reason: str = "",
    max_iterations: int = 5,
    hop_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, str]:
    _SYSTEM = """Senior wealth-advisory analyst. Each turn: respond {{"explore": N}} (max depth 6, never repeat same depth) if promising chains are cut off, or return final JSON. Return immediately for LOW nodes."""

    _USER = """GOAL: {goal}
FLAGGED BECAUSE: {score_reason}
DATA (depth={depth}):
{context}

━━ TRIAGE ━━
CAPITAL-CONTROL: owner, owns, director, shareholder, founder, settlor, beneficiary, trustee, business ownership, business profit, asset contributor, has account, sale of business
FAMILY: father, mother, parent, child, son, daughter, spouse, husband, wife, brother, sister, sibling, family relationship
LOW-POWER: works for, works at, employee, employer, shared address, shared email, shared phone, mentions, revoked

HIGH: ≥1 capital-control edge
MEDIUM: no capital-control + ≥1 family edge to a node that has capital-control edges (verify in hops)
LOW: everything else → 2–3 sentences on why no capital control, recommended_action = "Not recommended for outreach." Return JSON immediately.

━━ BRIEFING (HIGH / MEDIUM only) ━━
"insight" must contain three parts:
1. WHO (2 sentences): name + CAPITAL-CONTROL roles/accounts only. Never mention LOW-POWER edges.
2. CHAINS (2+, different themes): literal entity names from relations/hops as a step-by-step path —
   A [edge] → B [edge] → C — financial implication for the GOAL.
   Only CAPITAL-CONTROL or FAMILY edges at every step. Use hops for multi-hop paths.
   MEDIUM: one chain must show the family member's capital and why this node is the entry path.
3. WHY CONTACT: "Because [named path + evidence], therefore [consequence]" → specific product.

"recommended_action": 3 items — verb + specific entity/amount from data + product/service.

━━ EXAMPLE — HIGH ━━
insight: "John Lee is a director of Beta Corp and a shareholder of Delta Financial. He holds a $4.2M account linked to Epsilon Trust.
Chain 1 — Control: John Lee [director] → Beta Corp ← [60% owned by] Alpha Holdings → Beta Corp sold $12M to Gamma Industries — Lee controls reinvestment of that capital.
Chain 2 — Wealth: John Lee [shareholder] → Delta Financial [beneficiary of] → Epsilon Trust [holds] → Lee's $4.2M account — corporate equity and trust structures need coordinated advice.
Because Lee directs Beta Corp ($12M sale) and holds $4.2M through Epsilon Trust, he needs reinvestment structuring and estate planning."
recommended_action: "1. Call re: $12M Beta Corp sale — controls capital allocation. 2. Review $4.2M Epsilon Trust account — integrated trust-portfolio strategy. 3. Meet re: Delta Financial shareholding — succession and liquidity planning."

━━ EXAMPLE — MEDIUM ━━
insight: "Mary Chen has no capital-control edges. She is the spouse of David Chen.
Chain 1 — Family access: Mary Chen [spouse of] → David Chen [founder of] → Horizon Capital ($28M AUM) [owns 40% of] → Jade Properties — Mary is the entry path to David's $28M.
Because David controls $28M via Horizon Capital, Mary is the natural introduction for estate and succession planning."
recommended_action: "1. Invite Mary to wealth seminar — family financial wellness. 2. Request intro to David re: Horizon Capital ($28M) succession. 3. Propose joint estate review — $28M in business assets needs protection."

━━ RETURN ━━
{{"influence_level": "HIGH|MEDIUM|LOW", "insight": "...", "recommended_action": "..."}}
Or: {{"explore": N}}"""

    graph = artifacts.graph
    depth = 1
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]

    for _ in range(max_iterations):
        ctx = deep_node_context(graph, node, max_hops=depth)
        if hop_callback:
            hop_callback(depth, "fetching", f"{len(ctx.get('relations', []))} direct relations, {sum(len(v) for v in ctx.get('hops', {}).values())} extended paths")

        messages.append({"role": "user", "content": _USER.format(
            goal=final_goal, score_reason=score_reason, depth=depth,
            context=json.dumps(ctx, ensure_ascii=False, default=str),
        )})
        raw = strip_fences(llm.chat.completions.create(
            model=config.llm_model, messages=messages, temperature=0.2,
        ).choices[0].message.content)
        messages.append({"role": "assistant", "content": raw})

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            break

        if "explore" in result:
            if min(int(result["explore"]), 6) <= depth:
                break
            depth += 1
            if hop_callback:
                hop_callback(depth - 1, "explore_requested", f"→ Hop {depth}")
            continue

        influence = result.get("influence_level", "MEDIUM")
        if hop_callback:
            hop_callback(depth, "final", influence)
        return {"influence_level": influence, "insight": result.get("insight", ""), "recommended_action": result.get("recommended_action", "")}

    messages.append({"role": "user", "content": "Produce the final briefing now. Return ONLY the JSON with influence_level, insight, and recommended_action."})
    result = json.loads(strip_fences(llm.chat.completions.create(
        model=config.llm_model, messages=messages, temperature=0.2,
    ).choices[0].message.content))
    influence = result.get("influence_level", "MEDIUM")
    if hop_callback:
        hop_callback(depth, "final", influence)
    return {"influence_level": influence, "insight": result.get("insight", ""), "recommended_action": result.get("recommended_action", "")}


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
