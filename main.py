import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers.cross_encoder import CrossEncoder

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".secrets" / "deepseek.env")
load_dotenv(ROOT / ".env")


_cross_encoder_model: CrossEncoder | None = None


@dataclass
class Thread:
        label: str
        pri: float
        lines: list[str] = field(default_factory=list)


CLASSIFY_PROMPT = """\
You are a graph-intelligence analyst. Classify each edge into exactly ONE
category and assess its urgency.

CATEGORIES (use these exact labels):
    CONTROL   – ownership, directorship, shareholding, trusteeship, officer role,
                             beneficial ownership, power of attorney, any corporate/legal control
    FAMILY    – any family, kinship, or domestic relationship
    SHARED_ID – shared address, phone, email, registration, or any identifier
                             that co-links separate entities
    RISK      – sanctions, PEP flags, adverse media, regulatory warnings, licence
                             revocations, criminal records, compliance red flags
    EVENT     – news mentions, lawsuits, bankruptcies, deaths, acquisitions, IPOs,
                             succession events, or any other notable occurrence
    OTHER     – anything that does not clearly fit the above

For every edge return three fields:
    cat – one of the category labels above
    hot – true ONLY if it signals a TIME-SENSITIVE opportunity or threat
                (e.g. recent death / succession / active lawsuit / new sanction /
                live acquisition / bankruptcy filing)
    pri – priority 0.0–1.0  (1.0 = most critical to an analyst)

EDGES:
{edges_json}

Return ONLY a JSON array, same length and order as input. Example:
[{{"cat":"CONTROL","hot":false,"pri":0.7}}, {{"cat":"EVENT","hot":true,"pri":0.95}}]"""

NARRATIVE_PROMPT = """\
You are an intelligence analyst. Write a concise briefing for:
TARGET: {target}
GOAL: {goal}

STRICT RULES:
- ONLY describe relationships explicitly present in the GRAPH INTELLIGENCE below.
- Indirect connections (marked "via FAMILY" or "via CONTROL") are multi-hop.
    Describe them as indirect - do NOT treat them as direct relationships.
- Do NOT infer, fabricate, or speculate about connections not shown in the data.
- If a multi-hop event has unclear relevance to TARGET, acknowledge the uncertainty.

Cover:
    (1) Identity & controlled assets
        (2) Events creating opportunity or risk - state the actual connection path
    (3) Recommended approach angle

Prioritise ⚡ items only where genuinely relevant. Narrative form, no lists.

GRAPH INTELLIGENCE:
{ctx}

Return ONLY JSON:
{{"insight": "...", "recommended_action": "..."}}"""

CHUNK_SIZE = 50


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
    node_names: list[str]
    edges: list[tuple[str, str, dict[str, Any]]]


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

    node_names = list(graph.nodes)

    edges = [(u, v, d) for u, v, d in graph.edges(data=True)]

    if status_callback:
        status_callback("Artifacts ready.")
    print(f"Ready | {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return GraphArtifacts(
        graph=graph,
        topology=topology,
        node_names=node_names,
        edges=edges,
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



def _node_label(node: dict[str, Any] | None, fallback: str) -> str:
    if not node:
        return fallback

    for key in ("Name", "name", "Title", "title", "label", "Label"):
        value = node.get(key)
        if value:
            label = str(value).strip()
            if label:
                node_type = node.get("Type") or node.get("type")
                if node_type:
                    type_text = str(node_type).strip()
                    if type_text:
                        return f"{label}[{type_text}]"
                return label

    node_type = node.get("Type") or node.get("type")
    if node_type:
        type_text = str(node_type).strip()
        if type_text:
            return f"{fallback}[{type_text}]"
    return fallback


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return str(edge["source"]), str(edge["edge_type"]), str(edge["target"])


def _edge_desc(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = str(edge["source"])
    target = str(edge["target"])
    desc: dict[str, Any] = {
        "src": _node_label(nodes.get(source), source),
        "rel": str(edge["edge_type"]),
        "tgt": _node_label(nodes.get(target), target),
    }
    if edge.get("edge_info"):
        desc["info"] = str(edge["edge_info"])[:250]

    for nid in (source, target):
        node = nodes.get(nid)
        if not node:
            continue
        for key in ("title", "status", "description", "Title", "Status", "Description"):
            value = node.get(key)
            if value:
                desc[f"{nid}_{key.lower()}"] = str(value)[:150]
    return desc


def _safe_parse(raw: str, expected: int) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if len(arr) == expected:
                validated: list[dict[str, Any]] = []
                for item in arr:
                    pri_value = item.get("pri", 0.3)
                    try:
                        pri = float(pri_value)
                    except (TypeError, ValueError):
                        pri = 0.3
                    validated.append({
                        "cat": str(item.get("cat", "OTHER")).strip().upper() or "OTHER",
                        "hot": bool(item.get("hot", False)),
                        "pri": max(0.0, min(1.0, pri)),
                    })
                return validated
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return [{"cat": "OTHER", "hot": False, "pri": 0.3}] * expected


def _collect(center: str, graph: nx.DiGraph, hops: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    nodes: dict[str, dict[str, Any]] = {}
    adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []
    queue: list[tuple[str, int]] = [(center, 0)]

    while queue:
        nid, depth = queue.pop(0)
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        if not graph.has_node(nid):
            continue

        nodes[nid] = dict(graph.nodes[nid])

        for source, target, data in list(graph.out_edges(nid, data=True)) + list(graph.in_edges(nid, data=True)):
            edge = {
                "source": str(source),
                "target": str(target),
                "edge_type": str(data.get("edge_type", "")),
                "edge_info": data.get("edge_info", ""),
            }
            edge_key = _edge_key(edge)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append(edge)
            adj[edge["source"]].append(edge)
            adj[edge["target"]].append(edge)

            neighbor = target if str(source) == nid else source
            if neighbor not in seen_nodes and depth < hops:
                queue.append((str(neighbor), depth + 1))

    return nodes, edges, adj


def _classify(edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], llm: OpenAI, model: str) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    for start in range(0, len(edges), CHUNK_SIZE):
        chunk = edges[start : start + CHUNK_SIZE]
        descs = [_edge_desc(edge, nodes) for edge in chunk]
        prompt = CLASSIFY_PROMPT.format(edges_json=json.dumps(descs, ensure_ascii=False, default=str))
        try:
            response = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = strip_fences(response.choices[0].message.content or "")
        except Exception as exc:
            if "context length" in str(exc).lower() and len(chunk) > 1:
                midpoint = len(chunk) // 2
                tags.extend(_classify(chunk[:midpoint], nodes, llm, model))
                tags.extend(_classify(chunk[midpoint:], nodes, llm, model))
                continue
            raise

        tags.extend(_safe_parse(raw, len(chunk)))

    return tags


def _eline(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    source = str(edge["source"])
    target = str(edge["target"])
    text = f"{_node_label(nodes.get(source), source)}—[{edge['edge_type']}]→{_node_label(nodes.get(target), target)}"
    if edge.get("edge_info"):
        return f"{text} | {str(edge['edge_info'])[:120]}"
    return text


def _threads(
    center: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    adj: dict[str, list[dict[str, Any]]],
    tags: list[dict[str, Any]],
) -> list[Thread]:
    edge_index = {_edge_key(edge): idx for idx, edge in enumerate(edges)}
    buckets: dict[str, Thread] = {}
    used: set[tuple[str, str, str]] = set()

    def bucket(label: str) -> Thread:
        thread = buckets.get(label)
        if thread is None:
            thread = Thread(label=label, pri=0.0, lines=[])
            buckets[label] = thread
        return thread

    for edge, tag in zip(edges, tags):
        if center not in (edge["source"], edge["target"]):
            continue

        cat = str(tag.get("cat", "OTHER")).strip().upper() or "OTHER"
        hot = bool(tag.get("hot", False))
        pri = float(tag.get("pri", 0.3))
        label = f"⚡{cat}" if hot else cat
        thread = bucket(label)
        thread.label = label
        thread.pri = max(thread.pri, pri)
        thread.lines.append(("⚡ " if hot else "") + _eline(edge, nodes))
        used.add(_edge_key(edge))

        if cat == "CONTROL":
            neighbor = edge["target"] if edge["source"] == center else edge["source"]
            for edge2 in adj.get(str(neighbor), []):
                edge2_key = _edge_key(edge2)
                if edge2_key in used:
                    continue
                other = edge2["target"] if edge2["source"] == neighbor else edge2["source"]
                if other == center:
                    continue
                linked_idx = edge_index.get(edge2_key)
                if linked_idx is not None and str(tags[linked_idx].get("cat", "")).strip().upper() == "CONTROL":
                    used.add(edge2_key)
                    thread.lines.append(" ↳ " + _eline(edge2, nodes))

        if cat == "FAMILY":
            relation = edge["target"] if edge["source"] == center else edge["source"]
            for edge2 in adj.get(str(relation), []):
                edge2_key = _edge_key(edge2)
                if edge2_key in used:
                    continue
                linked_idx = edge_index.get(edge2_key)
                if linked_idx is not None and bool(tags[linked_idx].get("hot", False)):
                    used.add(edge2_key)
                    relation_name = _node_label(nodes.get(str(relation)), str(relation))
                    thread.lines.append(f"  ⚡ via {relation_name}: {_eline(edge2, nodes)}")
                    thread.pri = max(thread.pri, float(tags[linked_idx].get("pri", 0.3)))

    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge, tag in zip(edges, tags):
        if str(tag.get("cat", "")).strip().upper() == "SHARED_ID":
            clusters[(str(edge["edge_type"]), str(edge["target"]))].add(str(edge["source"]))

    for (edge_type, target), sources in clusters.items():
        if len(sources) < 2:
            continue
        thread = bucket("⚡HIDDEN_LINK")
        thread.label = "⚡HIDDEN_LINK"
        thread.pri = max(thread.pri, 0.85)
        names = ", ".join(_node_label(nodes.get(source), source) for source in list(sources)[:10])
        thread.lines.append(f"⚡ {edge_type}: {len(sources)} entities share {_node_label(nodes.get(target), target)}: {names}")

    for edge, tag in zip(edges, tags):
        edge_key = _edge_key(edge)
        if edge_key in used:
            continue

        if not bool(tag.get("hot", False)):
            continue

        src, tgt = edge["source"], edge["target"]
        if center in (src, tgt):
            continue

        # Only keep 2-hop hot edges when the intermediary has a verified FAMILY/CONTROL bridge to center.
        for endpoint in (src, tgt):
            bridge = next(
                (
                    edge2
                    for edge2 in adj.get(str(endpoint), [])
                    if center in (edge2["source"], edge2["target"]) and _edge_key(edge2) != edge_key
                ),
                None,
            )
            if not bridge:
                continue

            bridge_idx = edge_index.get(_edge_key(bridge))
            if bridge_idx is None:
                continue

            bridge_cat = str(tags[bridge_idx].get("cat", "")).strip().upper()
            if bridge_cat not in {"FAMILY", "CONTROL"}:
                continue

            far = tgt if endpoint == src else src
            chain = (
                f"{_node_label(nodes.get(center), center)} "
                f"—[{bridge['edge_type']}]→ "
                f"{_node_label(nodes.get(endpoint), endpoint)} "
                f"—[{edge['edge_type']}]→ "
                f"{_node_label(nodes.get(far), far)}"
            )
            label = f"⚡{str(tag.get('cat', 'OTHER')).strip().upper() or 'OTHER'}_VIA_{bridge_cat}"
            thread = bucket(label)
            thread.label = label
            thread.pri = max(thread.pri, float(tag.get("pri", 0.3)) * 0.7)
            info = f" | {str(edge['edge_info'])[:120]}" if edge.get("edge_info") else ""
            thread.lines.append(f"⚡ (via {bridge_cat}): {chain}{info}")
            used.add(edge_key)
            break

    # Remainder: only unused direct edges attached to the center.
    remainder = [
        (edge, tags[idx])
        for idx, edge in enumerate(edges)
        if _edge_key(edge) not in used and center in (edge["source"], edge["target"])
    ]
    if remainder:
        thread = bucket("OTHER")
        thread.label = "OTHER"
        thread.pri = max(thread.pri, max(float(tag.get("pri", 0.3)) for _, tag in remainder))
        for edge, _ in remainder:
            thread.lines.append(_eline(edge, nodes))
            used.add(_edge_key(edge))

    result = [thread for thread in buckets.values() if thread.lines]
    result.sort(key=lambda item: item.pri, reverse=True)
    return result


def _serialize(threads: list[Thread], budget: int) -> str:
    blocks: list[str] = []
    total = 0
    for thread in threads:
        block = f"【{thread.label}】 pri={thread.pri:.2f}\n" + "\n".join(f"  {line}" for line in thread.lines[:20]) + "\n"
        if total + len(block) > budget:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


def _parse_narrative(raw: str) -> dict[str, str]:
    raw_text = strip_fences(raw)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"insight": raw_text, "recommended_action": ""}

    if isinstance(parsed, dict):
        return {
            "insight": str(parsed.get("insight", "")),
            "recommended_action": str(parsed.get("recommended_action", "")),
        }

    return {"insight": raw_text, "recommended_action": ""}


def analyze_node(
    node: str,
    artifacts: GraphArtifacts,
    final_goal: str,
    config: LeadGenConfig,
    llm: OpenAI,
    max_hops: int = 2,
    max_ctx: int = 6000,
    hop_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, str]:
    graph = artifacts.graph
    hop_limit = max(1, min(max_hops, 6))
    nodes, edges, adj = _collect(node, graph, hop_limit)

    if hop_callback:
        hop_callback(hop_limit, "fetching", f"{len(nodes)} nodes, {len(edges)} edges")

    tags = _classify(edges, nodes, llm, config.llm_model)
    threads = _threads(node, nodes, edges, adj, tags)
    ctx = _serialize(threads, max_ctx)

    prompt = NARRATIVE_PROMPT.format(
        target=_node_label(nodes.get(node), node),
        goal=final_goal,
        ctx=ctx,
    )
    response = llm.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    result = _parse_narrative(response.choices[0].message.content or "")

    if hop_callback:
        hop_callback(hop_limit, "final", result.get("recommended_action", ""))

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
    artifacts = load_artifacts("COI_Template.xlsx", "topology.md", config)
    llm = init_llm_client()
    final_goal = "Identify high-potential leads for financial services based on their relationships and attributes in the graph."
    node = "Hubertus von Baumbach"
    print("\n=== analyze_node test ===")
    print(analyze_node(node, artifacts, final_goal, config, llm))

    # targets = generate_targets(
    #     excel_path="COI_Template.xlsx",
    #     final_goal=final_goal,
    #     topology_path="topology.md",
    #     config=config,
    # )
    # print("\n=== High-Priority Target List ===")
    # for target in targets[:10]:
    #     print(f"  {target['node_name']} | {target['score']} | {target['reason']}")
