from __future__ import annotations

import json
from typing import Any, Callable

import networkx as nx
import pandas as pd
from openai import OpenAI


def strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped


def relation_text(src: str, edge_type: Any, dst: str) -> str:
    label = str(edge_type).strip()
    key = " ".join(label.replace("_", " ").replace("-", " ").split()).casefold()

    templates = {
        "owns": "{src} owns {dst}",
        "owner": "{src} is an owner of {dst}",
        "shareholder": "{src} is a shareholder of {dst}",
        "director": "{src} is a director of {dst}",
        "employee": "{src} is an employee of {dst}",
        "employer": "{src} is the employer of {dst}",
        "works at": "{src} works at {dst}",
        "works for": "{src} works for {dst}",
        "partner": "{src} is a partner of {dst}",
        "founder": "{src} is the founder of {dst}",
        "father": "{src} is the father of {dst}",
        "mother": "{src} is the mother of {dst}",
        "child": "{src} is a child of {dst}",
        "son": "{src} is the son of {dst}",
        "daughter": "{src} is the daughter of {dst}",
        "spouse": "{src} is the spouse of {dst}",
        "husband": "{src} is the husband of {dst}",
        "wife": "{src} is the wife of {dst}",
        "shared address": "{src} shares an address with {dst}",
        "shared email": "{src} shares an email with {dst}",
        "shared phone": "{src} shares a phone number with {dst}",
        "mentions": "{src} is mentioned with {dst}",
        "transaction": "{src} has a transaction with {dst}",
        "beneficiary": "{src} is a beneficiary of {dst}",
        "trustee": "{src} is a trustee of {dst}",
        "settlor": "{src} is the settlor of {dst}",
    }

    template = templates.get(key)
    if template:
        return template.format(src=src, dst=dst)
    return f"{src} has a '{label}' relationship to {dst}"


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

    return {
        "node": node,
        "attrs": attrs,
        "relations": out_rels + in_rels,
        "second_hop": second_hop[:20],
    }


def analyze_node(
    node: str,
    artifacts: Any,
    final_goal: str,
    config: Any,
    llm: OpenAI,
    score_reason: str = "",
    hop_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, str]:
    context = deep_node_context(artifacts.graph, node)
    context_json = json.dumps(context, ensure_ascii=False, default=str)

    if hop_callback:
        hop_callback(1, "fetching", f"{len(context['relations'])} relations, {len(context['second_hop'])} second-hop")

    prompt = f"""
    You are a senior wealth management advisor analyzing a graph-derived client opportunity.

    ### GOAL:
    {final_goal}

    ### NODE CONTEXT:
    {context_json}

    ### INITIAL TARGETING REASON:
    {score_reason}

    Write a professional client briefing using exactly two sections.

    Interpret relation_text as the preferred plain-English description of each relationship.
    Use edge_type, source, target, and attrs as structured evidence to verify the direction and meaning.

    Insight:
    - Explain who this person or organization is based on the graph.
    - Summarize the financial situation, business ownership, cash movement, relationships, and any life-event or liquidity signal visible in the data.
    - Cite actual evidence from the node context.

    Recommended Action:
    - Provide 3 to 4 concrete next-step actions.
    - Each action must be specific to this client and tied to the goal.
    - Explain why each action fits the evidence.

    Return ONLY JSON:
    {{"insight": "...", "recommended_action": "..."}}
    """
    response = llm.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    result = _parse_narrative(response.choices[0].message.content or "")

    if hop_callback:
        hop_callback(1, "final", result.get("recommended_action", ""))

    return {
        "insight": result.get("insight", ""),
        "recommended_action": result.get("recommended_action", ""),
    }


if __name__ == "__main__":
    from main import LeadGenConfig, init_llm_client, load_artifacts

    config = LeadGenConfig(debug=True, auto_write_rules=False)
    final_goal = "Identify high-potential leads for financial services based on their relationships and attributes in the graph."

    artifacts = load_artifacts("COI_Template.xlsx", "topology.md", config)
    llm = init_llm_client()

    node = "Hubertus von Baumbach"
    print("\n=== analyze_node test ===")
    print(analyze_node(node, artifacts, final_goal, config, llm))