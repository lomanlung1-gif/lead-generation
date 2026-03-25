"""SkillContext — the sandboxed API surface exposed to skill ``detect`` functions.

Skills call ``ctx.*`` methods instead of touching the graph directly.  This
gives us a single choke-point for validation and makes skills portable.
"""
from __future__ import annotations

import re
from typing import Any

import networkx as nx

from models import Lead, Severity


class SkillContext:
    """Provides the ``ctx`` API used inside skill ``detect`` functions."""

    # ── node helpers ───────────────────────────────────────────────────────

    @staticmethod
    def nodes_by_type(G: nx.MultiDiGraph, node_type: str) -> list[str]:
        """Return all node ids whose ``type`` attribute equals *node_type*."""
        return [
            n
            for n, data in G.nodes(data=True)
            if data.get("type") == node_type
        ]

    @staticmethod
    def in_neighbors(
        G: nx.MultiDiGraph, node: str, edge_type: str | None = None
    ) -> list[str]:
        """Return predecessors of *node*, optionally filtered by *edge_type*."""
        if edge_type is None:
            return list(G.predecessors(node))
        return [
            u
            for u, _, data in G.in_edges(node, data=True)
            if data.get("type") == edge_type
        ]

    @staticmethod
    def out_neighbors(
        G: nx.MultiDiGraph, node: str, edge_type: str | None = None
    ) -> list[str]:
        """Return successors of *node*, optionally filtered by *edge_type*."""
        if edge_type is None:
            return list(G.successors(node))
        return [
            v
            for _, v, data in G.out_edges(node, data=True)
            if data.get("type") == edge_type
        ]

    @staticmethod
    def edges_by_type(
        G: nx.MultiDiGraph, edge_type: str
    ) -> list[tuple[str, str, dict]]:
        """Return all edges whose ``type`` attribute equals *edge_type*."""
        return [
            (u, v, data)
            for u, v, data in G.edges(data=True)
            if data.get("type") == edge_type
        ]

    @staticmethod
    def node_attr(
        G: nx.MultiDiGraph, node: str, key: str, default: Any = None
    ) -> Any:
        """Return attribute *key* of *node*, or *default* if absent."""
        return G.nodes[node].get(key, default) if node in G else default

    # ── graph analytics ────────────────────────────────────────────────────

    @staticmethod
    def subgraph_hops(
        G: nx.MultiDiGraph, entity: str, hops: int
    ) -> set[str]:
        """Return all nodes reachable from *entity* within *hops* steps."""
        visited: set[str] = set()
        frontier = {entity}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(G.successors(node))
                next_frontier.update(G.predecessors(node))
            next_frontier -= visited
            next_frontier.discard(entity)
            visited |= frontier
            frontier = next_frontier
        visited |= frontier
        return visited

    @staticmethod
    def find_cycles(
        G: nx.MultiDiGraph, edge_type: str | None = None
    ) -> list[list[str]]:
        """Return simple cycles in the graph, optionally restricted to *edge_type* edges."""
        if edge_type is not None:
            sub = nx.MultiDiGraph()
            sub.add_nodes_from(G.nodes(data=True))
            sub.add_edges_from(
                (u, v, data)
                for u, v, data in G.edges(data=True)
                if data.get("type") == edge_type
            )
            view = sub
        else:
            view = G
        return list(nx.simple_cycles(view))

    @staticmethod
    def betweenness(
        G: nx.MultiDiGraph, top_pct: float = 0.1
    ) -> list[tuple[str, float]]:
        """Return the top *top_pct* fraction of nodes by betweenness centrality."""
        bc = nx.betweenness_centrality(G)
        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_bc) * top_pct))
        return sorted_bc[:k]

    @staticmethod
    def community_of(G: nx.MultiDiGraph, node: str) -> list[str]:
        """Return the weakly-connected component containing *node*."""
        for component in nx.weakly_connected_components(G):
            if node in component:
                return list(component)
        return [node]

    # ── data helpers ───────────────────────────────────────────────────────

    @staticmethod
    def parse_amount(label: str) -> float:
        """Parse a currency-like string into a float.

        Examples::

            ctx.parse_amount("¥1,234,567.89")  → 1234567.89
            ctx.parse_amount("USD 500k")        → 500000.0
        """
        if not label:
            return 0.0
        text = str(label)
        # Handle shorthand suffixes (k / m / b)
        multiplier = 1.0
        lower = text.lower()
        if lower.endswith("k"):
            multiplier = 1_000.0
            text = text[:-1]
        elif lower.endswith("m"):
            multiplier = 1_000_000.0
            text = text[:-1]
        elif lower.endswith("b"):
            multiplier = 1_000_000_000.0
            text = text[:-1]
        # Strip non-numeric characters except decimal point
        numeric = re.sub(r"[^\d.]", "", text)
        try:
            return float(numeric) * multiplier
        except ValueError:
            return 0.0

    # ── lead factory ───────────────────────────────────────────────────────

    @staticmethod
    def lead(
        title: str,
        severity: str | Severity,
        score: float,
        entities: list[str],
        evidence: list[str],
        actions: list[str],
        skill_id: str = "",
    ) -> Lead:
        """Construct a :class:`~models.Lead` instance.

        The *skill_id* is normally injected by the runtime after the fact, so
        skills can omit it.
        """
        return Lead(
            skill_id=skill_id,
            title=title,
            severity=Severity(severity) if isinstance(severity, str) else severity,
            score=float(score),
            entities=list(entities),
            evidence=list(evidence),
            actions=list(actions),
        )
