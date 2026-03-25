"""SkillDiscoverer — uses an LLM to propose new skills based on graph topology.

The discoverer analyses the graph structure (node types, edge types, degree
distribution, existing skill ids) and asks an LLM to write a new ``.skill.md``
file that might uncover novel compliance patterns.

LLM back-end
------------
The implementation uses the OpenAI Chat Completions API.  If the
``OPENAI_API_KEY`` environment variable is not set the discoverer will log a
warning and return an empty list, making it safe to run in environments without
an API key.
"""
from __future__ import annotations

import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import networkx as nx

from models import Skill

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a compliance analyst who specialises in detecting fraud, money
    laundering, and ownership-chain anomalies in knowledge graphs.

    The user will provide you with a JSON summary of a knowledge graph
    (node types, edge types, existing skill ids, and basic statistics).

    Your task is to write ONE new skill in the following .skill.md format:

    ```meta
    id: <snake_case_id>
    version: "1.0"
    severity: <CRITICAL|HIGH|MEDIUM|LOW>
    tags: [<tag1>, <tag2>]
    enabled: true
    ```

    ## Description
    <2-3 sentences describing what this skill detects>

    ```pattern
    node_types: [<list of relevant node types>]
    edge_types: [<list of relevant edge types>]
    ```

    ```python
    def detect(G, ctx):
        leads = []
        # Use ONLY ctx.* API methods
        # DO NOT use import, exec, eval, open, os, sys, or any dunder
        return leads
    ```

    ```test_cases
    - description: <short description>
      graph:
        nodes:
          - {id: A, type: <NodeType>}
        edges: []
      expected_leads: 0
    ```

    Rules:
    - The skill id must NOT be in the existing_skill_ids list.
    - Only use ctx.* API: nodes_by_type, in_neighbors, out_neighbors,
      edges_by_type, node_attr, subgraph_hops, find_cycles, betweenness,
      community_of, parse_amount, lead.
    - Do NOT use import, exec, eval, open, os, sys, globals, locals,
      getattr, setattr, delattr, hasattr, __builtins__, or any dunder.
    - Return the raw .skill.md text and nothing else.
    """
).strip()


class SkillDiscoverer:
    """Propose new skills using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        output_dir: str | Path = "skills",
    ) -> None:
        self._model = model
        self._output_dir = Path(output_dir)

    def discover(
        self,
        G: nx.MultiDiGraph,
        existing_skills: list[Skill],
        max_new: int = 3,
    ) -> list[str]:
        """Discover up to *max_new* new skill markdown texts.

        Parameters
        ----------
        G:
            The knowledge graph to analyse.
        existing_skills:
            Already-loaded skills (their ids are excluded from proposals).
        max_new:
            Maximum number of new skills to generate in one call.

        Returns
        -------
        list[str]
            Raw ``.skill.md`` markdown strings (not yet validated or saved).
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set; SkillDiscoverer will return no skills."
            )
            return []

        summary = self._graph_summary(G, existing_skills)
        results: list[str] = []

        try:
            import openai  # local import — only needed if discovery is used

            client = openai.OpenAI(api_key=api_key)
            for _ in range(max_new):
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(summary, ensure_ascii=False),
                        },
                    ],
                    # 0.7 balances creativity (diverse skill ideas) with
                    # coherence (syntactically valid skill.md output).
                    temperature=0.7,
                )
                content = response.choices[0].message.content or ""
                results.append(content)
                # Update summary so each subsequent call avoids duplicate ids
                new_id = self._extract_id(content)
                if new_id:
                    summary["existing_skill_ids"].append(new_id)
        except Exception:  # noqa: BLE001
            logger.exception("SkillDiscoverer LLM call failed.")

        return results

    def save_pending(self, skill_text: str, skill_id: str) -> Path:
        """Save a generated skill as a ``_pending_`` file for human review.

        Parameters
        ----------
        skill_text:
            Raw ``.skill.md`` markdown.
        skill_id:
            The id extracted from the skill.

        Returns
        -------
        Path
            The path of the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"_pending_{skill_id}.skill.md"
        path = self._output_dir / filename
        path.write_text(skill_text, encoding="utf-8")
        logger.info("Saved pending skill to '%s'.", path)
        return path

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _graph_summary(
        G: nx.MultiDiGraph,
        existing_skills: list[Skill],
    ) -> dict[str, Any]:
        node_types: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            t = str(data.get("type", "unknown"))
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: dict[str, int] = {}
        for _, _, data in G.edges(data=True):
            t = str(data.get("type", "unknown"))
            edge_types[t] = edge_types.get(t, 0) + 1

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "existing_skill_ids": [s.id for s in existing_skills],
        }

    @staticmethod
    def _extract_id(skill_text: str) -> str | None:
        import re

        m = re.search(r"id\s*:\s*(\S+)", skill_text)
        return m.group(1) if m else None
