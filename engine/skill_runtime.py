"""SkillRuntime — executes ``Skill`` objects against a graph in a sandbox.

Each skill's ``detect`` function is called with the graph and a
:class:`~engine.skill_context.SkillContext` instance.  The context is the
*only* object the skill can use to interact with the graph, which prevents
arbitrary code from accessing the file system or network.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

import networkx as nx

from engine.skill_context import SkillContext
from models import Lead, Skill

logger = logging.getLogger(__name__)


class SkillRuntime:
    """Run all enabled skills against a graph and collect :class:`~models.Lead` objects."""

    def __init__(self, ctx: SkillContext | None = None) -> None:
        self._ctx = ctx or SkillContext()

    def run_all(
        self,
        G: nx.MultiDiGraph,
        skills: list[Skill],
    ) -> list[Lead]:
        """Execute every enabled skill and return deduplicated leads.

        Parameters
        ----------
        G:
            The knowledge graph to analyse.
        skills:
            List of compiled :class:`~models.Skill` objects.

        Returns
        -------
        list[Lead]
            Sorted by score descending.
        """
        all_leads: list[Lead] = []
        for skill in skills:
            if not skill.enabled:
                continue
            leads = self.run_one(G, skill)
            all_leads.extend(leads)
        # Sort by score descending
        all_leads.sort(key=lambda lead: lead.score, reverse=True)
        return all_leads

    def run_one(
        self,
        G: nx.MultiDiGraph,
        skill: Skill,
    ) -> list[Lead]:
        """Execute a single skill safely.

        Parameters
        ----------
        G:
            The knowledge graph.
        skill:
            The skill to run.

        Returns
        -------
        list[Lead]
            Empty list on any error.
        """
        try:
            raw = skill.detect_fn(G, self._ctx)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Skill '%s' raised an exception:\n%s",
                skill.id,
                traceback.format_exc(),
            )
            return []

        leads: list[Lead] = []
        for item in raw or []:
            if not isinstance(item, Lead):
                logger.warning(
                    "Skill '%s' returned non-Lead item: %r", skill.id, item
                )
                continue
            # Stamp the skill id onto every lead it produced
            item.skill_id = skill.id
            leads.append(item)

        logger.info("Skill '%s' produced %d lead(s).", skill.id, len(leads))
        return leads
