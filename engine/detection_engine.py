"""DetectionEngine — main orchestrator.

Workflow
--------
1. Load all ``.skill.md`` files from the skills directory.
2. Validate each skill (AST + tests).
3. Run validated skills against the graph.
4. Optionally run LLM-based discovery (up to ``discovery_rounds`` times).
5. Newly discovered skills that pass validation are hot-loaded and re-run.
6. Return all leads sorted by score descending.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import networkx as nx

from engine.skill_discoverer import SkillDiscoverer
from engine.skill_loader import SkillLoader
from engine.skill_runtime import SkillRuntime
from engine.skill_validator import SkillValidator
from models import Lead, Skill

logger = logging.getLogger(__name__)


class DetectionEngine:
    """Orchestrate graph loading, skill execution, and lead generation."""

    def __init__(
        self,
        skills_dir: Union[str, Path] = "skills",
        discovery_rounds: int = 3,
        use_discovery: bool = False,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._discovery_rounds = discovery_rounds
        self._use_discovery = use_discovery
        self._llm_model = llm_model

        self._loader = SkillLoader()
        self._validator = SkillValidator()
        self._runtime = SkillRuntime()
        self._discoverer = SkillDiscoverer(
            model=llm_model,
            output_dir=self._skills_dir,
        )

    def run(self, G: nx.MultiDiGraph) -> list[Lead]:
        """Execute the full detection pipeline against *G*.

        Parameters
        ----------
        G:
            A ``nx.MultiDiGraph`` built by :class:`~graph_builder.GraphBuilder`.

        Returns
        -------
        list[Lead]
            All leads produced by validated skills, sorted by score descending.
        """
        # ── 1. Load & validate skills ──────────────────────────────────────
        skills = self._load_and_validate(self._skills_dir)
        logger.info("Active skills after initial load: %d", len(skills))

        # ── 2. Run initial detection ───────────────────────────────────────
        all_leads: list[Lead] = self._runtime.run_all(G, skills)

        # ── 3. Discovery loop (optional) ──────────────────────────────────
        if self._use_discovery:
            for round_num in range(1, self._discovery_rounds + 1):
                logger.info("Discovery round %d / %d", round_num, self._discovery_rounds)
                new_texts = self._discoverer.discover(G, skills)
                if not new_texts:
                    logger.info("No new skills proposed; stopping discovery.")
                    break
                new_skills = self._try_add_skills(new_texts, skills)
                if not new_skills:
                    logger.info("No new skills passed validation; stopping.")
                    break
                skills.extend(new_skills)
                # Re-run only the newly added skills to avoid duplicate leads
                round_leads = self._runtime.run_all(G, new_skills)
                all_leads.extend(round_leads)

        # ── 4. Sort & return ───────────────────────────────────────────────
        all_leads.sort(key=lambda lead: lead.score, reverse=True)
        logger.info("Total leads: %d", len(all_leads))
        return all_leads

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_and_validate(self, skills_dir: Path) -> list[Skill]:
        """Load skills from disk and validate each one."""
        raw_skills = SkillLoader.load_dir(skills_dir)
        valid_skills: list[Skill] = []
        for skill in raw_skills:
            result = self._validator.validate(skill)
            if result.passed:
                valid_skills.append(skill)
            else:
                logger.warning(
                    "Skill '%s' failed validation and will not run: %s",
                    skill.id,
                    result.errors,
                )
        return valid_skills

    def _try_add_skills(
        self, skill_texts: list[str], existing: list[Skill]
    ) -> list[Skill]:
        """Parse, validate, and save pending/active new skills."""
        existing_ids = {s.id for s in existing}
        new_valid: list[Skill] = []

        for text in skill_texts:
            try:
                skill = SkillLoader.parse(text, source="<llm_generated>")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse LLM-generated skill: %s", exc)
                self._discoverer.save_pending(text, "_unparseable_")
                continue

            if skill.id in existing_ids:
                logger.info("Skill '%s' already exists; skipping.", skill.id)
                continue

            result = self._validator.validate(skill)
            if result.passed:
                logger.info("LLM skill '%s' passed validation; activating.", skill.id)
                # Save as a proper skill file
                path = self._skills_dir / f"{skill.id}.skill.md"
                path.write_text(text, encoding="utf-8")
                skill.enabled = True
                new_valid.append(skill)
                existing_ids.add(skill.id)
            else:
                logger.warning(
                    "LLM skill '%s' failed validation; saving as pending.",
                    skill.id,
                )
                self._discoverer.save_pending(text, skill.id)

        return new_valid
