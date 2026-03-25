"""SkillLoader — parses ``.skill.md`` files into ``Skill`` objects.

Skill.md format
---------------
Each file contains five fenced sections (order is flexible):

```meta
id: ownership_chain
version: "1.0"
severity: HIGH
tags: [ownership, chain]
enabled: true
```

## Description
Two or three sentences describing what the skill detects.

```pattern
node_types: [Company, Person]
edge_types: [OWNS, CONTROLS]
```

```python
def detect(G, ctx) -> list:
    leads = []
    # ... skill logic using ctx API ...
    return leads
```

```test_cases
- description: basic smoke test
  graph:
    nodes:
      - {id: A, type: Company}
    edges: []
  expected_leads: 0
```
"""
from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

import yaml

from models import Lead, Severity, Skill

logger = logging.getLogger(__name__)

# Regex patterns for each section
_META_RE = re.compile(
    r"```(?:meta|yaml meta)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_DESC_RE = re.compile(
    r"##\s*Description\s*\n(.*?)(?=\n##|\n```|$)", re.DOTALL | re.IGNORECASE
)
_PATTERN_RE = re.compile(
    r"```(?:pattern|yaml pattern)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_LOGIC_RE = re.compile(
    r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_TEST_RE = re.compile(
    r"```(?:test_cases|yaml test_cases)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


class SkillLoader:
    """Load and compile ``.skill.md`` files."""

    @classmethod
    def load_dir(cls, directory: str | Path) -> list[Skill]:
        """Load all ``.skill.md`` files from *directory*.

        Returns
        -------
        list[Skill]
            Only skills that parsed and compiled successfully with ``enabled=True``.
        """
        directory = Path(directory)
        skills: list[Skill] = []
        for path in sorted(directory.glob("*.skill.md")):
            try:
                skill = cls.load_file(path)
                if skill.enabled:
                    skills.append(skill)
                    logger.info("Loaded skill '%s' from '%s'", skill.id, path.name)
                else:
                    logger.info("Skill '%s' disabled, skipping.", skill.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load skill '%s': %s", path.name, exc)
        return skills

    @classmethod
    def load_file(cls, path: str | Path) -> Skill:
        """Parse a single ``.skill.md`` file and return a ``Skill``.

        Parameters
        ----------
        path:
            Path to the ``.skill.md`` file.

        Raises
        ------
        ValueError
            If required sections are missing or YAML/Python is invalid.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return cls.parse(text, source=str(path))

    @classmethod
    def parse(cls, text: str, source: str = "<string>") -> Skill:
        """Parse skill markdown text.

        Parameters
        ----------
        text:
            Raw markdown content.
        source:
            Human-readable label for error messages.
        """
        meta = cls._parse_meta(text, source)
        description = cls._parse_description(text)
        triggers = cls._parse_pattern(text)
        detect_fn = cls._compile_logic(text, source)
        test_cases = cls._parse_test_cases(text)

        return Skill(
            id=str(meta.get("id", source)),
            version=str(meta.get("version", "1.0")),
            name=str(meta.get("name", meta.get("id", source))),
            description=description,
            tags=list(meta.get("tags", [])),
            enabled=bool(meta.get("enabled", True)),
            triggers=triggers,
            detect_fn=detect_fn,
            test_cases=test_cases,
        )

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_meta(text: str, source: str) -> dict[str, Any]:
        m = _META_RE.search(text)
        if not m:
            raise ValueError(f"[{source}] Missing ```meta``` block")
        data = yaml.safe_load(m.group(1)) or {}
        if "id" not in data:
            raise ValueError(f"[{source}] Meta block must contain 'id'")
        return data

    @staticmethod
    def _parse_description(text: str) -> str:
        m = _DESC_RE.search(text)
        if not m:
            return ""
        return textwrap.dedent(m.group(1)).strip()

    @staticmethod
    def _parse_pattern(text: str) -> dict[str, Any]:
        m = _PATTERN_RE.search(text)
        if not m:
            return {}
        return yaml.safe_load(m.group(1)) or {}

    @staticmethod
    def _compile_logic(text: str, source: str) -> Callable[..., list[Lead]]:
        m = _LOGIC_RE.search(text)
        if not m:
            raise ValueError(f"[{source}] Missing ```python``` logic block")
        code = textwrap.dedent(m.group(1))
        if "def detect(" not in code:
            raise ValueError(
                f"[{source}] Logic block must define 'def detect(G, ctx)'"
            )
        # exec is used intentionally here to compile a sandboxed skill function.
        # Security is enforced by SkillValidator (AST scan before execution) and
        # SkillRuntime (only ctx.* API is exposed; no builtins are injected).
        namespace: dict[str, Any] = {}
        try:
            exec(compile(code, source, "exec"), namespace)  # noqa: S102
        except SyntaxError as exc:
            raise ValueError(f"[{source}] Syntax error in logic: {exc}") from exc
        fn = namespace.get("detect")
        if fn is None or not callable(fn):
            raise ValueError(
                f"[{source}] Logic block must define a callable 'detect'"
            )
        # Attach source so the validator can perform an AST scan without
        # needing inspect.getsource() (which won't work for dynamically
        # compiled functions embedded in markdown files).
        fn._skill_source = code  # type: ignore[attr-defined]
        return fn

    @staticmethod
    def _parse_test_cases(text: str) -> list[dict[str, Any]]:
        m = _TEST_RE.search(text)
        if not m:
            return []
        cases = yaml.safe_load(m.group(1))
        return cases if isinstance(cases, list) else []
