"""SkillValidator — static AST security scanner + test-case runner.

Security rules
--------------
The following names are **forbidden** inside a skill's Python logic block:

* ``import`` / ``__import__``
* ``exec`` / ``eval``
* ``open`` / ``compile``
* ``os`` / ``sys`` / ``subprocess``
* ``__builtins__`` / ``globals`` / ``locals``
* ``getattr`` / ``setattr`` / ``delattr`` / ``hasattr``

If any of these appear the skill is rejected before it can be loaded.

Test cases
----------
Each test case in the YAML block is executed by building a tiny in-memory
graph and calling the skill's ``detect`` function.  The test passes if the
number of returned leads matches ``expected_leads``.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import networkx as nx

from engine.skill_context import SkillContext
from models import Lead, Skill

logger = logging.getLogger(__name__)

# Names that are never allowed inside skill code
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "import",
        "__import__",
        "exec",
        "eval",
        "open",
        "compile",
        "os",
        "sys",
        "subprocess",
        "__builtins__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)


class ValidationResult:
    """Result of validating a single skill."""

    def __init__(self, passed: bool, errors: list[str]) -> None:
        self.passed = passed
        self.errors = errors

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<ValidationResult {status} errors={self.errors}>"


class SkillValidator:
    """Validate a :class:`~models.Skill` via AST scanning and test execution."""

    def __init__(self, ctx: SkillContext | None = None) -> None:
        self._ctx = ctx or SkillContext()

    def validate(self, skill: Skill) -> ValidationResult:
        """Run all validation checks on *skill*.

        Parameters
        ----------
        skill:
            A compiled skill (``detect_fn`` must be set).

        Returns
        -------
        ValidationResult
        """
        errors: list[str] = []

        # 1. AST safety scan
        ast_errors = self._ast_scan(skill)
        errors.extend(ast_errors)

        # 2. Test cases
        if not ast_errors:
            test_errors = self._run_test_cases(skill)
            errors.extend(test_errors)

        passed = len(errors) == 0
        if passed:
            logger.info("Skill '%s' validation PASSED.", skill.id)
        else:
            logger.warning(
                "Skill '%s' validation FAILED: %s", skill.id, errors
            )
        return ValidationResult(passed, errors)

    # ── AST scanning ───────────────────────────────────────────────────────

    @staticmethod
    def _ast_scan(skill: Skill) -> list[str]:
        """Return a list of security violation descriptions."""
        # Prefer the source attached by SkillLoader over inspect.getsource(),
        # which cannot reliably retrieve source from dynamically-compiled fns.
        source: str | None = getattr(skill.detect_fn, "_skill_source", None)
        if source is None:
            import inspect  # local import — allowed in engine code

            try:
                source = inspect.getsource(skill.detect_fn)
            except (OSError, TypeError):
                # Source unavailable; skip AST check
                return []

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc}"]

        errors: list[str] = []
        for node in ast.walk(tree):
            # Forbidden imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                errors.append("Forbidden: import statement found")
            # Forbidden function calls / names
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                errors.append(f"Forbidden name: '{node.id}'")
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
                errors.append(f"Forbidden attribute access: '.{node.attr}'")
            # Forbidden string that looks like __dunder__
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("__") and node.value.endswith("__"):
                    errors.append(
                        f"Forbidden dunder string literal: '{node.value}'"
                    )
        return errors

    # ── test cases ─────────────────────────────────────────────────────────

    def _run_test_cases(self, skill: Skill) -> list[str]:
        errors: list[str] = []
        for i, case in enumerate(skill.test_cases):
            try:
                err = self._run_single_case(skill, case, index=i)
                if err:
                    errors.append(err)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Test case {i} raised exception: {exc}")
        return errors

    def _run_single_case(
        self, skill: Skill, case: dict[str, Any], index: int
    ) -> str | None:
        """Run one test case; return an error string or ``None`` on success."""
        G = self._build_test_graph(case.get("graph", {}))
        expected = case.get("expected_leads", 0)

        desc = case.get("description", f"case {index}")
        try:
            result = skill.detect_fn(G, self._ctx)
        except Exception as exc:  # noqa: BLE001
            return f"Test case {index} ({desc!r}): exception: {exc}"

        actual = len(result) if result else 0
        if actual != expected:
            desc = case.get("description", f"case {index}")
            return (
                f"Test case {index!r} ({desc!r}): "
                f"expected {expected} lead(s), got {actual}"
            )
        return None

    @staticmethod
    def _build_test_graph(graph_spec: dict[str, Any]) -> nx.MultiDiGraph:
        """Build a tiny graph from a test-case ``graph`` spec."""
        G: nx.MultiDiGraph = nx.MultiDiGraph()
        for node in graph_spec.get("nodes", []):
            node_id = str(node.get("id", ""))
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})
        for edge in graph_spec.get("edges", []):
            src = str(edge.get("source", edge.get("from", "")))
            tgt = str(edge.get("target", edge.get("to", "")))
            attrs = {
                k: v
                for k, v in edge.items()
                if k not in ("source", "target", "from", "to")
            }
            G.add_edge(src, tgt, **attrs)
        return G
