"""Data models for the compliance graph detection system."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Lead:
    """A detection result produced by a Skill."""

    skill_id: str
    title: str
    severity: Severity
    score: float  # 0-100
    entities: list[str]
    evidence: list[str]
    actions: list[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError(f"score must be in [0, 100], got {self.score}")
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)


@dataclass
class Skill:
    """A loaded and compiled detection skill."""

    id: str
    version: str
    name: str
    description: str
    tags: list[str]
    enabled: bool
    triggers: dict[str, Any]
    detect_fn: Callable[..., list[Lead]]
    test_cases: list[dict[str, Any]]
