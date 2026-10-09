from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    agent_id: str
    agent_role: str
    task_id: str

    status: Literal[
        "PASS",
        "CONDITIONAL_PASS",
        "REJECT",
        "ERROR",
    ]

    confidence: float = Field(ge=0.0, le=1.0)

    findings: list[str] = []
    evidence: list[dict[str, Any]] = []

    issues: list[str] = []
    critical_objections: list[str] = []

    approved_for_next_stage: bool = False

    required_actions: list[str] = []

    metadata: dict[str, Any] = {}