from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentMessage(BaseModel):
    agent_id: str = Field(min_length=1)
    agent_role: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

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

    @field_validator("agent_id", "agent_role", "task_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_approval_consistency(self):
        if self.status in {"PASS", "CONDITIONAL_PASS"}:
            if not self.approved_for_next_stage:
                raise ValueError(
                    "Successful agent status requires downstream approval"
                )

        if self.status in {"REJECT", "ERROR"}:
            if self.approved_for_next_stage:
                raise ValueError(
                    "Rejected or errored agent cannot approve downstream processing"
                )

        return self
