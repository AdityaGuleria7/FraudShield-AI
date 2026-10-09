from dataclasses import dataclass


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    role: str
    risk_level: str


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    agent_id: str
    action: str
    resource: str
    reason: str


class PolicyEngine:

    AGENTS = {
        "DATA_ANALYST_001": AgentIdentity(
            agent_id="DATA_ANALYST_001",
            role="Data Analyst",
            risk_level="LOW",
        ),
        "DATA_REVIEWER_001": AgentIdentity(
            agent_id="DATA_REVIEWER_001",
            role="Data Analyst Guardian",
            risk_level="MEDIUM",
        ),
    }

    # Explicit least-privilege matrix:
    # agent -> action -> resources
    PERMISSIONS = {
        "DATA_ANALYST_001": {
            "READ": {
                "TRANSACTION_DATA",
                "AGGREGATED_DATA",
            },
        },
        "DATA_REVIEWER_001": {
            "READ": {
                "TRANSACTION_DATA",
                "ANALYSIS_RESULTS",
            },
        },
    }

    def get_agent(self, agent_id: str) -> AgentIdentity | None:
        return self.AGENTS.get(agent_id)

    def check(
        self,
        agent_id: str,
        action: str,
        resource: str = "UNSPECIFIED",
    ) -> PolicyDecision:

        agent = self.get_agent(agent_id)

        if agent is None:
            return PolicyDecision(
                allowed=False,
                agent_id=agent_id,
                action=action,
                resource=resource,
                reason="Unknown agent. Access denied by default.",
            )

        allowed_resources = self.PERMISSIONS.get(
            agent_id,
            {},
        ).get(action, set())

        if resource in allowed_resources:
            return PolicyDecision(
                allowed=True,
                agent_id=agent_id,
                action=action,
                resource=resource,
                reason="Action and resource are permitted by policy.",
            )

        return PolicyDecision(
            allowed=False,
            agent_id=agent_id,
            action=action,
            resource=resource,
            reason=(
                f"Agent '{agent_id}' is not permitted to "
                f"perform '{action}' on resource '{resource}'."
            ),
        )
