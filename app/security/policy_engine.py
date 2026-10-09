from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    agent_id: str
    action: str
    reason: str


class PolicyEngine:

    POLICIES = {
        "DATA_ANALYST_001": {
            "READ_TRANSACTION_DATA",
            "READ_AGGREGATED_DATA",
        },
        "DATA_REVIEWER_001": {
            "READ_TRANSACTION_DATA",
            "READ_ANALYSIS_RESULTS",
        },
    }

    def check(self, agent_id: str, action: str) -> PolicyDecision:

        allowed_actions = self.POLICIES.get(agent_id, set())

        if action in allowed_actions:
            return PolicyDecision(
                allowed=True,
                agent_id=agent_id,
                action=action,
                reason="Action is permitted by policy.",
            )

        return PolicyDecision(
            allowed=False,
            agent_id=agent_id,
            action=action,
            reason="Action is not permitted for this agent.",
        )
