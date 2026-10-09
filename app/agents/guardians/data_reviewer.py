from app.models.agent_contract import AgentMessage


class DataReviewerAgent:

    agent_id = "DATA_REVIEWER_001"
    agent_role = "Data Analyst Guardian"

    def review(self, result: AgentMessage) -> AgentMessage:

        issues = []

        if result.confidence < 0.70:
            issues.append(
                "Agent confidence is below required threshold."
            )

        if not result.evidence:
            issues.append(
                "No supporting evidence provided."
            )

        if issues:
            return AgentMessage(
                agent_id=self.agent_id,
                agent_role=self.agent_role,
                task_id=result.task_id,
                status="REJECT",
                confidence=0.95,
                issues=issues,
                critical_objections=issues,
                approved_for_next_stage=False,
                required_actions=[
                    "Data Analyst must provide stronger evidence."
                ],
            )

        return AgentMessage(
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            task_id=result.task_id,
            status="PASS",
            confidence=0.95,
            findings=[
                "Data Analyst output passed independent review."
            ],
            evidence=[
                {
                    "type": "guardian_review",
                    "reviewed_agent": result.agent_id,
                }
            ],
            approved_for_next_stage=True,
        )
