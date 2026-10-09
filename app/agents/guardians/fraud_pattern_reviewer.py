from app.models.agent_contract import AgentMessage


class FraudPatternReviewerAgent:

    agent_id = "FRAUD_PATTERN_REVIEWER_001"
    agent_role = "Fraud Pattern Analyst Guardian"

    def review(self, result: AgentMessage) -> AgentMessage:

        if not result.evidence:
            return AgentMessage(
                agent_id=self.agent_id,
                agent_role=self.agent_role,
                task_id=result.task_id,
                status="REJECT",
                confidence=0.95,
                issues=[
                    "Fraud Pattern Analyst provided no evidence."
                ],
                critical_objections=[
                    "Suspicious finding cannot be validated "
                    "without supporting evidence."
                ],
                approved_for_next_stage=False,
                required_actions=[
                    "Provide deterministic pattern evidence."
                ],
            )

        invalid_evidence = [
            evidence
            for evidence in result.evidence
            if evidence.get("type")
            != "deterministic_pattern_detection"
        ]

        if invalid_evidence:
            return AgentMessage(
                agent_id=self.agent_id,
                agent_role=self.agent_role,
                task_id=result.task_id,
                status="REJECT",
                confidence=0.95,
                issues=[
                    "Unexpected evidence type detected."
                ],
                critical_objections=[
                    "Evidence source does not match the approved "
                    "deterministic pattern engine."
                ],
                approved_for_next_stage=False,
            )

        return AgentMessage(
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            task_id=result.task_id,
            status="PASS",
            confidence=0.95,
            findings=[
                "Fraud Pattern Analyst evidence passed "
                "independent guardian review."
            ],
            evidence=[
                {
                    "type": "guardian_validation",
                    "reviewed_agent": result.agent_id,
                    "validated_evidence_count": len(
                        result.evidence
                    ),
                }
            ],
            approved_for_next_stage=True,
        )
