from app.models.agent_contract import AgentMessage


class FraudPatternAnalystAgent:

    agent_id = "FRAUD_PATTERN_ANALYST_001"
    agent_role = "Fraud Pattern Analyst"

    def analyze(
        self,
        task_id: str,
        findings: list[dict],
    ) -> AgentMessage:

        if not findings:
            return AgentMessage(
                agent_id=self.agent_id,
                agent_role=self.agent_role,
                task_id=task_id,
                status="PASS",
                confidence=0.90,
                findings=[
                    "No configured suspicious patterns detected."
                ],
                evidence=[],
                approved_for_next_stage=True,
            )

        pattern_names = [
            finding["pattern"]
            for finding in findings
        ]

        evidence = [
            {
                "type": "deterministic_pattern_detection",
                "pattern": finding["pattern"],
                "details": finding,
            }
            for finding in findings
        ]

        return AgentMessage(
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            task_id=task_id,
            status="CONDITIONAL_PASS",
            confidence=0.95,
            findings=[
                f"Detected {len(findings)} suspicious pattern(s).",
                f"Patterns: {', '.join(pattern_names)}",
            ],
            evidence=evidence,
            issues=[
                "Pattern detection indicates suspicious activity "
                "but does not independently establish fraud."
            ],
            approved_for_next_stage=True,
            required_actions=[
                "Perform independent guardian review.",
                "Correlate with additional fraud signals.",
            ],
        )
