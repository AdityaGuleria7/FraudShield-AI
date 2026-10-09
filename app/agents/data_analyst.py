from app.models.agent_contract import AgentMessage


class DataAnalystAgent:

    agent_id = "DATA_ANALYST_001"
    agent_role = "Data Analyst"

    def analyze(self, task_id: str, data: dict) -> AgentMessage:

        if not data:
            return AgentMessage(
                agent_id=self.agent_id,
                agent_role=self.agent_role,
                task_id=task_id,
                status="REJECT",
                confidence=1.0,
                issues=["No data supplied"],
                critical_objections=[
                    "Cannot perform analysis without data"
                ],
                approved_for_next_stage=False,
                required_actions=[
                    "Provide valid input data"
                ],
            )

        return AgentMessage(
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            task_id=task_id,
            status="PASS",
            confidence=0.90,
            findings=[
                f"Received {len(data)} data fields for analysis."
            ],
            evidence=[
                {
                    "type": "input_validation",
                    "source": self.agent_id,
                }
            ],
            approved_for_next_stage=True,
        )
