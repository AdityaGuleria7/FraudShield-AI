from app.agents.data_analyst import DataAnalystAgent
from app.agents.guardians.data_reviewer import DataReviewerAgent
from app.audit.ledger import AuditLedger


class FraudShieldOrchestrator:

    def __init__(self, audit_path: str = "audit_ledger.jsonl"):
        self.data_analyst = DataAnalystAgent()
        self.data_reviewer = DataReviewerAgent()
        self.audit_ledger = AuditLedger(audit_path)

    def run(self, task_id: str, data: dict):

        analysis = self.data_analyst.analyze(
            task_id=task_id,
            data=data,
        )

        self.audit_ledger.record(
            agent_id=analysis.agent_id,
            agent_role=analysis.agent_role,
            task_id=analysis.task_id,
            action="DATA_ANALYSIS",
            status=analysis.status,
            confidence=analysis.confidence,
            data={
                "findings": analysis.findings,
                "evidence": analysis.evidence,
                "issues": analysis.issues,
            },
        )

        review = self.data_reviewer.review(
            analysis
        )

        self.audit_ledger.record(
            agent_id=review.agent_id,
            agent_role=review.agent_role,
            task_id=review.task_id,
            action="GUARDIAN_REVIEW",
            status=review.status,
            confidence=review.confidence,
            data={
                "findings": review.findings,
                "evidence": review.evidence,
                "issues": review.issues,
                "critical_objections": review.critical_objections,
                "approved_for_next_stage": review.approved_for_next_stage,
            },
        )

        return {
            "analysis": analysis,
            "review": review,
        }
