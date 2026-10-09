from app.agents.data_analyst import DataAnalystAgent
from app.agents.fraud_pattern_analyst import FraudPatternAnalystAgent
from app.agents.guardians.data_reviewer import DataReviewerAgent
from app.agents.guardians.fraud_pattern_reviewer import (
    FraudPatternReviewerAgent,
)
from app.audit.ledger import AuditLedger
from app.fraud.evidence_engine import EvidenceEngine
from app.fraud.pattern_engine import FraudPatternEngine
from app.investigation.case_builder import InvestigationCaseBuilder


class FraudShieldOrchestrator:

    def __init__(self, audit_path: str = "audit_ledger.jsonl"):
        self.data_analyst = DataAnalystAgent()
        self.data_reviewer = DataReviewerAgent()

        self.pattern_engine = FraudPatternEngine()
        self.evidence_engine = EvidenceEngine()
        self.case_builder = InvestigationCaseBuilder()
        self.fraud_pattern_analyst = FraudPatternAnalystAgent()
        self.fraud_pattern_reviewer = FraudPatternReviewerAgent()

        self.audit_ledger = AuditLedger(audit_path)

    def run(self, task_id: str, data: dict):

        # STAGE 1 — DATA ANALYSIS

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

        # STAGE 2 — DATA ANALYST GUARDIAN

        review = self.data_reviewer.review(analysis)

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
                "approved_for_next_stage": (
                    review.approved_for_next_stage
                ),
            },
        )

        # STAGE 3 — FRAUD PATTERN DETECTION

        transactions = data.get("transactions", [])

        velocity_findings = self.pattern_engine.detect_velocity(
            transactions
        )

        new_device_findings = (
            self.pattern_engine.detect_new_device_high_value(
                transactions
            )
        )

        impossible_travel_findings = (
            self.pattern_engine.detect_impossible_travel(
                transactions
            )
        )

        account_takeover_findings = (
            self.pattern_engine.detect_account_takeover(
                transactions
            )
        )

        pattern_findings = (
            velocity_findings
            + new_device_findings
            + impossible_travel_findings
            + account_takeover_findings
        )

        # STAGE 4 — EVIDENCE CORRELATION

        evidence_assessment = self.evidence_engine.assess(
            pattern_findings
        )

        self.audit_ledger.record(
            agent_id="EVIDENCE_ENGINE_001",
            agent_role="Evidence Correlation Engine",
            task_id=task_id,
            action="EVIDENCE_CORRELATION",
            status="PASS",
            confidence=1.0,
            data=evidence_assessment,
        )

        # STAGE 5 — INVESTIGATION CASE BUILDER

        investigation_case = self.case_builder.build(
            task_id=task_id,
            evidence_assessment=evidence_assessment,
            findings=pattern_findings,
        )

        self.audit_ledger.record(
            agent_id="INVESTIGATION_CASE_BUILDER_001",
            agent_role="Investigation Case Builder",
            task_id=task_id,
            action="INVESTIGATION_CASE_BUILT",
            status="PASS",
            confidence=1.0,
            data={
                "case_id": investigation_case["case_id"],
                "status": investigation_case["status"],
                "risk_score": investigation_case["risk_score"],
                "risk_level": investigation_case["risk_level"],
                "decision": investigation_case["decision"],
                "patterns": investigation_case["patterns"],
            },
        )

        # STAGE 6 — FRAUD PATTERN ANALYST

        pattern_analysis = self.fraud_pattern_analyst.analyze(
            task_id=task_id,
            findings=pattern_findings,
        )

        self.audit_ledger.record(
            agent_id=pattern_analysis.agent_id,
            agent_role=pattern_analysis.agent_role,
            task_id=pattern_analysis.task_id,
            action="FRAUD_PATTERN_ANALYSIS",
            status=pattern_analysis.status,
            confidence=pattern_analysis.confidence,
            data={
                "findings": pattern_analysis.findings,
                "evidence": pattern_analysis.evidence,
                "issues": pattern_analysis.issues,
                "required_actions": (
                    pattern_analysis.required_actions
                ),
            },
        )

        # STAGE 5 — FRAUD PATTERN GUARDIAN

        pattern_review = self.fraud_pattern_reviewer.review(
            pattern_analysis
        )

        self.audit_ledger.record(
            agent_id=pattern_review.agent_id,
            agent_role=pattern_review.agent_role,
            task_id=pattern_review.task_id,
            action="FRAUD_PATTERN_GUARDIAN_REVIEW",
            status=pattern_review.status,
            confidence=pattern_review.confidence,
            data={
                "findings": pattern_review.findings,
                "evidence": pattern_review.evidence,
                "issues": pattern_review.issues,
                "critical_objections": (
                    pattern_review.critical_objections
                ),
                "approved_for_next_stage": (
                    pattern_review.approved_for_next_stage
                ),
            },
        )

        return {
            "analysis": analysis,
            "review": review,
            "pattern_analysis": pattern_analysis,
            "pattern_review": pattern_review,
            "pattern_findings": pattern_findings,
            "evidence_assessment": evidence_assessment,
            "investigation_case": investigation_case,
        }
