from app.agents.data_analyst import DataAnalystAgent
from app.agents.fraud_pattern_analyst import FraudPatternAnalystAgent
from app.agents.guardians.data_reviewer import DataReviewerAgent
from app.agents.guardians.fraud_pattern_reviewer import (
    FraudPatternReviewerAgent,
)
from app.audit.ledger import AuditLedger
from app.fraud.evidence_engine import EvidenceEngine
from app.fraud.graph_detector import FraudGraphDetector
from app.fraud.pattern_engine import FraudPatternEngine
from app.investigation.case_builder import InvestigationCaseBuilder
from app.data_quality.validator import DataQualityValidator
from app.security.operation_registry import OperationRegistry
from app.security.secure_executor import SecureExecutor


class FraudShieldOrchestrator:

    def __init__(self, audit_path: str = "audit_ledger.jsonl"):
        self.data_analyst = DataAnalystAgent()
        self.data_reviewer = DataReviewerAgent()

        self.pattern_engine = FraudPatternEngine()
        self.graph_detector = FraudGraphDetector()
        self.evidence_engine = EvidenceEngine()
        self.case_builder = InvestigationCaseBuilder()
        self.data_quality_validator = DataQualityValidator()
        self.fraud_pattern_analyst = FraudPatternAnalystAgent()
        self.fraud_pattern_reviewer = FraudPatternReviewerAgent()

        self.audit_ledger = AuditLedger(audit_path)

        self.operation_registry = OperationRegistry()
        def fraud_pattern_analysis_operation(
            analysis_task_id: str,
            findings: list[dict],
        ):
            return self.fraud_pattern_analyst.analyze(
                task_id=analysis_task_id,
                findings=findings,
            )

        self._fraud_pattern_analysis_operation = (
            fraud_pattern_analysis_operation
        )
        self.operation_registry.register(
            action="READ",
            resource="PATTERN_FINDINGS",
            operation=self._fraud_pattern_analysis_operation,
        )
        self.secure_executor = SecureExecutor(
            operation_registry=self.operation_registry,
            audit_path=f"{audit_path}.security.jsonl",
        )

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

        if (
            review.status != "PASS"
            or not review.approved_for_next_stage
        ):
            return {
                "analysis": analysis,
                "review": review,
                "data_quality": None,
                "pattern_analysis": None,
                "pattern_review": None,
                "pattern_findings": [],
                "graph_analysis": None,
                "evidence_assessment": None,
                "investigation_case": None,
            }

        # STAGE 3 — DATA QUALITY GATE

        transactions = data.get("transactions", [])

        data_quality = self.data_quality_validator.validate(
            transactions
        )

        self.audit_ledger.record(
            agent_id="DATA_QUALITY_001",
            agent_role="Data Quality Validator",
            task_id=task_id,
            action="DATA_QUALITY_VALIDATION",
            status=data_quality["status"],
            confidence=(
                data_quality["quality_score"] / 100
            ),
            data=data_quality,
        )

        if data_quality["status"] == "REJECT":
            return {
                "analysis": analysis,
                "review": review,
                "data_quality": data_quality,
                "pattern_analysis": None,
                "pattern_review": None,
                "pattern_findings": [],
                "graph_analysis": None,
                "evidence_assessment": None,
                "investigation_case": None,
            }

        # STAGE 3B — SUPPLEMENTARY GRAPH ANALYSIS
        # Graph signals are audited separately and never alter risk scoring.
        graph_analysis = self.graph_detector.analyze(transactions)

        self.audit_ledger.record(
            agent_id="FRAUD_GRAPH_DETECTOR_001",
            agent_role="Relationship Graph Analyzer",
            task_id=task_id,
            action="GRAPH_ANALYSIS",
            status="PASS",
            confidence=1.0,
            data=graph_analysis,
        )

        # STAGE 4 — FRAUD PATTERN DETECTION

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

        # STAGE 5 — EVIDENCE CORRELATION

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

        # STAGE 6 — INVESTIGATION CASE BUILDER

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

        # STAGE 7 — FRAUD PATTERN ANALYST

        decision, pattern_analysis = self.secure_executor.execute(
            agent_id=self.fraud_pattern_analyst.agent_id,
            action="READ",
            resource="PATTERN_FINDINGS",
            operation=self._fraud_pattern_analysis_operation,
            task_id=task_id,
            analysis_task_id=task_id,
            findings=pattern_findings,
        )

        if not decision.allowed or pattern_analysis is None:
            raise PermissionError(
                "Fraud pattern analysis was denied by security policy."
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

        # STAGE 8 — FRAUD PATTERN GUARDIAN

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

        if (
            pattern_review.status != "PASS"
            or not pattern_review.approved_for_next_stage
        ):
            return {
                "analysis": analysis,
                "review": review,
                "pattern_analysis": pattern_analysis,
                "pattern_review": pattern_review,
                "pattern_findings": pattern_findings,
                "graph_analysis": graph_analysis,
                "evidence_assessment": evidence_assessment,
                "investigation_case": None,
            }

        # STAGE 9 — PLAIN-LANGUAGE EXPLANATIONS
        # Runs after all gatekeeping stages. Never blocks decisions.

        from app.explainability.plain_language import (
            explain_findings,
            build_executive_summary,
        )

        explained_findings = explain_findings(pattern_findings)
        executive_summary = build_executive_summary(
            evidence_assessment, explained_findings
        )

        return {
            "analysis": analysis,
            "review": review,
            "data_quality": data_quality,
            "pattern_analysis": pattern_analysis,
            "pattern_review": pattern_review,
            "pattern_findings": explained_findings,
            "graph_analysis": graph_analysis,
            "evidence_assessment": evidence_assessment,
            "investigation_case": investigation_case,
            "executive_summary": executive_summary,
        }
