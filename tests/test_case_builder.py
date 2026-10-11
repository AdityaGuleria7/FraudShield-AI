from app.investigation.case_builder import InvestigationCaseBuilder


def test_builds_open_investigation_case():

    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 47,
        "risk_level": "MEDIUM",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "patterns": [
            "ACCOUNT_TAKEOVER",
            "NEW_DEVICE_HIGH_VALUE",
        ],
        "overlapping_transactions": {
            "TX6000": [
                "ACCOUNT_TAKEOVER",
                "NEW_DEVICE_HIGH_VALUE",
            ]
        },
        "evidence_count": 2,
        "evidence_quality": "STRONG",
        "provenance": [
            {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "1",
            },
            {
                "detector": "new_device_high_value",
                "method": "deterministic_rule",
                "version": "1",
            },
        ],
    }

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "transaction_id": "TX6000",
        },
        {
            "pattern": "NEW_DEVICE_HIGH_VALUE",
            "transaction_id": "TX6000",
        },
    ]

    case = builder.build(
        task_id="CASE-TEST-001",
        evidence_assessment=assessment,
        findings=findings,
    )

    assert case["case_id"].startswith("CASE-")
    assert len(case["case_id"]) == 17
    assert case["task_id"] == "CASE-TEST-001"
    assert case["status"] == "OPEN_FOR_INVESTIGATION"
    assert case["risk_score"] == 47
    assert case["risk_level"] == "MEDIUM"
    assert case["decision"] == "REQUIRES_FURTHER_INVESTIGATION"
    assert case["patterns"] == [
        "ACCOUNT_TAKEOVER",
        "NEW_DEVICE_HIGH_VALUE",
    ]
    assert case["independent_patterns"] == []
    assert case["overlapping_patterns"] == [
        "ACCOUNT_TAKEOVER",
        "NEW_DEVICE_HIGH_VALUE",
    ]
    assert case["overlapping_transactions"] == {
        "TX6000": [
            "ACCOUNT_TAKEOVER",
            "NEW_DEVICE_HIGH_VALUE",
        ]
    }
    assert case["evidence_count"] == 2
    assert case["evidence_quality"] == "STRONG"
    assert case["findings"] == findings
    assert case["required_actions"] == [
        "Investigate supporting evidence."
    ]


def test_builds_no_action_case_when_risk_is_zero():

    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 0,
        "risk_level": "LOW",
        "decision": "NO_SUSPICIOUS_ACTIVITY",
        "patterns": [],
        "overlapping_transactions": {},
        "evidence_count": 0,
        "evidence_quality": "NONE",
    }

    case = builder.build(
        task_id="CASE-CLEAN-001",
        evidence_assessment=assessment,
        findings=[],
    )

    assert case["case_id"].startswith("CASE-")
    assert case["status"] == "NO_ACTION_REQUIRED"
    assert case["risk_score"] == 0
    assert case["risk_level"] == "LOW"
    assert case["patterns"] == []
    assert case["independent_patterns"] == []
    assert case["overlapping_patterns"] == []
    assert case["overlapping_transactions"] == {}
    assert case["required_actions"] == []

def test_investigation_case_preserves_detector_provenance():
    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 35,
        "risk_level": "MEDIUM",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "patterns": ["ACCOUNT_TAKEOVER"],
        "overlapping_transactions": {},
        "evidence_count": 1,
        "evidence_quality": "STRONG",
        "provenance": [
            {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "1",
            }
        ],
    }

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "transaction_id": "TX7000",
        }
    ]

    case = builder.build(
        task_id="CASE-PROV-001",
        evidence_assessment=assessment,
        findings=findings,
    )

    assert case["provenance"][0]["detector"] == "account_takeover"
    assert case["provenance"][0]["method"] == "deterministic_rule"
    assert case["provenance"][0]["version"] == "1"

def test_investigation_case_preserves_multiple_detector_provenance_records():
    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 60,
        "risk_level": "HIGH",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "patterns": [
            "ACCOUNT_TAKEOVER",
            "IMPOSSIBLE_TRAVEL",
        ],
        "overlapping_transactions": {},
        "evidence_count": 2,
        "evidence_quality": "STRONG",
        "provenance": [
            {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "1",
            },
            {
                "detector": "impossible_travel",
                "method": "deterministic_rule",
                "version": "1",
            },
        ],
    }

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "transaction_id": "TX8000",
        },
        {
            "pattern": "IMPOSSIBLE_TRAVEL",
            "transaction_id": "TX8001",
        },
    ]

    case = builder.build(
        task_id="CASE-PROV-002",
        evidence_assessment=assessment,
        findings=findings,
    )

    provenance_by_detector = {
        item["detector"]: item
        for item in case["provenance"]
    }

    assert len(case["provenance"]) == 2
    assert provenance_by_detector["account_takeover"]["method"] == "deterministic_rule"
    assert provenance_by_detector["account_takeover"]["version"] == "1"
    assert provenance_by_detector["impossible_travel"]["method"] == "deterministic_rule"
    assert provenance_by_detector["impossible_travel"]["version"] == "1"

def test_investigation_case_rejects_missing_provenance_for_risky_case():
    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 35,
        "risk_level": "MEDIUM",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "patterns": ["ACCOUNT_TAKEOVER"],
        "overlapping_transactions": {},
        "evidence_count": 1,
        "evidence_quality": "STRONG",
    }

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "transaction_id": "TX9000",
        }
    ]

    try:
        builder.build(
            task_id="CASE-INTEGRITY-001",
            evidence_assessment=assessment,
            findings=findings,
        )
    except ValueError as exc:
        assert "provenance" in str(exc).lower()
    else:
        raise AssertionError(
            "Risky investigation case must reject missing provenance"
        )

def test_investigation_case_rejects_evidence_count_mismatch():
    builder = InvestigationCaseBuilder()

    assessment = {
        "risk_score": 35,
        "risk_level": "MEDIUM",
        "decision": "REQUIRES_FURTHER_INVESTIGATION",
        "patterns": ["ACCOUNT_TAKEOVER"],
        "overlapping_transactions": {},
        "evidence_count": 2,
        "evidence_quality": "STRONG",
        "provenance": [
            {
                "detector": "account_takeover",
                "method": "deterministic_rule",
                "version": "1",
            }
        ],
    }

    findings = [
        {
            "pattern": "ACCOUNT_TAKEOVER",
            "transaction_id": "TX10000",
        }
    ]

    try:
        builder.build(
            task_id="CASE-INTEGRITY-002",
            evidence_assessment=assessment,
            findings=findings,
        )
    except ValueError as exc:
        assert "evidence count" in str(exc).lower()
    else:
        raise AssertionError(
            "Investigation case must reject mismatched evidence count"
        )

