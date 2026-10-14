from app.agents.data_analyst import DataAnalystAgent


def test_reports_transaction_count():
    agent = DataAnalystAgent()

    result = agent.analyze(
        task_id="TEST-DATA-001",
        data={
            "transactions": [
                {"transaction_id": "TX-001"},
                {"transaction_id": "TX-002"},
                {"transaction_id": "TX-003"},
            ]
        },
    )

    assert result.status == "PASS"
    assert result.approved_for_next_stage is True
    assert any("3 transactions" in finding for finding in result.findings)


def test_supports_generic_dictionary_input():
    agent = DataAnalystAgent()

    result = agent.analyze(
        task_id="TEST-DATA-002",
        data={
            "transaction_id": "TX-001",
            "amount": 1500,
            "country": "IN",
        },
    )

    assert result.status == "PASS"
    assert result.approved_for_next_stage is True


def test_rejects_empty_input():
    agent = DataAnalystAgent()

    result = agent.analyze(task_id="TEST-DATA-003", data={})

    assert result.status == "REJECT"
    assert result.approved_for_next_stage is False
    assert "No data supplied" in result.issues
