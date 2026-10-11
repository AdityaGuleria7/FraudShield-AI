from typing import Any


class ThresholdPolicyGate:
    def __init__(
        self,
        minimum_threshold: float = 0.30,
        maximum_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= minimum_threshold <= maximum_threshold <= 1.0:
            raise ValueError(
                "Threshold policy bounds must satisfy "
                "0.0 <= minimum_threshold <= maximum_threshold <= 1.0"
            )

        self.minimum_threshold = minimum_threshold
        self.maximum_threshold = maximum_threshold

    def validate(
        self,
        threshold: float,
        validation: dict[str, Any],
        optimization_basis: str,
        false_positive_cost: float | None = None,
        false_negative_cost: float | None = None,
    ) -> dict[str, Any]:
        issues = []

        if optimization_basis != "validation_out_of_sample":
            issues.append("optimization_provenance")

        if (
            false_positive_cost is None
            or false_negative_cost is None
            or false_positive_cost < 0
            or false_negative_cost < 0
        ):
            issues.append("cost_configuration")

        if not self.minimum_threshold <= threshold <= self.maximum_threshold:
            issues.append("threshold_range")

        if validation.get("status") != "PASS":
            issues.append("model_validation_status")

        if validation.get("approved") is not True:
            issues.append("model_validation_approval")

        if issues:
            return {
                "status": "REJECT",
                "approved": False,
                "issues": issues,
            }

        return {
            "status": "PASS",
            "approved": True,
            "issues": [],
            "threshold": threshold,
            "minimum_threshold": self.minimum_threshold,
            "maximum_threshold": self.maximum_threshold,
            "optimization_basis": optimization_basis,
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
        }
