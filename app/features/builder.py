from typing import Any

from app.features.baseline import BaselineFeatureBuilder
from app.features.behavioral import BehavioralFeatureBuilder
from app.features.category import CategoryFeatureBuilder
from app.features.relationships import RelationshipFeatureBuilder
from app.features.temporal import TemporalFeatureBuilder
from app.features.velocity import VelocityFeatureBuilder


class UnifiedFeatureBuilder:
    def __init__(self) -> None:
        self.behavioral_builder = BehavioralFeatureBuilder()
        self.velocity_builder = VelocityFeatureBuilder()
        self.relationship_builder = RelationshipFeatureBuilder()
        self.baseline_builder = BaselineFeatureBuilder()
        self.temporal_builder = TemporalFeatureBuilder()
        self.category_builder = CategoryFeatureBuilder()

    def build(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        behavioral = self.behavioral_builder.build(transactions)
        velocity = self.velocity_builder.build(transactions)
        relationships = self.relationship_builder.build(transactions)
        baseline = self.baseline_builder.build(transactions)
        temporal = self.temporal_builder.build(transactions)
        category = self.category_builder.build(transactions)

        features: dict[str, dict[str, Any]] = {}

        for transaction in transactions:
            transaction_id = transaction["transaction_id"]

            features[transaction_id] = {
                **behavioral[transaction_id],
                **velocity[transaction_id],
                **relationships[transaction_id],
                **baseline[transaction_id],
                **temporal[transaction_id],
                **category[transaction_id],
            }

        return features
