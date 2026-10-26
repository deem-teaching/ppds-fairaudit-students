"""Fairness auditing utilities for pipeline evaluation."""

from .matcher import PredictionMatcher
from .auditor import FairnessAuditor, AuditReport, AttributeAuditResult
from .groups import UnprivilegedGroups

__all__ = [
    "PredictionMatcher",
    "FairnessAuditor",
    "AuditReport",
    "AttributeAuditResult",
    "UnprivilegedGroups",
]
